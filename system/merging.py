# -*- coding: utf-8 -*-
"""
system/merging.py — 線上 merge：以 pool150 全池合成一份 exact dense delta

ADR-0002。三種方法的權重運算從 MoEA-Trainer-delivery 搬入：
  ta         src/task_arithmetic/runtime.py::materialize_task_arithmetic
  ties       src/adamerging/exact_ties.py
  dare_ties_ta  src/dare_ties/runtime.py::materialize_dare_ties
  pico_ta       src/pico/task_arithmetic.py
  lora_lego     src/lego/merger.py
             （含 src/dare_ties/pool50.py 的 mask 與 election，改用 pool150 命名）

只搬純權重運算；producer 的 prepare/infer 生命週期、評測與 run binding 留在
producer。超參數鎖成封板值（見 SEALED），使用者選方法不調參，因此對固定的
pool150 產物是決定性的。

輸出是 `{module}.delta_weight` 的 safetensors，與 system/merged_model.py 驗證的
dense_delta_v1 格式相同；打包成完整 artifact 由 system/merged_model.py 的
write_dense_delta_artifact 負責。
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

from system.adapter_pool import (
    AdapterPool,
    layer_index_from_key,
    module_name_from_key,
    paired_key,
    sha256,
)

#: 封板超參數（MoEA-Trainer-delivery/src/moea_repro/conditions.py）。
#: 使用者只選方法，不調參——這讓產物對固定的 pool150 是決定性的。
SEALED = {
    "ta": {"lambda": 1.0, "reduction": "mean", "tile_rows": 16},
    "ties_only": {"lambda": 0.3, "density": 0.2, "reduction": "disjoint_mean",
                  "tile_rows": 4},
    "dare_ties_ta": {"density": 1.0, "lambda": 0.25, "sign_method": "total",
                     "rescale": True, "tile_rows": 16, "mask_block_rows": 8},
    "pico_ta": {"reduction": "mean", "eps": 1e-12, "tile_rows": 16},
    "lora_lego": {"output_rank": 16, "output_ref_rank": 8, "lego_seed": 0,
                  "n_init": 10, "max_iter": 300, "parameter_reweight": True,
                  "output_reweight": True, "eps": 1e-12, "tile_rows": 16},
}
CANONICAL_SEED = 42
MERGE_METHODS = tuple(SEALED)


def stable_seed(*parts) -> int:
    encoded = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)


def _stacks(pool: AdapterPool, a_key: str, device, torch):
    """把全池同一模組的 lora_A / lora_B 疊成 [adapters, ...] 張量。"""
    b_key = paired_key(a_key)
    a_stack = torch.stack([weights[a_key] for weights in pool.weights]).to(
        device=device, dtype=torch.float32)
    b_stack = torch.stack([weights[b_key] for weights in pool.weights]).to(
        device=device, dtype=torch.float32)
    return a_stack, b_stack


# ---------------------------------------------------------------------------
# Task Arithmetic
# ---------------------------------------------------------------------------
def materialize_task_arithmetic(pool, output_path, *, coefficient, device="cpu",
                                tile_rows=16, output_dtype=None):
    """全池 LoRA delta 的係數加權和（exact，不做稀疏化）。"""
    import torch
    output_dtype = output_dtype or torch.bfloat16
    if tile_rows < 1:
        raise ValueError("tile_rows 必須為正")
    compute_device = torch.device(device)
    tensors = {}
    for a_key in pool.a_keys:
        a_stack, b_stack = _stacks(pool, a_key, compute_device, torch)
        output = torch.empty((b_stack.shape[1], a_stack.shape[2]),
                             dtype=output_dtype, device="cpu")
        for start in range(0, b_stack.shape[1], tile_rows):
            end = min(start + tile_rows, b_stack.shape[1])
            tile = torch.bmm(b_stack[:, start:end], a_stack).sum(dim=0)
            tile.mul_(pool.scaling * float(coefficient))
            output[start:end].copy_(tile.to(device="cpu", dtype=output_dtype))
        tensors[f"{module_name_from_key(a_key)}.delta_weight"] = output.contiguous()
    return _save(tensors, output_path, "exact_task_arithmetic",
                 {"coefficient": float(coefficient)})


# ---------------------------------------------------------------------------
# TIES
# ---------------------------------------------------------------------------
def trim_elect_weighted_sum(task_vectors, thresholds, coefficients, *,
                            reduction="fixed_weight_sum"):
    """全域門檻修剪 → 總和取號選舉 → 加權合併。"""
    import torch
    if task_vectors.ndim != 3:
        raise ValueError("預期 task_vectors=[tasks, rows, columns]")
    task_count = task_vectors.shape[0]
    if tuple(thresholds.shape) != (task_count,):
        raise ValueError("每個任務需要一個全域門檻")
    if tuple(coefficients.shape) != (task_count,):
        raise ValueError("每個任務需要一個係數")
    trimmed = torch.where(
        task_vectors.abs() >= thresholds[:, None, None],
        task_vectors, torch.zeros_like(task_vectors))
    elected = torch.sign(trimmed.sum(dim=0))
    agreement = ((torch.sign(trimmed) == elected.unsqueeze(0))
                 & (elected.unsqueeze(0) != 0))
    selected = torch.where(agreement, trimmed, torch.zeros_like(trimmed))
    merged = torch.einsum("tij,t->ij", selected, coefficients.to(selected.dtype))
    if reduction == "disjoint_mean":
        merged = merged / agreement.sum(dim=0).clamp_min(1)
    elif reduction != "fixed_weight_sum":
        raise ValueError(f"不支援的 TIES reduction：{reduction}")
    return merged, {
        "trimmed_coordinates": int((trimmed != 0).sum().item()),
        "selected_coordinates": int((selected != 0).sum().item()),
        "elected_zero_coordinates": int((elected == 0).sum().item()),
    }


def compute_global_thresholds(pool, density, device, *, progress=False):
    """逐任務算出「保留前 density 比例座標」的全域門檻。"""
    import torch
    if not 0.0 < density <= 1.0:
        raise ValueError("density 必須落在 (0, 1]")
    shapes, total = [], 0
    for a_key in pool.a_keys:
        b_key = paired_key(a_key)
        out_features = pool.weights[0][b_key].shape[0]
        in_features = pool.weights[0][a_key].shape[1]
        shapes.append((a_key, b_key, out_features, in_features))
        total += out_features * in_features
    keep = max(1, int(total * density))
    magnitudes = torch.empty(total, dtype=torch.float32, device=device)
    thresholds = []
    for task_index in range(len(pool.names)):
        offset = 0
        for a_key, b_key, out_features, in_features in shapes:
            a = pool.weights[task_index][a_key].to(device=device, dtype=torch.float32)
            b = pool.weights[task_index][b_key].to(device=device, dtype=torch.float32)
            size = out_features * in_features
            magnitudes[offset:offset + size].copy_(
                (b @ a).mul_(pool.scaling).abs_().reshape(-1))
            offset += size
            del a, b
        threshold = (magnitudes.min() if density == 1.0
                     else torch.kthvalue(magnitudes, total - keep + 1).values)
        thresholds.append(float(threshold.item()))
        if progress and (task_index + 1) % 25 == 0:
            print(f"[merge] 門檻 {task_index + 1}/{len(pool.names)}", flush=True)
    del magnitudes
    return torch.tensor(thresholds, dtype=torch.float32)


def materialize_ties(pool, output_path, *, thresholds, coefficients, tile_rows,
                     reduction="disjoint_mean", device="cpu", output_dtype=None):
    """把修剪選舉後的 exact dense task vector 串流成一份 delta。"""
    import torch
    output_dtype = output_dtype or torch.bfloat16
    compute_device = torch.device(device)
    task_count = len(pool.names)
    threshold_tensor = torch.as_tensor(
        thresholds, dtype=torch.float32, device=compute_device)
    coefficient_tensor = torch.as_tensor(
        coefficients, dtype=torch.float32, device=compute_device)
    if tuple(threshold_tensor.shape) != (task_count,):
        raise ValueError(f"預期 {task_count} 個門檻")
    # 係數可以是全域的 [tasks]（ties_only 用固定 lambda），也可以是逐層的
    # [layers, tasks]（adamerging_pp 用學出來的係數）。
    if coefficient_tensor.ndim == 1:
        if tuple(coefficient_tensor.shape) != (task_count,):
            raise ValueError(f"預期 {task_count} 個係數")
    elif coefficient_tensor.ndim == 2:
        if tuple(coefficient_tensor.shape) != (len(pool.a_keys), task_count):
            raise ValueError(
                f"預期逐層係數 [{len(pool.a_keys)}, {task_count}]")
    else:
        raise ValueError("係數必須是全域 [tasks] 或逐層 [layers, tasks]")

    tensors, reports = {}, []
    for module_order, a_key in enumerate(pool.a_keys):
        a_stack, b_stack = _stacks(pool, a_key, compute_device, torch)
        out_features, in_features = b_stack.shape[1], a_stack.shape[2]
        output = torch.empty((out_features, in_features),
                             dtype=output_dtype, device="cpu")
        stats = {"trimmed_coordinates": 0, "selected_coordinates": 0,
                 "elected_zero_coordinates": 0}
        for start in range(0, out_features, tile_rows):
            end = min(out_features, start + tile_rows)
            vectors = torch.bmm(b_stack[:, start:end], a_stack).mul_(pool.scaling)
            module_coefficients = (
                coefficient_tensor if coefficient_tensor.ndim == 1
                else coefficient_tensor[module_order])
            merged, tile_stats = trim_elect_weighted_sum(
                vectors, threshold_tensor, module_coefficients,
                reduction=reduction)
            output[start:end].copy_(merged.to(output_dtype).cpu())
            for key, value in tile_stats.items():
                stats[key] += value
            del vectors, merged
        module_name = module_name_from_key(a_key)
        tensors[f"{module_name}.delta_weight"] = output.contiguous()
        reports.append({"layer_index": layer_index_from_key(a_key),
                        "module": module_name,
                        "shape": [out_features, in_features], **stats})
        del a_stack, b_stack
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()
    return _save(tensors, output_path,
                 f"global_trim_total_sign_{reduction}_exact_dense",
                 {"reduction": reduction,
                  "coefficient_scope": ("global" if coefficient_tensor.ndim == 1
                                        else "layer_wise"),
                  "module_reports": reports})


# ---------------------------------------------------------------------------
# DARE-TIES
# ---------------------------------------------------------------------------
def make_dare_mask(shape, density, seed, device):
    import torch
    if not 0.0 < density <= 1.0:
        raise ValueError("density 必須落在 (0, 1]")
    if density == 1.0:
        return torch.ones(shape, dtype=torch.bool, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(shape, generator=generator, device=device) < density


def elect_and_disjoint_merge(task_vectors, sign_method, lambda_):
    """取號選舉後，只對同號的貢獻者取平均。"""
    import torch
    if sign_method == "total":
        elected = torch.sign(task_vectors.sum(dim=0))
    elif sign_method == "frequency":
        elected = torch.sign(torch.sign(task_vectors).sum(dim=0))
    else:
        raise ValueError(f"未知的 sign method：{sign_method}")
    signs = torch.sign(task_vectors)
    positive = (signs > 0).any(dim=0)
    negative = (signs < 0).any(dim=0)
    agreement = (signs == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)
    contributor_count = agreement.sum(dim=0)
    numerator = torch.where(agreement, task_vectors, 0).sum(dim=0)
    merged = numerator / contributor_count.clamp_min(1)
    merged = torch.where(contributor_count > 0, merged, 0) * lambda_
    return merged, {
        "sign_conflict_coordinates": int((positive & negative).sum().item()),
        "elected_zero_coordinates": int((elected == 0).sum().item()),
        "active_coordinates": int((contributor_count > 0).sum().item()),
    }


def dare_ties_tile(task_vectors, *, density, lambda_, sign_method, seed,
                   layer_index, global_row_start, mask_block_rows, rescale=True):
    if task_vectors.ndim != 3:
        raise ValueError("預期 [adapters, rows, columns]")
    if global_row_start % mask_block_rows:
        raise ValueError("列起點必須對齊 mask_block_rows")
    pruned = task_vectors.clone()
    kept, total = 0, pruned.numel()
    for local_start in range(0, pruned.shape[1], mask_block_rows):
        rows = min(mask_block_rows, pruned.shape[1] - local_start)
        block_index = (global_row_start + local_start) // mask_block_rows
        block = pruned[:, local_start:local_start + rows]
        mask = make_dare_mask(tuple(block.shape), density,
                              stable_seed(seed, layer_index, block_index),
                              block.device)
        kept += int(mask.sum().item())
        block.mul_(mask)
        if rescale:
            block.div_(density)
    merged, stats = elect_and_disjoint_merge(pruned, sign_method, lambda_)
    stats.update({"dare_kept": kept, "dare_total": total})
    return merged, stats


def materialize_dare_ties(pool, output_path, *, density, coefficient, sign_method,
                          seed, rescale, tile_rows, mask_block_rows, device="cpu",
                          output_dtype=None):
    """DARE 隨機遮罩 → 取號選舉 → disjoint mean。"""
    import torch
    output_dtype = output_dtype or torch.bfloat16
    if tile_rows < 1 or tile_rows % mask_block_rows:
        raise ValueError("tile_rows 必須是 mask_block_rows 的正整數倍")
    compute_device = torch.device(device)
    tensors, reports = {}, []
    total_kept = total_masks = 0
    for module_order, a_key in enumerate(pool.a_keys):
        a_stack, b_stack = _stacks(pool, a_key, compute_device, torch)
        output = torch.empty((b_stack.shape[1], a_stack.shape[2]),
                             dtype=output_dtype, device="cpu")
        stats = {"dare_kept": 0, "dare_total": 0, "sign_conflict_coordinates": 0,
                 "elected_zero_coordinates": 0, "active_coordinates": 0}
        for start in range(0, b_stack.shape[1], tile_rows):
            end = min(start + tile_rows, b_stack.shape[1])
            vectors = torch.bmm(b_stack[:, start:end], a_stack).mul_(pool.scaling)
            merged, tile_stats = dare_ties_tile(
                vectors, density=density, lambda_=coefficient,
                sign_method=sign_method, seed=seed, layer_index=module_order,
                global_row_start=start, mask_block_rows=mask_block_rows,
                rescale=rescale)
            output[start:end].copy_(merged.to(output_dtype).cpu())
            for key, value in tile_stats.items():
                stats[key] += int(value)
            del vectors, merged
        module_name = module_name_from_key(a_key)
        tensors[f"{module_name}.delta_weight"] = output.contiguous()
        reports.append({"module": module_name,
                        "shape": [b_stack.shape[1], a_stack.shape[2]], **stats})
        total_kept += stats["dare_kept"]
        total_masks += stats["dare_total"]
        del a_stack, b_stack
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()
    return _save(tensors, output_path, "exact_dare_ties", {
        "density": float(density), "lambda": float(coefficient),
        "sign_method": sign_method, "seed": int(seed), "rescale": bool(rescale),
        "reduction": "disjoint_mean",
        "observed_dare_density": total_kept / max(total_masks, 1),
        "module_reports": reports,
    })


# ---------------------------------------------------------------------------
# PICO + Task Arithmetic
# ---------------------------------------------------------------------------
def _low_rank_frobenius_norm(a_factor, b_factor, scaling):
    """||scaling * B @ A||_F，不展開 B @ A。"""
    aat = a_factor @ a_factor.mT
    btb = b_factor.mT @ b_factor
    squared = (aat * btb.mT).sum().clamp_min(0)
    return squared.sqrt() * abs(float(scaling))


def _validate_stacks(a_stack, b_stack, *, minimum_tasks=1):
    if a_stack.ndim != 3 or b_stack.ndim != 3:
        raise ValueError("a_stack 與 b_stack 都必須是三階張量")
    if a_stack.shape[0] != b_stack.shape[0]:
        raise ValueError("a_stack 與 b_stack 的任務數不同")
    if a_stack.shape[1] != b_stack.shape[2]:
        raise ValueError("a_stack 與 b_stack 的來源 rank 不同")
    if a_stack.shape[0] < minimum_tasks:
        raise ValueError(f"至少需要 {minimum_tasks} 個 adapter")


def merge_pico_task_arithmetic_module(a_stack, b_stack, *, source_scaling,
                                      reduction="mean", eps=1e-12):
    """對單一模組做 PICO 校準、加總 task vector、再還原合併後的量級。"""
    import torch

    _validate_stacks(a_stack, b_stack, minimum_tasks=2)
    if source_scaling <= 0:
        raise ValueError("source_scaling 必須為正")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction 只能是 mean 或 sum")

    a = a_stack.float()
    b = b_stack.float()
    task_count, source_rank, in_features = a.shape
    out_features = b.shape[1]
    b_all = b.permute(1, 0, 2).reshape(out_features, task_count * source_rank)

    u, singular_values, _ = torch.linalg.svd(b_all, full_matrices=False)
    energy = singular_values.square()
    energy_sum = energy.sum()
    if float(energy_sum.detach().cpu()) <= eps:
        calibrated_b_all = b_all
    else:
        sharing_scores = energy / energy_sum
        attenuation = 1.0 / (1.0 + (task_count - 1) * sharing_scores)
        projection = u.mT @ b_all
        calibrated_b_all = b_all + u @ (
            (attenuation - 1.0).unsqueeze(1) * projection)

    calibrated_b_stack = (
        calibrated_b_all.reshape(out_features, task_count, source_rank)
        .permute(1, 0, 2).contiguous())
    source_norms = torch.stack([
        _low_rank_frobenius_norm(a[i], b[i], source_scaling)
        for i in range(task_count)])
    source_norm_mean = source_norms.mean()

    merged_a = a.reshape(task_count * source_rank, in_features).contiguous()
    merged_b = calibrated_b_stack.permute(1, 0, 2).reshape(
        out_features, task_count * source_rank).contiguous()
    if reduction == "mean":
        merged_b = merged_b / task_count
    calibrated_sum_norm = _low_rank_frobenius_norm(
        merged_a, merged_b, source_scaling)
    if float(calibrated_sum_norm.detach().cpu()) <= eps:
        restoration = torch.ones((), device=merged_b.device, dtype=merged_b.dtype)
    else:
        restoration = source_norm_mean / calibrated_sum_norm
    return merged_a, merged_b * restoration, float(restoration.detach().cpu())


def materialize_pico_task_arithmetic(pool, output_path, *, reduction="mean",
                                     eps=1e-12, device="cpu", tile_rows=16,
                                     output_dtype=None):
    """PICO 校準後的 exact dense delta。"""
    import torch
    output_dtype = output_dtype or torch.bfloat16
    compute_device = torch.device(device)
    tensors, reports = {}, []
    for a_key in pool.a_keys:
        a_stack, b_stack = _stacks(pool, a_key, compute_device, torch)
        merged_a, merged_b, restoration = merge_pico_task_arithmetic_module(
            a_stack, b_stack, source_scaling=pool.scaling,
            reduction=reduction, eps=eps)
        output = torch.empty((merged_b.shape[0], merged_a.shape[1]),
                             dtype=output_dtype, device="cpu")
        for start in range(0, merged_b.shape[0], tile_rows):
            end = min(start + tile_rows, merged_b.shape[0])
            tile = merged_b[start:end] @ merged_a
            tile.mul_(pool.scaling)     # PICO 的因子沒有折進 scaling，這裡要乘
            output[start:end].copy_(tile.to(device="cpu", dtype=output_dtype))
        module_name = module_name_from_key(a_key)
        tensors[f"{module_name}.delta_weight"] = output.contiguous()
        reports.append({"module": module_name,
                        "output_rank": int(merged_a.shape[0]),
                        "magnitude_restoration": restoration})
        del a_stack, b_stack, merged_a, merged_b
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()
    return _save(tensors, output_path, "pico_task_arithmetic",
                 {"reduction": reduction, "module_reports": reports})


# ---------------------------------------------------------------------------
# LoRA-Lego
# ---------------------------------------------------------------------------
def merge_lego_module(a_stack, b_stack, *, source_scaling, output_rank,
                      output_ref_rank, seed=0, n_init=10, max_iter=300,
                      parameter_reweight=True, output_reweight=True, eps=1e-12):
    """把逐 rank 的 MSU 分群，重建成一個 rank-k 的 LoRA 模組。"""
    import numpy as np
    import torch
    from sklearn.cluster import KMeans

    _validate_stacks(a_stack, b_stack)
    if source_scaling <= 0 or output_rank <= 0 or output_ref_rank <= 0:
        raise ValueError("scaling 與 rank 都必須為正")
    task_count, source_rank, in_features = a_stack.shape
    out_features = b_stack.shape[1]
    pool_size = task_count * source_rank
    if output_rank > pool_size:
        raise ValueError("output_rank 不能超過 MSU 的總數")

    adapter_weights = np.full(task_count, 1.0 / task_count, dtype=np.float64)
    # sqrt(scaling) 折進 A 與 B 兩邊，因此重建出來的 delta 已含 scaling，
    # 之後不可再乘一次。
    scale_root = math.sqrt(float(source_scaling))
    a = a_stack.detach().float().cpu().numpy() * scale_root
    b = b_stack.detach().float().cpu().numpy() * scale_root
    msus = np.concatenate([
        a.reshape(pool_size, in_features),
        b.transpose(0, 2, 1).reshape(pool_size, out_features),
    ], axis=1).astype(np.float32, copy=False)
    sample_weights = np.repeat(adapter_weights, source_rank)

    kmeans = KMeans(n_clusters=output_rank, random_state=seed,
                    n_init=n_init, max_iter=max_iter)
    labels = kmeans.fit_predict(msus, sample_weight=sample_weights)
    centers = kmeans.cluster_centers_.astype(np.float32, copy=True)

    cluster_sizes = []
    for cluster_index in range(output_rank):
        mask = labels == cluster_index
        cluster_sizes.append(int(mask.sum()))
        if not parameter_reweight:
            continue
        center_norm = float(np.linalg.norm(centers[cluster_index], ord=np.inf))
        member_norms = np.linalg.norm(msus[mask], ord=np.inf, axis=1)
        member_weights = sample_weights[mask]
        target_norm = float(
            np.sum(member_norms * member_weights) / member_weights.sum())
        centers[cluster_index] *= target_norm / max(center_norm, eps)

    a_merged = centers[:, :in_features]
    b_merged = centers[:, in_features:].T
    output_factor = math.sqrt(float(output_ref_rank) / float(output_rank))
    if output_reweight:
        b_merged = b_merged * output_factor
    else:
        output_factor = 1.0
    return (torch.from_numpy(a_merged).contiguous(),
            torch.from_numpy(b_merged).contiguous(),
            tuple(cluster_sizes), float(kmeans.inertia_), int(kmeans.n_iter_),
            output_factor)


def materialize_lego(pool, output_path, *, output_rank, output_ref_rank, seed,
                     n_init, max_iter, parameter_reweight, output_reweight,
                     eps=1e-12, tile_rows=16, output_dtype=None):
    """把每個模組的 MSU 分群後重建成 exact dense delta。"""
    import torch
    output_dtype = output_dtype or torch.bfloat16
    tensors, reports = {}, []
    for a_key in pool.a_keys:
        b_key = paired_key(a_key)
        merged_a, merged_b, cluster_sizes, inertia, iterations, factor = (
            merge_lego_module(
                torch.stack([w[a_key] for w in pool.weights]),
                torch.stack([w[b_key] for w in pool.weights]),
                source_scaling=pool.scaling, output_rank=output_rank,
                output_ref_rank=output_ref_rank, seed=seed, n_init=n_init,
                max_iter=max_iter, parameter_reweight=parameter_reweight,
                output_reweight=output_reweight, eps=eps))
        output = torch.empty((merged_b.shape[0], merged_a.shape[1]),
                             dtype=output_dtype, device="cpu")
        for start in range(0, merged_b.shape[0], tile_rows):
            end = min(start + tile_rows, merged_b.shape[0])
            # scaling 已折進因子，這裡不再乘。
            output[start:end].copy_(
                (merged_b[start:end] @ merged_a).to(output_dtype))
        module_name = module_name_from_key(a_key)
        tensors[f"{module_name}.delta_weight"] = output.contiguous()
        reports.append({"module": module_name, "output_rank": int(output_rank),
                        "cluster_sizes": list(cluster_sizes),
                        "inertia": inertia, "iterations": iterations,
                        "output_reweight_factor": factor})
    return _save(tensors, output_path, "lora_lego",
                 {"output_rank": int(output_rank), "module_reports": reports})


# ---------------------------------------------------------------------------
# 共用
# ---------------------------------------------------------------------------
def _save(tensors, output_path, method, extra):
    from safetensors.torch import save_file
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(destination), metadata={"method": method})
    return {"status": "complete", "method": method,
            "module_count": len(tensors),
            "output_path": str(destination),
            "output_bytes": destination.stat().st_size,
            "output_sha256": sha256(destination), **extra}


def merge_pool_adamerging(pool, output_path, learned_coefficients, *,
                          device="cpu", output_dtype=None, progress=False,
                          density=0.2, tile_rows=16, reduction="disjoint_mean"):
    """用已學到的逐層係數走 TIES 合成（adamerging_pp 的第二階段）。

    第一階段的係數最佳化在 system/adamerging.py；這裡只負責把係數變成權重，
    用的是與 ties_only 完全相同、已對 producer 驗證過的合成路徑。
    """
    import torch

    thresholds = compute_global_thresholds(
        pool, density, torch.device(device), progress=progress)
    report = materialize_ties(
        pool, output_path, thresholds=thresholds,
        coefficients=learned_coefficients, tile_rows=tile_rows,
        reduction=reduction, device=device, output_dtype=output_dtype)
    report["method"] = "adamerging_pp"
    report["density"] = float(density)
    return report


def merge_pool(method, pool, output_path, *, device="cpu", output_dtype=None,
               progress=False):
    """以封板超參數對整個 pool150 執行指定的 merge 方法。"""
    if method not in SEALED:
        raise ValueError(
            f"不支援的 merge 方法：{method!r}；可用 {', '.join(MERGE_METHODS)}")
    sealed = SEALED[method]
    if method == "ta":
        # reduction=mean 要把 lambda 攤到每個任務上——直接用 lambda 會讓輸出
        # 放大 len(pool) 倍。producer 的 prepare() 在呼叫端做這件事，搬過來時
        # 必須一起搬，否則兩邊產物不可能相同。
        if sealed["reduction"] == "mean":
            coefficient = float(sealed["lambda"]) / len(pool.names)
        elif sealed["reduction"] == "sum":
            coefficient = float(sealed["lambda"])
        else:
            raise ValueError(
                f"不支援的 Task Arithmetic reduction：{sealed['reduction']!r}")
        return materialize_task_arithmetic(
            pool, output_path, coefficient=coefficient,
            device=device, tile_rows=sealed["tile_rows"],
            output_dtype=output_dtype)
    if method == "ties_only":
        import torch
        thresholds = compute_global_thresholds(
            pool, sealed["density"], torch.device(device), progress=progress)
        coefficients = [sealed["lambda"]] * len(pool.names)
        return materialize_ties(
            pool, output_path, thresholds=thresholds, coefficients=coefficients,
            tile_rows=sealed["tile_rows"], reduction=sealed["reduction"],
            device=device, output_dtype=output_dtype)
    if method == "dare_ties_ta":
        return materialize_dare_ties(
            pool, output_path, density=sealed["density"],
            coefficient=sealed["lambda"], sign_method=sealed["sign_method"],
            seed=CANONICAL_SEED, rescale=sealed["rescale"],
            tile_rows=sealed["tile_rows"],
            mask_block_rows=sealed["mask_block_rows"],
            device=device, output_dtype=output_dtype)
    if method == "pico_ta":
        return materialize_pico_task_arithmetic(
            pool, output_path, reduction=sealed["reduction"],
            eps=sealed["eps"], device=device, tile_rows=sealed["tile_rows"],
            output_dtype=output_dtype)
    # lora_lego：KMeans 在 CPU 上執行，device 對它沒有作用。
    return materialize_lego(
        pool, output_path, output_rank=sealed["output_rank"],
        output_ref_rank=sealed["output_ref_rank"], seed=sealed["lego_seed"],
        n_init=sealed["n_init"], max_iter=sealed["max_iter"],
        parameter_reweight=sealed["parameter_reweight"],
        output_reweight=sealed["output_reweight"], eps=sealed["eps"],
        tile_rows=sealed["tile_rows"], output_dtype=output_dtype)

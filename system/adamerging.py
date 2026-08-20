# -*- coding: utf-8 -*-
"""
system/adamerging.py — AdaMerging++ 的逐層係數最佳化

從 MoEA-Trainer-delivery 搬入（`src/adamerging/pool150.py` 與
`src/adamerging/runtime.py` 的最佳化部分、`src/adamerging/loss.py`）。

`adamerging_pp` 與其他 baseline 的差別在於它**不是純權重運算**：它要把模型載進 GPU、
對校準資料跑 500 次迭代，學出一組「逐層 × 逐任務」的合併係數，最後才用這組係數走
TIES 的合成流程（`system/merging.py` 的 `compute_global_thresholds` 與
`materialize_ties`，後者支援逐層係數）。

**這是建置階段的動作，不是服務階段的。** 產物落地並通過驗證之後才可被選用；回答請求
時不會做這裡的任何事情（CONTEXT.md 不變式 5、6）。

目標函數是 prompt 上的 token 熵，**不看標準答案**——`objective_uses_target_labels`
在封板設定裡是 false。校準資料取自 `dataset/train_data/`，每個任務 50 筆。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

#: 封板設定（MoEA-Trainer-delivery/src/moea_repro/conditions.py 的 adamerging_pp）。
SEALED_ADAMERGING = {
    "optimization_model": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "ties": {"density": 0.2, "tile_rows": 16, "reduction": "disjoint_mean"},
    "lambda_init": 0.3,
    "coefficient_parameterization": "projected_nonnegative",
    "coefficient_min": 0.0,
    "coefficient_max": 1.0,
    "lr": 0.001,
    "max_iterations": 500,
    "micro_batch_size": 1,
    "max_length": 8192,
    "gradient_clip_norm": 1.0,
    "tasks_per_iteration": 150,
    "examples_per_selected_task": 32,
    "calibration_per_task": 50,
    "validation_per_task": 32,
    "file_pattern": "{task}_train.json",
    "source_prompts_include_target": True,
}


def stable_seed(*parts) -> int:
    encoded = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)


def truncate_head_tail(token_ids, budget):
    """超出預算時保留頭尾兩端的證據，而不是直接砍尾巴。"""
    if budget < 1:
        raise ValueError("token 預算必須為正")
    if len(token_ids) <= budget:
        return token_ids, False
    head = (budget + 1) // 2
    tail = budget - head
    return token_ids[:head] + (token_ids[-tail:] if tail else []), True


# ---------------------------------------------------------------------------
# 校準資料
# ---------------------------------------------------------------------------
def _record(task_name, item, *, source_prompts_include_target):
    prompt = str(item["full_prompt"])
    target = str(item["output"])
    if source_prompts_include_target:
        stripped_prompt = prompt.rstrip()
        stripped_target = target.strip()
        if not stripped_target:
            raise ValueError(f"{task_name}/{item['instance_id']}：target 是空的")
        if not stripped_prompt.endswith(stripped_target):
            # 這一步是防答案外洩：訓練樣本的 prompt 尾端必須精確等於它的 output，
            # 才知道要移除哪一段。對不上就拒絕，不猜。
            raise ValueError(
                f"{task_name}/{item['instance_id']}：prompt 尾端不等於其 target，"
                "拒絕使用可能外洩答案的校準樣本")
        prompt = stripped_prompt[: -len(stripped_target)]
    return {"task": task_name, "instance_id": item["instance_id"],
            "prompt": prompt, "target": target}


def build_source_split(train_root, task_names, *, calibration_per_task,
                       validation_per_task, seed, file_pattern,
                       source_prompts_include_target):
    """從訓練資料抽出校準堆與驗證堆，逐任務、種子固定、兩堆不重疊。"""
    root = Path(train_root)
    calibration, validation, selections = [], [], {}
    for task_name in task_names:
        path = root / file_pattern.format(task=task_name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task_key") != task_name:
            raise ValueError(
                f"{path}：task_key={payload.get('task_key')!r}，預期 {task_name!r}")
        instances = list(payload["instances"])
        if len(instances) <= calibration_per_task:
            raise ValueError(
                f"{task_name}：{len(instances)} 筆不足以取 {calibration_per_task} "
                "筆校準再加驗證")
        order = list(range(len(instances)))
        random.Random(stable_seed(seed, task_name)).shuffle(order)
        calibration_indices = sorted(order[:calibration_per_task])
        available = len(instances) - calibration_per_task
        validation_count = min(validation_per_task, available)
        validation_indices = sorted(
            order[calibration_per_task:calibration_per_task + validation_count])
        if set(calibration_indices) & set(validation_indices):
            raise AssertionError(f"{task_name}：校準堆與驗證堆重疊")
        for target, indices in ((calibration, calibration_indices),
                                (validation, validation_indices)):
            target.extend(
                _record(task_name, instances[i],
                        source_prompts_include_target=source_prompts_include_target)
                for i in indices)
        selections[task_name] = {
            "available": len(instances),
            "calibration_ids": [instances[i]["instance_id"]
                                for i in calibration_indices],
            "validation_ids": [instances[i]["instance_id"]
                               for i in validation_indices],
        }
    serialized = json.dumps(selections, sort_keys=True, separators=(",", ":"))
    metadata = {
        "selection_seed": seed,
        "task_count": len(task_names),
        "calibration_count": len(calibration),
        "validation_count": len(validation),
        "file_pattern": file_pattern,
        "source_prompts_include_target": source_prompts_include_target,
        "selection_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }
    return calibration, validation, metadata


# ---------------------------------------------------------------------------
# 係數模組與掛載
# ---------------------------------------------------------------------------
def inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus 的初值必須為正")
    return math.log(math.expm1(value))


def _build_modules():
    """延後 import torch，讓不需要最佳化的路徑不必付這個代價。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class LayerWiseAdaMerging(nn.Module):
        """每一層、每一個 adapter 各一個非負係數。"""

        def __init__(self, num_layers, num_tasks, initial_value,
                     parameterization="projected_nonnegative", maximum_value=None):
            super().__init__()
            if parameterization not in ("projected_nonnegative", "softplus"):
                raise ValueError(f"未知的係數參數化方式：{parameterization}")
            self.parameterization = parameterization
            self.maximum_value = maximum_value
            initial_raw = (initial_value if parameterization == "projected_nonnegative"
                           else inverse_softplus(initial_value))
            self.raw_coefficients = nn.Parameter(
                torch.full((num_layers, num_tasks), initial_raw))

        @property
        def coefficients(self):
            if self.parameterization == "softplus":
                return functional.softplus(self.raw_coefficients)
            return self.raw_coefficients

        @torch.no_grad()
        def project_(self):
            if self.parameterization == "projected_nonnegative":
                self.raw_coefficients.clamp_(min=0.0, max=self.maximum_value)

    class WeightedLoRALinear(nn.Module):
        """一層之內 K 個 rank-r task vector 的精確加權和。"""

        def __init__(self, base_layer, a_stack, b_stack, scaling,
                     coefficient_module, layer_index):
            super().__init__()
            if a_stack.ndim != 3 or b_stack.ndim != 3:
                raise ValueError("預期 A=[K,r,in] 與 B=[K,out,r]")
            if (a_stack.shape[0] != b_stack.shape[0]
                    or a_stack.shape[1] != b_stack.shape[2]):
                raise ValueError("堆疊的 LoRA 形狀不相容")
            self.base_layer = base_layer
            self.register_buffer("a_stack", a_stack, persistent=False)
            self.register_buffer("b_stack", b_stack, persistent=False)
            self.scaling = float(scaling)
            self.coefficient_module = coefficient_module
            self.layer_index = int(layer_index)

        def forward(self, inputs):
            base_output = self.base_layer(inputs)
            compute = inputs.to(self.a_stack.dtype)
            task_count, rank, in_features = self.a_stack.shape
            a_concat = self.a_stack.reshape(task_count * rank, in_features)
            down = functional.linear(compute, a_concat)
            coefficients = self.coefficient_module.coefficients[self.layer_index]
            weighted_b = self.b_stack * coefficients.to(
                self.b_stack.dtype)[:, None, None]
            out_features = self.b_stack.shape[1]
            b_concat = weighted_b.permute(1, 0, 2).reshape(
                out_features, task_count * rank)
            delta = functional.linear(down, b_concat) * self.scaling
            return base_output + delta.to(base_output.dtype)

    return LayerWiseAdaMerging, WeightedLoRALinear


def attach_weighted_lora(model, pool, coefficient_module, device,
                         weighted_linear_cls, dtype=None):
    """把每個 down_proj 換成加權 LoRA 版本，並凍結模型本身的參數。"""
    import torch

    from system.adapter_pool import layer_index_from_key, module_name_from_key, paired_key

    dtype = dtype or torch.bfloat16
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    attached = []
    for a_key in pool.a_keys:
        b_key = paired_key(a_key)
        module_name = module_name_from_key(a_key)
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        base_layer = getattr(parent, child_name)
        a_stack = torch.stack([item[a_key] for item in pool.weights]).to(
            device=device, dtype=dtype)
        b_stack = torch.stack([item[b_key] for item in pool.weights]).to(
            device=device, dtype=dtype)
        setattr(parent, child_name, weighted_linear_cls(
            base_layer, a_stack, b_stack, pool.scaling, coefficient_module,
            layer_index_from_key(a_key)))
        attached.append(module_name)
    if len(attached) != len(pool.a_keys):
        raise RuntimeError(
            f"預期掛上 {len(pool.a_keys)} 個模組，實際 {len(attached)} 個")
    return attached


# ---------------------------------------------------------------------------
# 目標函數：prompt 上的 token 熵（不看標準答案）
# ---------------------------------------------------------------------------
def _entropy_per_token(logits):
    """H(p)=logsumexp(x)-E_p[x]，分塊算以避免三份全詞彙表的暫存。"""
    import torch

    log_normalizer = torch.logsumexp(logits, dim=-1)
    weighted_logits = torch.zeros_like(log_normalizer)
    for start in range(0, logits.shape[-1], 16384):
        chunk = logits[..., start:start + 16384]
        probabilities = (chunk - log_normalizer.unsqueeze(-1)).exp()
        weighted_logits = weighted_logits + (probabilities * chunk).sum(dim=-1)
    return log_normalizer - weighted_logits


def token_entropy_sum(logits, attention_mask):
    from torch.utils.checkpoint import checkpoint

    entropy = checkpoint(_entropy_per_token, logits, use_reentrant=False)
    return (entropy * attention_mask).sum()


# ---------------------------------------------------------------------------
# 編碼與抽樣
# ---------------------------------------------------------------------------
def encode_prompt(tokenizer, record, max_length):
    original = tokenizer.encode(record["prompt"], add_special_tokens=True)
    encoded, truncated = truncate_head_tail(original, max_length)
    return {**record, "input_ids": encoded, "prompt_truncated": truncated,
            "original_prompt_tokens": len(original)}


def collate(records, pad_token_id):
    import torch

    width = max(len(item["input_ids"]) for item in records)
    input_ids, masks = [], []
    for item in records:
        padding = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        masks.append([1] * len(item["input_ids"]) + [0] * padding)
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long)}


def length_batches(records, batch_size):
    """長度相近的排在一起，減少 padding 浪費。順序是決定性的。"""
    ordered = sorted(records, key=lambda item: (
        len(item["input_ids"]), item["task"], item["instance_id"]))
    for start in range(0, len(ordered), batch_size):
        yield ordered[start:start + batch_size]


def sample_calibration(records, *, tasks_per_iteration, examples_per_task,
                       seed, iteration):
    """每一輪抽一批任務、每個任務抽固定筆數。純由 seed 與 iteration 決定。"""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["task"]].append(record)
    all_tasks = sorted(grouped, key=lambda value: int(value[4:]))
    if len(all_tasks) % tasks_per_iteration:
        raise ValueError("tasks_per_iteration 必須整除任務總數")
    ordered = list(all_tasks)
    random.Random(stable_seed(seed, "task-cycle", iteration - 1)).shuffle(ordered)
    chosen = ordered[:tasks_per_iteration]
    selected = []
    for task_name in chosen:
        candidates = grouped[task_name]
        if len(candidates) < examples_per_task:
            raise ValueError(
                f"{task_name}：需要 {examples_per_task} 筆校準樣本，只有 {len(candidates)} 筆")
        rng = random.Random(stable_seed(seed, "calibration", iteration, task_name))
        selected.extend(rng.sample(candidates, examples_per_task))
    return selected, chosen


def coefficient_summary(module):
    values = module.coefficients.detach().float().cpu()
    return {"min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "near_zero_fraction": float((values < 1e-6).float().mean()),
            "over_one_fraction": float((values > 1).float().mean())}


# ---------------------------------------------------------------------------
# 最佳化
# ---------------------------------------------------------------------------
def optimize_coefficients(pool, train_root, work_dir, *, seed, device="cuda",
                          sealed=None, progress_every=10):
    """學出逐層係數並存檔；可中斷續跑。回傳 [layers, tasks] 的係數與摘要。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sealed = sealed or SEALED_ADAMERGING
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "training_state.pt"
    trace_path = work_dir / "optimization_trace.json"

    calibration_records, _validation, split_meta = build_source_split(
        train_root, pool.names,
        calibration_per_task=sealed["calibration_per_task"],
        validation_per_task=sealed["validation_per_task"],
        seed=seed, file_pattern=sealed["file_pattern"],
        source_prompts_include_target=sealed["source_prompts_include_target"])
    print(f"[adamerging] 校準 {len(calibration_records)} 筆"
          f"（指紋 {split_meta['selection_sha256'][:16]}）", flush=True)

    compute_device = torch.device(device)
    name = sealed["optimization_model"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map="auto", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True)
    model.eval()
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    layer_wise_cls, weighted_linear_cls = _build_modules()
    coefficients = layer_wise_cls(
        len(pool.a_keys), len(pool.names), float(sealed["lambda_init"]),
        str(sealed["coefficient_parameterization"]),
        maximum_value=float(sealed["coefficient_max"])).to(compute_device)
    attach_weighted_lora(model, pool, coefficients, compute_device,
                         weighted_linear_cls)

    max_length = int(sealed["max_length"])
    encoded = [encode_prompt(tokenizer, item, max_length)
               for item in calibration_records]
    truncated = [item["instance_id"] for item in encoded if item["prompt_truncated"]]
    if truncated:
        # 截斷會改變目標函數看到的內容，不能靜默接受。
        raise ValueError(f"{len(truncated)} 筆校準樣本超過 {max_length} tokens")

    optimizer = torch.optim.Adam([coefficients.raw_coefficients],
                                 lr=float(sealed["lr"]), betas=(0.9, 0.999),
                                 weight_decay=0.0)
    start_iteration = 0
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else []
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("source_split_sha256") != split_meta["selection_sha256"]:
            raise ValueError("既有的訓練檔點對應不同的校準資料，拒絕續跑")
        start_iteration = int(state["iteration"])
        coefficients.raw_coefficients.data.copy_(
            state["raw_coefficients"].to(compute_device))
        optimizer.load_state_dict(state["optimizer"])
        print(f"[adamerging] 自第 {start_iteration} 次迭代續跑", flush=True)

    maximum = int(sealed["max_iterations"])
    micro_batch = int(sealed["micro_batch_size"])
    for iteration in range(start_iteration + 1, maximum + 1):
        optimizer.zero_grad(set_to_none=True)
        selected, chosen = sample_calibration(
            encoded, tasks_per_iteration=int(sealed["tasks_per_iteration"]),
            examples_per_task=int(sealed["examples_per_selected_task"]),
            seed=seed, iteration=iteration)
        total_tokens = sum(len(item["input_ids"]) for item in selected)
        entropy_sum = 0.0
        started = time.perf_counter()
        for items in length_batches(selected, micro_batch):
            batch = {key: value.to(compute_device)
                     for key, value in collate(items, tokenizer.pad_token_id).items()}
            logits = model(**batch).logits
            entropy = token_entropy_sum(logits, batch["attention_mask"])
            (entropy / total_tokens).backward()
            entropy_sum += float(entropy.detach().item())
            del logits, entropy
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            [coefficients.raw_coefficients], float(sealed["gradient_clip_norm"])))
        optimizer.step()
        coefficients.project_()

        entry = {"iteration": iteration,
                 "train_entropy": entropy_sum / max(total_tokens, 1),
                 "gradient_norm_before_clip": gradient_norm,
                 "selected_tasks": len(chosen),
                 "selected_examples": len(selected),
                 "coefficients": coefficient_summary(coefficients),
                 "iteration_seconds": time.perf_counter() - started}
        trace.append(entry)
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        temporary = state_path.with_suffix(".pt.tmp")
        torch.save({"source_split_sha256": split_meta["selection_sha256"],
                    "iteration": iteration,
                    "raw_coefficients": coefficients.raw_coefficients.detach().cpu(),
                    "optimizer": optimizer.state_dict()}, temporary)
        temporary.replace(state_path)
        if iteration % progress_every == 0 or iteration == maximum:
            print(f"[adamerging] {iteration}/{maximum} "
                  f"entropy={entry['train_entropy']:.5f} "
                  f"({entry['iteration_seconds']:.1f}s)", flush=True)

    learned = coefficients.coefficients.detach().cpu().clone()
    summary = {"status": "complete", "completed_iterations": maximum,
               "resumed_from_iteration": start_iteration,
               "optimization_used_target_labels": False,
               "coefficient_summary": coefficient_summary(coefficients),
               "source_split": split_meta,
               "sealed": {key: value for key, value in sealed.items()
                          if key != "ties"}}
    (work_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    del model, tokenizer, coefficients, optimizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return learned, summary

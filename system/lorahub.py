# -*- coding: utf-8 -*-
"""
system/lorahub.py — LoRAHub 的 CMA-ES 權重搜尋

從 MoEA-Trainer-delivery 搬入（`src/lorahub/algorithm.py` 與 `src/lorahub/runtime.py`
的擬合部分）。

與其他 baseline 最大的不同：**LoRAHub 的產物綁定一組具體的示範樣本。** 它從 150 個
adapter 裡隨機挑 20 個，再用 CMA-ES 搜尋 20 個權重，讓這 20 個 LoRA 的加權和在那幾筆
示範樣本上的 cross-entropy 最低。換一批樣本就得重搜一次。

因此它**不能當通用的線上拒絕選項**——沒有一組權重對得上任意請求。它的定位是
benchmark 的受測對象：為某一份固定資料擬合一次，測完就有意義。產出的 manifest 一定
會記錄「針對哪些樣本、哪個種子」，避免被誤當通用權重使用。

示範樣本由呼叫端提供（`[{"prompt", "output", "instance_id"}, ...]`），本模組不綁定
任何特定資料集的讀取方式。

產物是 LoRA（與來源同 rank），不是 dense delta——`compose_state_dict` 只是把 20 個
state_dict 加權相加，形狀不變。對應 SMoEA 契約的 peft_adapter_v1 格式。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import OrderedDict
from pathlib import Path

#: 封板設定（MoEA-Trainer-delivery/src/moea_repro/conditions.py 的 lorahub）。
SEALED_LORAHUB = {
    "candidate_count": 20,
    "shots": 5,
    "generations": 40,
    "population_size": 12,
    "initial_sigma": 0.05,
    "weight_lower": -1.5,
    "weight_upper": 1.5,
    "initial_weights": 0.0,
    "l1_coefficient": 0.05,
    "l1_reduction": "sum",
    "optimizer_package_version": "4.4.4",
    "expected_objective_evaluations_per_run": 480,
    "adapt_max_tokens": 2048,
    "run_seeds": [1, 2, 3],
    "example_seeds": {"1": 42, "2": 43, "3": 44},
    "optimizer_seeds": {"1": 42, "2": 43, "3": 44},
}


class LoRAHubError(RuntimeError):
    """LoRAHub 的設定或資料不符合封板定義。"""


def select_candidate_adapters(adapter_names, count, seed):
    """從池中挑出候選 adapter。對應官方的 random.seed(seed); random.sample(...)。"""
    if count > len(adapter_names):
        raise LoRAHubError(
            f"要挑 {count} 個候選，池裡只有 {len(adapter_names)} 個")
    return random.Random(seed).sample(list(adapter_names), count)


def paper_l1_regularization(weights, coefficient=0.05):
    """論文的目標函數：alpha 乘上係數絕對值的總和。"""
    return float(coefficient) * sum(abs(float(value)) for value in weights)


def validate_paper_settings(settings):
    """設定一旦偏離論文定義就直接失敗，不讓它悄悄變成另一個方法。"""
    dimensions = int(settings["candidate_count"])
    generations = int(settings["generations"])
    population = int(settings["population_size"])
    expected_population = 4 + math.floor(3 * math.log(dimensions))
    evaluations = generations * population
    if dimensions != 20:
        raise LoRAHubError(f"論文定義 N=20，得到 {dimensions}")
    if generations != 40:
        raise LoRAHubError(f"論文定義 K=40 代，得到 {generations}")
    if population != expected_population:
        raise LoRAHubError(
            f"維度 {dimensions} 的族群數應為 {expected_population}，設定是 {population}")
    if evaluations != int(settings["expected_objective_evaluations_per_run"]):
        raise LoRAHubError("代數 × 族群數與宣告的評估次數不符")
    if settings.get("l1_reduction") != "sum":
        raise LoRAHubError("論文的目標函數要求 alpha * sum(abs(weights))")
    return {
        "candidate_count": dimensions,
        "generations": generations,
        "population_size": population,
        "objective_evaluations_per_run": evaluations,
        "shots_per_objective_evaluation": int(settings["shots"]),
    }


def compose_state_dict(weights, selected_names, cache):
    """把選中的 LoRA state_dict 依權重相加。對應官方的參數內插。"""
    if len(weights) != len(selected_names) or not selected_names:
        raise LoRAHubError("weights 與 selected_names 必須等長且非空")
    keys = list(cache[selected_names[0]].keys())
    final = {}
    for index, name in enumerate(selected_names):
        state = cache[name]
        if list(state.keys()) != keys:
            raise LoRAHubError(f"{name} 的 state_dict 鍵與其他不同")
        weight = float(weights[index])
        if index == 0:
            final = {key: weight * state[key] for key in keys}
        else:
            for key in keys:
                final[key].add_(state[key], alpha=weight)
    return final


def load_adapter_cache(pool, selected_names, device="cpu"):
    """讀進選中 adapter 的 LoRA 張量，並檢查它們形狀一致。"""
    from safetensors.torch import load_file

    by_name = dict(zip(pool.names, pool.paths))
    cache = OrderedDict()
    expected_keys = None
    expected_shapes = None
    for name in selected_names:
        path = Path(by_name[name]) / "adapter_model.safetensors"
        state = load_file(str(path), device=str(device))
        keys = list(state.keys())
        shapes = {key: tuple(value.shape) for key, value in state.items()}
        if expected_keys is None:
            expected_keys, expected_shapes = keys, shapes
        elif keys != expected_keys or shapes != expected_shapes:
            raise LoRAHubError(f"LoRA 張量不相容：{path}")
        cache[name] = state
    return cache


def build_causal_lm_batch(tokenizer, examples, max_tokens, device):
    """把示範樣本編成 prompt 遮罩後的 causal-LM 批次；只截左側的 prompt。"""
    import torch

    rows, truncations = [], []
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    for example in examples:
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=True,
                               truncation=False)["input_ids"]
        output_ids = tokenizer(example["output"], add_special_tokens=False,
                               truncation=False)["input_ids"]
        if eos is not None:
            output_ids = output_ids + [eos]
        original = len(prompt_ids) + len(output_ids)
        # 目標 token 一定保留，只截 prompt 的左側。
        output_ids = output_ids[:min(len(output_ids), max_tokens)]
        prompt_budget = max_tokens - len(output_ids)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
        rows.append((prompt_ids + output_ids,
                     [-100] * len(prompt_ids) + output_ids))
        truncations.append({"instance_id": example.get("instance_id"),
                            "original_tokens": original,
                            "used_tokens": len(prompt_ids) + len(output_ids),
                            "truncated": original > max_tokens})
    width = max(len(input_ids) for input_ids, _ in rows)
    input_batch, attention_batch, label_batch = [], [], []
    for input_ids, labels in rows:
        amount = width - len(input_ids)
        input_batch.append(input_ids + [pad] * amount)
        attention_batch.append([1] * len(input_ids) + [0] * amount)
        label_batch.append(labels + [-100] * amount)
    return {
        "input_ids": torch.tensor(input_batch, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_batch, dtype=torch.long, device=device),
        "labels": torch.tensor(label_batch, dtype=torch.long, device=device),
    }, truncations


class PaperCMAESOptimizer:
    """逐代明確執行的 CMA-ES，對應論文文字的解讀。"""

    def __init__(self, model, cache, selected_names, batch, settings):
        self.model = model
        self.cache = cache
        self.selected_names = list(selected_names)
        self.batch = batch
        self.settings = settings
        self.trajectory = []

    def score(self, weights):
        import torch
        from peft.utils.save_and_load import set_peft_model_state_dict

        state = compose_state_dict(weights, self.selected_names, self.cache)
        result = set_peft_model_state_dict(self.model, state)
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if unexpected:
            raise LoRAHubError(f"PEFT 出現未預期的鍵：{unexpected[:3]}")
        with torch.inference_mode():
            cross_entropy = float(self.model(**self.batch).loss.detach().float())
        regularization = paper_l1_regularization(
            weights, float(self.settings["l1_coefficient"]))
        objective = cross_entropy + regularization
        self.trajectory.append({
            "evaluation": len(self.trajectory) + 1,
            "cross_entropy_batch_mean": cross_entropy,
            "l1_regularization": regularization,
            "objective": objective,
            "weights": [float(value) for value in weights],
        })
        return objective

    def run(self, progress_every=40):
        import cma
        import numpy as np

        expected_version = str(self.settings["optimizer_package_version"])
        actual_version = str(getattr(cma, "__version__", "unknown"))
        if actual_version != expected_version:
            # 演化策略的軌跡與套件版本綁定，版本不同就不是同一次搜尋。
            raise LoRAHubError(
                f"cma 版本 {actual_version} 與封板的 {expected_version} 不符")

        seed = int(self.settings["optimizer_seed"])
        random.seed(seed)
        np.random.seed(seed)
        count = len(self.selected_names)
        population = int(self.settings["population_size"])
        optimizer = cma.CMAEvolutionStrategy(
            [float(self.settings["initial_weights"])] * count,
            float(self.settings["initial_sigma"]),
            {"bounds": [float(self.settings["weight_lower"]),
                        float(self.settings["weight_upper"])],
             "popsize": population, "seed": seed, "verbose": -9})

        best_weights, best_objective = None, float("inf")
        generations = int(self.settings["generations"])
        for generation in range(1, generations + 1):
            candidates = optimizer.ask()
            if len(candidates) != population:
                raise LoRAHubError(
                    f"CMA-ES 回傳 {len(candidates)} 個候選，預期 {population}")
            losses = []
            for candidate_index, candidate in enumerate(candidates):
                objective = self.score(candidate)
                self.trajectory[-1].update({"generation": generation,
                                            "candidate_index": candidate_index})
                losses.append(objective)
                if objective < best_objective:
                    best_objective = objective
                    best_weights = [float(value) for value in candidate]
            optimizer.tell(candidates, losses)
            if generation % progress_every == 0 or generation == generations:
                print(f"[lorahub] 第 {generation}/{generations} 代 "
                      f"目前最佳目標值 {best_objective:.5f}", flush=True)

        expected = int(self.settings["expected_objective_evaluations_per_run"])
        if len(self.trajectory) != expected:
            raise LoRAHubError(
                f"CMA-ES 做了 {len(self.trajectory)} 次評估，預期 {expected}")
        if best_weights is None:
            raise LoRAHubError("CMA-ES 沒有產生任何候選")
        return best_weights, best_objective


def examples_fingerprint(examples):
    """示範樣本的指紋。產物的身分必須包含它，否則會被誤當通用權重。"""
    serialized = json.dumps(
        [{"instance_id": item.get("instance_id"), "prompt": item["prompt"],
          "output": item["output"]} for item in examples],
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def adapt(pool, examples, output_dir, *, base_model, run_seed=1, device="cuda",
          settings=None, torch_dtype=None):
    """為一組示範樣本搜尋權重，寫出合成後的 LoRA。

    回傳的報告含示範樣本指紋與所有種子，用來標明這份權重是為誰擬合的。
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = dict(settings or SEALED_LORAHUB)
    settings["example_seed"] = int(settings["example_seeds"][str(run_seed)])
    settings["optimizer_seed"] = int(settings["optimizer_seeds"][str(run_seed)])
    policy = validate_paper_settings(settings)
    if len(examples) != int(settings["shots"]):
        raise LoRAHubError(
            f"預期 {settings['shots']} 筆示範樣本，得到 {len(examples)}")

    selected_names = select_candidate_adapters(
        pool.names, int(settings["candidate_count"]), settings["example_seed"])
    print(f"[lorahub] 候選 {len(selected_names)} 個（seed {settings['example_seed']}）",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto",
        torch_dtype=torch_dtype or torch.bfloat16, low_cpu_mem_usage=True)
    by_name = dict(zip(pool.names, pool.paths))
    model = PeftModel.from_pretrained(
        base, str(by_name[selected_names[0]]), is_trainable=False)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    compute_device = next(model.parameters()).device
    batch, truncations = build_causal_lm_batch(
        tokenizer, examples, int(settings["adapt_max_tokens"]), compute_device)
    truncated = [item for item in truncations if item["truncated"]]
    if truncated:
        raise LoRAHubError(
            f"{len(truncated)} 筆示範樣本被截斷；截斷會改變擬合目標")

    cache = load_adapter_cache(pool, selected_names, device=compute_device)
    optimizer = PaperCMAESOptimizer(model, cache, selected_names, batch, settings)
    best_weights, best_objective = optimizer.run()

    final_state = compose_state_dict(best_weights, selected_names, cache)
    from peft.utils.save_and_load import set_peft_model_state_dict
    set_peft_model_state_dict(model, final_state)
    model.config.use_cache = True
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (output_dir / filename).is_file():
            raise LoRAHubError(f"LoRAHub 沒有寫出 {filename}")

    report = {
        "status": "complete",
        "method": "lorahub",
        "run_seed": run_seed,
        "example_seed": settings["example_seed"],
        "optimizer_seed": settings["optimizer_seed"],
        "selected_adapter_names": selected_names,
        "fewshot_instance_ids": [item.get("instance_id") for item in examples],
        # 這份權重只對這組樣本有意義——指紋是它身分的一部分。
        "fewshot_sha256": examples_fingerprint(examples),
        "best_weights": best_weights,
        "minimum_observed_objective": best_objective,
        "objective_evaluations": len(optimizer.trajectory),
        "policy": policy,
        "tokenization": truncations,
        "output_dir": str(output_dir),
    }
    (output_dir / "lorahub_info.json").write_text(
        json.dumps({**report, "trajectory": optimizer.trajectory},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    del model, base, tokenizer, cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return report

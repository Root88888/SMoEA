# -*- coding: utf-8 -*-
"""
system/inference.py

【生成執行層】路由決定去向之後的執行者：載入 base model
（unsloth/Meta-Llama-3.1-8B）、按任務熱切換 LoRA adapter、生成輸出。
對應架構圖的「ID Adapter → Inference → Output」一段。

生成行為照 MoEA-Trainer/inference_script2.py 原設定收編：
  prompt      = 樣本的 full_prompt 原文（base model 不套 chat template），
                前補 bos token
  tokenize    = 左 padding、truncation
  解碼        = do_sample=False、max_new_tokens=512、
                eos 併集（end_of_text 與 tokenizer eos 都保留）、
                stop_strings 防 few-shot 續寫、取 input 之後段、
                skip_special_tokens 解碼

adapter 目錄結構（本 repo convention，扁平）：
  adapter/task{N}/                  直接含 adapter 檔（adapter_config.json…）
  adapter/task{N}/checkpoint-*/     或多 checkpoint——自動 sort -V 取最新

【拒絕分支接口】ensure_rejection() 依 system.rejection_method 啟用 base、
預先產生的 artifact、Direct Arrow 或 Taskwise-K16 Arrow。Serving 不在 query 時
重新 merge，也不自動挑選最新 run；Direct Arrow 可在啟動後由 raw adapters 算一次
prototype，Taskwise-K16 則必須使用預先產生的 routing assets。

【scale up】adapter 逐任務獨立、熱切換 O(1) 換卡不重載 base；
任務擴充只需在 adapter/ 底下加目錄。
"""

import copy
import glob
import os
import re

from system.merged_model import (
    DenseDeltaController,
    MergedModelError,
    load_merged_model_artifact,
    validate_base_model_config,
    validate_inference_config,
)
from system.arrow_runtime import (
    ArrowController,
    ArrowRuntimeError,
    load_arrow_runtime_artifact,
)

STOP_STRINGS = ["\nQ:", "\nQuestion:", "\n\n\n"]   # 照 inference_script2


def _sort_v_key(path):
    """checkpoint-235 式名稱的自然排序鍵（等效 sort -V）。"""
    nums = re.findall(r"\d+", os.path.basename(path))
    return [int(x) for x in nums] if nums else [0]


def resolve_adapter_path(adapter_dir, task_key):
    """task{N} → 實際 adapter 路徑。直含 adapter 檔用之；
    多 checkpoint 取最新（sort -V 尾）。"""
    root = os.path.join(adapter_dir, task_key)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"找不到 adapter 目錄 {root}（convention：adapter/{task_key}/）")
    if os.path.exists(os.path.join(root, "adapter_config.json")):
        return root
    ckpts = sorted(glob.glob(os.path.join(root, "checkpoint-*")),
                   key=_sort_v_key)
    if not ckpts:
        raise FileNotFoundError(
            f"{root} 內無 adapter_config.json 亦無 checkpoint-*")
    return ckpts[-1]


class InferenceEngine:
    """base model 常駐、adapter 熱切換的生成引擎。"""

    def __init__(self, cfg):
        self.cfg = cfg["system"]
        self.model = None
        self.tokenizer = None
        self._loaded_adapters = {}     # task_key -> adapter_name
        self._active = None            # 目前生效的 adapter_name（None=純 base）
        merged_dir = (
            self.cfg.get("rejection_artifact_dir")
            or self.cfg.get("merged_model_dir")
        )
        configured_method = self.cfg.get("rejection_method")
        # Backward compatibility for existing deployments that only set the
        # old merged_model_dir field.
        self._rejection_method = (
            str(configured_method)
            if configured_method is not None
            else ("artifact" if merged_dir else "base")
        )
        supported = {"base", "artifact", "arrow", "taskwise_k16_arrow"}
        if self._rejection_method not in supported:
            raise MergedModelError(
                "system.rejection_method must be one of "
                + ", ".join(sorted(supported))
            )
        if (
            self.cfg.get("merged_model_required", False)
            and configured_method in {None, "artifact"}
            and not merged_dir
        ):
            raise MergedModelError(
                "production 要求 merged model，但 system.merged_model_dir 未設定")
        if self._rejection_method == "artifact" and not merged_dir:
            raise MergedModelError(
                "rejection_method=artifact 但未設定 system.rejection_artifact_dir")
        self._merged_artifact = (
            load_merged_model_artifact(
                merged_dir,
                expected_base_model=self.cfg["base_model"],
            )
            if self._rejection_method == "artifact"
            else None
        )
        if self._merged_artifact is not None:
            validate_base_model_config(self._merged_artifact)
            validate_inference_config(self._merged_artifact, self.cfg)
        self._merged_adapter_name = "__selected_merged__"
        self._dense_controller = None
        self._arrow_artifact = None
        self._arrow_controller = None
        if self._rejection_method in {"arrow", "taskwise_k16_arrow"}:
            self._arrow_artifact = load_arrow_runtime_artifact(
                self._rejection_method,
                router_dir=self.cfg.get("rejection_router_dir"),
                adapter_manifest=self.cfg.get("rejection_adapter_manifest"),
                adapter_root=self.cfg.get("rejection_adapter_root"),
                expected_base_model=self.cfg["base_model"],
            )
            self._arrow_controller = ArrowController(self._arrow_artifact)

    # ------------------------------------------------------------------
    # 載入
    # ------------------------------------------------------------------
    def _base_revision_kwargs(self):
        if self._merged_artifact is None:
            return {}
        revision = self._merged_artifact.base_model_revision
        if revision in {"local", "unresolved"}:
            return {}
        return {"revision": revision}

    def load_base(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = self.cfg["base_model"]
        print(f"[system] 載入 base model {name} "
              f"({'4bit' if self.cfg['load_in_4bit'] else self.cfg['dtype']})…",
              flush=True)
        revision_kwargs = self._base_revision_kwargs()
        self.tokenizer = AutoTokenizer.from_pretrained(name, **revision_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"   # 照原設定
        kw = {"device_map": "auto", **revision_kwargs}
        if self.cfg["load_in_4bit"]:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["torch_dtype"] = getattr(torch, self.cfg["dtype"])
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw)
        self.model.eval()
        self._load_merged_weights()
        print(f"[system] base model 就緒 "
              f"({next(self.model.parameters()).device})", flush=True)

    def _load_merged_weights(self):
        if self._merged_artifact is None:
            return
        artifact = self._merged_artifact
        if artifact.format == "dense_delta_v1":
            self._dense_controller = DenseDeltaController.attach(
                self.model, artifact)
            self._dense_controller.disable()
            return
        if artifact.format == "peft_adapter_v1":
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                str(artifact.directory),
                adapter_name=self._merged_adapter_name,
            )
            self._disable_peft_layers()
            return
        raise MergedModelError(f"不支援 merged model 格式 {artifact.format!r}")

    def _disable_peft_layers(self):
        tuner = getattr(self.model, "base_model", self.model)
        if hasattr(tuner, "disable_adapter_layers"):
            tuner.disable_adapter_layers()

    def _enable_peft_layers(self):
        tuner = getattr(self.model, "base_model", self.model)
        if hasattr(tuner, "enable_adapter_layers"):
            tuner.enable_adapter_layers()

    def ensure_adapter(self, task_key):
        """切到指定任務的 adapter（未載過則熱載入）。"""
        self.load_base()
        if self._arrow_controller is not None:
            self._arrow_controller.disable()
        if self._dense_controller is not None:
            self._dense_controller.disable()
        if task_key not in self._loaded_adapters:
            from peft import PeftModel
            path = resolve_adapter_path(self.cfg["adapter_dir"], task_key)
            print(f"[system] 載入 adapter {task_key} ← {path}", flush=True)
            if not hasattr(self.model, "peft_config"):
                self.model = PeftModel.from_pretrained(
                    self.model, path, adapter_name=task_key)
            else:
                self.model.load_adapter(path, adapter_name=task_key)
            self._loaded_adapters[task_key] = task_key
        self._enable_peft_layers()
        if self._active != task_key:
            self.model.set_adapter(task_key)
            self._active = task_key

    def ensure_merged(self):
        """切到啟動時明確選定的 merged model，不重載 base model。"""
        if self._merged_artifact is None:
            raise MergedModelError(
                "拒絕分支未選擇 merged artifact")
        self.load_base()
        if self._arrow_controller is not None:
            self._arrow_controller.disable()
        artifact = self._merged_artifact
        if artifact.format == "dense_delta_v1":
            self._disable_peft_layers()
            self._dense_controller.enable()
        else:
            if self._dense_controller is not None:
                self._dense_controller.disable()
            self._enable_peft_layers()
            self.model.set_adapter(self._merged_adapter_name)
        self._active = self._merged_adapter_name
        return {
            "method": "artifact",
            "condition_id": artifact.condition_id,
            "run_id": artifact.run_id,
            "format": artifact.format,
        }

    def ensure_base(self):
        """切到沒有任何 task／merge／Arrow update 的 base model。"""
        self.load_base()
        if self._arrow_controller is not None:
            self._arrow_controller.disable()
        if self._dense_controller is not None:
            self._dense_controller.disable()
        self._disable_peft_layers()
        self._active = None
        return {
            "method": "base",
            "condition_id": "base",
            "run_id": None,
            "format": "base_model",
        }

    def ensure_arrow(self):
        """切到已選定的 Arrow condition，保留同一份 base model。"""
        if self._arrow_artifact is None or self._arrow_controller is None:
            raise ArrowRuntimeError("拒絕分支未選擇 Arrow condition")
        self.load_base()
        if self._dense_controller is not None:
            self._dense_controller.disable()
        self._disable_peft_layers()
        self._arrow_controller.attach(self.model)
        self._arrow_controller.enable()
        self._active = f"__{self._arrow_artifact.condition_id}__"
        return {
            "method": self._arrow_artifact.condition_id,
            "condition_id": self._arrow_artifact.condition_id,
            "run_id": self._arrow_artifact.run_id,
            "format": "arrow_routing_v1",
            "expert_count": len(self._arrow_artifact.adapter_paths),
        }

    def ensure_rejection(self):
        """Activate the configured rejection method and return its identity."""
        if self._rejection_method == "base":
            return self.ensure_base()
        if self._rejection_method == "artifact":
            return self.ensure_merged()
        return self.ensure_arrow()

    def load_adapters_merged(self, weights, merged_name="merged"):
        """【拒絕分支用】以權重合成多個任務 adapter 並切換生效。

        weights: {task_key: float}，例 {"task23": 0.6, "task10": 0.4}。
        合成方式 PEFT add_weighted_adapter(combination_type="linear")。
        回傳生效的 adapter 名稱。
        """
        self.load_base()
        for tk in weights:
            self.ensure_adapter(tk)     # 確保成員都已載入
        names = list(weights.keys())
        ws = [float(weights[n]) for n in names]
        if merged_name in getattr(self.model, "peft_config", {}):
            self.model.delete_adapter(merged_name)
        self.model.add_weighted_adapter(
            adapters=names, weights=ws, adapter_name=merged_name,
            combination_type="linear")
        self.model.set_adapter(merged_name)
        self._active = merged_name
        return merged_name

    def unload(self):
        """釋放模型（分時載卸模式用）。"""
        import gc
        import torch
        if self._dense_controller is not None:
            self._dense_controller.close()
            self._dense_controller = None
        if self._arrow_controller is not None:
            self._arrow_controller.close()
        self.model = None
        self.tokenizer = None
        self._loaded_adapters, self._active = {}, None
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 生成（照 inference_script2 原設定）
    # ------------------------------------------------------------------
    def _generation_config(self, overrides=None):
        overrides = overrides or {}
        gc_ = copy.deepcopy(self.model.generation_config)
        gc_.max_new_tokens = int(
            overrides.get("max_new_tokens", self.cfg["max_new_tokens"])
        )
        gc_.pad_token_id = self.tokenizer.pad_token_id
        eos = gc_.eos_token_id
        eos_ids = [] if eos is None else ([eos] if isinstance(eos, int)
                                          else list(eos))
        if (self.tokenizer.eos_token_id is not None
                and self.tokenizer.eos_token_id not in eos_ids):
            eos_ids.append(self.tokenizer.eos_token_id)
        gc_.eos_token_id = (None if not eos_ids else
                            eos_ids[0] if len(eos_ids) == 1 else eos_ids)
        gc_.max_length = None
        gc_.do_sample = False          # 貪婪、可重現
        gc_.temperature = None
        gc_.top_p = None
        gc_.stop_strings = list(overrides.get("stop_strings", STOP_STRINGS))
        return gc_

    def _with_bos(self, prompts):
        bos = self.tokenizer.bos_token
        return [
            prompt
            if (not bos or prompt.startswith(bos))
            else f"{bos}{prompt}"
            for prompt in prompts
        ]

    def prompt_lengths(self, prompts):
        """Return unpadded token counts under the production prompt format."""
        encoded = self.tokenizer(
            self._with_bos(prompts),
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        if getattr(encoded, "ndim", 0) == 1:
            return [int(encoded.shape[-1])]
        if len(encoded) > 0 and isinstance(encoded[0], int):
            return [len(encoded)]
        return [len(tokens) for tokens in encoded]

    def generate(self, prompts, generation_overrides=None):
        """list[str]（full_prompt 原文）→ list[str] 模型輸出。"""
        import torch
        assert self.model is not None, "先 ensure_adapter/load_base"
        overrides = generation_overrides or {}
        max_input_tokens = int(
            overrides.get("max_input_tokens", self.cfg["max_input_tokens"])
        )
        ps = self._with_bos(prompts)
        if overrides.get("reject_prompt_truncation", False):
            lengths = self.prompt_lengths(prompts)
            too_long = [length for length in lengths if length > max_input_tokens]
            if too_long:
                raise ValueError(
                    f"{len(too_long)} prompts exceed max_input_tokens="
                    f"{max_input_tokens}; refusing truncation"
                )
        inputs = self.tokenizer(
            ps, return_tensors="pt", padding=True, truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=False)   # bos 已手動補，避免重複
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, generation_config=self._generation_config(overrides),
                tokenizer=self.tokenizer)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

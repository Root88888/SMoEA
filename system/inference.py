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
from dataclasses import dataclass

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
from system.registry import load_registry

SUPPORTED_REJECTION_METHODS = ("base", "artifact", "arrow", "taskwise_k16_arrow")
CONFIGURED_SELECTION_ID = "(config)"

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


@dataclass(frozen=True)
class RejectionSelection:
    """一組**已完整驗證**、可以啟用的拒絕方法設定。

    只有通過驗證的設定才會被建成 RejectionSelection，因此持有一個就等於
    「這個選擇可以安全啟用」。切換時先建新的、成功了才換掉舊的——驗證失敗
    時現行選擇原封不動（ADR-0001）。
    """

    id: str
    method: str
    merged_artifact: object = None
    arrow_artifact: object = None

    def identity(self):
        """寫進批次輸出、互動模式顯示用的來源身分。"""
        if self.method == "base":
            return {"method": "base", "condition_id": "base",
                    "run_id": None, "format": "base_model"}
        if self.method == "artifact":
            artifact = self.merged_artifact
            return {"method": "artifact",
                    "condition_id": artifact.condition_id,
                    "run_id": artifact.run_id,
                    "format": artifact.format}
        artifact = self.arrow_artifact
        return {"method": artifact.condition_id,
                "condition_id": artifact.condition_id,
                "run_id": artifact.run_id,
                "format": "arrow_routing_v1",
                "expert_count": len(artifact.adapter_paths)}


class InferenceEngine:
    """base model 常駐、adapter 熱切換的生成引擎。"""

    def __init__(self, cfg):
        self.cfg = cfg["system"]
        self.model = None
        self.tokenizer = None
        self._loaded_adapters = {}     # task_key -> adapter_name
        self._active = None            # 目前生效的 adapter_name（None=純 base）
        self._merged_adapter_name = "__selected_merged__"
        self._dense_controller = None
        self._arrow_controller = None
        self._attached = False
        registry_path = self.cfg.get("artifact_registry")
        self._registry = (
            load_registry(registry_path) if registry_path else None)
        self._selection = self._validated_selection(
            CONFIGURED_SELECTION_ID,
            method=self.cfg.get("rejection_method"),
            artifact_dir=self.cfg.get("rejection_artifact_dir"),
            router_dir=self.cfg.get("rejection_router_dir"),
            adapter_manifest=self.cfg.get("rejection_adapter_manifest"),
            adapter_root=self.cfg.get("rejection_adapter_root"),
        )

    # ------------------------------------------------------------------
    # 拒絕方法的選擇（ADR-0001）
    # ------------------------------------------------------------------
    def _validated_selection(self, selection_id, *, method, artifact_dir,
                             router_dir, adapter_manifest, adapter_root):
        """完整驗證一組設定並回傳 RejectionSelection；失敗則丟例外、不動現狀。"""
        method = "base" if method is None else str(method)
        if method not in SUPPORTED_REJECTION_METHODS:
            raise MergedModelError(
                "system.rejection_method must be one of "
                + ", ".join(sorted(SUPPORTED_REJECTION_METHODS)))
        if method == "base":
            return RejectionSelection(id=selection_id, method="base")
        if method == "artifact":
            if not artifact_dir:
                raise MergedModelError(
                    "rejection_method=artifact 但未設定 system.rejection_artifact_dir")
            artifact = load_merged_model_artifact(
                artifact_dir, expected_base_model=self.cfg["base_model"])
            validate_base_model_config(artifact)
            validate_inference_config(artifact, self.cfg)
            return RejectionSelection(
                id=selection_id, method="artifact", merged_artifact=artifact)
        artifact = load_arrow_runtime_artifact(
            method,
            router_dir=router_dir,
            adapter_manifest=adapter_manifest,
            adapter_root=adapter_root,
            expected_base_model=self.cfg["base_model"],
        )
        return RejectionSelection(
            id=selection_id, method=method, arrow_artifact=artifact)

    def available_rejections(self):
        """registry 宣告的可選項目；未設定 registry 時為空。"""
        return () if self._registry is None else self._registry.entries

    def current_rejection(self):
        """目前選定的拒絕方法身分（尚未啟用也可查）。"""
        return {"id": self._selection.id, **self._selection.identity()}

    def select_rejection(self, entry_id):
        """依 registry id 換掉拒絕方法。驗證失敗時現行選擇不受影響。"""
        if self._registry is None:
            raise MergedModelError(
                "未設定 system.artifact_registry，無法依 id 選擇 artifact")
        entry = self._registry.get(entry_id)
        overrides = entry.as_system_overrides()
        selection = self._validated_selection(
            entry.id,
            method=overrides["rejection_method"],
            artifact_dir=overrides["rejection_artifact_dir"],
            router_dir=overrides["rejection_router_dir"],
            adapter_manifest=overrides["rejection_adapter_manifest"],
            adapter_root=overrides["rejection_adapter_root"],
        )
        # 驗證已通過，才動現行狀態。
        self._detach_selection()
        self._selection = selection
        self._attach_selection()
        return self.current_rejection()

    # ------------------------------------------------------------------
    # 載入
    # ------------------------------------------------------------------
    def _base_revision_kwargs(self):
        if self._selection.merged_artifact is None:
            return {}
        revision = self._selection.merged_artifact.base_model_revision
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
        self._attach_selection()
        print(f"[system] base model 就緒 "
              f"({next(self.model.parameters()).device})", flush=True)

    def _attach_selection(self):
        """把目前選定的 artifact 掛上已載入的 base model（停用狀態）。

        一次只掛一組權重：dense delta 每個 3.76 GB，同時常駐多組是浪費。
        切換的成本是重掛一次，base model 全程不重載。
        """
        if self.model is None or self._attached:
            return
        artifact = self._selection.merged_artifact
        if artifact is not None:
            if artifact.format == "dense_delta_v1":
                self._dense_controller = DenseDeltaController.attach(
                    self.model, artifact)
                self._dense_controller.disable()
            elif artifact.format == "peft_adapter_v1":
                from peft import PeftModel
                if not hasattr(self.model, "peft_config"):
                    self.model = PeftModel.from_pretrained(
                        self.model, str(artifact.directory),
                        adapter_name=self._merged_adapter_name)
                else:
                    self.model.load_adapter(
                        str(artifact.directory),
                        adapter_name=self._merged_adapter_name)
                self._disable_peft_layers()
            else:
                raise MergedModelError(
                    f"不支援 merged model 格式 {artifact.format!r}")
        if self._selection.arrow_artifact is not None:
            self._arrow_controller = ArrowController(
                self._selection.arrow_artifact)
        self._attached = True

    def _detach_selection(self):
        """卸下目前選定 artifact 的權重，讓下一個選擇能乾淨掛上。"""
        if self._dense_controller is not None:
            self._dense_controller.close()      # 移除 forward hook
            self._dense_controller = None
        if self._arrow_controller is not None:
            self._arrow_controller.close()
            self._arrow_controller = None
        if (self.model is not None
                and self._selection.merged_artifact is not None
                and self._selection.merged_artifact.format == "peft_adapter_v1"
                and self._merged_adapter_name in getattr(
                    self.model, "peft_config", {})):
            self.model.delete_adapter(self._merged_adapter_name)
        if self._active == self._merged_adapter_name:
            self._active = None
        self._attached = False

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

    def _activate_artifact(self):
        """切到目前選定的 merged artifact，不重載 base model。"""
        artifact = self._selection.merged_artifact
        if artifact is None:
            raise MergedModelError("拒絕分支未選擇 merged artifact")
        self.load_base()
        if self._arrow_controller is not None:
            self._arrow_controller.disable()
        if artifact.format == "dense_delta_v1":
            self._disable_peft_layers()
            self._dense_controller.enable()
        else:
            if self._dense_controller is not None:
                self._dense_controller.disable()
            self._enable_peft_layers()
            self.model.set_adapter(self._merged_adapter_name)
        self._active = self._merged_adapter_name
        return self._selection.identity()

    def _activate_base(self):
        """切到沒有任何 task／merge／Arrow update 的 base model。"""
        self.load_base()
        if self._arrow_controller is not None:
            self._arrow_controller.disable()
        if self._dense_controller is not None:
            self._dense_controller.disable()
        self._disable_peft_layers()
        self._active = None
        return self._selection.identity()

    def _activate_arrow(self):
        """切到目前選定的 Arrow condition，保留同一份 base model。"""
        if self._selection.arrow_artifact is None or self._arrow_controller is None:
            raise ArrowRuntimeError("拒絕分支未選擇 Arrow condition")
        self.load_base()
        if self._dense_controller is not None:
            self._dense_controller.disable()
        self._disable_peft_layers()
        self._arrow_controller.attach(self.model)
        self._arrow_controller.enable()
        self._active = f"__{self._selection.arrow_artifact.condition_id}__"
        return self._selection.identity()

    def ensure_rejection(self):
        """啟用目前選定的拒絕方法並回傳其身分。"""
        if self._selection.method == "base":
            return self._activate_base()
        if self._selection.method == "artifact":
            return self._activate_artifact()
        return self._activate_arrow()

    def unload(self):
        """釋放模型（分時載卸模式用）。"""
        import gc
        import torch
        self._detach_selection()
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

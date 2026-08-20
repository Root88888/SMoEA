# -*- coding: utf-8 -*-
"""
system/adapter_pool.py — pool150 的 adapter 清單載入與驗證

線上 merge 的輸入（ADR-0002）。本模組從 MoEA-Trainer-delivery 搬入
（`src/adapter_inventory.py` 與 `src/adamerging/pool150.py` 的載入部分），
只保留 serving 需要的部分：manifest 解析、LoRA 因子讀取、以及跨 adapter 的
一致性檢查。訓練、抽樣與實驗生命週期的程式碼一律不搬。

本專案只有一個 adapter 池：**pool150**（adapter slot task0–task48 與
task50–task150，task49 未指派）。程式碼中不使用 pool50 命名——那個池是歷史
實驗，不在交付範圍。

manifest 格式（與 producer 的 `pool150_in_domain_manifest.json` 相同）：

    {"schema_version": 1, "pool_id": "pool150_in_domain", "count": 150,
     "adapters": [{"name": "task0", "path": "adapter/task0"}, ...]}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CANONICAL_ADAPTER_COUNT = 150
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


class AdapterPoolError(ValueError):
    """adapter manifest 或 LoRA 權重不符合 pool150 契約。"""


# ---------------------------------------------------------------------------
# LoRA 鍵名
# ---------------------------------------------------------------------------
def paired_key(a_key: str) -> str:
    """lora_A 的鍵名 → 同一模組的 lora_B 鍵名。"""
    return a_key.replace(".lora_A.", ".lora_B.")


def module_name_from_key(a_key: str) -> str:
    """LoRA 鍵名 → base model 中的模組名（去掉 PEFT 包裝前綴）。"""
    name = a_key.split(".lora_A.", 1)[0]
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def layer_index_from_key(a_key: str) -> int:
    """LoRA 鍵名 → transformer 層編號。"""
    marker = ".layers."
    if marker not in a_key:
        raise AdapterPoolError(f"LoRA 鍵名沒有層編號：{a_key}")
    return int(a_key.split(marker, 1)[1].split(".", 1)[0])


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload) -> None:
    """原子寫出 JSON（先寫 .tmp 再 replace）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def adapter_signature(config: dict) -> dict:
    """merge 前必須逐項相同的 LoRA 設定。"""
    return {
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "r": int(config["r"]),
        "lora_alpha": float(config["lora_alpha"]),
        "target_modules": sorted(config["target_modules"]),
        "use_rslora": bool(config.get("use_rslora", False)),
        "use_dora": bool(config.get("use_dora", False)),
        "fan_in_fan_out": bool(config.get("fan_in_fan_out", False)),
        "bias": config.get("bias"),
    }


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdapterEntry:
    name: str
    path: Path


@dataclass(frozen=True)
class AdapterInventory:
    """已驗證的有序 adapter 清單。順序即 merge 的輸入順序。"""

    manifest_path: Path
    entries: tuple[AdapterEntry, ...]
    pool_id: str | None

    @property
    def names(self) -> list[str]:
        return [entry.name for entry in self.entries]

    @property
    def paths(self) -> list[Path]:
        return [entry.path for entry in self.entries]


def load_adapter_inventory(
    manifest_path: str | Path,
    *,
    expected_count: int | None = CANONICAL_ADAPTER_COUNT,
    required_files: Iterable[str] = REQUIRED_ADAPTER_FILES,
    path_root: str | Path | None = None,
) -> AdapterInventory:
    """讀一份 manifest 並強制其有序 adapter 池的不變式。"""
    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterPoolError(f"讀不到 adapter manifest {path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise AdapterPoolError(f"adapter manifest {path} 必須是 JSON 物件")
    raw_entries = payload.get("adapters")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise AdapterPoolError(f"adapter manifest {path} 的 adapters 不得為空")

    resolved_root = Path(path_root) if path_root is not None else path.parent
    entries: list[AdapterEntry] = []
    for index, item in enumerate(raw_entries):
        if not isinstance(item, dict) or not item.get("name") or not item.get("path"):
            raise AdapterPoolError(
                f"adapter manifest {path} 第 {index} 項缺少 name 或 path")
        entry_path = Path(item["path"])
        if not entry_path.is_absolute():
            entry_path = resolved_root / entry_path
        entries.append(AdapterEntry(name=str(item["name"]), path=entry_path))

    names = [entry.name for entry in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise AdapterPoolError(f"adapter 名稱重複：{duplicates}")
    path_strings = [str(entry.path) for entry in entries]
    duplicate_paths = sorted(
        {item for item in path_strings if path_strings.count(item) > 1})
    if duplicate_paths:
        raise AdapterPoolError(f"adapter 路徑重複：{duplicate_paths}")

    declared = payload.get("count")
    if declared is not None and int(declared) != len(entries):
        raise AdapterPoolError(
            f"manifest 宣告 count={declared}，實際 {len(entries)} 筆")
    if expected_count is not None and len(entries) != int(expected_count):
        raise AdapterPoolError(
            f"預期 {expected_count} 個 adapter，找到 {len(entries)} 個")

    missing = [
        str(entry.path / filename)
        for entry in entries
        for filename in tuple(required_files)
        if not (entry.path / filename).is_file()
    ]
    if missing:
        preview = "、".join(missing[:5])
        suffix = f"（另有 {len(missing) - 5} 個）" if len(missing) > 5 else ""
        raise FileNotFoundError(f"adapter 檔案缺漏：{preview}{suffix}")

    pool_id = payload.get("pool_id")
    return AdapterInventory(
        manifest_path=path,
        entries=tuple(entries),
        pool_id=None if pool_id is None else str(pool_id),
    )


# ---------------------------------------------------------------------------
# LoRA 權重
# ---------------------------------------------------------------------------
@dataclass
class AdapterPool:
    """已載入的 pool150 LoRA 因子與其身分資訊。"""

    names: list[str]
    paths: list[Path]
    weights: list[dict]
    configs: list[dict]
    scaling: float
    rank: int
    alpha: float
    a_keys: list[str]
    source_files: list[dict]
    pool_id: str | None

    def fingerprint(self) -> dict:
        """pool 的完整指紋：換掉任何一個 adapter 都會改變它。

        寫進 artifact 身分，避免「同一個 run_id 對應到兩份不同權重」——
        producer 只記錄 adapter_ids，換掉某個 adapter 的權重看不出來。
        """
        digest = hashlib.sha256()
        for item in self.source_files:
            digest.update(f"{item['name']}:{item['sha256']}".encode("utf-8"))
        return {
            "id": self.pool_id,
            "adapter_ids": list(self.names),
            "adapter_count": len(self.names),
            "adapters_sha256": digest.hexdigest(),
        }


def load_adapter_pool(
    manifest_path: str | Path,
    *,
    expected_adapters: int = CANONICAL_ADAPTER_COUNT,
    path_root: str | Path | None = None,
) -> AdapterPool:
    """載入 pool150 的全部 LoRA 因子，並檢查它們可以被 merge。"""
    from safetensors.torch import load_file

    inventory = load_adapter_inventory(
        manifest_path, expected_count=expected_adapters, path_root=path_root)
    names, paths = inventory.names, inventory.paths

    configs = [
        json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        for path in paths
    ]
    signatures = [adapter_signature(item) for item in configs]
    if any(item != signatures[0] for item in signatures[1:]):
        differing = [
            names[index] for index, item in enumerate(signatures)
            if item != signatures[0]
        ]
        raise AdapterPoolError(
            f"LoRA 設定不一致，無法 merge：{differing[:5]} 與 {names[0]} 不同")
    signature = signatures[0]
    if signature["use_rslora"] or signature["use_dora"] or signature["fan_in_fan_out"]:
        raise AdapterPoolError("只支援標準非 DoRA 的 LoRA task vector")

    weights = []
    expected_keys = None
    source_files = []
    for name, path in zip(names, paths):
        tensor_path = path / "adapter_model.safetensors"
        current = load_file(str(tensor_path), device="cpu")
        keys = sorted(current)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise AdapterPoolError(f"{tensor_path} 的張量鍵與其他 adapter 不同")
        unexpected = [
            key for key in keys if ".lora_A." not in key and ".lora_B." not in key]
        if unexpected:
            raise AdapterPoolError(f"{tensor_path} 含非 LoRA 張量：{unexpected}")
        weights.append(current)
        source_files.append(
            {"name": name, "path": str(tensor_path), "sha256": sha256(tensor_path)})

    a_keys = sorted(key for key in expected_keys if ".lora_A." in key)
    if any(paired_key(key) not in expected_keys for key in a_keys):
        raise AdapterPoolError("有 lora_A 沒有配對的 lora_B")
    layers = sorted(layer_index_from_key(key) for key in a_keys)
    if layers != list(range(len(a_keys))):
        raise AdapterPoolError(
            f"預期每層恰好一個被改寫的模組，實得層編號 {layers[:5]}…")

    return AdapterPool(
        names=names,
        paths=paths,
        weights=weights,
        configs=configs,
        scaling=float(signature["lora_alpha"]) / int(signature["r"]),
        rank=int(signature["r"]),
        alpha=float(signature["lora_alpha"]),
        a_keys=a_keys,
        source_files=source_files,
        pool_id=inventory.pool_id,
    )


def manifest_from_adapter_dir(adapter_dir: str | Path, *, task_ids=None) -> dict:
    """由 repo 的 `adapter/task{N}/` 慣例推導出有序 manifest。

    交付時公司方可能只拿到 adapter 目錄而沒有 manifest。順序**依任務編號遞增**，
    與 producer 的 pool150 順序一致——順序會影響 merge 的結果，不能任意排。
    多 checkpoint 的目錄取編號最大的那個（與 system/inference.py 同一規則）。
    """
    import glob
    import re

    root = Path(adapter_dir)
    if not root.is_dir():
        raise AdapterPoolError(f"找不到 adapter 目錄 {root}")
    found = {}
    for entry in root.iterdir():
        match = re.fullmatch(r"task(\d+)", entry.name)
        if not match or not entry.is_dir():
            continue
        if (entry / "adapter_config.json").is_file():
            found[int(match.group(1))] = entry
            continue
        checkpoints = sorted(
            glob.glob(str(entry / "checkpoint-*")),
            key=lambda path: [int(x) for x in re.findall(r"\d+", Path(path).name)])
        if checkpoints:
            found[int(match.group(1))] = Path(checkpoints[-1])
    if not found:
        raise AdapterPoolError(
            f"{root} 下找不到任何 task{{N}}/ adapter（需含 adapter_config.json "
            f"或 checkpoint-*/）")
    wanted = sorted(found) if task_ids is None else sorted(task_ids)
    missing = [task for task in wanted if task not in found]
    if missing:
        raise AdapterPoolError(f"adapter 目錄缺少 task {missing[:5]}")
    return {
        "schema_version": 1,
        "pool_id": f"pool{len(wanted)}_from_adapter_dir",
        "count": len(wanted),
        "adapters": [{"name": f"task{task}", "path": str(found[task].resolve())}
                     for task in wanted],
    }

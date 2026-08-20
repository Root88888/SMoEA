# -*- coding: utf-8 -*-
"""
system/registry.py — Artifact Registry：可選 rejection artifact 的宣告清單

ADR-0001：SMoEA 不掃描 runs 目錄、不自動選「最新的 run」。使用者能選什麼，
完全由這份明確宣告的清單決定——沒有被列出的目錄，即使存在也不可選。

registry.json：

    {
      "schema_version": 1,
      "entries": [
        {"id": "base", "method": "base"},
        {"id": "ties", "method": "artifact", "source": "prepared",
         "artifact_dir": "ties_only/e3de085e/prepare/merged_model",
         "description": "TIES-only，pool150 全池"},
        {"id": "arrow", "method": "arrow",
         "adapter_manifest": "/data/pool150/manifest.json",
         "adapter_root": "/data/pool150"}
      ]
    }

相對路徑以 registry 檔所在目錄為基準，且不得以 `..` 逃出該目錄；絕對路徑照用
（操作者寫絕對路徑是明確宣告，與 --set system.rejection_artifact_dir=/data/… 同一層級）。

本模組只做宣告的解析與形式檢查，**不碰權重**。artifact 的 schema、base model 指紋、
dtype 與 checksum 驗證仍由 system/merged_model.py 與 system/arrow_runtime.py 負責，
發生在該項目被選用之前。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from system.merged_model import MergedModelError

SCHEMA_VERSION = 1

#: 每種方法必須提供哪些路徑欄位（其一即可的以 tuple 表示）。
_REQUIRED_PATHS: dict[str, tuple[str, ...]] = {
    "base": (),
    "artifact": ("artifact_dir",),
    "arrow": ("adapter_manifest",),
    "taskwise_k16_arrow": ("router_dir",),
}

_PATH_FIELDS = (
    "artifact_dir", "router_dir", "adapter_manifest", "adapter_root",
)

#: artifact 是怎麼來的。影響的是可追溯性的敘述，不影響驗證強度——
#: 兩種來源都要通過同一套 checksum 與 schema 驗證。
SOURCES = {
    "prepared": "現成",      # 離線預備或自遠端取得
    "runtime_merged": "線上生成",
}


class RegistryError(MergedModelError):
    """Registry 的宣告缺漏、重複或不安全。"""


@dataclass(frozen=True)
class RegistryEntry:
    """registry 中的一個可選項目。"""

    id: str
    method: str
    description: str | None
    source: str | None
    artifact_dir: Path | None
    router_dir: Path | None
    adapter_manifest: Path | None
    adapter_root: Path | None

    def as_system_overrides(self) -> dict[str, Any]:
        """轉成 InferenceEngine 認得的 system 設定片段。"""
        return {
            "rejection_method": self.method,
            "rejection_artifact_dir": _text(self.artifact_dir),
            "rejection_router_dir": _text(self.router_dir),
            "rejection_adapter_manifest": _text(self.adapter_manifest),
            "rejection_adapter_root": _text(self.adapter_root),
        }

    def summary(self) -> str:
        """`:artifact list` 用的單行說明。"""
        location = next(
            (str(path) for path in (self.artifact_dir, self.router_dir,
                                    self.adapter_manifest) if path is not None),
            "（不需外部檔案）",
        )
        origin = f"[{SOURCES[self.source]}] " if self.source else ""
        tail = f"  {self.description}" if self.description else ""
        return f"{self.id:16s} {origin}{self.method:18s} {location}{tail}"


@dataclass(frozen=True)
class Registry:
    """一份已解析的 registry。"""

    path: Path
    entries: tuple[RegistryEntry, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.entries)

    def get(self, entry_id: str) -> RegistryEntry:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        raise RegistryError(
            f"registry 沒有 id={entry_id!r} 的項目；可選：{', '.join(self.ids)}")

    def listing(self) -> str:
        header = f"registry {self.path}（{len(self.entries)} 項）"
        return "\n".join([header, *(f"  {e.summary()}" for e in self.entries)])


def _text(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _resolve(root: Path, value: Any, label: str, entry_id: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RegistryError(f"{entry_id}：{label} 必須是非空字串")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if ".." in candidate.parts:
        raise RegistryError(f"{entry_id}：{label} 不得以 .. 逃出 registry 目錄：{value!r}")
    return (root / candidate).resolve()


def path_for_registry(registry_path: str | Path, target: str | Path) -> str:
    """把要登記的路徑轉成 load_registry 解得開的形式。

    寫入端（merge_pool150.py、fetch_artifact.py）手上的通常是「相對於 cwd」
    的路徑——例如 configs 的 `artifact_root: artifacts` 就是相對的，於是登記
    成 `artifacts/<method>/<run>/…`。但本檔的契約是相對路徑以 **registry 檔
    所在目錄** 為基準，那份登記再解析一次就疊成 `artifacts/artifacts/…`，
    選用時找不到檔案。收斂在寫入端做：落在 registry 目錄底下的存相對路徑
    （registry 與 artifact 一起搬走仍然解得開），在外面的存絕對路徑。
    """
    root = Path(registry_path).resolve().parent
    resolved = Path(target).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _entry_from(payload: Any, root: Path, seen: set[str]) -> RegistryEntry:
    if not isinstance(payload, dict):
        raise RegistryError("registry 的每個項目都必須是 JSON 物件")
    entry_id = payload.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        raise RegistryError("registry 項目缺少非空的 id")
    if entry_id in seen:
        raise RegistryError(f"registry 的 id 重複：{entry_id!r}")
    seen.add(entry_id)

    method = payload.get("method")
    if method not in _REQUIRED_PATHS:
        raise RegistryError(
            f"{entry_id}：method 必須是 "
            + "、".join(sorted(_REQUIRED_PATHS))
            + f"；得到 {method!r}")

    for field in _REQUIRED_PATHS[method]:
        if not payload.get(field):
            raise RegistryError(f"{entry_id}：method={method} 必須提供 {field}")

    resolved: dict[str, Path | None] = {}
    for field in _PATH_FIELDS:
        value = payload.get(field)
        resolved[field] = (
            _resolve(root, value, field, entry_id) if value is not None else None)

    description = payload.get("description")
    if description is not None and not isinstance(description, str):
        raise RegistryError(f"{entry_id}：description 必須是字串")
    source = payload.get("source")
    if source is not None and source not in SOURCES:
        raise RegistryError(
            f"{entry_id}：source 必須是 {'、'.join(SOURCES)}；得到 {source!r}")

    return RegistryEntry(
        id=entry_id,
        method=method,
        description=description,
        source=source,
        artifact_dir=resolved["artifact_dir"],
        router_dir=resolved["router_dir"],
        adapter_manifest=resolved["adapter_manifest"],
        adapter_root=resolved["adapter_root"],
    )


def load_registry(path: str | Path) -> Registry:
    """讀取並檢查 registry 宣告；不驗證權重本身。"""
    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"讀不到 artifact registry {registry_path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise RegistryError("registry 必須是一個 JSON 物件")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RegistryError(
            f"不支援的 registry schema：{payload.get('schema_version')!r}")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise RegistryError("registry 至少要宣告一個項目")

    root = registry_path.parent.resolve()
    seen: set[str] = set()
    entries = tuple(_entry_from(item, root, seen) for item in raw_entries)
    return Registry(path=registry_path.resolve(), entries=entries)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把舊 producer 版本的 merged model manifest 補成 SMoEA 的契約。

早期 producer 寫出的 result.json 與 SMoEA 的驗證器有兩處不符，權重本身完全有效——
缺的只是宣告：

1. **沒有 inference 物件。** torch_dtype 取自權重檔中 tensor 的實際 dtype，
   quantization 固定 "none"（dense delta 不量化）。
2. **modules 只有 name，沒有 tensor_name。** producer 把兩者合寫成一個帶
   `.delta_weight` 後綴的張量名；SMoEA 需要分開的「模型內模組名」與「檔案內張量名」。

兩項都由權重檔本身推導、再與之對帳，不做字串猜測：張量名必須確實存在於
safetensors，且補完後 modules 的張量清單必須與檔案完全一致。任一項對不上就拒絕遷移。

只改寫 result.json，不動權重檔，因此 weights 的 sha256 保持不變。

  python scripts/migrate_artifact_manifest.py <artifact_dir> [...] [--dry-run]
  python scripts/migrate_artifact_manifest.py --scan <root> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from system.merged_model import (  # noqa: E402
    MergedModelError,
    load_merged_model_artifact,
)


class MigrationError(RuntimeError):
    """The artifact cannot be migrated without guessing."""


DELTA_SUFFIX = ".delta_weight"


def read_weight_file(directory: Path, payload: dict) -> dict[str, str]:
    """回傳權重檔中 {張量名: dtype}，作為遷移的唯一事實來源。"""
    weights = payload.get("weights")
    if not isinstance(weights, list) or len(weights) != 1:
        raise MigrationError("dense delta 應該只有一個權重檔")
    relative = weights[0].get("path")
    if not isinstance(relative, str) or not relative:
        raise MigrationError("權重項目沒有 path")
    path = directory / relative
    if not path.is_file():
        raise MigrationError(f"權重檔不存在：{path}")
    from safetensors import safe_open

    from system.merged_model import _SAFETENSORS_DTYPES

    with safe_open(str(path), framework="pt") as handle:
        names = {}
        for key in handle.keys():
            raw = str(handle.get_slice(key).get_dtype())
            # safetensors 用 BF16 這種寫法，torch 與 manifest 用 bfloat16。
            # 兩邊混用會讓 validate_inference_config 比對失敗。
            torch_name = _SAFETENSORS_DTYPES.get(raw)
            if torch_name is None:
                raise MigrationError(f"未知的張量 dtype：{raw!r}")
            names[key] = torch_name
        return names


def infer_block(file_dtypes: dict[str, str]) -> dict:
    """從權重檔中 tensor 的實際 dtype 推導 inference 物件。"""
    dtypes = set(file_dtypes.values())
    if len(dtypes) != 1:
        raise MigrationError(f"權重檔的 dtype 不一致：{sorted(dtypes)}")
    return {"torch_dtype": dtypes.pop(), "quantization": "none"}


def split_module_names(modules: list, file_dtypes: dict[str, str]) -> list:
    """把 producer 合寫的張量名拆成 name（模組）與 tensor_name（張量）。"""
    if not isinstance(modules, list) or not modules:
        raise MigrationError("manifest 沒有 modules 清單")
    split = []
    for module in modules:
        if not isinstance(module, dict):
            raise MigrationError("modules 的項目必須是物件")
        if module.get("tensor_name"):
            split.append(module)
            continue
        name = module.get("name")
        if name not in file_dtypes:
            raise MigrationError(
                f"modules 的 name={name!r} 不是權重檔中的張量，無法安全推導")
        if not name.endswith(DELTA_SUFFIX):
            raise MigrationError(f"張量名 {name!r} 沒有預期的 {DELTA_SUFFIX} 後綴")
        split.append({**module,
                      "name": name[: -len(DELTA_SUFFIX)],
                      "tensor_name": name})
    tensor_names = {module["tensor_name"] for module in split}
    if tensor_names != set(file_dtypes):
        raise MigrationError("補完後的張量清單與權重檔不一致")
    return split


def migrate(directory: Path, *, dry_run: bool) -> str:
    manifest_path = directory / "result.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(f"讀不到 {manifest_path}：{exc}") from exc
    modules = payload.get("modules")
    needs_inference = not isinstance(payload.get("inference"), dict)
    needs_split = isinstance(modules, list) and any(
        isinstance(item, dict) and not item.get("tensor_name") for item in modules)
    if not needs_inference and not needs_split:
        return "skip  已符合契約"

    file_dtypes = read_weight_file(directory, payload)
    block = infer_block(file_dtypes) if needs_inference else payload["inference"]
    split = split_module_names(modules, file_dtypes) if needs_split else modules
    todo = ", ".join(
        part for part, needed in (("inference", needs_inference),
                                  ("tensor_name", needs_split)) if needed)
    if dry_run:
        return f"would  補上 {todo}（dtype={block['torch_dtype']}）"

    # 依 result.json 既有的鍵序插在 base_model 之後，維持與 producer 相同的版面。
    migrated = {}
    for key, value in payload.items():
        migrated[key] = split if key == "modules" else value
        if key == "base_model":
            migrated["inference"] = block
    if "inference" not in migrated:
        migrated["inference"] = block

    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(migrated, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary, manifest_path)

    # 寫完立刻以正式驗證器複驗，確保補出來的是真的能用的 artifact。
    try:
        load_merged_model_artifact(directory)
    except MergedModelError as exc:
        raise MigrationError(f"補寫後仍未通過驗證：{exc}") from exc
    return f"ok    補上 inference={block}"


def find_artifacts(root: Path) -> list[Path]:
    return sorted(
        path.parent for path in root.rglob("result.json")
        if path.parent.name == "merged_model"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="*", type=Path,
                        help="artifact 目錄（含 result.json）")
    parser.add_argument("--scan", type=Path, default=None,
                        help="遞迴掃描此根目錄下所有 merged_model/ artifact")
    parser.add_argument("--dry-run", action="store_true",
                        help="只報告會做什麼，不寫檔")
    args = parser.parse_args()

    targets = list(args.directories)
    if args.scan is not None:
        targets.extend(find_artifacts(args.scan))
    if not targets:
        parser.error("請指定 artifact 目錄，或用 --scan <root>")

    failures = 0
    for directory in targets:
        try:
            print(f"{migrate(directory, dry_run=args.dry_run)}  {directory}")
        except MigrationError as exc:
            failures += 1
            print(f"FAIL  {exc}  {directory}")
    if failures:
        raise SystemExit(f"{failures} 個 artifact 遷移失敗")


if __name__ == "__main__":
    main()

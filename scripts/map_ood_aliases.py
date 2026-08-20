#!/usr/bin/env python3
"""Map benchmark-native OOD IDs to SMoEA internal aliases without renaming data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def map_ood_task149(dataset_dir: str | Path) -> Path:
    dataset = Path(dataset_dir)
    sources = [
        dataset / "ood_test_data" / f"task149_test.{extension}"
        for extension in ("json", "jsonl")
    ]
    sources = [path for path in sources if path.is_file()]
    aliases = [
        dataset / "test_data" / f"task9149_test.{extension}"
        for extension in ("json", "jsonl")
    ]
    existing_aliases = [path for path in aliases if path.exists()]
    if len(sources) > 1:
        raise RuntimeError("OOD task149 同時存在 JSON 與 JSONL，請只保留一種格式")
    if not sources:
        if len(existing_aliases) == 1:
            return existing_aliases[0]
        raise FileNotFoundError(
            "缺 dataset/ood_test_data/task149_test.json（或 .jsonl）；"
            "不要把 OOD task149 覆寫到 source task149 的 flat test file"
        )
    source = sources[0]
    alias = dataset / "test_data" / f"task9149_test{source.suffix}"
    conflicting = [path for path in existing_aliases if path != alias]
    if conflicting:
        raise RuntimeError(f"存在衝突的 task9149 alias：{conflicting[0]}")
    if alias.exists():
        if alias.resolve() != source.resolve():
            raise RuntimeError(f"{alias} 已存在但未指向 {source}")
        return alias
    alias.parent.mkdir(parents=True, exist_ok=True)
    relative_source = os.path.relpath(source, start=alias.parent)
    alias.symlink_to(relative_source)
    return alias


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    args = parser.parse_args()
    alias = map_ood_task149(args.dataset_dir)
    print(f"[OOD alias] task149 → internal task9149: {alias}")


if __name__ == "__main__":
    main()

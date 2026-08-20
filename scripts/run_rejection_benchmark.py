#!/usr/bin/env python3
"""Run the fixed 15-OOD data through SMoEA's production rejection runtime.

拒絕方法可以用 --artifact <id> 從 registry 指定，與互動模式的 `:rejection use` 和
批次模式的 --artifact 同一套；不指定時沿用設定檔的 system.rejection_method。
三個入口共用同一條路徑（見 CONTEXT.md 不變式 2）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from router.config import add_config_args, load_config  # noqa: E402
from system.benchmark import GROUPS, run_rejection_benchmark  # noqa: E402
from system.inference import InferenceEngine  # noqa: E402


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser())
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group", choices=["all", *GROUPS], default="all")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--artifact", default=None,
                        help="要測哪一個拒絕方法；值為 system.artifact_registry "
                             "中宣告的 id。不給則用設定檔的 rejection_method")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    if cfg["system"].get("load_in_4bit", False) or cfg["system"].get("dtype") != "bfloat16":
        parser.error(
            "15-OOD benchmark requires --set system.dtype=bfloat16 and "
            "system.load_in_4bit=false"
        )
    groups = GROUPS if args.group == "all" else (args.group,)
    engine = InferenceEngine(cfg)
    if args.artifact is not None:
        current = engine.select_rejection(args.artifact)
        print(f"[benchmark] 拒絕方法：{current['id']}"
              f"（{current['condition_id']}"
              + (f":{current['run_id']}" if current["run_id"] else "") + "）")
    report = run_rejection_benchmark(
        engine,
        benchmark_root=args.benchmark_root,
        output_dir=args.output_dir,
        groups=groups,
        batch_size=args.batch_size or int(cfg["system"]["batch_size"]),
        smoke=args.smoke,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

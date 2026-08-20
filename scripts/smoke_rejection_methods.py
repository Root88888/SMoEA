#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐一啟用清單檔裡的每個拒絕方法，各生成一次，確認它們真的能服務。

**完全跳過 Router**——不讀路由資產、不做路由判定，因此與 data.routing_text
的設定無關。它回答的問題只有一個：清單檔宣告的每個 condition，是不是真的載得起來
並且答得出東西。

base model 只載一次，方法之間熱切換。

  python scripts/smoke_rejection_methods.py --set system.dtype=bfloat16
  python scripts/smoke_rejection_methods.py --only ties_only,arrow
"""

from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from router.config import add_config_args, load_config  # noqa: E402
from system.inference import InferenceEngine  # noqa: E402
from system.merged_model import MergedModelError  # noqa: E402
from system.rejection import run_rejection  # noqa: E402

DEFAULT_PROMPT = (
    "Explain in two sentences why the sky appears blue during the day.\n\n"
    "Answer:"
)


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser())
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="要送去生成的提示；預設是一句通用問題")
    parser.add_argument("--only", default=None,
                        help="只測這幾個 id，逗號分隔；預設測清單檔全部")
    parser.add_argument("--max-new-tokens", type=int, default=48,
                        help="每個方法生成幾個 token（煙霧測試不需要長輸出）")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    engine = InferenceEngine(cfg)
    entries = engine.available_rejections()
    if not entries:
        raise SystemExit(
            "清單檔沒有可選項目；確認 system.artifact_registry 指到正確的檔案")
    wanted = ([item.strip() for item in args.only.split(",")]
              if args.only else [entry.id for entry in entries])
    print(f"[smoke] 要測 {len(wanted)} 個：{', '.join(wanted)}", flush=True)
    print(f"[smoke] 提示：{args.prompt[:60]!r}", flush=True)

    results = []
    for entry_id in wanted:
        print(f"\n{'=' * 60}\n[smoke] {entry_id}", flush=True)
        started = time.time()
        try:
            engine.select_rejection(entry_id)
            outputs, identity = run_rejection(engine, [args.prompt])
            elapsed = time.time() - started
            text = outputs[0].strip()
            status = "OK" if text else "空輸出"
            results.append((status, entry_id, identity, elapsed, text))
            print(f"[{identity['condition_id']}"
                  + (f":{identity['run_id']}" if identity["run_id"] else "")
                  + f"] ({elapsed:.1f}s)")
            print(text[:400] or "（模型沒有產生任何文字）")
        except (MergedModelError, Exception) as exc:  # noqa: BLE001
            results.append(("失敗", entry_id, None, time.time() - started,
                            f"{type(exc).__name__}: {exc}"))
            print(f"失敗：{type(exc).__name__}: {str(exc)[:200]}", flush=True)

    print(f"\n{'=' * 60}\n[smoke] 總結")
    for status, entry_id, identity, elapsed, detail in results:
        source = ""
        if identity:
            source = f"{identity['condition_id']}"
            if identity["run_id"]:
                source += f":{identity['run_id'][:12]}"
        print(f"  {status:6s} {entry_id:20s} {elapsed:6.1f}s  {source}")
    failures = [item for item in results if item[0] != "OK"]
    if failures:
        raise SystemExit(f"{len(failures)} 個方法沒有通過")
    print("[smoke] 全部通過")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把遠端 artifact 的 condition 名稱改成 producer 的正式名稱。

早期上傳時把 condition 正規化成自訂簡寫（ties、dare-ties），與 producer 的
condition_id（ties_only、dare_ties_ta）不一致。權重本身沒有問題，要改的只有
路徑與說明檔裡的 condition_id。

權重用 Hugging Face 的伺服器端複製搬移，不重新上傳——3.76 GB 不會經過你的網路。
說明檔重新產生後上傳（幾 KB），最後刪掉舊路徑。整批寫成一個 commit，
中途失敗不會留下半新半舊的狀態。

  python scripts/rename_remote_condition.py --repo <org>/<repo> \
      --rename ties=ties_only --rename dare-ties=dare_ties_ta
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SUFFIX = "prepare/merged_model"
FILES = ("dense_delta.safetensors", "result.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--rename", action="append", required=True,
                        metavar="舊名=新名")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationCopy,
        CommitOperationDelete,
        HfApi,
        hf_hub_download,
    )

    api = HfApi(token=args.token)
    existing = {s.rfilename for s in api.repo_info(args.repo).siblings}

    operations = []
    summary = []
    for pair in args.rename:
        old, _, new = pair.partition("=")
        if not old or not new:
            raise SystemExit(f"--rename 格式應為 舊名=新名，得到 {pair!r}")
        runs = sorted({
            name.split("/")[1] for name in existing
            if name.startswith(f"{old}/") and name.endswith("result.json")
        })
        if not runs:
            print(f"[rename] {old}：遠端沒有這個 condition，跳過")
            continue
        for run_id in runs:
            old_dir = f"{old}/{run_id}/{SUFFIX}"
            new_dir = f"{new}/{run_id}/{SUFFIX}"

            # 說明檔重新產生：condition_id 改成新名稱，其餘原封不動。
            local = hf_hub_download(args.repo, f"{old_dir}/result.json",
                                    token=args.token)
            payload = json.loads(open(local, encoding="utf-8").read())
            before = payload.get("condition_id")
            payload["condition_id"] = new
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8")
            json.dump(payload, handle, ensure_ascii=False, indent=1)
            handle.close()

            # 權重：伺服器端複製，不經過本機網路。
            operations.append(CommitOperationCopy(
                src_path_in_repo=f"{old_dir}/dense_delta.safetensors",
                path_in_repo=f"{new_dir}/dense_delta.safetensors"))
            operations.append(CommitOperationAdd(
                path_in_repo=f"{new_dir}/result.json",
                path_or_fileobj=handle.name))
            for filename in FILES:
                operations.append(CommitOperationDelete(
                    path_in_repo=f"{old_dir}/{filename}"))
            summary.append(f"  {old_dir} → {new_dir}"
                           f"（condition_id {before} → {new}）")

    if not operations:
        print("[rename] 沒有需要改的項目")
        return
    print("[rename] 將執行：")
    for line in summary:
        print(line)
    if args.dry_run:
        print("[rename] --dry-run：未變更遠端")
        return

    api.create_commit(
        repo_id=args.repo, operations=operations,
        commit_message="rename conditions to producer naming")
    print(f"[rename] 完成。遠端現有檔案：")
    for name in sorted(s.rfilename for s in api.repo_info(args.repo).siblings):
        print("  ", name)


if __name__ == "__main__":
    main()

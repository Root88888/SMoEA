#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把本機的 rejection artifact 上傳到私有 Hugging Face repo（ADR-0002 第 6 節）。

上傳只是**交付通道**：SMoEA 執行期永遠只讀本地目錄，不會去 HF 取檔。對應的
下載端是 scripts/fetch_artifact.py，兩邊用同一套版面：

    <condition>/<run_id>/prepare/merged_model/{result.json,dense_delta.safetensors}

dense delta 是 Llama-3.1-8B 的衍生物（只含 down_proj 的差值，不含 base model
本身）。散布時 model card 必須標示 Built with Llama、附 Llama 3.1 Community
License 與 Acceptable Use Policy 連結——本腳本會把這三項寫進 card。
`--private` 可改為私有 repo。

  hf auth login                  # 先登入（或設 HF_TOKEN）
  python scripts/push_artifact.py --repo <org>/<repo> \
      --artifact /shared/artifacts/ties/<run_id>/prepare/merged_model
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from system.merged_model import load_merged_model_artifact  # noqa: E402

CARD = """---
license: llama3.1
base_model: {base_model}
tags:
  - smoea
  - lora-merge
---

# SMoEA rejection artifacts

Built with Llama.

SMoEA Router 拒絕分支使用的 merged model artifact。每一份都是 `{base_model}`
的 **dense delta**：只含 `down_proj` 的差值，**不含 base model 本身**——沒有
base model 無法使用。

## 版面

    <condition>/<run_id>/prepare/merged_model/
        result.json                # 契約：base model 指紋、dtype、checksum、模組清單
        dense_delta.safetensors    # 權重差值

## 取用

    python scripts/fetch_artifact.py --repo {repo_id} --condition <condition> \\
        --artifact-root <本機路徑> --registry <本機路徑>/registry.json

下載後由 SMoEA 逐檔核對 `result.json` 記錄的大小與 sha256；不符即拒絕載入。

## 授權

Built with Llama。衍生自 Llama 3.1，受
[Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)
規範，並適用
[Acceptable Use Policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md)。
訓練資料來自 Natural Instructions（Apache-2.0）。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="目標 repo id（<org>/<name>）")
    parser.add_argument("--artifact", required=True, action="append",
                        help="要上傳的 merged_model 目錄；可重複指定")
    parser.add_argument("--private", action="store_true",
                        help="建為私有 repo；預設公開（見檔頭的授權說明）")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--dry-run", action="store_true",
                        help="只驗證與列出將上傳的內容，不真的上傳")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)

    # 先全部驗證過，再開始上傳——不把壞掉的 artifact 傳上去。
    # 這一段刻意不需要登入：本機驗證與登入狀態無關。
    planned = []
    for directory in args.artifact:
        artifact = load_merged_model_artifact(directory)
        remote = (f"{artifact.condition_id}/{artifact.run_id}"
                  f"/prepare/merged_model")
        size = sum(path.stat().st_size for path in artifact.weight_files)
        planned.append((directory, artifact, remote, size))
        print(f"[push] 已驗證 {artifact.condition_id}:{artifact.run_id} "
              f"（{size / 1e9:.2f} GB）→ {remote}")
    total = sum(item[3] for item in planned)
    visibility = "私有" if args.private else "公開"
    print(f"[push] 合計 {total / 1e9:.2f} GB，目標 {args.repo}（{visibility}）")

    if args.dry_run:
        print("[push] --dry-run：未上傳任何檔案")
        return

    try:
        who = api.whoami()
    except Exception as exc:
        raise SystemExit(
            f"尚未登入 Hugging Face（{type(exc).__name__}）。"
            "先執行 hf auth login，或設定 HF_TOKEN。")
    print(f"[push] 身分：{who['name']}")

    api.create_repo(args.repo, private=args.private, exist_ok=True,
                    repo_type="model")
    base_model = planned[0][1].base_model_name
    api.upload_file(
        path_or_fileobj=CARD.format(
            base_model=base_model, repo_id=args.repo).encode("utf-8"),
        path_in_repo="README.md", repo_id=args.repo, token=args.token)

    for directory, artifact, remote, _size in planned:
        print(f"[push] 上傳 {remote} …", flush=True)
        api.upload_folder(
            folder_path=str(directory), path_in_repo=remote,
            repo_id=args.repo, token=args.token,
            commit_message=f"{artifact.condition_id}:{artifact.run_id}")
    print(f"[push] 完成 → https://huggingface.co/{args.repo}（{visibility}）")


if __name__ == "__main__":
    main()

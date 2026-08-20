#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把本機的 pool150 上傳到 Hugging Face，作為使用者建置時的 adapter 來源。

對應的下載端是 scripts/fetch_adapter_pool.py。上傳只是**交付通道**：SMoEA
執行期永遠只讀本地 `adapter/task{N}/`，不會去 HF 取檔。

遠端版面（與本地 `adapter/` 一致，下載後直接就位）：

    pool150_manifest.json          # 有序清單、逐檔 sha256 與大小、共用的 LoRA 設定
    task0/adapter_config.json
    task0/adapter_model.safetensors
    ...

只上傳 adapter 本身需要的兩個檔。訓練殘留物（optimizer.pt、rng_state、
每個 checkpoint 各一份的 tokenizer.json）不傳——那些佔了目錄的大半，
serving 與合成都用不到。

**manifest 最後才上傳。** 中途失敗的 repo 沒有 manifest，fetch 端會直接
判定不可用，不會拿到半套的池。

上傳的是 Llama-3.1-8B 的 LoRA 衍生物（只含 down_proj 的低秩因子，不含
base model 本身）。公開散布時 model card 必須標示 Built with Llama、附
Llama 3.1 Community License 與 Acceptable Use Policy 連結——本腳本會把
這三項寫進 card。`--private` 可改為私有 repo。

  hf auth login                  # 先登入（或設 HF_TOKEN）
  python scripts/push_adapter_pool.py --repo <org>/<repo> --dry-run
  python scripts/push_adapter_pool.py --repo <org>/<repo>
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from router.config import add_config_args, load_config  # noqa: E402
from system.adapter_pool import (  # noqa: E402
    CANONICAL_ADAPTER_COUNT,
    REQUIRED_ADAPTER_FILES,
    adapter_signature,
    load_adapter_inventory,
    manifest_from_adapter_dir,
    sha256,
    write_json,
)

MANIFEST_NAME = "pool150_manifest.json"

CARD = """---
license: llama3.1
base_model: {base_model}
library_name: peft
tags:
  - smoea
  - lora
  - adapter-pool
---

# SMoEA adapter pool ({count} LoRA adapters)

Built with Llama.

SMoEA Router 的專家池：{count} 個逐任務訓練的 LoRA adapter，皆以
`{base_model}` 為底。**不含 base model 本身**——沒有 base model 無法使用。

| | |
|---|---|
| adapter 數 | {count} |
| rank (`r`) | {r} |
| `lora_alpha` | {alpha} |
| `target_modules` | {targets} |
| 每個 adapter | 約 {per_mib:.0f} MiB |
| 全池 | 約 {total_gib:.2f} GiB |

## 版面

    pool150_manifest.json          # 有序清單、逐檔 sha256 與大小、共用的 LoRA 設定
    task{{N}}/adapter_config.json
    task{{N}}/adapter_model.safetensors

清單的順序即合成時的輸入順序（依任務編號遞增），**順序會影響 merge 的結果**。

## 取用

    python scripts/fetch_adapter_pool.py --repo {repo_id}

下載後逐檔核對 manifest 記錄的大小與 sha256；不符即拒絕。
`bash scripts/setup_workspace.sh` 會在 adapter 不齊時自動代跑這一步。

## 授權

衍生自 Llama 3.1，受
[Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)
規範，並適用
[Acceptable Use Policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md)。
訓練資料來自 Natural Instructions（Apache-2.0）。
"""


def collect(inventory):
    """逐 adapter 算 checksum，並確認整池的 LoRA 設定一致。

    設定不一致的池不能合成（system/adapter_pool.py 會擋），與其讓使用者
    下載完才發現，不如在上傳前就擋下來。
    """
    entries = []
    signatures = {}
    for entry in inventory.entries:
        config = json.loads(
            (entry.path / "adapter_config.json").read_text(encoding="utf-8"))
        signature = adapter_signature(config)
        signatures[entry.name] = signature
        files = {}
        for filename in REQUIRED_ADAPTER_FILES:
            path = entry.path / filename
            files[filename] = {"size": path.stat().st_size,
                               "sha256": sha256(path)}
        entries.append({"name": entry.name, "path": entry.name,
                        "source": str(entry.path), "files": files})
        print(f"[push] {entry.name:>9}  "
              f"{files['adapter_model.safetensors']['size'] / 1024**2:6.2f} MiB  "
              f"{files['adapter_model.safetensors']['sha256'][:12]}", flush=True)

    reference = signatures[inventory.entries[0].name]
    divergent = sorted(name for name, item in signatures.items()
                       if item != reference)
    if divergent:
        raise SystemExit(
            f"整池的 LoRA 設定必須相同，以下與 {inventory.entries[0].name} 不同："
            f"{divergent[:5]}")
    return entries, reference


def main() -> None:
    parser = add_config_args()
    parser.add_argument("--repo", required=True, help="目標 repo id（<org>/<name>）")
    parser.add_argument("--adapter-dir", default=None,
                        help="要上傳哪一個 adapter 目錄；"
                             "預設取設定檔的 system.adapter_dir")
    parser.add_argument("--manifest", default=None,
                        help="改以現成的有序 manifest 指定要傳哪些 adapter")
    parser.add_argument("--expected-adapters", type=int,
                        default=CANONICAL_ADAPTER_COUNT)
    parser.add_argument("--private", action="store_true",
                        help="建為私有 repo；預設公開（見檔頭的授權說明）")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--dry-run", action="store_true",
                        help="只驗證與列出將上傳的內容，不真的上傳")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    if args.manifest and args.adapter_dir:
        parser.error("--manifest 與 --adapter-dir 只能擇一")
    if not args.manifest and not args.adapter_dir:
        args.adapter_dir = cfg["system"]["adapter_dir"]

    # 先全部驗證過再開始上傳——不把壞掉的池傳上去。
    # 這一段刻意不需要登入：本機驗證與登入狀態無關。
    staging = tempfile.mkdtemp(prefix="smoea-pool-")
    try:
        if args.adapter_dir:
            manifest_path = os.path.join(staging, "derived.json")
            write_json(manifest_path, manifest_from_adapter_dir(args.adapter_dir))
        else:
            manifest_path = args.manifest
        inventory = load_adapter_inventory(
            manifest_path, expected_count=args.expected_adapters)

        entries, signature = collect(inventory)
        total = sum(item["files"][name]["size"]
                    for item in entries for name in REQUIRED_ADAPTER_FILES)
        payload = {
            "schema_version": 1,
            "pool_id": inventory.pool_id or f"pool{len(entries)}",
            "count": len(entries),
            "base_model": signature["base_model_name_or_path"],
            "lora_signature": signature,
            "total_bytes": total,
            "adapters": [{k: v for k, v in item.items() if k != "source"}
                         for item in entries],
        }

        print(f"[push] {len(entries)} 個 adapter，合計 {total / 1024**3:.2f} GiB"
              f"（r={signature['r']}、alpha={signature['lora_alpha']:g}、"
              f"target={','.join(signature['target_modules'])}）")
        visibility = "私有" if args.private else "公開"
        print(f"[push] 目標 {args.repo}（{visibility}）")

        if args.dry_run:
            print("[push] --dry-run：未上傳任何檔案")
            return

        from huggingface_hub import HfApi

        api = HfApi(token=args.token)
        try:
            who = api.whoami()
        except Exception as exc:
            raise SystemExit(
                f"尚未登入 Hugging Face（{type(exc).__name__}）。"
                "先執行 hf auth login，或設定 HF_TOKEN。")
        print(f"[push] 身分：{who['name']}")
        api.create_repo(args.repo, private=args.private, exist_ok=True,
                        repo_type="model")

        # 用 symlink 疊出要上傳的版面：不複製 2.6 GiB，也不會誤傳
        # optimizer.pt 之類的訓練殘留物。
        tree = os.path.join(staging, "tree")
        for item in entries:
            target = os.path.join(tree, item["path"])
            os.makedirs(target, exist_ok=True)
            for filename in REQUIRED_ADAPTER_FILES:
                os.symlink(os.path.join(item["source"], filename),
                           os.path.join(target, filename))

        print(f"[push] 上傳 {len(entries)} × {len(REQUIRED_ADAPTER_FILES)} 個檔案 …",
              flush=True)
        api.upload_large_folder(repo_id=args.repo, folder_path=tree,
                                repo_type="model")

        # manifest 最後傳：中途失敗的 repo 沒有 manifest，fetch 端會直接擋下。
        api.upload_file(
            path_or_fileobj=(json.dumps(
                payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            path_in_repo=MANIFEST_NAME, repo_id=args.repo, token=args.token,
            commit_message=f"{payload['pool_id']}：{len(entries)} 個 adapter")
        per_mib = total / len(entries) / 2 / 1024**2
        api.upload_file(
            path_or_fileobj=CARD.format(
                base_model=payload["base_model"], count=len(entries),
                r=signature["r"], alpha=f"{signature['lora_alpha']:g}",
                targets=", ".join(f"`{m}`" for m in signature["target_modules"]),
                per_mib=per_mib, total_gib=total / 1024**3,
                repo_id=args.repo).encode("utf-8"),
            path_in_repo="README.md", repo_id=args.repo, token=args.token)
        print(f"[push] 完成 → https://huggingface.co/{args.repo}（{visibility}）")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()

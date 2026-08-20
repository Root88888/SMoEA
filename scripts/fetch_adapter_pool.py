#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""從 Hugging Face 取得 pool150，落成本地的 `adapter/task{N}/`。

**下載是 setup 階段的動作，不是 serving 的動作。** SMoEA 線上只讀本地
adapter 目錄，執行期不會對任何外部服務發出請求。`setup_workspace.sh` 在
adapter 不齊時會自動代跑本腳本；也可以單獨執行。

  python scripts/fetch_adapter_pool.py --repo <org>/<repo> --list
  python scripts/fetch_adapter_pool.py --repo <org>/<repo>

遠端版面由 scripts/push_adapter_pool.py 寫出：

    pool150_manifest.json          # 有序清單、逐檔 sha256 與大小、共用的 LoRA 設定
    task{N}/{adapter_config.json,adapter_model.safetensors}

manifest 是唯一真相：要下載哪些檔、順序為何、內容對不對，全看它。已在本機
且 sha256 相符的檔案直接跳過，所以中斷後重跑只補缺的部分。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from router.config import add_config_args, load_config  # noqa: E402
from system.adapter_pool import (  # noqa: E402
    AdapterPoolError,
    REQUIRED_ADAPTER_FILES,
    load_adapter_inventory,
    sha256,
    write_json,
)

MANIFEST_NAME = "pool150_manifest.json"


def remote_manifest(repo_id, revision, token):
    """取遠端 manifest。缺 manifest 的 repo 一律視為不可用。"""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(repo_id, MANIFEST_NAME, revision=revision,
                               token=token)
    except EntryNotFoundError:
        raise SystemExit(
            f"遠端 {repo_id} 沒有 {MANIFEST_NAME}——這不是一個 adapter pool "
            f"repo，或上一次上傳沒有完成。") from None
    except RepositoryNotFoundError:
        raise SystemExit(
            f"找不到 {repo_id}；若為私有 repo，先執行 hf auth login "
            f"或設定 HF_TOKEN。") from None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("adapters")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"遠端 {MANIFEST_NAME} 的 adapters 不得為空")
    return payload


def up_to_date(target, spec):
    """本機這一份是否已與 manifest 相符（大小先擋，相符才算 sha256）。"""
    if not os.path.isfile(target) or os.path.getsize(target) != spec["size"]:
        return False
    return sha256(target) == spec["sha256"]


def fetch_one(repo_id, entry, adapter_dir, *, revision, token):
    """下載一個 adapter 的兩個檔，逐檔核對 sha256。回傳實際下載的檔數。"""
    from huggingface_hub import hf_hub_download

    destination = os.path.join(adapter_dir, entry["path"])
    os.makedirs(destination, exist_ok=True)
    downloaded = 0
    for filename in REQUIRED_ADAPTER_FILES:
        spec = entry["files"][filename]
        target = os.path.join(destination, filename)
        if up_to_date(target, spec):
            continue
        remote = f"{entry['path']}/{filename}"
        local = hf_hub_download(repo_id, remote, revision=revision, token=token)
        # 先落成 .tmp 再 replace：中斷的下載不會被誤認為完整的 adapter。
        temporary = target + ".tmp"
        shutil.copyfile(local, temporary)
        actual = sha256(temporary)
        if actual != spec["sha256"]:
            os.unlink(temporary)
            raise AdapterPoolError(
                f"{remote} 的 sha256 不符：預期 {spec['sha256'][:12]}、"
                f"實得 {actual[:12]}")
        os.replace(temporary, target)
        downloaded += 1
    return downloaded


def main() -> None:
    parser = add_config_args()
    parser.add_argument("--repo", required=True, help="Hugging Face repo id")
    parser.add_argument("--adapter-dir", default=None,
                        help="落地到哪裡；預設取設定檔的 system.adapter_dir")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="私有 repo 的存取權杖；預設讀環境變數 HF_TOKEN")
    parser.add_argument("--workers", type=int, default=8,
                        help="同時下載幾個 adapter")
    parser.add_argument("--list", action="store_true",
                        help="只顯示遠端這個池是什麼，不下載")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    adapter_dir = args.adapter_dir or cfg["system"]["adapter_dir"]

    payload = remote_manifest(args.repo, args.revision, args.token)
    entries = payload["adapters"]
    signature = payload.get("lora_signature", {})
    total = payload.get("total_bytes") or sum(
        item["files"][name]["size"]
        for item in entries for name in REQUIRED_ADAPTER_FILES)
    print(f"[fetch] {args.repo}：{payload.get('pool_id')}，"
          f"{len(entries)} 個 adapter、{total / 1024**3:.2f} GiB")
    if signature:
        print(f"[fetch] base={payload.get('base_model')}、r={signature.get('r')}、"
              f"alpha={signature.get('lora_alpha')}、"
              f"target={','.join(signature.get('target_modules', []))}")
    if args.list:
        return

    os.makedirs(adapter_dir, exist_ok=True)
    print(f"[fetch] → {os.path.abspath(adapter_dir)}"
          f"（已存在且 sha256 相符者跳過）", flush=True)

    done = 0
    downloaded = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(fetch_one, args.repo, entry, adapter_dir,
                        revision=args.revision, token=args.token): entry
            for entry in entries
        }
        for future in as_completed(futures):
            downloaded += future.result()
            done += 1
            if done % 10 == 0 or done == len(entries):
                print(f"[fetch] {done}/{len(entries)}", flush=True)

    # manifest 也落地：合成腳本可以直接用它，順序與遠端完全一致。
    # path 改為相對，manifest 與 adapter 一起搬到別處仍然解得開。
    local_manifest = os.path.join(adapter_dir, MANIFEST_NAME)
    write_json(local_manifest, payload)

    # 用正式的驗證器複驗一次：缺檔、數量不符、順序重複在這裡就會被擋下。
    inventory = load_adapter_inventory(local_manifest,
                                       expected_count=payload.get("count"))
    print(f"[fetch] 已驗證 {len(inventory.entries)} 個 adapter"
          f"（本次下載 {downloaded} 個檔案）")
    print(f"[fetch] manifest：{local_manifest}")


if __name__ == "__main__":
    main()

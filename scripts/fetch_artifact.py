#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""從 Hugging Face repo 取得現成的 rejection artifact（ADR-0002 第 6 節）。

**下載是 setup 階段的動作，不是 serving 的動作。** SMoEA 線上只認本地目錄契約，
執行期不會對任何外部服務發出請求；本腳本把遠端檔案落成標準的
`prepare/merged_model/` 目錄之後，驗證與載入的路徑與現成 artifact 完全相同。

  python scripts/fetch_artifact.py --repo <org>/<repo> --condition ties \
      --artifact-root /shared/artifacts \
      --registry /shared/artifacts/registry.json --register-as ties

遠端 repo 的版面預期與本地相同：
  <condition>/<run_id>/prepare/merged_model/{result.json,dense_delta.safetensors}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from system.merged_model import (  # noqa: E402
    MergedModelError,
    load_merged_model_artifact,
)
from system.registry import load_registry  # noqa: E402

ARTIFACT_FILES = ("result.json", "dense_delta.safetensors")


def remote_run_ids(repo_id, condition, revision, token):
    """列出遠端某個 condition 底下有哪些 run。"""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, revision=revision, token=token)
    prefix = f"{condition}/"
    runs = {
        item[len(prefix):].split("/", 1)[0]
        for item in files
        if item.startswith(prefix) and item.endswith("result.json")
    }
    if not runs:
        raise SystemExit(
            f"遠端 {repo_id} 沒有 condition={condition} 的 artifact")
    return sorted(runs)


def fetch(repo_id, condition, run_id, artifact_root, *, revision, token):
    """把一份 artifact 下載成本地標準目錄，並以正式驗證器複驗。"""
    from huggingface_hub import hf_hub_download

    destination = os.path.join(artifact_root, condition, run_id,
                               "prepare", "merged_model")
    os.makedirs(destination, exist_ok=True)
    for filename in ARTIFACT_FILES:
        remote = f"{condition}/{run_id}/prepare/merged_model/{filename}"
        print(f"[fetch] {remote}", flush=True)
        local = hf_hub_download(repo_id, remote, revision=revision, token=token)
        target = os.path.join(destination, filename)
        # 先複製到 .tmp 再 replace：中斷的下載不會被誤認為完整 artifact。
        temporary = target + ".tmp"
        shutil.copyfile(local, temporary)
        os.replace(temporary, target)
    # checksum 由驗證器逐檔核對——下載途中的損壞在這裡就會被擋下。
    return load_merged_model_artifact(destination), destination


def register(registry_path, entry_id, artifact_dir, description):
    if os.path.exists(registry_path):
        payload = json.loads(open(registry_path, encoding="utf-8").read())
    else:
        payload = {"schema_version": 1, "entries": []}
    entry = {"id": entry_id, "method": "artifact", "source": "prepared",
             "artifact_dir": str(artifact_dir), "description": description}
    payload["entries"] = [
        item for item in payload["entries"] if item.get("id") != entry_id
    ] + [entry]
    temporary = f"{registry_path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, registry_path)
    load_registry(registry_path)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Hugging Face repo id")
    parser.add_argument("--condition", required=True,
                        help="要取得哪一個 condition（ta / ties_only / dare_ties_ta …）")
    parser.add_argument("--run-id", default=None,
                        help="指定 run；省略時遠端只有一個 run 才自動採用")
    parser.add_argument("--artifact-root", default=None,
                        help="下載到哪裡；--list 時不需要")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="私有 repo 的存取權杖；預設讀環境變數 HF_TOKEN")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--register-as", default=None)
    parser.add_argument("--list", action="store_true",
                        help="只列出遠端有哪些 run，不下載")
    args = parser.parse_args()

    runs = remote_run_ids(args.repo, args.condition, args.revision, args.token)
    if args.list:
        for run in runs:
            print(run)
        return
    if not args.artifact_root:
        parser.error("下載需要 --artifact-root")
    run_id = args.run_id
    if run_id is None:
        if len(runs) != 1:
            # 不自動挑「最新的」——那正是本專案刻意排除的行為。
            raise SystemExit(
                f"遠端有多個 run（{', '.join(runs)}），請用 --run-id 指定")
        run_id = runs[0]

    artifact, directory = fetch(args.repo, args.condition, run_id,
                                args.artifact_root,
                                revision=args.revision, token=args.token)
    print(f"[fetch] 已驗證：{artifact.condition_id}:{artifact.run_id}"
          f"（{len(artifact.modules)} modules，{artifact.torch_dtype}）")

    if args.registry:
        entry = register(args.registry, args.register_as or args.condition,
                         directory, f"{args.condition}，自 {args.repo} 取得")
        print(f"[fetch] 已登記進 {args.registry} → id={entry['id']}")


if __name__ == "__main__":
    main()

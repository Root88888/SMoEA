#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""以 pool150 全池線上合成一份 rejection artifact（ADR-0002）。

生成是**使用者明確觸發的獨立動作**：它不會發生在回答某一筆 query 的路徑上。
產物落地、通過驗證、登記進 registry 之後，才能被 `:rejection use <id>` 或
`--artifact <id>` 選用。

超參數鎖成封板值（見 system/merging.py 的 SEALED），使用者只選方法不調參。

編號（run_id）取產出檔案本身的 sha256 前 16 碼，不是從輸入推算的。原因是同樣的
輸入在不同型號的顯卡上會得到最後一位不同的結果（浮點加法換個順序算就差一位），
用輸入推算會讓兩份不同的檔案拿到同一個編號。取檔案雜湊則保證編號相同必然內容相同。

  python scripts/merge_pool150.py --method ties \
      --manifest /data/pool150/manifest.json \
      --artifact-root /shared/artifacts \
      --registry /shared/artifacts/registry.json --register-as ties
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from router.config import add_config_args, load_config  # noqa: E402
from system.adapter_pool import (  # noqa: E402
    load_adapter_pool,
    manifest_from_adapter_dir,
)
from system.merged_model import (  # noqa: E402
    MergedModelError,
    load_merged_model_artifact,
    resolve_base_model_identity,
    write_dense_delta_artifact,
)
from system.merging import (  # noqa: E402  # noqa: F401
    CANONICAL_SEED,
    MERGE_METHODS,
    SEALED,
    merge_pool,
    merge_pool_adamerging,
)
from system.registry import load_registry  # noqa: E402


def find_existing(artifact_root, method, pool_digest):
    """找出同一個 adapter 池、同一個方法算過的產物；沒有則回 None。"""
    import glob

    pattern = os.path.join(artifact_root, method, "*", "prepare", "merged_model")
    for candidate in sorted(glob.glob(pattern)):
        try:
            manifest = json.loads(
                open(os.path.join(candidate, "result.json"),
                     encoding="utf-8").read())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("adapter_pool", {}).get("adapters_sha256") == pool_digest:
            try:
                load_merged_model_artifact(candidate)
            except MergedModelError:
                continue
            return candidate
    return None


def register(registry_path, entry_id, artifact_dir, description,
             source="runtime_merged"):
    """把 artifact 登記進 registry；同名項目就地更新。"""
    if os.path.exists(registry_path):
        payload = json.loads(open(registry_path, encoding="utf-8").read())
    else:
        payload = {"schema_version": 1, "entries": []}
    entry = {"id": entry_id, "method": "artifact", "source": source,
             "artifact_dir": str(artifact_dir), "description": description}
    entries = [item for item in payload["entries"] if item.get("id") != entry_id]
    payload["entries"] = entries + [entry]
    temporary = f"{registry_path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, registry_path)
    load_registry(registry_path)          # 立刻複驗，壞掉的 registry 不落地
    return entry


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser())
    parser.add_argument("--method", required=True,
                        choices=list(MERGE_METHODS) + ["adamerging_pp", "lorahub"])
    parser.add_argument("--train-root", default=None,
                        help="adamerging_pp 的校準資料來源；預設 "
                             "<dataset_dir>/train_data")
    parser.add_argument("--examples", default=None,
                        help="lorahub 的示範樣本 JSON："
                             "[{\"instance_id\",\"prompt\",\"output\"}, ...]")
    parser.add_argument("--run-seed", type=int, default=1,
                        help="lorahub 的 run seed（封板為 1、2、3）")
    parser.add_argument("--manifest", default=None,
                        help="pool150 的有序 adapter manifest")
    parser.add_argument("--adapter-dir", default=None,
                        help="由 adapter/task{N}/ 慣例推導 manifest（依編號遞增排序）；"
                             "預設取設定檔的 system.adapter_dir")
    parser.add_argument("--adapter-root", default=None,
                        help="manifest 中相對路徑的根目錄")
    parser.add_argument("--artifact-root", default=None,
                        help="產物落地根目錄；預設取設定檔的 system.artifact_root")
    parser.add_argument("--device", default="cuda",
                        help="merge 的運算裝置（cpu 亦可，較慢）")
    parser.add_argument("--registry", default=None,
                        help="要登記進哪一份清單檔；預設取設定檔的 "
                             "system.artifact_registry。傳 none 可略過登記")
    parser.add_argument("--register-as", default=None,
                        help="登記用的 id；預設同 --method")
    parser.add_argument("--expected-adapters", type=int, default=150)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    if args.manifest and args.adapter_dir:
        parser.error("--manifest 與 --adapter-dir 只能擇一")
    if not args.manifest and not args.adapter_dir:
        # 設定檔已經定義了 adapter 的位置（InferenceEngine 也用同一個值），
        # 不該再要求使用者指定一次。
        args.adapter_dir = cfg["system"]["adapter_dir"]
    # 未指定時沿用設定檔，使用者不必重複打路徑。
    if args.artifact_root is None:
        args.artifact_root = cfg["system"].get("artifact_root")
        if not args.artifact_root:
            parser.error("未設定 system.artifact_root，請用 --artifact-root 指定")
    if args.registry is None:
        args.registry = cfg["system"].get("artifact_registry")
    if args.registry == "none":
        args.registry = None

    import torch

    manifest_path = args.manifest
    if args.adapter_dir:
        derived = manifest_from_adapter_dir(args.adapter_dir)
        manifest_path = os.path.join(args.artifact_root, "pool_manifest.json")
        os.makedirs(args.artifact_root, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(derived, handle, ensure_ascii=False, indent=2)
        print(f"[merge] 由 {args.adapter_dir} 推導 manifest → {manifest_path}",
              flush=True)

    print(f"[merge] 載入 adapter 池 ← {manifest_path}", flush=True)
    pool = load_adapter_pool(manifest_path,
                            expected_adapters=args.expected_adapters,
                            path_root=args.adapter_root)
    fingerprint = pool.fingerprint()
    print(f"[merge] {fingerprint['adapter_count']} 個 adapter，"
          f"rank={pool.rank} scaling={pool.scaling:.4f}，"
          f"指紋 {fingerprint['adapters_sha256'][:16]}", flush=True)

    dtype = cfg["system"]["dtype"]

    # 同一個池與方法算過就不重算。編號要算完才知道，所以改以池指紋比對既有產物。
    existing_dir = find_existing(args.artifact_root, args.method,
                                 fingerprint["adapters_sha256"])
    if existing_dir is not None:
        existing = load_merged_model_artifact(existing_dir)
        print(f"[merge] 同一個池已經算過，跳過重算："
              f"{existing.condition_id}:{existing.run_id}")
        if args.registry:
            entry = register(args.registry, args.register_as or args.method,
                             existing_dir,
                             f"{args.method}，pool150 全池，線上合成")
            print(f"[merge] 已登記進 {args.registry} → id={entry['id']}")
        return

    # 先寫進暫存區，拿到檔案雜湊之後才知道最終路徑。
    staging = os.path.join(args.artifact_root, ".staging",
                           f"{args.method}-{os.getpid()}")
    os.makedirs(staging, exist_ok=True)
    staged_delta = os.path.join(staging, "dense_delta.safetensors")
    started = time.time()
    if args.method == "lorahub":
        # LoRAHub 的產物是 PEFT adapter，而且綁定一組示範樣本——與其他方法分流。
        from system.lorahub import adapt
        from system.merged_model import write_peft_adapter_artifact
        if not args.examples:
            parser.error("lorahub 需要 --examples 指定示範樣本")
        examples = json.loads(open(args.examples, encoding="utf-8").read())
        adapter_dir = os.path.join(staging, "adapter")
        lorahub_report = adapt(
            pool, examples, adapter_dir, base_model=cfg["system"]["base_model"],
            run_seed=args.run_seed, device=args.device)
        # 編號取權重檔雜湊，與其他方法一致。
        import hashlib
        digest = hashlib.sha256()
        with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        run_id = digest.hexdigest()[:16]
        artifact_dir = os.path.join(args.artifact_root, args.method, run_id,
                                    "prepare", "merged_model")
        base = resolve_base_model_identity(cfg["system"]["base_model"])
        artifact = write_peft_adapter_artifact(
            artifact_dir, source=adapter_dir, condition_id="lorahub",
            run_id=run_id, base_model_name=base["name"],
            base_model_revision=base["revision"],
            base_model_config_sha256=base["config_sha256"],
            torch_dtype=dtype, adapter_pool=fingerprint,
            merge_report=lorahub_report)
        print(f"[merge] artifact 已驗證：{artifact.condition_id}:{artifact.run_id} "
              f"（PEFT，示範樣本指紋 {lorahub_report['fewshot_sha256'][:16]}）")
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        if args.registry:
            entry = register(args.registry, args.register_as or args.method,
                             artifact_dir,
                             f"LoRAHub，擬合於 {lorahub_report['fewshot_sha256'][:12]}")
            print(f"[merge] 已登記進 {args.registry} → id={entry['id']}")
        return

    if args.method == "adamerging_pp":
        # 兩階段：先學逐層係數（需要 GPU 與校準資料），再走 TIES 合成。
        from system.adamerging import SEALED_ADAMERGING, optimize_coefficients
        train_root = args.train_root or os.path.join(
            cfg["paths"]["dataset_dir"], "train_data")
        learned, opt_summary = optimize_coefficients(
            pool, train_root, os.path.join(staging, "optimization"),
            seed=CANONICAL_SEED, device=args.device)
        report = merge_pool_adamerging(
            pool, staged_delta, learned, device=args.device,
            output_dtype=getattr(torch, dtype), progress=True,
            density=SEALED_ADAMERGING["ties"]["density"],
            tile_rows=SEALED_ADAMERGING["ties"]["tile_rows"],
            reduction=SEALED_ADAMERGING["ties"]["reduction"])
        report["optimization"] = {
            key: value for key, value in opt_summary.items() if key != "sealed"}
    else:
        report = merge_pool(args.method, pool, staged_delta, device=args.device,
                            output_dtype=getattr(torch, dtype), progress=True)
    print(f"[merge] 權重合成完成（{time.time() - started:.0f}s，"
          f"{report['output_bytes'] / 1e9:.2f} GB）", flush=True)

    run_id = report["output_sha256"][:16]
    artifact_dir = os.path.join(args.artifact_root, args.method, run_id,
                                "prepare", "merged_model")
    os.makedirs(artifact_dir, exist_ok=True)
    delta_path = os.path.join(artifact_dir, "dense_delta.safetensors")
    os.replace(staged_delta, delta_path)
    import shutil
    shutil.rmtree(staging, ignore_errors=True)
    print(f"[merge] run_id={run_id}（取自檔案雜湊）→ {artifact_dir}", flush=True)

    base = resolve_base_model_identity(cfg["system"]["base_model"])
    artifact = write_dense_delta_artifact(
        artifact_dir,
        source=delta_path,
        condition_id=args.method,
        run_id=run_id,
        base_model_name=base["name"],
        base_model_revision=base["revision"],
        base_model_config_sha256=base["config_sha256"],
        torch_dtype=dtype,
        quantization="bitsandbytes_4bit" if cfg["system"]["load_in_4bit"] else "none",
        adapter_pool=fingerprint,
        merge_report={key: value for key, value in report.items()
                      if key != "module_reports"},
    )
    print(f"[merge] artifact 已驗證：{artifact.condition_id}:{artifact.run_id} "
          f"（{len(artifact.modules)} modules，{artifact.torch_dtype}）")

    if args.registry:
        entry = register(args.registry, args.register_as or args.method,
                         artifact_dir,
                         f"{args.method}，pool150 全池，線上合成")
        print(f"[merge] 已登記進 {args.registry} → id={entry['id']}")
        print(f"[merge] 使用：--artifact {entry['id']} 或 :rejection use {entry['id']}")


if __name__ == "__main__":
    main()

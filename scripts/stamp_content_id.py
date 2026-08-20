#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 producer artifact 重新標成以檔案雜湊為準的 run_id。

producer 的 run_id 不是內容決定的——同一份權重在兩次 run 底下會拿到兩個不同的 id。
改用產出檔案本身的 sha256 前 16 碼當 run_id：**編號相同就保證內容相同**，兩份不同
的檔案不可能撞號。

（早期版本改用「方法 + 超參數 + adapter 池指紋」推算編號，已證實不可行：同樣的輸入
在不同型號的顯卡上會得到最後一位不同的結果，那會讓兩份不同的檔案拿到同一個編號。
見 ADR-0002。）

仍要求提供 verify_against_producer.py 的報告，但用途改為記錄來歷：報告會被寫進
manifest，讓使用者知道這份權重與 SMoEA 自己算的差多少。差異超過一個 ulp 時拒絕，
那代表搬移的實作有問題，不該當成同一個方法交付。

權重以 hardlink 連過去（同一檔案系統時不佔額外空間），只有 result.json 是新寫的；
producer 的原始 run_id 保留在 provenance 欄位。

  python scripts/stamp_content_id.py --method ties \
      --producer <producer merged_model 目錄> --report <ties_report.json> \
      --adapter-dir adapter --out-root /shared/artifacts/staged
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from system.adapter_pool import (  # noqa: E402
    load_adapter_pool,
    manifest_from_adapter_dir,
)
from system.merged_model import write_dense_delta_artifact  # noqa: E402
from system.merging import CANONICAL_SEED, MERGE_METHODS, SEALED  # noqa: E402


def content_run_id(weight_path: str) -> str:
    """編號 = 產出檔案的 sha256 前 16 碼，與 merge_pool150.py 同一套算法。"""
    digest = hashlib.sha256()
    with open(weight_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(MERGE_METHODS))
    parser.add_argument("--producer", required=True)
    parser.add_argument("--report", required=True,
                        help="verify_against_producer.py 的報告；必須是 PASS")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--expected-adapters", type=int, default=150)
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.adapter_dir):
        parser.error("請擇一提供 --manifest 或 --adapter-dir")

    report = json.loads(open(args.report, encoding="utf-8").read())
    if report.get("method") != args.method:
        raise SystemExit(
            f"報告的方法是 {report.get('method')!r}，與 --method {args.method!r} 不符")
    difference = report.get("difference") or {}
    if report.get("bitwise_identical"):
        verdict = "位元完全相同"
    elif difference.get("all_differences_within_one_ulp"):
        verdict = (f"差異在一個 ulp 之內"
                   f"（{difference.get('differing_elements'):,}/"
                   f"{difference.get('total_elements'):,} 個元素）")
    else:
        raise SystemExit(
            f"{args.report} 的差異超過一個 ulp，代表搬移的實作與 producer 不等價；"
            "先查清楚再交付")
    print(f"[stamp] 等價驗證：{args.method} —— {verdict}"
          f"（producer {report['producer_run_id']}）")

    manifest_path = args.manifest
    if args.adapter_dir:
        os.makedirs(args.out_root, exist_ok=True)
        manifest_path = os.path.join(args.out_root, "pool_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest_from_adapter_dir(args.adapter_dir), handle)
    pool = load_adapter_pool(manifest_path,
                            expected_adapters=args.expected_adapters)
    fingerprint = pool.fingerprint()
    if fingerprint["adapters_sha256"] != report.get("pool_fingerprint"):
        raise SystemExit(
            "adapter 池與驗證當時不同（指紋不符）；請以同一個池重新驗證")

    source_weight = os.path.join(args.producer, "dense_delta.safetensors")
    run_id = content_run_id(source_weight)
    destination = os.path.join(args.out_root, args.method, run_id,
                               "prepare", "merged_model")
    os.makedirs(destination, exist_ok=True)
    print(f"[stamp] run_id = {run_id}（取自檔案雜湊）→ {destination}")

    producer_manifest = json.loads(
        open(os.path.join(args.producer, "result.json"), encoding="utf-8").read())
    target_weight = os.path.join(destination, "dense_delta.safetensors")
    if not os.path.exists(target_weight):
        try:
            os.link(source_weight, target_weight)     # 同檔案系統：不佔空間
            print("[stamp] 權重以 hardlink 連結（未複製）")
        except OSError:
            shutil.copyfile(source_weight, target_weight)
            print("[stamp] 權重已複製（跨檔案系統，無法 hardlink）")

    base = producer_manifest["base_model"]
    artifact = write_dense_delta_artifact(
        destination,
        source=target_weight,
        condition_id=producer_manifest["condition_id"],  # 沿用 producer 的名稱
        run_id=run_id,
        base_model_name=base["name"],
        base_model_revision=base["revision"],
        base_model_config_sha256=base["config_sha256"],
        torch_dtype=producer_manifest["inference"]["torch_dtype"],
        quantization=producer_manifest["inference"]["quantization"],
        producer=producer_manifest.get("producer"),
        adapter_pool=fingerprint,
        merge_report={
            "sealed": SEALED[args.method],
            "seed": CANONICAL_SEED,
            "provenance": {
                "producer_condition_id": producer_manifest["condition_id"],
                "producer_run_id": producer_manifest["run_id"],
                "verified_bitwise_identical": bool(report.get("bitwise_identical")),
                "verification_verdict": verdict,
                "verification_report": os.path.abspath(args.report),
            },
        },
    )
    print(f"[stamp] 已驗證：{artifact.condition_id}:{artifact.run_id} "
          f"（{len(artifact.modules)} modules，{artifact.torch_dtype}）")
    print(f"[stamp] 上傳：python scripts/push_artifact.py --repo <org>/<repo> "
          f"--artifact {destination}")


if __name__ == "__main__":
    main()

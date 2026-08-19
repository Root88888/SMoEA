#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""驗證 SMoEA 的線上 merge 與 producer 的封板產物是否等價（ADR-0002）。

搬進 SMoEA 的權重運算必須重現 MoEA-Trainer 的結果，否則「線上生成」與「現成
artifact」就是兩個不同的東西，run_id 也不能共用。本腳本對同一個 adapter 池跑一次
merge，把 `dense_delta.safetensors` 的 sha256 與 producer 的記錄逐一比對。

sha256 不同時不會直接判定失敗——producer 在 GPU 上以 fp32 累加，累加順序不同可能
造成末位差異。因此另外逐元素比對最大絕對差與相異比例，用來分辨「實作錯誤」與
「捨入差異」。判定門檻寫在報告裡，由人決定是否接受。

  python scripts/verify_against_producer.py --method ta \
      --adapter-dir adapter --producer <producer merged_model 目錄> \
      --work-dir /shared/verify --device cuda
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
from system.merged_model import load_merged_model_artifact  # noqa: E402
from system.merging import MERGE_METHODS, merge_pool  # noqa: E402


def compare(mine_path, theirs_path):
    """逐元素比對兩份 dense delta，回傳差異統計。

    只給最大差值不夠判斷——實作寫錯與浮點捨入的最大差值可能一樣大，差在
    「有多少元素受影響」與「差多少格」。因此另外統計每個相異元素差了幾個
    ulp（該數值在 bfloat16 下能表示的最小間隔）：全部落在 1 ulp 就是捨入，
    出現大於 1 ulp 的就要當成實作問題查。
    """
    import torch
    from safetensors.torch import load_file

    mine = load_file(mine_path)
    theirs = load_file(theirs_path)
    if set(mine) != set(theirs):
        return {"tensor_names_match": False,
                "only_mine": sorted(set(mine) - set(theirs))[:5],
                "only_theirs": sorted(set(theirs) - set(mine))[:5]}
    worst = 0.0
    differing = 0
    total = 0
    magnitude = 0.0
    ulp_counts: dict[int, int] = {}
    for key in sorted(mine):
        a = mine[key].float()
        b = theirs[key].float()
        difference = (a - b).abs()
        mask = difference != 0
        worst = max(worst, float(difference.max()))
        differing += int(mask.sum())
        total += difference.numel()
        magnitude = max(magnitude, float(b.abs().max()))
        if not bool(mask.any()):
            continue
        # bfloat16 有 8 位有效位數，故 ulp(x) = 2^(floor(log2|x|) - 7)。
        reference = torch.maximum(a[mask].abs(), b[mask].abs())
        exponent = torch.floor(torch.log2(reference.clamp_min(1e-30)))
        ulp = torch.pow(2.0, exponent - 7)
        steps = torch.round(difference[mask] / ulp).to(torch.int64)
        for value, count in zip(*torch.unique(steps, return_counts=True)):
            key_value = int(value)
            ulp_counts[key_value] = ulp_counts.get(key_value, 0) + int(count)
    return {
        "tensor_names_match": True,
        "max_abs_difference": worst,
        "differing_elements": differing,
        "total_elements": total,
        "differing_fraction": differing / total if total else 0.0,
        "producer_max_abs_value": magnitude,
        "difference_in_ulp": {str(k): ulp_counts[k] for k in sorted(ulp_counts)},
        "all_differences_within_one_ulp": all(
            abs(k) <= 1 for k in ulp_counts),
    }


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser())
    parser.add_argument("--method", required=True, choices=list(MERGE_METHODS))
    parser.add_argument("--producer", required=True,
                        help="producer 的 merged_model 目錄")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-adapters", type=int, default=150)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    if bool(args.manifest) == bool(args.adapter_dir):
        parser.error("請擇一提供 --manifest 或 --adapter-dir")

    import torch

    os.makedirs(args.work_dir, exist_ok=True)
    manifest_path = args.manifest
    if args.adapter_dir:
        manifest_path = os.path.join(args.work_dir, "pool_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest_from_adapter_dir(args.adapter_dir), handle)

    print(f"[verify] 方法 {args.method}｜裝置 {args.device}", flush=True)
    started = time.time()
    pool = load_adapter_pool(manifest_path,
                            expected_adapters=args.expected_adapters)
    print(f"[verify] 池載入 {len(pool.names)} 個 adapter "
          f"（{time.time() - started:.0f}s，scaling={pool.scaling}）", flush=True)

    producer = load_merged_model_artifact(args.producer)
    print(f"[verify] producer：{producer.condition_id}:{producer.run_id} "
          f"（{producer.torch_dtype}）", flush=True)

    output = os.path.join(args.work_dir, f"{args.method}.safetensors")
    started = time.time()
    report = merge_pool(args.method, pool, output, device=args.device,
                        output_dtype=getattr(torch, producer.torch_dtype),
                        progress=True)
    elapsed = time.time() - started
    print(f"[verify] 合成完成（{elapsed:.0f}s）", flush=True)

    expected = producer.manifest["weights"][0]["sha256"]
    identical = report["output_sha256"] == expected
    result = {
        "method": args.method,
        "device": args.device,
        "elapsed_seconds": round(elapsed, 1),
        "adapter_count": len(pool.names),
        "pool_fingerprint": pool.fingerprint()["adapters_sha256"],
        "producer_condition_id": producer.condition_id,
        "producer_run_id": producer.run_id,
        "producer_sha256": expected,
        "smoea_sha256": report["output_sha256"],
        "bitwise_identical": identical,
    }
    if not identical:
        result["difference"] = compare(
            output, str(producer.weight_files[0]))

    report_path = os.path.join(args.work_dir, f"{args.method}_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if identical:
        print(f"\n[verify] PASS {args.method}：位元完全相同")
    else:
        print(f"\n[verify] DIFF {args.method}：sha256 不同，見上方逐元素比對")
    print(f"[verify] 報告 → {report_path}")


if __name__ == "__main__":
    main()

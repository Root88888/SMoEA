#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_router_assets.py

【router 離線建置一條龍】依序執行並把全部資產落地到 assets/：
  1. 掃 dataset 目錄推導任務集合（ID = train_data 有檔；OOD = 只有測試檔）
  2. 嵌入：ID train / ID test / OOD test 逐任務快取（已有快取自動跳過）
  3. 指紋堆切分 → 任務指紋 → 多質心（silhouette 選 k）
  4. 路由單位（union-find）
  5. TF-IDF 詞彙索引（只 fit 指紋堆）
  6. 校準分數庫（每單位排序 margin + 分位摘要，直接輸出可檢視 json）
  7. 質心 npz + 摘要、單位示例、建置 metadata

同種子同輸入必同輸出；落地檔與線上系統使用的物件同源一致。
說明書 assets/unit_descriptions.json 為獨立資產（LLM 生成＋人工校訂），
不在本腳本產出範圍；缺檔時結尾會提示。

【執行】repo 根目錄；首次執行需 GPU 與網路（下載嵌入模型），
嵌入快取齊備後純 CPU 數分鐘：
  python scripts/build_router_assets.py 2>&1 | tee results/build_log.txt
  # 只建線上服務資產、不要求 benchmark test_data：--serving-only
  # 只補嵌入不重建其餘：--embed_only
  # 覆蓋單項設定：--set units.sim_threshold=0.97

【scale up】任務擴充後同指令重跑；嵌入缺哪補哪、其餘資產全量重建
（重建為 CPU 級快速作業，確保單位表/校準庫與新任務集合一致）。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router.config import (  # noqa: E402
    config_from_cli,
    discover_serving_tasks,
    discover_tasks,
)
from router.core import Router, DESCRIPTIONS  # noqa: E402
from router.embedding import ensure_task_embeddings  # noqa: E402
from router.units import units_summary  # noqa: E402


def add_build_args(parser):
    parser.add_argument("--embed_only", action="store_true")
    parser.add_argument(
        "--serving-only",
        action="store_true",
        help=("只建立線上 router 所需的訓練嵌入與資產；"
              "不要求 benchmark test_data"),
    )


def main():
    cfg, args = config_from_cli(add_build_args)
    if args.serving_only:
        id_tasks = discover_serving_tasks(cfg)
        ood_tasks = []
        print(f"線上任務集合：ID {len(id_tasks)} 個"
              f"（由 {cfg['paths']['dataset_dir']}/train_data 掃描；"
              "不讀 benchmark test_data）")
    else:
        id_tasks, ood_tasks = discover_tasks(cfg)
        print(f"任務集合：ID {len(id_tasks)} 個、OOD {len(ood_tasks)} 個"
              f"（由 {cfg['paths']['dataset_dir']} 掃描推導）")

    # ---- 嵌入快取 ----
    n1 = ensure_task_embeddings(cfg, id_tasks, test=False)
    n2 = [] if args.serving_only else ensure_task_embeddings(
        cfg, id_tasks, test=True)
    n3 = [] if args.serving_only else ensure_task_embeddings(
        cfg, ood_tasks, test=True)
    print(f"[embed] 補算 train {len(n1)} / ID test {len(n2)} / "
          f"OOD test {len(n3)} 個任務（其餘用既有快取）")
    if args.embed_only:
        return

    # ---- 建置與落地 ----
    r = Router.build(cfg, id_tasks)
    r.save()
    us = units_summary(r.units, r.id_tasks)
    print(f"[units] {us['n_units']} 單位、多成員 {us['n_multi_member']} 組："
          f"{us['multi_member_units']}")
    multi_k = {f"t{t}": k for t, k in r.k_by_task.items() if k > 1}
    print(f"[centroids] 多質心任務 {len(multi_k)}/{len(id_tasks)}: {multi_k}")
    print(f"[calibration] 校準樣本 {r.cal_margin.size} 筆、"
          f"{len(r.units)} 組")

    dp = os.path.join(cfg["paths"]["assets_dir"], DESCRIPTIONS)
    if not os.path.exists(dp):
        print(f"\n[注意] 缺單位說明書 {dp}——送審裁決（eval_router 的 "
              f"score 段）需要它；請放入後再跑評測。")
    # 建置摘要
    out = os.path.join(cfg["paths"]["assets_dir"], "build_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_id_tasks": len(id_tasks), "n_ood_tasks": len(ood_tasks),
                   "id_tasks": id_tasks, "ood_tasks": ood_tasks,
                   "units": us, "k_by_task": multi_k,
                   "n_calibration": int(r.cal_margin.size)},
                  f, ensure_ascii=False, indent=1)
    print(f"[done] 建置摘要 → {out}")


if __name__ == "__main__":
    main()

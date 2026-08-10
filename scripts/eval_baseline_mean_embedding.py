#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/eval_baseline_mean_embedding.py

【Baseline：Pure Embedding（平均嵌入比對＋手訂相似度門檻）】
系統定義（刻意最簡）：
  離線：每任務全量 train 嵌入取平均、L2 → 單一任務指紋
        （無多質心、無路由單位、無校準、無詞彙訊號、無 LLM）
  線上：query 對全部指紋取 cosine；top-1 相似度 < tau 拒絕，否則路由 top-1
計分與 router 完全同尺：router.metrics 共用計分（單位投影表按 router
資產的單位表；OOD gt 依 configs 同兩模式）。

【三段執行】(純 CPU；需先跑 build_router_assets 備妥嵌入快取與單位表)
  python scripts/eval_baseline_mean_embedding.py --mode score_dist \
      2>&1 | tee results/baseline_meanemb_dist_log.txt
  python scripts/eval_baseline_mean_embedding.py --mode sweep \
      2>&1 | tee results/baseline_meanemb_sweep_log.txt
  python scripts/eval_baseline_mean_embedding.py --mode eval --tau 0.XX \
      2>&1 | tee results/baseline_meanemb_eval_log.txt
  產出：results/baseline_meanemb_scores.npz（分數快取）
        results/baseline_meanemb_results_tau{X}.json

【scale up】指紋數、投影表、分母全部資料驅動；任務擴充後三段重跑。
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router import data_io, metrics  # noqa: E402
from router.config import config_from_cli, discover_tasks  # noqa: E402
from router.core import Router  # noqa: E402

NPZ = "baseline_meanemb_scores.npz"


def build_scores(cfg, id_tasks, ood_tasks, t_index, npz_path):
    fp = []
    for t in id_tasks:
        e = data_io.load_embeddings(cfg, t)          # 全量 train
        m = e.mean(axis=0)
        fp.append(m / np.linalg.norm(m))
    FP = np.stack(fp)

    def score(tasks):
        top1, arg, src = [], [], []
        for t in tasks:
            e = data_io.load_embeddings(cfg, t, test=True)
            S = e @ FP.T
            top1.append(S.max(axis=1))
            arg.append(S.argmax(axis=1))
            src.append(np.full(e.shape[0], t))
        return (np.concatenate(top1), np.concatenate(arg),
                np.concatenate(src))
    id_t, id_a, id_s = score(id_tasks)
    oo_t, oo_a, oo_s = score(ood_tasks)
    np.savez_compressed(npz_path, id_top1=id_t, id_arg=id_a,
                        id_y=np.array([t_index[t] for t in id_s]),
                        ood_top1=oo_t, ood_arg=oo_a, ood_src=oo_s)
    print(f"[score] 分數已存 {npz_path}")


def main():
    cfg, args = config_from_cli(lambda p: (
        p.add_argument("--mode", required=True,
                       choices=["score_dist", "sweep", "eval"]),
        p.add_argument("--tau", type=float, default=None),
        p.add_argument("--sweep_grid",
                       default="0.60,0.65,0.70,0.72,0.74,0.76,0.78,"
                               "0.80,0.82,0.84,0.86,0.88,0.90")))
    rd = cfg["paths"]["results_dir"]
    os.makedirs(rd, exist_ok=True)
    rt = Router.load(cfg)                 # 只取單位投影表（僅計分用）
    id_tasks, ood_tasks = discover_tasks(cfg)
    t_index = {t: i for i, t in enumerate(id_tasks)}
    npz_path = os.path.join(rd, NPZ)
    if args.mode == "score_dist" or not os.path.exists(npz_path):
        build_scores(cfg, id_tasks, ood_tasks, t_index, npz_path)
    Z = np.load(npz_path)
    id_top1, id_arg, id_y = Z["id_top1"], Z["id_arg"], Z["id_y"]
    oo_top1, oo_arg, oo_src = Z["ood_top1"], Z["ood_arg"], Z["ood_src"]

    if args.mode == "score_dist":
        print(f"\n== top-1 相似度分布（ID {id_top1.size} / "
              f"OOD {oo_top1.size}）==")
        print(f"{'分位':<8}{'ID':<10}{'OOD':<10}")
        for q in [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
            print(f"{q:<8}{np.quantile(id_top1, q):<10.4f}"
                  f"{np.quantile(oo_top1, q):<10.4f}")
        print(f"\n無拒絕裸 acc（top-1 即路由）：task 級 "
              f"{float((id_arg == id_y).mean()):.4f}")
        bins = np.arange(0.50, 1.001, 0.02)
        hi, _ = np.histogram(id_top1, bins=bins)
        ho, _ = np.histogram(oo_top1, bins=bins)
        mi, mo = max(hi.max(), 1), max(ho.max(), 1)
        for i in range(len(bins) - 1):
            print(f"[{bins[i]:.2f},{bins[i+1]:.2f})  "
                  f"ID:{'#'*int(40*hi[i]/mi):<42}"
                  f"OOD:{'#'*int(40*ho[i]/mo)}")
        return

    gt = data_io.load_ood_groundtruth(
        cfg["evaluation"].get("ood_groundtruth"))

    def evaluate(tau):
        pt = np.where(id_top1 >= tau, id_arg, -2)
        ood_pred = {}
        for o in ood_tasks:
            m = oo_src == o
            ood_pred[o] = np.where(oo_top1[m] >= tau, oo_arg[m], -2)
        return metrics.compute_metrics(
            pt, id_y, rt.unit_of, id_tasks, ood_pred, ood_gt=gt,
            ood_top_routes=cfg["evaluation"].get("ood_top_routes", 3))

    if args.mode == "sweep":
        keys = None
        for tau in [float(x) for x in args.sweep_grid.split(",")]:
            m = evaluate(tau)["micro"]
            if keys is None:
                keys = list(m)
                print(f"{'tau':<7}" + "".join(f"{k:<16}" for k in keys))
            print(f"{tau:<7}" + "".join(
                f"{(m[k] if m[k] is not None else float('nan')):<16.4f}"
                for k in keys))
        return

    assert args.tau is not None, "--mode eval 需指定 --tau"
    rep = evaluate(args.tau)
    rep["baseline"] = "mean_embedding_top1_tau"
    rep["tau"] = args.tau
    print(f"\n== Pure Embedding（tau={args.tau}）micro 指標 ==")
    for k, v in rep["micro"].items():
        print(f"  {k:<18}{v}")
    out = os.path.join(rd, f"baseline_meanemb_results_tau{args.tau}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"[done] → {out}")


if __name__ == "__main__":
    main()

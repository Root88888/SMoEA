#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/eval_baseline_bm25_voting.py

【Baseline：BM25 Sample-level Retrieval Voting】（fork 定稿版為底改造）
  離線：全部 ID 任務全量 train 樣本（field、截 doc_clip_chars）建 BM25 索引
        score(q,d) = Σ_t IDF(t)·TF(t,d)(k1+1) / (TF(t,d)+k1(1−b+b·|d|/avgdl))
        IDF 負值截零；k1=1.5、b=0.75
  線上：query 對全部樣本算 BM25 → top-20 樣本按所屬任務加權投票
        ratio = weight_top1 / Σ weight_all；ratio ≥ 門檻路由 top-1，否則拒
計分與 router 完全同尺（router.metrics 共用；OOD gt 依 configs 同兩模式）。

【三段執行】(純 CPU；score 段建索引＋全量打分約 10–30 分鐘)
  python scripts/eval_baseline_bm25_voting.py --mode score_dist \
      2>&1 | tee results/baseline_bm25_dist_log.txt
  python scripts/eval_baseline_bm25_voting.py --mode sweep \
      2>&1 | tee results/baseline_bm25_sweep_log.txt
  python scripts/eval_baseline_bm25_voting.py --mode eval --ratio_tau 0.XX \
      2>&1 | tee results/baseline_bm25_eval_log.txt
  產出：results/baseline_bm25_scores.npz、baseline_bm25_results_tau{X}.json

【scale up】索引、投影表、分母全部資料驅動；任務擴充後三段重跑。
"""

import json
import os
import re
import sys
import time

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router import data_io, metrics  # noqa: E402
from router.config import config_from_cli, discover_tasks  # noqa: E402
from router.core import Router  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]{2,}")
K1, B, TOPK = 1.5, 0.75, 20
NPZ = "baseline_bm25_scores.npz"


def build_scores(cfg, id_tasks, ood_tasks, t_index, npz_path):
    clip = cfg["data"]["doc_clip_chars"]
    t0 = time.time()
    docs_tok, doc_task = [], []
    for t in id_tasks:
        for txt in data_io.load_task_texts(cfg, t):
            docs_tok.append(TOKEN.findall(txt.lower()))
            doc_task.append(t_index[t])
    doc_task = np.array(doc_task)
    N = len(docs_tok)
    vocab, rows, cols, vals = {}, [], [], []
    dl = np.zeros(N, dtype=np.float32)
    for d, toks in enumerate(docs_tok):
        dl[d] = len(toks)
        cnt = {}
        for w in toks:
            cnt[w] = cnt.get(w, 0) + 1
        for w, c in cnt.items():
            j = vocab.setdefault(w, len(vocab))
            rows.append(d); cols.append(j); vals.append(c)
    Vn = len(vocab)
    TF = sparse.csr_matrix((np.array(vals, dtype=np.float32),
                            (np.array(rows), np.array(cols))),
                           shape=(N, Vn))
    n_t = np.asarray((TF > 0).sum(axis=0)).ravel()
    idf = np.log((N - n_t + 0.5) / (n_t + 0.5)).astype(np.float32)
    idf = np.maximum(idf, 0.0)  # 負 IDF 截零（超常見字防護）
    avgdl = float(dl.mean())
    norm_d = (K1 * (1 - B + B * dl / avgdl)).astype(np.float32)
    W = TF.tocoo()
    w_vals = (W.data * (K1 + 1)) / (W.data + norm_d[W.row])
    Wc = sparse.csc_matrix((w_vals, (W.row, W.col)), shape=(N, Vn))
    print(f"[index] {N} docs, vocab {Vn}, avgdl {avgdl:.0f} "
          f"({time.time()-t0:.0f}s)")

    def score(tasks):
        ratios, args_, srcs = [], [], []
        for tag in tasks:
            for txt in data_io.load_task_texts(cfg, tag, test=True):
                q_terms = {vocab[w] for w in
                           set(TOKEN.findall(txt.lower()[:clip]))
                           if w in vocab}
                if not q_terms:
                    ratios.append(0.0); args_.append(-1)
                    srcs.append(tag); continue
                cq = np.fromiter(q_terms, dtype=np.int64)
                s = np.asarray(Wc[:, cq] @ idf[cq]).ravel()
                top = np.argpartition(-s, min(TOPK, s.size - 1))[:TOPK]
                wts = {}
                for d in top:
                    if s[d] > 0:
                        wts[doc_task[d]] = wts.get(doc_task[d], 0.0) \
                            + float(s[d])
                if not wts:
                    ratios.append(0.0); args_.append(-1)
                    srcs.append(tag); continue
                best = max(wts, key=wts.get)
                ratios.append(wts[best] / sum(wts.values()))
                args_.append(int(best))
                srcs.append(tag)
        return (np.array(ratios, dtype=np.float32), np.array(args_),
                np.array(srcs))

    id_r, id_a, id_s = score(id_tasks)
    oo_r, oo_a, oo_s = score(ood_tasks)
    np.savez_compressed(npz_path, id_ratio=id_r, id_arg=id_a,
                        id_y=np.array([t_index[t] for t in id_s]),
                        ood_ratio=oo_r, ood_arg=oo_a, ood_src=oo_s)
    print(f"[score] 全量打分完成 ({time.time()-t0:.0f}s) → {npz_path}")


def main():
    cfg, args = config_from_cli(lambda p: (
        p.add_argument("--mode", required=True,
                       choices=["score_dist", "sweep", "eval"]),
        p.add_argument("--ratio_tau", type=float, default=None),
        p.add_argument("--sweep_grid",
                       default="0.30,0.40,0.50,0.60,0.70,0.80,0.85,"
                               "0.90,0.95")))
    rd = cfg["paths"]["results_dir"]
    os.makedirs(rd, exist_ok=True)
    rt = Router.load(cfg)                 # 只取單位投影表（僅計分用）
    id_tasks, ood_tasks = discover_tasks(cfg)
    t_index = {t: i for i, t in enumerate(id_tasks)}
    npz_path = os.path.join(rd, NPZ)
    if args.mode == "score_dist" or not os.path.exists(npz_path):
        build_scores(cfg, id_tasks, ood_tasks, t_index, npz_path)
    Z = np.load(npz_path)
    id_r, id_a, id_y = Z["id_ratio"], Z["id_arg"], Z["id_y"]
    oo_r, oo_a, oo_src = Z["ood_ratio"], Z["ood_arg"], Z["ood_src"]

    if args.mode == "score_dist":
        print(f"\n== 投票 ratio 分布（ID {id_r.size} / OOD {oo_r.size}）==")
        print(f"{'分位':<8}{'ID':<10}{'OOD':<10}")
        for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]:
            print(f"{q:<8}{np.quantile(id_r, q):<10.4f}"
                  f"{np.quantile(oo_r, q):<10.4f}")
        print(f"\n無拒絕裸 acc：task 級 "
              f"{float((id_a == id_y).mean()):.4f}")
        bins = np.arange(0, 1.001, 0.05)
        hi, _ = np.histogram(id_r, bins=bins)
        ho, _ = np.histogram(oo_r, bins=bins)
        mi, mo = max(hi.max(), 1), max(ho.max(), 1)
        for i in range(len(bins) - 1):
            print(f"[{bins[i]:.2f},{bins[i+1]:.2f})  "
                  f"ID:{'#'*int(40*hi[i]/mi):<42}"
                  f"OOD:{'#'*int(40*ho[i]/mo)}")
        return

    gt = data_io.load_ood_groundtruth(
        cfg["evaluation"].get("ood_groundtruth"))

    def evaluate(tau):
        pt = np.where((id_r >= tau) & (id_a >= 0), id_a, -2)
        ood_pred = {}
        for o in ood_tasks:
            m = oo_src == o
            ood_pred[o] = np.where((oo_r[m] >= tau) & (oo_a[m] >= 0),
                                   oo_a[m], -2)
        return metrics.compute_metrics(
            pt, id_y, rt.unit_of, id_tasks, ood_pred, ood_gt=gt,
            ood_top_routes=cfg["evaluation"].get("ood_top_routes", 3))

    if args.mode == "sweep":
        keys = None
        for tau in [float(x) for x in args.sweep_grid.split(",")]:
            m = evaluate(tau)["micro"]
            if keys is None:
                keys = list(m)
                print(f"{'ratio_tau':<11}" + "".join(f"{k:<16}"
                                                     for k in keys))
            print(f"{tau:<11}" + "".join(
                f"{(m[k] if m[k] is not None else float('nan')):<16.4f}"
                for k in keys))
        return

    assert args.ratio_tau is not None, "--mode eval 需指定 --ratio_tau"
    rep = evaluate(args.ratio_tau)
    rep["baseline"] = "bm25_top20_weighted_voting"
    rep["params"] = {"k1": K1, "b": B, "topk": TOPK,
                     "ratio_tau": args.ratio_tau}
    print(f"\n== BM25 Voting（ratio_tau={args.ratio_tau}）micro 指標 ==")
    for k, v in rep["micro"].items():
        print(f"  {k:<18}{v}")
    out = os.path.join(rd,
                       f"baseline_bm25_results_tau{args.ratio_tau}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"[done] → {out}")


if __name__ == "__main__":
    main()

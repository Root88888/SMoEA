#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/export_report_data.py

【報表數據匯出】把三張結果大表所需的全部數據榨成單一 json：
  1. 主評測 per-task 細分（ID：四區 × 路由正確/錯誤/拒絕；
     OOD：四區流向 ＋ 拒絕率 ＋ top-3 去向）——由 decide 產物
     ＋分數檔重放取得（與 verify_flow_table 同路徑）；
  2. 五個 ablation 變體的 micro 與 per-task 指標（讀各變體結果 json）；
  3. 兩支 baseline 的 micro 與 per-task 指標。
缺檔容忍：哪個變體/基線尚未跑完就跳過並在 missing 欄列示。

【執行】評測（至少主評測三段）完成後於 repo 根目錄：
  python scripts/export_report_data.py
  → results/report_data.json（另存時間戳副本）
"""

import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router import conformal, data_io, fingerprint  # noqa: E402
from router.config import config_from_cli, discover_tasks  # noqa: E402
from router.core import Router  # noqa: E402


def replay_main(cfg, rt, id_tasks, ood_tasks, rd):
    """重放主評測：逐 source 還原 pred，彙整 per-task 細分。"""
    z = np.load(os.path.join(rd, "eval_zones.npz"))
    scores = {}
    with open(os.path.join(rd, "escalation_scores.jsonl"),
              encoding="utf-8") as f:
        for l in f:
            if l.strip():
                s = json.loads(l)
                scores[(s["source"], s["row"])] = s

    def pred_of(src, emb):
        tS = fingerprint.task_sims(emb, rt.C, rt.owner, len(rt.id_tasks))
        dec = {"zone": z[f"{src}__zone"], "b1": z[f"{src}__b1"], "tS": tS,
               "esc": {}}
        for r in np.where(dec["zone"] == conformal.ZONE_ESCALATE)[0]:
            s = scores.get((src, int(r)))
            if s:
                dec["esc"][int(r)] = {"units": s["units"],
                                      "p_yes": s["p_yes"]}
        p, miss = rt.finalize(dec, count_missing=True)
        assert miss == 0, f"{src} 缺分 {miss} 筆"
        return dec["zone"], p

    # ID 側（單一 test 來源、y 提供任務身分）
    te = np.concatenate([data_io.load_embeddings(cfg, t, test=True)
                         for t in id_tasks])
    zone, pred = pred_of("test", te)
    y = z["test__y"]
    unit_of = np.zeros(len(id_tasks), dtype=int)
    for u, g in enumerate(rt.units):
        unit_of[np.array(g)] = u
    up = np.where(pred >= 0, unit_of[np.clip(pred, 0, None)], -9)
    id_rows = {}
    for i, t in enumerate(id_tasks):
        m = y == i
        zt, pt, upt = zone[m], pred[m], up[m]
        uy = unit_of[i]
        # ok/bad 以 unit 級判定（表格藍格加總 = unit 級 acc 的設計不變量）
        row = {"n": int(m.sum())}
        for zk, zname in ((conformal.ZONE_FLOOR, "direct"),
                          (conformal.ZONE_GREEN, "green")):
            mm = zt == zk
            row[zname] = {"ok": int((upt[mm] == uy).sum()),
                          "bad": int((pt[mm] >= 0).sum()
                                     - (upt[mm] == uy).sum())}
        row["red"] = {"rej": int((zt == conformal.ZONE_RED).sum())}
        me = zt == conformal.ZONE_ESCALATE
        row["esc"] = {"ok": int((upt[me] == uy).sum()),
                      "bad": int(((pt[me] >= 0) & (upt[me] != uy)).sum()),
                      "rej": int((pt[me] < 0).sum())}
        id_rows[str(id_tasks[i])] = row

    ood_rows = {}
    for o in ood_tasks:
        src = f"ood_t{o}"
        e = data_io.load_embeddings(cfg, o, test=True)
        zone, pred = pred_of(src, e)
        row = {"n": int(len(pred)),
               "direct": int((zone == conformal.ZONE_FLOOR).sum()),
               "green": int((zone == conformal.ZONE_GREEN).sum()),
               "red": int((zone == conformal.ZONE_RED).sum()),
               "esc_route": int(((zone == conformal.ZONE_ESCALATE)
                                 & (pred >= 0)).sum()),
               "esc_rej": int(((zone == conformal.ZONE_ESCALATE)
                               & (pred < 0)).sum())}
        ood_rows[str(o)] = row
    return id_rows, ood_rows


def load_result_json(rd, suffix=""):
    p = os.path.join(rd, f"router_eval_results{suffix}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def latest(pattern):
    fs = sorted(glob.glob(pattern), key=os.path.getmtime)
    return fs[-1] if fs else None


def main():
    cfg, _ = config_from_cli()
    rd = cfg["paths"]["results_dir"]
    id_tasks, ood_tasks = discover_tasks(cfg)
    rt = Router.load(cfg)
    missing = []

    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_id_tasks": len(id_tasks), "n_ood_tasks": len(ood_tasks)}

    main_rep = load_result_json(rd)
    assert main_rep, "缺主評測結果——先跑 --mode run"
    id_rows, ood_rows = replay_main(cfg, rt, id_tasks, ood_tasks, rd)
    # OOD 行為欄位（top_routes 等）直接沿用結果 json
    for t, row in ood_rows.items():
        row.update({k: main_rep["per_task_ood"][t][k]
                    for k in ("reject_rate", "route_rate",
                              "top_routes", "concentration")})
    for t, row in id_rows.items():
        row.update({k: main_rep["per_task_id"][t][k]
                    for k in ("acc_task", "acc_unit", "reject_rate")})
    out["main"] = {"micro": main_rep["micro"],
                   "flow_table": main_rep["flow_table"],
                   "id_rows": id_rows, "ood_rows": ood_rows}

    out["ablation"] = {}
    for ab in ("no_multicentroid", "no_direct", "no_lexical",
               "gray_reject", "gray_route"):
        rep = load_result_json(rd, f"__{ab}")
        if rep is None:
            missing.append(f"ablation:{ab}")
            continue
        out["ablation"][ab] = {
            "micro": rep["micro"],
            "per_task_id": {t: {"acc_task": r["acc_task"],
                                "acc_unit": r["acc_unit"]}
                            for t, r in rep["per_task_id"].items()},
            "per_task_ood": {t: {"reject_rate": r["reject_rate"],
                                 "top_routes": r["top_routes"][:1]}
                             for t, r in rep["per_task_ood"].items()}}

    out["baselines"] = {}
    for name, pat in (("pure_embedding", "baseline_meanemb_results*.json"),
                      ("bm25", "baseline_bm25_results*.json")):
        p = latest(os.path.join(rd, pat))
        if p is None:
            missing.append(f"baseline:{name}")
            continue
        rep = json.load(open(p, encoding="utf-8"))
        out["baselines"][name] = {
            "source_file": os.path.basename(p),
            "micro": rep["micro"],
            "per_task_id": {t: {"acc_task": r["acc_task"],
                                "acc_unit": r["acc_unit"]}
                            for t, r in rep["per_task_id"].items()},
            "per_task_ood": {t: {"reject_rate": r["reject_rate"],
                                 "top_routes": r["top_routes"][:1]}
                             for t, r in rep["per_task_ood"].items()}}

    out["missing"] = missing
    op = os.path.join(rd, "report_data.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    import shutil
    shutil.copy(op, op.replace(".json",
                f"_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    print(f"[export] 主評測 {len(id_rows)}+{len(ood_rows)} 任務細分；"
          f"變體 {len(out['ablation'])}/5、基線 {len(out['baselines'])}/2"
          + (f"；缺 {missing}" if missing else "；齊全"))
    print(f"[done] → {op}（含時間戳副本）")


if __name__ == "__main__":
    main()

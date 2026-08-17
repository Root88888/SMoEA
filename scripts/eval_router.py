#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/eval_router.py

【router 主評測】三段式（送審 LLM 打分是唯一 GPU 重活，獨立成段、
可中斷續跑）：

  --mode decide  (CPU)  全測試集分區，送審佇列與分區摘要落地
                        → results/eval_zones.npz、escalation_queue.jsonl
  --mode score   (GPU)  對送審佇列打是非題分（可中斷續跑、逐筆 append）
                        → results/escalation_scores.jsonl
  --mode run     (CPU)  合併分數 → 六指標（micro）＋ per-task ＋ 六流向表
                        → results/router_eval_results.json

OOD 指標兩模式（由 configs 的 evaluation.ood_groundtruth 決定）：
  設路徑   → 完整六指標（overall / id_task / id_unit / ood_all /
             應拒 OOD 拒絕率 / 應路由 OOD 正確路由率）
  null     → overall 與 OOD acc 取消；OOD 段輸出行為描述
             （拒絕率 / top-3 路由去向與占比 / 集中度）——
             覆核 top_routes 後可手寫任務級 gt 檔重跑得完整指標。

【執行】(依序三段；score 段需 GPU 與裁決模型)
  python scripts/eval_router.py --mode decide 2>&1 | tee results/eval_decide_log.txt
  python scripts/eval_router.py --mode score  2>&1 | tee results/eval_score_log.txt
  python scripts/eval_router.py --mode run    2>&1 | tee results/eval_run_log.txt
  # 管線自測（無 GPU）可在 score 段加 --fake_verifier（結果不具評測意義）

【scale up】任務數、送審量、分母全部資料驅動；score 段隨送審量線性
成長、checkpoint 續跑已內建。
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router import conformal, data_io, metrics  # noqa: E402
from router.config import config_from_cli, discover_tasks  # noqa: E402
from router.core import Router  # noqa: E402

QUEUE = "escalation_queue.jsonl"
ABL_SUFFIX = ""
GRAY_MODE = "verdict"


def _q(rd):
    return os.path.join(rd, QUEUE.replace(".jsonl", ABL_SUFFIX + ".jsonl"))
SCORES = "escalation_scores.jsonl"
ZONES = "eval_zones.npz"


def _sources(cfg, rt, id_tasks, ood_tasks):
    """依序產出 (source_tag, texts, emb)；ID 合併為一源、OOD 逐任務。"""
    te_txt, te_emb, y = [], [], []
    for t in id_tasks:
        txt = data_io.load_task_texts(cfg, t, test=True)
        te_txt += txt
        te_emb.append(data_io.load_embeddings(cfg, t, test=True))
        y += [rt.t_index[t]] * len(txt)
    yield "test", te_txt, np.concatenate(te_emb), np.array(y)
    for o in ood_tasks:
        txt = data_io.load_task_texts(cfg, o, test=True)
        yield (f"ood_t{o}", txt,
               data_io.load_embeddings(cfg, o, test=True), None)


def mode_decide(cfg, rt, id_tasks, ood_tasks, rd):
    zpath = os.path.join(rd, ZONES)
    qpath = _q(rd)
    store, n_esc = {}, 0
    with open(qpath, "w", encoding="utf-8") as fq:
        line_no = 0
        for src, txt, emb, y in _sources(cfg, rt, id_tasks, ood_tasks):
            d = rt.decide(txt, emb=emb)
            store[f"{src}__zone"] = d["zone"]
            store[f"{src}__b1"] = d["b1"]
            store[f"{src}__margin"] = d["margin"]
            store[f"{src}__pval"] = d["pval"]
            store[f"{src}__top3"] = d["top3_units"]
            if y is not None:
                store[f"{src}__y"] = y
            zc = {conformal.ZONE_NAMES[k]:
                  int((d["zone"] == k).sum()) for k in range(4)}
            print(f"[decide] {src}: n={len(txt)} zones={zc}")
            for r in np.where(d["zone"] == conformal.ZONE_ESCALATE)[0]:
                fq.write(json.dumps(
                    {"line_no": line_no, "source": src, "row": int(r),
                     "query": txt[int(r)][:2000],
                     "units": [int(u) for u in d["top3_units"][int(r)]]},
                    ensure_ascii=False) + "\n")
                line_no += 1
                n_esc += 1
    np.savez_compressed(zpath, **store)
    print(f"[decide] 分區 → {zpath}；送審佇列 {n_esc} 筆 → {qpath}")


def mode_score(cfg, rt, rd, fake=False):
    qpath, spath = _q(rd), os.path.join(rd, SCORES)
    with open(qpath, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    done = set()
    if os.path.exists(spath):
        with open(spath, encoding="utf-8") as f:
            done = {(j["source"], j["row"]) for l in f if l.strip()
                    for j in (json.loads(l),)}   # 以 (source,row) 去重：
            #   不同 ablation 變體的佇列行號各自獨立，line_no 不可當 key
    print(f"佇列 {len(lines)} 筆，已完成 {len(done)}")
    descs = rt.load_descriptions()
    if fake:
        from router.verifier import make_fake_scorer
        scorer = make_fake_scorer()
        print("[warn] --fake_verifier：偽分數，僅供管線自測，結果無評測意義")
    else:
        from router.verifier import load_llm, make_scorer
        tok, model = load_llm(cfg["verifier"]["model_path"])
        scorer = make_scorer(tok, model)
    from router.verifier import score_candidate
    t0, n = time.time(), 0
    every = cfg["verifier"]["checkpoint_every"]
    with open(spath, "a", encoding="utf-8") as fo:
        for ln in lines:
            if (ln["source"], ln["row"]) in done:
                continue
            ps = [score_candidate(scorer,
                                  descs[str(u)]["description"],
                                  rt.unit_examples.get(str(u), ""),
                                  ln["query"])
                  for u in ln["units"]]
            fo.write(json.dumps({"line_no": ln["line_no"],
                                 "source": ln["source"], "row": ln["row"],
                                 "units": ln["units"], "p_yes": ps},
                                ensure_ascii=False) + "\n")
            n += 1
            if n % every == 0:
                fo.flush()
                print(f"  {n} 筆 ({n/(time.time()-t0):.2f}/s)", flush=True)
    print(f"[score] 完成 → {spath}")


def mode_run(cfg, rt, id_tasks, ood_tasks, rd):
    Z = np.load(os.path.join(rd, ZONES))
    scores = {}
    spath = os.path.join(rd, SCORES)
    if os.path.exists(spath):
        with open(spath, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    s = json.loads(l)
                    scores[(s["source"], s["row"])] = s

    # 重放 tS：讀嵌入快取（CPU 快速），不重算嵌入；unit→task 還原需要它
    from router import fingerprint as fpm

    def _finalize_with_ts(src, emb):
        tS = fpm.task_sims(emb, rt.C, rt.owner, len(rt.id_tasks))
        esc = {r: s for (s2, r), s in scores.items() if s2 == src}
        dec = {"zone": Z[f"{src}__zone"], "b1": Z[f"{src}__b1"],
               "top3_units": Z[f"{src}__top3"], "tS": tS, "esc": esc}
        return rt.finalize(dec, count_missing=True, gray=GRAY_MODE)

    # ID
    te_emb = np.concatenate([data_io.load_embeddings(cfg, t, test=True)
                             for t in id_tasks])
    pred_id, miss = _finalize_with_ts("test", te_emb)
    y_id = Z["test__y"]
    ood_pred, ood_zone = {}, {}
    for o in ood_tasks:
        e = data_io.load_embeddings(cfg, o, test=True)
        p, m2 = _finalize_with_ts(f"ood_t{o}", e)
        miss += m2
        ood_pred[o] = p
        ood_zone[o] = Z[f"ood_t{o}__zone"]
    assert miss == 0, (f"送審缺分 {miss} 筆——先跑 --mode score 完成打分"
                       f"（或確認 {SCORES} 完整）")

    gt = data_io.load_ood_groundtruth(
        cfg["evaluation"].get("ood_groundtruth"))
    rep = metrics.compute_metrics(
        pred_id, y_id, rt.unit_of, rt.id_tasks, ood_pred, ood_gt=gt,
        ood_top_routes=cfg["evaluation"].get("ood_top_routes", 3))

    # 六流向表（ID / OOD / 總體 三口徑）
    def flow(zsets, preds):
        c = dict(direct=0, green=0, red=0, esc_route=0, esc_rej=0)
        for z, p in zip(zsets, preds):
            c["direct"] += int((z == 0).sum())
            c["green"] += int((z == 1).sum())
            c["red"] += int((z == 3).sum())
            e = z == 2
            c["esc_route"] += int((e & (p >= 0)).sum())
            c["esc_rej"] += int((e & (p < 0)).sum())
        return c
    cid = flow([Z["test__zone"]], [pred_id])
    cood = flow([ood_zone[o] for o in ood_tasks],
                [ood_pred[o] for o in ood_tasks])
    ni, no = int(y_id.size), sum(int(ood_pred[o].size) for o in ood_tasks)
    rep["flow_table"] = {"id": {**cid, "n": ni},
                         "ood": {**cood, "n": no},
                         "total": {k: cid[k] + cood[k] for k in cid} |
                                  {"n": ni + no}}
    for part, n in [("id", ni), ("ood", no), ("total", ni + no)]:
        s = sum(v for k, v in rep["flow_table"][part].items() if k != "n")
        assert s == n, f"{part} 流向合計 {s} != {n}"
    print("[check] 三口徑流向各自合計 = 樣本數（完整分割）")

    print("\n== micro 指標 ==")
    for k, v in rep["micro"].items():
        print(f"  {k:<18}{v}")
    print("== 六流向（ID / OOD / 總體）==")
    for k in ("direct", "green", "red", "esc_route", "esc_rej"):
        print(f"  {k:<11}{cid[k]:>7}  {cood[k]:>7}  {cid[k]+cood[k]:>7}")

    out = os.path.join(rd, f"router_eval_results{ABL_SUFFIX}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    import shutil
    ts = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy(out, out.replace(".json", f"_{ts}.json"))
    print(f"[done] → {out}（含時間戳副本）")


def main():
    cfg, args = config_from_cli(lambda p: (
        p.add_argument("--mode", required=True,
                       choices=["decide", "score", "run"]),
        p.add_argument("--fake_verifier", action="store_true"),
        p.add_argument("--ablate", default="none",
                       choices=["none", "no_multicentroid", "no_direct",
                                "no_lexical", "gray_reject", "gray_route"],
                       help="ablation 變體（見檔頭；no_multicentroid 需先以 "
                            "k_max=1 build 變體資產至 assets_ablate_nomc/）")))
    rd = cfg["paths"]["results_dir"]
    os.makedirs(rd, exist_ok=True)
    ab = args.ablate
    if ab == "no_multicentroid":
        if cfg["paths"]["assets_dir"] == "assets":     # 使用者 --set 過則尊重之
            cfg["paths"]["assets_dir"] = "assets_ablate_nomc"
    elif ab == "no_direct":
        cfg["thresholds"]["transfer_floor"] = 999.0
    elif ab == "no_lexical":
        cfg["thresholds"]["use_lexical"] = False
    globals()["ABL_SUFFIX"] = "" if ab == "none" else f"__{ab}"
    globals()["GRAY_MODE"] = ("reject" if ab == "gray_reject" else
                              "route" if ab == "gray_route" else "verdict")
    rt = Router.load(cfg)
    id_tasks, ood_tasks = discover_tasks(cfg)
    assert id_tasks == rt.id_tasks, \
        "dataset 任務集合與資產不一致——請先重跑 build_router_assets"
    if args.mode == "decide":
        mode_decide(cfg, rt, id_tasks, ood_tasks, rd)
    elif args.mode == "score":
        mode_score(cfg, rt, rd, fake=args.fake_verifier)
    else:
        mode_run(cfg, rt, id_tasks, ood_tasks, rd)


if __name__ == "__main__":
    main()

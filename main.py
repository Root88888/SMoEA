#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — SMoEA 主程式：query → Router → adapter → inference → output

架構圖對應：Router 分流之後，路由樣本
由 system/InferenceEngine 載對應任務 adapter 生成（本檔完成）；拒絕
樣本進 Model Merging 分支（接口 system/rejection.py，待實作——
目前拒絕樣本會印明去向並跳過生成）。

【互動模式】單筆 query 跑完整流程並顯示 router 逐步判定：
  python main.py --mode interactive
  送審裁決模型依 configs system.verifier_mode：
    resident_4bit  裁決 LLM 以 4bit 常駐、與生成模型共存（預設，秒級）
    swap           分時載卸（精度同封板評測、單筆送審多 ~1 分鐘）
    off            不載裁決，送審一律拒絕（fail-closed）

【批次模式】跑 dataset 測試檔全部（或指定任務），逐筆路由＋生成：
  python main.py --mode batch [--tasks 3,7,10] [--limit 50]
  → results/main_batch_outputs.jsonl（每行：instance_id / query /
    router 診斷 / 去向 / 模型輸出；拒絕樣本 output=null、留診斷）
  流程分三段執行以省模型切換：全量分區 → 送審打分 → 按任務分組生成。

【scale up】任務、adapter、樣本量全部資料驅動；生成按任務分組批次、
adapter 熱切換 O(1)。
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router import conformal, data_io  # noqa: E402
from router.config import config_from_cli, discover_tasks  # noqa: E402
from router.core import Router  # noqa: E402
from system.inference import InferenceEngine  # noqa: E402

ZONE_LABEL = {0: "直判路由 (margin>floor)", 1: "綠區路由",
              2: "送審", 3: "紅區拒絕"}


# ---------------------------------------------------------------------------
# 裁決模型（互動模式）
# ---------------------------------------------------------------------------
def make_verifier(cfg, mode):
    """依 verifier_mode 回傳 scorer 或 None（off / swap 延後載）。"""
    if mode != "resident_4bit":
        return None
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from router.verifier import make_scorer
    path = cfg["verifier"]["model_path"]
    print(f"[verifier] 4bit 常駐載入 {path} …", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16))
    model.eval()
    return make_scorer(tok, model)


def swap_verify(cfg, rt, engine, texts, dec):
    """swap 模式：卸生成模型 → 載裁決（float16、同封板）→ 打分 → 卸。"""
    from router.verifier import load_llm, make_scorer
    engine.unload()
    tok, model = load_llm(cfg["verifier"]["model_path"])
    rt.escalate(dec, texts, make_scorer(tok, model), rt.load_descriptions())
    del model, tok
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 診斷
# ---------------------------------------------------------------------------
def diagnosis_of(rt, dec, row):
    """單筆 decide 結果 → 標準化診斷 dict（互動列印與拒絕分支共用）。"""
    top_u = [int(u) for u in dec["top3_units"][row]]
    tS = dec["tS"][row]
    top_tasks, top_sims = [], []
    for u in top_u:
        g = rt.units[u]
        ti = g[int(np.argmax(tS[g]))]
        top_tasks.append(f"task{rt.id_tasks[ti]}")
        top_sims.append(round(float(tS[g].max()), 4))
    esc = dec.get("esc", {}).get(row)
    return {"zone": int(dec["zone"][row]),
            "margin": round(float(dec["margin"][row]), 4),
            "pval": round(float(dec["pval"][row]), 4),
            "lex_agree": bool(~dec["lex_disagree"][row]),
            "top_units": top_u, "top_tasks": top_tasks,
            "top_sims": top_sims,
            "esc_p_yes": esc["p_yes"] if esc else None}


def print_diagnosis(d):
    print(f"[Router] margin={d['margin']}  p={d['pval']}  "
          f"詞彙一致{'✓' if d['lex_agree'] else '✗'}")
    print(f"[Router] top-3：" + "  ".join(
        f"{t}(sim {s})" for t, s in zip(d["top_tasks"], d["top_sims"])))
    if d["esc_p_yes"] is not None:
        print(f"[Router] 送審裁決 p_yes：" + "  ".join(
            f"{t}:{p}" for t, p in zip(d["top_tasks"], d["esc_p_yes"])))


# ---------------------------------------------------------------------------
# 互動模式
# ---------------------------------------------------------------------------
def run_interactive(cfg, rt, preload=True):
    engine = InferenceEngine(cfg)
    vmode = cfg["system"]["verifier_mode"]
    scorer = make_verifier(cfg, vmode)
    descs = rt.load_descriptions() if vmode != "off" else None
    if descs is not None:
        miss = [u for u in range(len(rt.units)) if str(u) not in descs]
        if miss:
            raise ValueError(
                f"單位說明書缺 {len(miss)} 個單位（如 unit {miss[:5]}）——"
                f"assets/unit_descriptions.json 與當前路由資產版本不符。"
                f"常見原因：dataset 任務集合與說明書生成時不同"
                f"（多/少了任務）。請對齊任務集合重 build，"
                f"或更新說明書後重試")
    if preload:
        print("[warmup] 預載嵌入模型…", flush=True)
        rt._embed(["warmup"])          # bge 首載也在開場完成
        if vmode != "swap":
            # 重載入全部集中在開場：使用者輸入後的等待只剩
            # adapter 熱切換（秒級）與生成本身。swap 模式不預載
            # 生成模型（其設計本來就是用到才載、判完即卸）。
            engine.load_base()
    print("\n===== SMoEA 互動模式（輸入 query，exit 離開）=====")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        dec = rt.decide([q])
        z = int(dec["zone"][0])
        if z == conformal.ZONE_ESCALATE:
            if vmode == "resident_4bit":
                rt.escalate(dec, [q], scorer, descs)
            elif vmode == "swap":
                swap_verify(cfg, rt, engine, [q], dec)
            else:
                print("[Router] 送審（verifier_mode=off → fail-closed 拒絕）")
        d = diagnosis_of(rt, dec, 0)
        print_diagnosis(d)
        pred, _ = rt.finalize(dec, count_missing=True)
        if pred[0] >= 0:
            task_key = f"task{rt.id_tasks[int(pred[0])]}"
            print(f"[Router] 判定：{ZONE_LABEL[z]} → {task_key}")
            engine.ensure_adapter(task_key)
            t0 = time.time()
            out = engine.generate([q])[0]
            print(f"[Output] ({time.time()-t0:.1f}s)\n{out}")
        else:
            d["zone"] = "red" if z == conformal.ZONE_RED else "esc_rej"
            print(f"[Router] 判定：{ZONE_LABEL[z] if z != 2 else '送審→拒絕'}"
                  f" → 進入 Adapter Merging 分支")
            try:
                from system.rejection import handle_rejection
                out = handle_rejection(q, engine)
                print(f"[Output]\n{out}")
            except NotImplementedError as e:
                print(f"[System] {e}")


# ---------------------------------------------------------------------------
# 批次模式
# ---------------------------------------------------------------------------
def run_batch(cfg, rt, tasks_arg, limit):
    id_tasks, ood_tasks = discover_tasks(cfg)
    wanted = ([int(x) for x in tasks_arg.split(",")]
              if tasks_arg else id_tasks + ood_tasks)
    rows = []                       # (task, instance_id, query_text, prompt)
    for t in wanted:
        recs = data_io.load_task_records(cfg, t, test=True)
        if limit:
            recs = recs[:limit]
        field = cfg["data"]["field"]
        for r in recs:
            rows.append((t, r.get("instance_id", f"task{t}-?"),
                         str(r.get(field, ""))[:3000],
                         r.get("full_prompt", r.get(field, ""))))
    print(f"[batch] {len(wanted)} 任務、{len(rows)} 筆")

    texts = [r[2] for r in rows]
    # 嵌入：優先用建置時的快取（與評測同一份、免現算）；缺才現算
    embs = []
    for t in wanted:
        n_t = sum(1 for r in rows if r[0] == t)
        try:
            e = data_io.load_embeddings(cfg, t, test=True)[:n_t]
            assert e.shape[0] == n_t
            embs.append(e)
        except (FileNotFoundError, AssertionError):
            embs = None
            break
    dec = rt.decide(texts,
                    emb=(np.concatenate(embs) if embs else None))
    n_esc = int((dec["zone"] == conformal.ZONE_ESCALATE).sum())
    if n_esc:
        print(f"[batch] 送審 {n_esc} 筆 → 載裁決模型打分（float16 同封板）")
        engine_tmp = InferenceEngine(cfg)   # 尚未載，unload 為 no-op
        swap_verify(cfg, rt, engine_tmp, texts, dec)
    pred, miss = rt.finalize(dec, count_missing=True)
    assert miss == 0

    engine = InferenceEngine(cfg)
    outputs = [None] * len(rows)
    routed = {}
    for i, p in enumerate(pred):
        if p >= 0:
            routed.setdefault(f"task{rt.id_tasks[int(p)]}", []).append(i)
    bs = int(cfg["system"]["batch_size"])
    for task_key, idxs in routed.items():
        engine.ensure_adapter(task_key)
        for s in range(0, len(idxs), bs):
            chunk = idxs[s:s + bs]
            outs = engine.generate([rows[i][3] for i in chunk])
            for i, o in zip(chunk, outs):
                outputs[i] = o
        print(f"[batch] {task_key}: {len(idxs)} 筆生成完")

    rd = cfg["paths"]["results_dir"]
    os.makedirs(rd, exist_ok=True)
    out_path = os.path.join(rd, "main_batch_outputs.jsonl")
    n_rej = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (t, iid, _q, prompt) in enumerate(rows):
            d = diagnosis_of(rt, dec, i)
            p = int(pred[i])
            if p < 0:
                n_rej += 1
            f.write(json.dumps(
                {"source_task": f"task{t}", "instance_id": iid,
                 "routed_to": (f"task{rt.id_tasks[p]}" if p >= 0 else None),
                 "diagnosis": d, "output": outputs[i]},
                ensure_ascii=False) + "\n")
    import shutil
    ts = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy(out_path, out_path.replace(".jsonl", f"_{ts}.jsonl"))
    print(f"[batch] 路由生成 {len(rows)-n_rej} 筆、拒絕 {n_rej} 筆"
          f"（拒絕樣本 output=null，待 Adapter Merging 分支）")
    print(f"[done] → {out_path}（含時間戳副本）")


def main():
    cfg, args = config_from_cli(lambda p: (
        p.add_argument("--mode", required=True,
                       choices=["interactive", "batch"]),
        p.add_argument("--tasks", default=None,
                       help="批次模式限定任務，如 3,7,10；預設全部"),
        p.add_argument("--limit", type=int, default=None,
                       help="批次模式每任務最多筆數（試跑用）"),
        p.add_argument("--no_preload", action="store_true",
                       help="互動模式不預載模型（只看路由判定的輕量用法）")))
    rt = Router.load(cfg)
    print(f"[Router] 資產已載：{len(rt.id_tasks)} 任務、"
          f"{len(rt.units)} 路由單位")
    if args.mode == "interactive":
        run_interactive(cfg, rt, preload=not args.no_preload)
    else:
        run_batch(cfg, rt, args.tasks, args.limit)


if __name__ == "__main__":
    main()

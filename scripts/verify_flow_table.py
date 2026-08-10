#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/verify_flow_table.py

【六流向 × 三口徑分流表：獨立重放驗證】台達驗收時「數字可信」的機器
證明。計算路徑刻意獨立：不讀主評測的分區落地檔（eval_zones.npz），
而是由 Router.load + decide 重放分區、由分數檔重放送審裁決，算出
六流向表後才與 results/router_eval_results.json 的 flow_table 逐格
對帳——雙路徑一致才印 PASS。

  流向：直判路由 / 綠區路由 / 紅區拒絕 / 送審→路由 / 送審→拒絕
  口徑：ID 側 / OOD 側 / 評測集總體

【執行】(主評測三段跑完之後；純 CPU 數分鐘)
  python scripts/verify_flow_table.py 2>&1 | tee results/verify_flow_log.txt

【scale up】表列與口徑由資料驅動、零改動。
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router import conformal, data_io  # noqa: E402
from router.config import config_from_cli, discover_tasks  # noqa: E402
from router.core import Router  # noqa: E402


def main():
    cfg, _ = config_from_cli()
    rd = cfg["paths"]["results_dir"]
    rt = Router.load(cfg)
    id_tasks, ood_tasks = discover_tasks(cfg)

    # 送審分數（重放裁決用）
    scores = {}
    spath = os.path.join(rd, "escalation_scores.jsonl")
    with open(spath, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                s = json.loads(l)
                scores[(s["source"], s["row"])] = s

    theta = cfg["thresholds"]["theta_verify"]
    KEYS = ["direct", "green", "red", "esc_route", "esc_rej"]

    def count(src, texts, emb):
        d = rt.decide(texts, emb=emb)          # 獨立重放分區
        z = d["zone"]
        c = dict.fromkeys(KEYS, 0)
        c["direct"] = int((z == conformal.ZONE_FLOOR).sum())
        c["green"] = int((z == conformal.ZONE_GREEN).sum())
        c["red"] = int((z == conformal.ZONE_RED).sum())
        miss = 0
        for r in np.where(z == conformal.ZONE_ESCALATE)[0]:
            s = scores.get((src, int(r)))
            if s is None:
                miss += 1
                continue
            j = int(np.argmax(s["p_yes"]))
            c["esc_route" if s["p_yes"][j] >= theta else "esc_rej"] += 1
        assert miss == 0, f"{src} 送審缺分 {miss} 筆，無法驗證"
        return c, z.shape[0]

    te_txt, te_emb = [], []
    for t in id_tasks:
        te_txt += data_io.load_task_texts(cfg, t, test=True)
        te_emb.append(data_io.load_embeddings(cfg, t, test=True))
    cid, ni = count("test", te_txt, np.concatenate(te_emb))
    cood, no = dict.fromkeys(KEYS, 0), 0
    for o in ood_tasks:
        txt = data_io.load_task_texts(cfg, o, test=True)
        co, n = count(f"ood_t{o}", txt,
                      data_io.load_embeddings(cfg, o, test=True))
        no += n
        for k in KEYS:
            cood[k] += co[k]
    ctot = {k: cid[k] + cood[k] for k in KEYS}

    NAMES = {"direct": "直判路由 (margin>floor)",
             "green": "綠區路由 (p>=p_hi 且詞彙一致)",
             "red": "紅區拒絕 (p<p_lo)",
             "esc_route": "送審->路由", "esc_rej": "送審->拒絕"}
    print(f"\n{'流向':<32}{'ID側':<18}{'OOD側':<18}{'總體':<18}")
    for k in KEYS:
        a, b, c = cid[k], cood[k], ctot[k]
        print(f"{NAMES[k]:<32}{a/ni:>7.4f} ({a:>5})  "
              f"{b/max(no,1):>7.4f} ({b:>5})  "
              f"{c/(ni+no):>7.4f} ({c:>5})")
    print(f"{'樣本數':<32}{ni:>15}{no:>18}{ni+no:>18}")
    for c, n, nm in [(cid, ni, "ID"), (cood, no, "OOD"),
                     (ctot, ni + no, "總體")]:
        assert sum(c.values()) == n, f"{nm} 合計 != {n}"
    print("[check] 三口徑各自合計 = 樣本數（完整分割）")

    # 與主評測結果逐格對帳
    with open(os.path.join(rd, "router_eval_results.json"),
              encoding="utf-8") as f:
        ft = json.load(f)["flow_table"]
    bad = [(nm, k, mine[k], js[k]) for nm, mine, js in
           [("id", cid, ft["id"]), ("ood", cood, ft["ood"])]
           for k in KEYS if mine[k] != js[k]]
    if bad:
        for nm, k, m, j in bad:
            print(f"[FAIL] {nm}.{k}: 重放 {m} != 主評測 {j}")
        raise SystemExit(1)
    print(f"[PASS] 獨立重放與主評測 flow_table 逐格一致"
          f"（{len(KEYS)*2} 格全對）")


if __name__ == "__main__":
    main()

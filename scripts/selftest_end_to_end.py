#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/selftest_end_to_end.py

【端到端管線自測】零 GPU、零下載、零真實資料——在暫存目錄生成合成
dataset 與假嵌入快取，把完整管線走一遍：

  1. 合成資料：6 個 ID 任務（含一對攣生）+ 2 個 OOD（一真陌生、一近親），
     每任務數十筆、各任務用不同詞彙集（讓 TF-IDF 有訊號）；
     假嵌入 = 任務中心方向 + 噪聲（決定性種子），直接寫入 assets/
     快取 → build 的嵌入段自動跳過、不觸發模型下載。
  2. build_router_assets：建置 + 資產落地。
  3. eval_router 三段（score 段 --fake_verifier）：
     先無 gt 模式（驗 overall/OOD acc 鍵取消、行為描述輸出），
     再寫任務級 gt 檔跑有 gt 模式（驗七鍵齊全）。
  4. verify_flow_table：獨立重放對帳 PASS。
  5. 兩支 baseline：score_dist + eval 各跑一次。

【執行】repo 根目錄：python scripts/selftest_end_to_end.py
全部通過印 ALL PASS、結束碼 0。此腳本也是新環境（如國網）部署後的
第一道驗收——它通過代表依賴、路徑、管線全部就緒。
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)


WORDS = {  # 每任務專屬詞彙集（TF-IDF 訊號）
    0: "translate french sentence grammar verb",
    1: "translate france phrase grammar verbs",     # 攣生（與 t0 幾乎同）
    2: "classify sentiment review positive negative movie",
    3: "summarize article paragraph short abstract news",
    4: "arithmetic add subtract number integer math",
    5: "extract entity person location organization name",
    901: "quantum entanglement physics wavefunction operator",  # OOD 真陌生
    902: "classify sentiment comment positive negative film",   # OOD 近親 t2
}
DIRS = {}  # 任務中心方向


def _center(task, dim, rng):
    if task not in DIRS:
        v = rng.randn(dim)
        DIRS[task] = v / np.linalg.norm(v)
    return DIRS[task]


def make_synth(root):
    rng = np.random.RandomState(7)
    dim = 32
    ds_tr = os.path.join(root, "dataset", "train_data")
    ds_te = os.path.join(root, "dataset", "test_data")
    ad = os.path.join(root, "assets")
    for d in (ds_tr, ds_te, ad, os.path.join(root, "results")):
        os.makedirs(d, exist_ok=True)
    # 攣生：t1 中心 = t0 中心加微擾；OOD 近親：t902 中心 = t2 加中擾；
    # OOD 真陌生 t901：對「全部 ID 中心等距」（= ID 中心平均方向）——
    # 等距 ⇒ top1≈top2 ⇒ margin 貼地 ⇒ 走統計區被紅區/送審攔（真實系統
    # 中「高分零 margin」型陌生的合成復刻）。噪聲 0.30 讓部分 ID 樣本
    # 也落入統計帶，四區才都有流量可測。
    for t in range(6):
        _center(t, dim, rng)
    DIRS[1] = DIRS[0] + 0.02 * rng.randn(dim)
    DIRS[1] /= np.linalg.norm(DIRS[1])
    DIRS[902] = DIRS[2] + 0.25 * rng.randn(dim)
    DIRS[902] /= np.linalg.norm(DIRS[902])
    eq = DIRS[3] + DIRS[4]      # 兩個 ID 中心的正中點：中垂面上
    DIRS[901] = eq / np.linalg.norm(eq)  # top1≈top2 ⇒ margin 貼地
    NOISE = {t: 0.30 for t in list(range(6)) + [902]}
    NOISE[0] = NOISE[1] = 0.12  # 攣生對：噪聲小才守得住指紋相似度>0.97
    NOISE[901] = 0.03   # 緊貼中垂面，margin 貼地（0.10 相對直判線太寬）

    def emit(task, n, test):
        c = _center(task, dim, rng)
        words = WORDS[task].split()
        texts, embs = [], []
        for i in range(n):
            k = rng.randint(4, 8)
            texts.append(" ".join(rng.choice(words, k)) + f" sample {i}")
            e = c + NOISE[task] * rng.randn(dim)
            embs.append(e / np.linalg.norm(e))
        sub = ds_te if test else ds_tr
        stem = f"task{task}_{'test' if test else 'train'}.json"
        with open(os.path.join(sub, stem), "w", encoding="utf-8") as f:
            json.dump([{"input": t} for t in texts], f)
        np.savez_compressed(
            os.path.join(ad, (f"emb_test_task{task}" if test
                              else f"emb_task{task}") + ".npz"),
            emb=np.stack(embs).astype(np.float32))

    for t in range(6):
        emit(t, 90, test=False)
        emit(t, 40, test=True)
    for o in (901, 902):
        emit(o, 40, test=True)

    # 假單位說明書（單位數 = build 後才知；先逐任務寫、build 後補齊單位鍵）
    # 簡化：說明書按單位編號，於 build 之後由本測試補寫。
    # config：指向合成目錄、CPU、假模型名（不會被載入）
    cfgp = os.path.join(root, "config.yaml")
    shutil.copy(os.path.join(REPO, "configs", "default.yaml"), cfgp)
    txt = open(cfgp, encoding="utf-8").read()
    txt = txt.replace("dataset_dir: dataset",
                      f"dataset_dir: {root}/dataset")
    txt = txt.replace("assets_dir: assets", f"assets_dir: {root}/assets")
    txt = txt.replace("results_dir: results", f"results_dir: {root}/results")
    txt = txt.replace("device: cuda", "device: cpu")
    open(cfgp, "w", encoding="utf-8").write(txt)
    return cfgp, ad


def run(cmd, **kw):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, **kw)
    sys.stdout.write(r.stdout[-2200:])
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


def main():
    root = tempfile.mkdtemp(prefix="smoea_e2e_")
    print(f"合成環境：{root}")
    cfgp, ad = make_synth(root)
    py = sys.executable
    C = ["--config", cfgp]

    # ---- build ----
    out = run([py, "scripts/build_router_assets.py"] + C)
    assert "缺單位說明書" in out
    meta = json.load(open(os.path.join(ad, "router_assets_meta.json")))
    units = meta["units"]
    twin_unit = [g for g in units if len(g) > 1]
    assert twin_unit and set(twin_unit[0]) == {0, 1}, \
        f"攣生任務未併組：{units}"
    print(f"[E2E-PASS] build：攣生 t0/t1 併為單位、資產落地齊全")

    # 補假說明書（每單位一條）
    descs = {str(u): {"description": f"synthetic unit {u}: "
             + " ".join(WORDS[meta['id_tasks'][g[0]]].split()[:3])}
             for u, g in enumerate(units)}
    with open(os.path.join(ad, "unit_descriptions.json"), "w",
              encoding="utf-8") as f:
        json.dump({"descriptions": descs}, f)

    # ---- eval 三段（無 gt 模式）----
    run([py, "scripts/eval_router.py", "--mode", "decide"] + C)
    run([py, "scripts/eval_router.py", "--mode", "score",
         "--fake_verifier"] + C)
    run([py, "scripts/eval_router.py", "--mode", "run"] + C)
    rep = json.load(open(os.path.join(root, "results",
                                      "router_eval_results.json")))
    m = rep["micro"]
    assert rep["metadata"]["mode"] == "no_ood_gt"
    for banned in ("overall_acc", "ood_acc", "ood_rej_acc",
                   "ood_route_acc"):
        assert banned not in m
    assert m["id_acc_unit"] > 0.9, f"合成資料 ID unit acc 異常低：{m}"
    r901 = rep["per_task_ood"]["901"]["reject_rate"]
    r902 = rep["per_task_ood"]["902"]["reject_rate"]
    # fake verifier 均勻分數下，送審拒絕率理論值 = P(max of 3 < θ=0.5)
    # = 0.125，故真陌生（幾乎全送審）拒絕率期望 ~0.12±；斷言驗「相對序
    # 且非零」而非絕對水準（絕對水準是真 LLM 的事）。
    assert r901 > 0.08 and r901 > 2 * r902, \
        f"真陌生應明顯高拒：t901={r901} t902={r902}"
    assert "top_routes" in rep["per_task_ood"]["902"]
    print(f"[E2E-PASS] eval（無 gt）：acc 鍵取消、真陌生拒 {r901} > "
          f"近親拒 {r902}、行為描述齊全")

    # ---- verify（獨立重放對帳）----
    out = run([py, "scripts/verify_flow_table.py"] + C)
    assert "[PASS]" in out
    print("[E2E-PASS] verify_flow_table：雙路徑逐格一致")

    # ---- 有 gt 模式 ----
    gtp = os.path.join(root, "ood_gt.json")
    json.dump({"results": {"901": {"standard_answer": "reject"},
                           "902": {"standard_answer": "route_to_task2"}}},
              open(gtp, "w"))
    run([py, "scripts/eval_router.py", "--mode", "run"] + C
        + ["--set", f"evaluation.ood_groundtruth={gtp}"])
    rep = json.load(open(os.path.join(root, "results",
                                      "router_eval_results.json")))
    m = rep["micro"]
    assert rep["metadata"]["mode"] == "with_ood_gt"
    for need in ("overall_acc", "ood_acc", "ood_rej_acc", "ood_route_acc"):
        assert need in m and m[need] is not None
    assert m["ood_rej_acc"] > 0.08   # fake verifier max-of-3 理論 ~0.12
    print(f"[E2E-PASS] eval（有 gt）：七鍵齊全 "
          f"overall={m['overall_acc']} 應拒={m['ood_rej_acc']} "
          f"應路由={m['ood_route_acc']}")

    # ---- 兩支 baseline ----
    run([py, "scripts/eval_baseline_mean_embedding.py",
         "--mode", "score_dist"] + C)
    run([py, "scripts/eval_baseline_mean_embedding.py",
         "--mode", "eval", "--tau", "0.72"] + C
        + ["--set", f"evaluation.ood_groundtruth={gtp}"])
    run([py, "scripts/eval_baseline_bm25_voting.py",
         "--mode", "score_dist"] + C)
    run([py, "scripts/eval_baseline_bm25_voting.py",
         "--mode", "eval", "--ratio_tau", "0.5"] + C)
    b = json.load(open(os.path.join(
        root, "results", "baseline_meanemb_results_tau0.72.json")))
    assert b["metadata"]["mode"] == "with_ood_gt"
    b2 = json.load(open(os.path.join(
        root, "results", "baseline_bm25_results_tau0.5.json")))
    assert b2["metadata"]["mode"] == "no_ood_gt"   # 未帶 gt 覆蓋
    print("[E2E-PASS] baselines：兩支三模式管線可跑、兩模式計分正確")

    shutil.rmtree(root)
    print("\nALL PASS（合成環境已清理）")


if __name__ == "__main__":
    main()

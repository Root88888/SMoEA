#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/selftest_core_modules.py

【底層模組自測】不需資料集、純合成輸入，驗證五件關鍵性質：
  1. units：union-find 遞移閉包（A~B、B~C ⇒ A,B,C 同單位）
  2. conformal.group_pvals：可交換合成資料下 p 值 super-uniform
     （P(p < α) ≤ α，多個 α 檢查）
  3. conformal.zones：四區優先序（直判 > 紅 > 送審 > 綠；詞彙不一致
     只把綠拉下送審、不動紅區）
  4. metrics 有 gt 模式：手算小例逐鍵對帳
  5. metrics 無 gt 模式：overall/OOD acc 鍵「不存在」（定案：取消而非 null）、
     top_routes 與 concentration 正確

【執行】repo 根目錄下：python scripts/selftest_core_modules.py
全部 PASS 才結束碼 0。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router import conformal, metrics, units  # noqa: E402


def test_units():
    fp = np.eye(4, 8, dtype=np.float32)          # 4 個正交指紋
    fp[1] = fp[0] * 0.999 + fp[1] * 0.001        # t1 ≈ t0
    fp[2] = fp[1] * 0.999 + fp[2] * 0.001        # t2 ≈ t1（但 t2·t0 較低）
    fp = fp / np.linalg.norm(fp, axis=1, keepdims=True)
    us, unit_of, pairs = units.build_units(fp, sim_threshold=0.97)
    assert unit_of[0] == unit_of[1] == unit_of[2], "遞移閉包失敗"
    assert unit_of[3] != unit_of[0], "不相似任務被誤併"
    assert len(us) == 2
    print("[PASS] units：遞移閉包與獨立單位正確")


def test_pvals_super_uniform():
    rng = np.random.RandomState(0)
    n_cal, n_test, trials = 400, 2000, 30
    hits = {0.02: [], 0.05: [], 0.10: []}
    for _ in range(trials):
        cal = rng.randn(n_cal)                   # 同分布 → 可交換
        te = rng.randn(n_test)
        g0 = np.zeros(n_cal, dtype=int)
        g1 = np.zeros(n_test, dtype=int)
        pv = conformal.group_pvals(g1, te, g0, cal, 1)
        for a in hits:
            hits[a].append(float((pv < a).mean()))
    for a, v in hits.items():
        m = float(np.mean(v))
        # 均值應貼著 a 以下（super-uniform 精確保證 + 抽樣波動容差）
        assert m <= a + 0.01, f"P(p<{a}) 均值 {m:.4f} 超出容差"
    print("[PASS] conformal：p 值 super-uniform（30 次重抽、3 個 α）")


def test_zones_priority():
    thr = {"transfer_floor": 0.10, "p_lo": 0.02, "p_hi": 0.10}
    margin = np.array([0.20, 0.01, 0.01, 0.01, 0.01, 0.20])
    pval = np.array([0.001, 0.001, 0.05, 0.50, 0.50, 0.50])
    lexd = np.array([True, False, True, False, True, True])
    z = conformal.zones(margin, pval, lexd, thr)
    # 逐筆：0 直判蓋過低p｜1 紅｜2 送審帶｜3 綠｜4 綠被詞彙拉下送審｜5 直判蓋過詞彙
    assert z.tolist() == [0, 3, 2, 1, 2, 0], f"分區優先序錯誤：{z.tolist()}"
    print("[PASS] zones：四區優先序（直判最高、詞彙票只動綠區）")


def _toy_setup():
    id_tasks = [0, 9, 22]                        # 任務 id（索引 0,1,2）
    unit_of = np.array([0, 0, 1])                # t0,t9 攣生同單位
    y = np.array([0, 0, 1, 1, 2, 2])
    #        對:0  攣生:1  對:1  拒:-2  錯:0  對:2
    pred = np.array([0, 1, 1, -2, 0, 2])
    ood_pred = {901: np.array([-2, -2, -2, 0]),      # 應拒，3/4 拒
                902: np.array([1, 1, 1, 0, -2])}     # 應路由 t9，3/5 對
    return id_tasks, unit_of, y, pred, ood_pred


def test_metrics_with_gt():
    id_tasks, unit_of, y, pred, ood_pred = _toy_setup()
    r = metrics.compute_metrics(pred, y, unit_of, id_tasks, ood_pred,
                                ood_gt={901: -1, 902: 9})
    m = r["micro"]
    # 手算：pred=[0,1,1,-2,0,2] vs y=[0,0,1,1,2,2]
    #   task 級對 3 筆（位置 0,2,5）；unit 級對 4 筆（位置 1 攣生救回）
    assert m["id_acc_task"] == round(3 / 6, 4)
    assert m["id_acc_unit"] == round(4 / 6, 4)
    assert m["id_reject_rate"] == round(1 / 6, 4)
    assert m["ood_rej_acc"] == 0.75
    assert m["ood_route_acc"] == 0.6
    assert m["ood_acc"] == round(6 / 9, 4)
    assert m["overall_acc"] == round((4 + 3 + 3) / 15, 4)
    assert r["metadata"]["mode"] == "with_ood_gt"
    assert r["metadata"]["denominators"]["overall"] == 15
    print("[PASS] metrics（有 gt）：七鍵手算逐一對帳")


def test_metrics_no_gt():
    id_tasks, unit_of, y, pred, ood_pred = _toy_setup()
    r = metrics.compute_metrics(pred, y, unit_of, id_tasks, ood_pred,
                                ood_gt=None)
    m = r["micro"]
    for banned in ("overall_acc", "ood_acc", "ood_rej_acc",
                   "ood_route_acc"):
        assert banned not in m, f"無 gt 模式不應存在鍵 {banned}"
    assert m["id_acc_task"] == round(3 / 6, 4)
    assert m["ood_reject_rate"] == round(4 / 9, 4)
    row = r["per_task_ood"]["902"]
    assert row["route_rate"] == 0.8
    assert row["top_routes"][0]["task"] == "t9"      # 路由 4 筆中 3 筆去 t9
    assert row["concentration"] == 0.75
    assert r["metadata"]["mode"] == "no_ood_gt"
    print("[PASS] metrics（無 gt）：acc 鍵取消、行為描述與集中度正確")


if __name__ == "__main__":
    test_units()
    test_pvals_super_uniform()
    test_zones_priority()
    test_metrics_with_gt()
    test_metrics_no_gt()
    print("\n全部自測 PASS")

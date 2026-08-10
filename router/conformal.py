# -*- coding: utf-8 -*-
"""
router/conformal.py

【共形校準與四區分流】router 的統計核心，四件事：

1. split_indices：train 樣本逐任務切三堆（指紋/校準/驗證）。校準堆與
   指紋堆的分配是「內容盲」隨機切分、且打分函數（指紋）只由指紋堆建成
   ——校準查詢與測試查詢對打分函數同為建置局外人，這兩點是可交換性
   （conformal validity）成立的程序性前提，不可省略或混堆。

2. margin：分數 = 全域 unit margin（top-1 相似度 − top-2 相似度），
   讀作「第一名的甩開幅度／唯一認領者存在性」。margin 型分數對全域
   風格偏移一階免疫（兩名同漲同跌、差值穩定）。

3. group_pvals：Mondrian 共形 p 值——按「預測 top-1 單位」分組（分組
   函數是輸入的可測函數，可交換性保持），p = (1 + #{該組校準分數 ≥ q})
   / (n_g + 1)。白話：拿這筆查詢的被認領明確度，跟該單位正牌成員隊伍
   比站位；p 是「比它更突兀的正牌成員占比」。可交換前提下 p 值
   super-uniform，故 P(p < α) ≤ α ——紅區誤拒率受控的出處。

4. zones：四區分流（傳回碼 0 直判 / 1 綠 / 2 送審 / 3 紅）：
     margin > transfer_floor            → 0 直判路由（標籤語意釘死的可遷移線）
     p < p_lo                           → 3 紅區拒絕（定理預算，預設 0.02）
     p_lo ≤ p < p_hi                    → 2 送審（成本旋鈕上界，預設 0.10）
     其餘（p ≥ p_hi）且詞彙一致          → 1 綠區路由
     其餘但詞彙 top-1 ≠ 嵌入 top-1       → 2 送審（詞彙第二票只升級不放行）

（split_indices / top2 / group_pvals / zones 邏輯自 v4→v7 鏈原樣收編；
 分區優先序與 verify 腳本的六流向定義一致。）

【scale up】組數 = 單位數，隨資料自動增長；校準樣本量不足的組其
p 值自然偏保守（分母 +1），無需特判。
"""

import numpy as np


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------
def split_indices(n, fractions, seed):
    """單一任務 n 筆樣本 → (指紋堆, 校準堆, 驗證堆) 三組索引。

    呼叫慣例：seed 傳 cfg 的 split.seed + task_id（逐任務獨立洗牌、
    整體可重現）。fractions 例 [0.6, 0.2, 0.2]。
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_fp = int(round(fractions[0] * n))
    n_cal = int(round(fractions[1] * n))
    return perm[:n_fp], perm[n_fp:n_fp + n_cal], perm[n_fp + n_cal:]


# ---------------------------------------------------------------------------
# margin
# ---------------------------------------------------------------------------
def top2(mat):
    """逐列 top-2：回傳 (idx1, val1, idx2, val2)。"""
    i1 = mat.argmax(axis=1)
    ar = np.arange(mat.shape[0])
    v1 = mat[ar, i1]
    m2 = mat.copy()
    m2[ar, i1] = -np.inf
    i2 = m2.argmax(axis=1)
    v2 = m2[ar, i2]
    return i1, v1, i2, v2


def unit_margins(uS, n_top=3):
    """從單位相似度矩陣 (n, U) 算決策所需全部量。

    回傳 dict：
      b1     (n,)  全域 top-1 單位（margin 的認領者、Mondrian 分組鍵、
                   綠區/直判的路由目標）
      margin (n,)  全域 unit margin = top1 − top2
      top    (n, n_top) 全域前 n_top 單位（送審候選；與 margin 同一次
                   排序產出，慢路徑零重算）
    """
    b1, v1, _, v2 = top2(uS)
    order = np.argsort(-uS, axis=1)[:, :n_top]
    return {"b1": b1, "margin": v1 - v2, "top": order}


# ---------------------------------------------------------------------------
# Mondrian 共形 p 值
# ---------------------------------------------------------------------------
def group_pvals(q_groups, q_scores, c_groups, c_scores, n_groups):
    """逐 Mondrian 組的共形 p 值。

    分數方向慣例：傳入「非一致性分數」（越大越突兀）。本系統的用法是
    傳 -margin（margin 越小越突兀）。
    p = (1 + #{該組校準分數 ≥ 查詢分數}) / (n_g + 1)。
    """
    pv = np.ones(len(q_scores))
    for g in range(n_groups):
        cs = np.sort(c_scores[c_groups == g])
        if len(cs) == 0:
            continue
        m = q_groups == g
        if not m.any():
            continue
        cnt = len(cs) - np.searchsorted(cs, q_scores[m], side="left")
        pv[m] = (1 + cnt) / (len(cs) + 1)
    return pv


def margin_pvals(q_b1, q_margin, cal_b1, cal_margin, n_units):
    """本系統標準用法的包裝：margin 越小越突兀 → 取負送入 group_pvals。"""
    return group_pvals(q_b1, -q_margin, cal_b1, -cal_margin, n_units)


# ---------------------------------------------------------------------------
# 校準分數庫（建置產物；dump 供人工檢視，線上物件逐位元一致）
# ---------------------------------------------------------------------------
def build_calibration_bank(cal_b1, cal_margin, units, id_tasks):
    """回傳可序列化的校準庫 dict（每單位：排序 margin + 分位摘要）。"""
    bank = {}
    for u in range(len(units)):
        m = cal_b1 == u
        arr = np.sort(cal_margin[m]).round(6)
        bank[str(u)] = {
            "tasks": [f"t{id_tasks[i]}" for i in units[u]],
            "n": int(arr.size),
            "margin_sorted": arr.tolist(),
            "quantiles": {q: round(float(np.quantile(arr, float(q))), 4)
                          for q in ("0.02", "0.10", "0.50", "0.90")}
            if arr.size else {},
        }
    return {"grouping": "predicted top-1 unit (Mondrian)",
            "score": "global margin (top1 - top2, unit level)",
            "units": bank}


# ---------------------------------------------------------------------------
# 四區分流
# ---------------------------------------------------------------------------
ZONE_FLOOR, ZONE_GREEN, ZONE_ESCALATE, ZONE_RED = 0, 1, 2, 3
ZONE_NAMES = {0: "direct", 1: "green", 2: "escalate", 3: "red"}


def zones(margin, pval, lex_disagree, thresholds):
    """四區分流；優先序：直判 > 紅 > 送審 > 綠（詞彙不一致把綠拉下送審）。

    參數
      margin        (n,)   全域 unit margin
      pval          (n,)   Mondrian 共形 p 值
      lex_disagree  (n,)   bool，詞彙 top-1 ≠ 嵌入 top-1
      thresholds    dict   需含 transfer_floor / p_lo / p_hi
    回傳 (n,) int zone 碼。
    """
    z = np.full(len(pval), ZONE_GREEN, dtype=int)
    z[(pval >= thresholds["p_lo"]) & (pval < thresholds["p_hi"])] = \
        ZONE_ESCALATE
    z[pval < thresholds["p_lo"]] = ZONE_RED
    z[(z == ZONE_GREEN) & lex_disagree] = ZONE_ESCALATE
    z[margin > thresholds["transfer_floor"]] = ZONE_FLOOR
    return z

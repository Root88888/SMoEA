# -*- coding: utf-8 -*-
"""
router/units.py

【路由單位】指紋 cosine 相似度超過門檻（預設 0.97）的任務以 union-find
遞移閉包併為一個路由單位（route unit）。攣生任務（如同一資料源拆出的
兩個近乎相同的任務）在嵌入空間無法區分，若各自為政會把 margin 分布拖出
貼地長尾、撐鬆 Mondrian 門檻；併成單位後 margin 在單位層計算，區分工作
交給單位內還原（unit 內 task 相似度 argmax）。

本模組同時服務兩種使用者：
  - router 本體：單位是 margin / 校準 / 裁決的作用層；
  - baselines：無單位概念，但計分需要「按同一張單位表投影」的 unit 級
    acc 以與 router 同尺——build_units 的輸出即該投影表。
（收編自 v3/v4 的 build_units 與兩支 baseline 各自複製的 union-find 段。）

【scale up】單位數是相似度結構的結果、非設定值；任務擴充自動適應。
"""

import numpy as np


def build_units(fingerprints, sim_threshold):
    """從任務指紋矩陣建路由單位。

    參數
      fingerprints  : (T, d) 各任務指紋（L2 歸一化），列序 = id_tasks 序
      sim_threshold : 相似度門檻，S[i,j] > threshold 即併組

    回傳
      units        : list[list[int]]，每單位的任務「索引」清單
                     （大單位在前、同大小按最小索引排序）
      unit_of      : (T,) 每任務索引所屬的單位編號
      merged_pairs : [(i, j, sim), ...] 觸發合併的配對（建置報告用）
    """
    T = fingerprints.shape[0]
    S = fingerprints @ fingerprints.T
    parent = list(range(T))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged_pairs = []
    for i in range(T):
        for j in range(i + 1, T):
            if S[i, j] > sim_threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
                merged_pairs.append((i, j, round(float(S[i, j]), 4)))
    groups = {}
    for i in range(T):
        groups.setdefault(find(i), []).append(i)
    units = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    unit_of = np.zeros(T, dtype=int)
    for u, g in enumerate(units):
        for i in g:
            unit_of[i] = u
    return units, unit_of, merged_pairs


def units_summary(units, id_tasks):
    """人類可讀摘要：多成員單位以任務名列出。"""
    multi = [[f"t{id_tasks[i]}" for i in g] for g in units if len(g) > 1]
    return {"n_units": len(units), "n_multi_member": len(multi),
            "multi_member_units": multi}

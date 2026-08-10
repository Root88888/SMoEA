# -*- coding: utf-8 -*-
"""
router/fingerprint.py

【任務指紋與多質心】只用「指紋堆」樣本（split 第一堆）建：
  - 任務指紋 = 指紋堆嵌入平均（L2）——路由單位建構用；
  - 多質心   = 指紋堆內 KMeans、k 以 silhouette 在 [2, k_max] 掃描重選，
    最佳 silhouette < min_silhouette 則不展開（k=1，用平均）。
    k 在指紋堆內重選（而非沿用全量資料的分群結果）是為了避免校準污染

【scale up】逐任務獨立；任務擴充自動適應。
"""

import numpy as np


def task_fingerprint(emb_fp):
    """指紋堆平均、L2 歸一化。"""
    m = emb_fp.mean(axis=0)
    return (m / np.linalg.norm(m)).astype(np.float32)


def select_centroids_inpile(emb_fp, k_max, min_sil, sil_sample, seed):
    """指紋堆內選 k 並回傳 (質心矩陣 (k,d) L2, k)。k=1 時質心=平均。"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n = emb_fp.shape[0]
    best_k, best_sil = 1, -1.0
    for k in range(2, min(k_max, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=3, random_state=seed)
        lab = km.fit_predict(emb_fp)
        if np.bincount(lab, minlength=k).min() < 2:
            continue
        try:
            s = silhouette_score(emb_fp, lab,
                                 sample_size=min(sil_sample, n),
                                 random_state=seed)
        except ValueError:
            continue
        if s > best_sil:
            best_k, best_sil = k, s
    if best_sil < min_sil:
        m = emb_fp.mean(axis=0, keepdims=True)
        return (m / np.linalg.norm(m, axis=1, keepdims=True)
                ).astype(np.float32), 1
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(emb_fp)
    c = km.cluster_centers_
    return (c / np.linalg.norm(c, axis=1, keepdims=True)
            ).astype(np.float32), best_k


def stack_centroids(centroids_by_task, id_tasks):
    """把逐任務質心疊成 (M, d) 矩陣＋擁有者索引 (M,)（相似度向量化用）。"""
    cols, owner = [], []
    for ti, t in enumerate(id_tasks):
        for c in centroids_by_task[t]:
            cols.append(c)
            owner.append(ti)
    return np.stack(cols), np.array(owner)


def task_sims(Q, C, owner, n_tasks):
    """query 對每任務的相似度 = 對該任務全部質心取 max。(n, T)。"""
    sims = Q @ C.T
    out = np.full((Q.shape[0], n_tasks), -np.inf, dtype=np.float32)
    for ti in range(n_tasks):
        cols = np.where(owner == ti)[0]
        out[:, ti] = sims[:, cols].max(axis=1)
    return out


def unit_sims(tS, units):
    """任務相似度 → 單位相似度（單位內取 max）。(n, U)。"""
    out = np.full((tS.shape[0], len(units)), -np.inf, dtype=np.float32)
    for u, g in enumerate(units):
        out[:, u] = tS[:, g].max(axis=1)
    return out

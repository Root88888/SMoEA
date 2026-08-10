# -*- coding: utf-8 -*-
"""
router/lexical.py

【詞彙訊號】TF-IDF 單位質心比對，做為嵌入訊號之外的第二票：
在指紋堆文本上 fit TF-IDF，逐路由單位取質心（L2）；查詢時回傳詞彙
top-1 單位。決策層用法：綠區樣本若「詞彙 top-1 ≠ 嵌入 top-1」則拉下
送審（用字露餡是嵌入看不見的警報，如 t1622 案例）——詞彙訊號只升級
不放行，不儲存分數、不設門檻。
（LexUnit 自 v4/v5 原樣收編；fit 只用指紋堆、對校準與測試為建置局外人，
 這是共形可交換前提的配套條件之一，不可改成全量 fit。）
"""

import numpy as np


class LexUnit:
    def __init__(self, fp_texts_by_task, id_tasks, units, max_features):
        """fp_texts_by_task: {task_id: [指紋堆文本]}；units 為任務索引分組。"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        docs, owner = [], []
        for ti, t in enumerate(id_tasks):
            for s in fp_texts_by_task[t]:
                docs.append(s)
                owner.append(ti)
        self.vec = TfidfVectorizer(max_features=max_features, min_df=2,
                                   norm="l2", sublinear_tf=True)
        X = self.vec.fit_transform(docs)
        owner = np.array(owner)
        cents = []
        for u, g in enumerate(units):
            rows = np.isin(owner, g)
            c = np.asarray(X[rows].mean(axis=0)).ravel()
            nrm = np.linalg.norm(c)
            cents.append(c / nrm if nrm > 0 else c)
        self.Ucent = np.stack(cents).astype(np.float32)

    def top1_unit(self, texts):
        """回傳每筆文本的詞彙 top-1 單位編號（(n,) int 陣列）。"""
        X = self.vec.transform(texts)
        sims = X @ self.Ucent.T
        return np.asarray(sims.argmax(axis=1)).ravel()

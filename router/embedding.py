# -*- coding: utf-8 -*-
"""
router/embedding.py

【嵌入層】sentence-transformers 封裝（預設 BAAI/bge-large-en-v1.5，
field=input）。模型延遲載入：ensure_task_embeddings 若快取已存在
（assets/emb_task{t}.npz / emb_test_task{t}.npz）直接跳過、完全不碰
模型——因此在已有快取的環境（含測試環境）零 GPU、零下載。

【scale up】逐任務快取、缺哪補哪；任務擴充只需重跑 build，既有快取
不重算。
"""

import os

import numpy as np

from . import data_io


class Embedder:
    """延遲載入的嵌入模型。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            name = self.cfg["embedding"]["model_name"]
            dev = self.cfg["embedding"].get("device", "cuda")
            try:
                self._model = SentenceTransformer(name, device=dev)
            except Exception:
                self._model = SentenceTransformer(name, device="cpu")
        return self._model

    def encode(self, texts):
        """回傳 L2 歸一化嵌入 (n, d) float32。"""
        model = self._load()
        e = model.encode(texts,
                         batch_size=self.cfg["embedding"]["batch_size"],
                         show_progress_bar=True,
                         convert_to_numpy=True,
                         normalize_embeddings=True)
        return e.astype(np.float32)


def ensure_task_embeddings(cfg, task_ids, test=False, embedder=None):
    """確保每個任務的嵌入快取存在；缺的現算補齊。回傳補算的任務清單。"""
    missing = [t for t in task_ids
               if not os.path.exists(data_io.emb_cache_path(cfg, t,
                                                            test=test))]
    if not missing:
        return []
    emb = embedder or Embedder(cfg)
    for t in missing:
        texts = data_io.load_task_texts(cfg, t, test=test)
        print(f"[embed] task{t} ({'test' if test else 'train'}) "
              f"{len(texts)} 筆")
        data_io.save_embeddings(cfg, t, emb.encode(texts), test=test)
    return missing

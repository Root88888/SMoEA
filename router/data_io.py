# -*- coding: utf-8 -*-
"""
router/data_io.py

【資料層】樣本檔讀取、路由文字準備、檔名解析、嵌入快取讀寫、OOD 標記檔解析。
來源：BM25 baseline（fork 定稿版）的 read_texts / find_file 原樣收編——
支援三種樣本檔格式（json array / dict 包裹 / jsonl）與多種檔名 stem，
其中 task{t}_train.json / task{t}_test.json 是 repo dataset 的標準命名。

【OOD 標記檔格式】（任務級、選用；由 top-3 去向人工覆核後手寫）
  {"results": {"9013": {"standard_answer": "reject"},
               "149":  {"standard_answer": "route_to_task9"}, ...}}
"""

import glob
import json
import os
import re

import numpy as np

from .config import routing_text_mode


# ---------------------------------------------------------------------------
# 樣本檔
# ---------------------------------------------------------------------------
def read_texts(path, field, clip_chars, strip_output=False):
    """讀樣本檔的 field 欄位。支援：json array / dict 包裹的樣本列表 / jsonl。"""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):               # 找 dict 裡的樣本列表
            for key in ("data", "samples", "instances", "examples"):
                if isinstance(obj.get(key), list):
                    obj = obj[key]
                    break
            else:
                lists = [v for v in obj.values() if isinstance(v, list)]
                if len(lists) == 1:
                    obj = lists[0]
                elif obj and all(isinstance(v, dict)
                                 for v in obj.values()):
                    obj = list(obj.values())    # dict-of-records
                else:
                    raise ValueError(
                        f"{path} 是 dict 但找不到唯一樣本列表，"
                        f"頂層鍵：{list(obj.keys())[:8]}")
        assert isinstance(obj, list), f"{path} 頂層非列表"
        records = obj
    except json.JSONDecodeError:                # jsonl
        records = [json.loads(l) for l in raw.splitlines() if l.strip()]
    texts = []
    for r in records:
        text = str(r.get(field, ""))
        if strip_output:
            output = str(r.get("output", ""))
            if not output or not text.endswith(output):
                raise ValueError(
                    f"{path}: 訓練 full_prompt 未以 output 結尾，"
                    "無法安全建立不含答案的 route prompt")
            text = text[:-len(output)]
        texts.append(text[:clip_chars])
    return texts


def find_file(dirpath, task_id, test=False):
    """找任務樣本檔；標準命名 task{t}_train.json / task{t}_test.json 優先。"""
    t = task_id
    stems = ([f"task{t}_test", f"test_task{t}", f"task{t}"] if test else
             [f"task{t}_train", f"train_task{t}", f"task{t}"])
    for stem in stems:
        for ext in (".jsonl", ".json"):
            p = os.path.join(dirpath, stem + ext)
            if os.path.exists(p):
                return p
    hits = sorted(glob.glob(os.path.join(dirpath, f"*task{t}[_.]*")))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"task{t} in {dirpath}（已試 {stems} × .jsonl/.json）")


def load_task_texts(cfg, task_id, test=False):
    """依設定讀一個任務的路由文字。

    answer_free_full_prompt 模式在 train 資料把尾端的標準 output 精確移除；
    test 的 full_prompt 本來就不含答案，因此原樣使用。
    """
    ds = cfg["paths"]["dataset_dir"]
    sub = "test_data" if test else "train_data"
    path = find_file(os.path.join(ds, sub), task_id, test=test)
    mode = routing_text_mode(cfg)
    if mode == "answer_free_full_prompt":
        return read_texts(
            path,
            "full_prompt",
            cfg["data"]["doc_clip_chars"],
            strip_output=not test,
        )
    return read_texts(path, mode, cfg["data"]["doc_clip_chars"])


# ---------------------------------------------------------------------------
# 嵌入快取（assets_dir 下 emb_task{t}.npz / emb_test_task{t}.npz，鍵 "emb"）
# ---------------------------------------------------------------------------
def emb_cache_path(cfg, task_id, test=False):
    stem = f"emb_test_task{task_id}" if test else f"emb_task{task_id}"
    return os.path.join(cfg["paths"]["assets_dir"], stem + ".npz")


def load_embeddings(cfg, task_id, test=False):
    """讀嵌入快取，原樣回傳（僅 astype float32）。

    注意：不可對回傳值再做 L2 normalize。快取已歸一化，重複歸一化
    的浮點擾動會改變 KMeans 分群結果，破壞質心結構的可重現性。"""
    z = np.load(emb_cache_path(cfg, task_id, test=test))
    return z["emb"].astype(np.float32)


def save_embeddings(cfg, task_id, emb, test=False):
    os.makedirs(cfg["paths"]["assets_dir"], exist_ok=True)
    np.savez_compressed(emb_cache_path(cfg, task_id, test=test),
                        emb=emb.astype(np.float32))


# ---------------------------------------------------------------------------
# OOD 任務級標記檔（選用）
# ---------------------------------------------------------------------------
def load_ood_groundtruth(path):
    """回傳 {task_id: -1(應拒) | target_task_id}；path 為 None 時回傳 None。"""
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        res = json.load(f)["results"]
    out = {}
    for k, v in res.items():
        ans = str(v["standard_answer"])
        if ans == "reject":
            out[int(k)] = -1
        else:
            out[int(k)] = int(re.sub(r"[^0-9]", "", ans))
    return out


def load_task_records(cfg, task_id, test=False):
    """讀一個任務的完整樣本紀錄（instances 原欄位：input / output /
    instance_id / full_prompt…）。生成執行層（system/）用它取
    full_prompt 當 prompt；路由層仍走 load_task_texts 準備路由文字。"""
    ds = cfg["paths"]["dataset_dir"]
    sub = "test_data" if test else "train_data"
    path = find_file(os.path.join(ds, sub), task_id, test=test)
    with open(path, encoding="utf-8") as f:
        obj = json.load(f) if not path.endswith(".jsonl") else \
            [json.loads(l) for l in f if l.strip()]
    if isinstance(obj, dict):
        for k in ("instances", "data", "examples", "samples"):
            if k in obj:
                return obj[k]
        raise ValueError(f"{path}: dict 無 instances 類鍵")
    return obj

# -*- coding: utf-8 -*-
"""
router/config.py

【設定載入】configs/*.yaml 是全部超參數與路徑的唯一定義處；本模組負責：
  1. load_config(path)：讀 yaml 成巢狀 dict，並套用指令列 --set a.b=c 覆蓋。
  2. discover_tasks(cfg)：掃 dataset 目錄自動推導任務集合——
       ID 任務   = train_data 有訓練檔的任務
       OOD 任務  = test_data 有測試檔、但 train_data 無訓練檔的任務
     任務數不寫死在任何地方；擴充任務只需把資料檔放進 dataset/ 即可。

【scale up】任務擴充後零改動；新任務由目錄掃描自動納入。
"""

import argparse
import copy
import os
import re

import yaml

_TASK_RE = re.compile(r"task(\d+)")


def load_config(path="configs/default.yaml", overrides=None):
    """讀 yaml 設定；overrides 為 ["a.b=c", ...] 形式的覆蓋清單。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)  # 自動轉型（數字/布林/null）
    return cfg


def add_config_args(parser=None):
    """給入口腳本共用的兩個參數：--config 與 --set。"""
    p = parser or argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="覆蓋單項設定，例：--set thresholds.p_lo=0.01")
    return p


def config_from_cli(extra_args_fn=None):
    """入口腳本標準流程：解參數 → 載設定。回傳 (cfg, args)。"""
    p = add_config_args()
    if extra_args_fn:
        extra_args_fn(p)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    return cfg, args


def _scan_task_ids(dirpath, suffix_hint):
    """掃目錄取任務 id 集合；接受 task{t}_train.json / task{t}_test.jsonl 等。"""
    ids = set()
    if not os.path.isdir(dirpath):
        return ids
    for fn in os.listdir(dirpath):
        if suffix_hint not in fn and f"task" not in fn:
            continue
        m = _TASK_RE.search(fn)
        if m and fn.endswith((".json", ".jsonl")):
            ids.add(int(m.group(1)))
    return ids


def discover_tasks(cfg):
    """回傳 (id_tasks, ood_tasks)，皆為排序後的 int list。

    定義：train_data 有檔 → ID；只在 test_data 有檔 → OOD。
    """
    ds = cfg["paths"]["dataset_dir"]
    train_ids = _scan_task_ids(os.path.join(ds, "train_data"), "train")
    test_ids = _scan_task_ids(os.path.join(ds, "test_data"), "test")
    id_tasks = sorted(train_ids)
    ood_tasks = sorted(test_ids - train_ids)
    if not id_tasks:
        raise FileNotFoundError(
            f"{ds}/train_data 找不到任何 task 訓練檔（task{{t}}_train.json）")
    return id_tasks, ood_tasks

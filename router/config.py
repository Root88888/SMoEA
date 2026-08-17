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


def _declared_ood_tasks(dataset_dir):
    marker = os.path.join(dataset_dir, "ood_tasks.txt")
    if not os.path.exists(marker):
        return set()
    declared = set()
    with open(marker, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                declared.add(int(line))
    return declared


def discover_serving_tasks(cfg):
    """Find tasks needed to build the online router from training data only.

    Unlike :func:`discover_tasks`, this path intentionally does not require
    benchmark test files. Tasks declared as OOD are excluded even if a train
    file is present, matching the full evaluation build.
    """
    ds = cfg["paths"]["dataset_dir"]
    train_ids = _scan_task_ids(os.path.join(ds, "train_data"), "train")
    if not train_ids:
        raise FileNotFoundError(
            f"{ds}/train_data 找不到任何 task 訓練檔（task{{t}}_train.json）")
    return sorted(train_ids - _declared_ood_tasks(ds))


def discover_tasks(cfg):
    """回傳 (id_tasks, ood_tasks)，皆為排序後的 int list。

    兩種模式：
    - dataset/ood_tasks.txt 存在（建議、repo 自帶）：**宣告即真相**
      ——列於其中者為 OOD，train 檔存在與否皆忽略（專案 convention：
      OOD 任務的訓練資料照常存放，僅身分由宣告決定）。仍檢查兩種
      真錯置：宣告 OOD 卻缺 test 檔；有 test 沒 train 卻未宣告。
    - 無標記檔：退回自動推導（train_data 有檔 → ID；只在
      test_data 有檔 → OOD）。
    """
    ds = cfg["paths"]["dataset_dir"]
    train_ids = _scan_task_ids(os.path.join(ds, "train_data"), "train")
    test_ids = _scan_task_ids(os.path.join(ds, "test_data"), "test")
    if not train_ids:
        raise FileNotFoundError(
            f"{ds}/train_data 找不到任何 task 訓練檔（task{{t}}_train.json）")

    marker = os.path.join(ds, "ood_tasks.txt")
    if os.path.exists(marker):
        declared = _declared_ood_tasks(ds)
        missing = sorted(declared - test_ids)
        if missing:
            raise ValueError(
                f"任務 {missing} 宣告為 OOD 但 test_data 缺其測試檔")
        undeclared = sorted(test_ids - train_ids - declared)
        if undeclared:
            raise ValueError(
                f"任務 {undeclared} 只有測試檔但未宣告於 {marker}——"
                f"若為 OOD 請補進該檔；若為 ID 請補其 train 檔")
        return sorted(train_ids - declared), sorted(declared)

    return sorted(train_ids), sorted(test_ids - train_ids)

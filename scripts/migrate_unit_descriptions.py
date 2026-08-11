#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/migrate_unit_descriptions.py

【單位說明書遷移】任務集合擴充後（如 50→150），單位表由 build 重新
聚類產生、編號整體重排；本腳本以「單位成員的任務集合」為鍵，把舊
說明書可沿用的描述搬進新結構，產出待人工補完的骨架：

  - 成員集合與舊單位完全相同 → 沿用舊描述（status: kept）
  - 含舊任務但成員有變（新任務併入/拆分）→ 附上舊描述供參考、
    標記需重寫（status: rewrite，description 置 TODO）
  - 全新任務組成 → 標記待生成（status: new，description 置 TODO）

【執行】build 完成後於 repo 根目錄：
  python scripts/migrate_unit_descriptions.py \\
      --old assets/unit_descriptions.json \\
      --out assets/unit_descriptions_150task_skeleton.json
校訂完成後將骨架檔改名/覆蓋為 assets/unit_descriptions.json 即生效
（記得移除所有 TODO 與 status/_old_description 輔助欄位可留可刪，
程式僅讀 descriptions[uid]["description"]）。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router import data_io  # noqa: E402
from router.config import load_config  # noqa: E402


def task_display_name(cfg, t):
    """讀任務檔的 task_name 欄（無則回 task{t}）。"""
    try:
        ds = cfg["paths"]["dataset_dir"]
        path = data_io.find_file(os.path.join(ds, "train_data"), t)
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and obj.get("task_name"):
            return str(obj["task_name"])
    except Exception:
        pass
    return f"task{t}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--old", default="assets/unit_descriptions.json")
    ap.add_argument("--out",
                    default="assets/unit_descriptions_skeleton.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    meta = json.load(open(os.path.join(cfg["paths"]["assets_dir"],
                                       "router_assets_meta.json")))
    id_tasks = meta["id_tasks"]
    new_units = [[id_tasks[i] for i in g] for g in meta["units"]]

    old = json.load(open(args.old, encoding="utf-8"))
    old_by_members = {}
    for uid, members in old.get("units", {}).items():
        d = old["descriptions"].get(uid, {})
        old_by_members[frozenset(int(t) for t in members)] = d

    out = {"note": f"{len(id_tasks)}-task 版骨架（migrate 產出，"
                   "TODO 待人工補完後啟用）",
           "units": {}, "descriptions": {}}
    n_kept = n_rewrite = n_new = 0
    for uid, members in enumerate(new_units):
        key = frozenset(members)
        entry = {"tasks": [f"t{t}" for t in members],
                 "task_names": [task_display_name(cfg, t) for t in members]}
        if key in old_by_members:
            entry["description"] = old_by_members[key]["description"]
            entry["status"] = "kept"
            n_kept += 1
        else:
            overlap = [m for m in old_by_members
                       if m & key]
            if overlap:
                entry["description"] = "TODO（成員變動，需重寫）"
                entry["status"] = "rewrite"
                entry["_old_description"] = [
                    old_by_members[m]["description"] for m in overlap]
                n_rewrite += 1
            else:
                entry["description"] = "TODO"
                entry["status"] = "new"
                n_new += 1
        out["units"][str(uid)] = members
        out["descriptions"][str(uid)] = entry

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[migrate] 新單位 {len(new_units)} 個："
          f"沿用 {n_kept}、需重寫 {n_rewrite}、待生成 {n_new}")
    print(f"[out] → {args.out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
router/metrics.py

【六指標共用計分】router、全部 baseline、未來任何新系統都用同一個
compute_metrics 計分，杜絕各腳本自帶計分邏輯漂移（收編自 router 主評測
與兩支 baseline 逐行等價的 evaluate 複製體）。

預測表示法（全系統統一）：pred 為 int 陣列，>= 0 是路由目標的
task 索引（id_tasks 序），< 0 是拒絕。

兩種模式（由 ood_gt 是否提供決定）：

A. 有 OOD 任務級標記（ood_gt = {task_id: -1(應拒) | target_task_id}）
   micro 七鍵（全樣本平均）：
     overall_acc      全體行為正確率（ID 以 unit 級計、OOD 以行為正確計）
     id_acc_task      ID task 級路由正確率
     id_acc_unit      ID unit 級路由正確率
     id_reject_rate   ID 誤拒率
     ood_acc          OOD 全體行為正確率
     ood_rej_acc      應拒 OOD 的拒絕率
     ood_route_acc    應路由 OOD 的正確路由率
   per_task_ood 每任務：行為正確率、拒絕率。

B. 無標記（ood_gt=None；150-task 制度的預設）
   【設計決策】overall 與三個 OOD acc「取消」而非標 null——
   micro 只含 id_acc_task / id_acc_unit / id_reject_rate 加上
   ood_reject_rate（純描述、全 OOD 樣本平均）。
   per_task_ood 每任務輸出行為描述：
     n / reject_rate / route_rate /
     top_routes: 前 k 個路由去向 [{task, share_of_routed}] /
     concentration: top-1 去向占被路由樣本的比例
   讀表方式：人工只需覆核 top_routes 是否合理近親（比對空間由全任務
   縮到 k）；覆核結論可寫成任務級標記檔重跑，即得 A 模式完整指標。

metadata 一律標明 mode 與各指標分母（計算域標註規範）。

【scale up】分母、任務清單、去向表全部資料驅動，零改動。
"""

import numpy as np


def _per_task_id(pred, y_task, unit_of, id_tasks):
    pu = np.where(pred >= 0, unit_of[np.clip(pred, 0, None)], -2)
    yu = unit_of[y_task]
    out = {}
    for ti, t in enumerate(id_tasks):
        m = y_task == ti
        out[str(t)] = {
            "n": int(m.sum()),
            "acc_task": round(float((pred[m] == ti).mean()), 4),
            "acc_unit": round(float((pu[m] == yu[m]).mean()), 4),
            "reject_rate": round(float((pred[m] < 0).mean()), 4)}
    return out, pu, yu


def _ood_behavior_row(pred, id_tasks, top_k):
    """單一 OOD 任務的行為描述（無標記模式的 per-task 列）。"""
    n = int(pred.size)
    routed = pred >= 0
    n_routed = int(routed.sum())
    row = {"n": n,
           "reject_rate": round(float((~routed).mean()), 4),
           "route_rate": round(float(routed.mean()), 4)}
    if n_routed:
        vals, cnt = np.unique(pred[routed], return_counts=True)
        order = np.argsort(-cnt)
        row["top_routes"] = [
            {"task": f"t{id_tasks[int(vals[i])]}",
             "share_of_routed": round(float(cnt[i] / n_routed), 4)}
            for i in order[:top_k]]
        row["concentration"] = row["top_routes"][0]["share_of_routed"]
    else:
        row["top_routes"] = []
        row["concentration"] = None
    return row


def compute_metrics(pred_task, y_task, unit_of, id_tasks,
                    ood_pred, ood_gt=None, ood_top_routes=3):
    """全系統共用計分。

    參數
      pred_task     (n_id,)  ID 側預測（task 索引；<0 = 拒絕）
      y_task        (n_id,)  ID 側標籤（task 索引）
      unit_of       (T,)     任務索引 → 單位編號投影表
      id_tasks      list     任務索引 → 任務 id（報表命名用）
      ood_pred      {ood_task_id: (n_o,) 預測陣列}
      ood_gt        {ood_task_id: -1 | target_task_id} 或 None
      ood_top_routes 無標記模式 per-task 報告的去向數

    回傳 {"micro": ..., "per_task_id": ..., "per_task_ood": ...,
          "metadata": ...}
    """
    per_id, pu, yu = _per_task_id(pred_task, y_task, unit_of, id_tasks)
    n_id = int(y_task.size)
    id_task_ok = int((pred_task == y_task).sum())
    id_unit_ok = int((pu == yu).sum())
    id_rej = float((pred_task < 0).mean())
    t_index = {t: i for i, t in enumerate(id_tasks)}

    # ---------------- A. 有任務級標記：完整六指標 ----------------
    if ood_gt is not None:
        rej_ok = rej_n = rt_ok = rt_n = 0
        per_ood = {}
        for o, pred in ood_pred.items():
            g = ood_gt[o]
            rej = float((pred < 0).mean())
            if g == -1:
                ok = int((pred < 0).sum())
                rej_ok += ok
                rej_n += pred.size
            else:
                ok = int((pred == t_index[g]).sum())
                rt_ok += ok
                rt_n += pred.size
            per_ood[str(o)] = {
                "n": int(pred.size),
                "gt": "reject" if g == -1 else f"t{g}",
                "correct_rate": round(ok / pred.size, 4),
                "reject_rate": round(rej, 4)}
        n_ood = rej_n + rt_n
        micro = {
            "overall_acc": round((id_unit_ok + rej_ok + rt_ok)
                                 / (n_id + n_ood), 4),
            "id_acc_task": round(id_task_ok / n_id, 4),
            "id_acc_unit": round(id_unit_ok / n_id, 4),
            "id_reject_rate": round(id_rej, 4),
            "ood_acc": round((rej_ok + rt_ok) / n_ood, 4),
            "ood_rej_acc": round(rej_ok / rej_n, 4) if rej_n else None,
            "ood_route_acc": round(rt_ok / rt_n, 4) if rt_n else None}
        meta = {"mode": "with_ood_gt",
                "note": "all micro metrics are sample-averaged",
                "denominators": {"overall": n_id + n_ood, "id": n_id,
                                 "ood": n_ood, "ood_rej": rej_n,
                                 "ood_route": rt_n}}
        return {"micro": micro, "per_task_id": per_id,
                "per_task_ood": per_ood, "metadata": meta}

    # ---------------- B. 無標記：ID acc + OOD 行為描述 ----------------
    per_ood = {}
    ood_rej_num = ood_n = 0
    for o, pred in ood_pred.items():
        per_ood[str(o)] = _ood_behavior_row(pred, id_tasks, ood_top_routes)
        ood_rej_num += int((pred < 0).sum())
        ood_n += int(pred.size)
    micro = {
        "id_acc_task": round(id_task_ok / n_id, 4),
        "id_acc_unit": round(id_unit_ok / n_id, 4),
        "id_reject_rate": round(id_rej, 4),
        "ood_reject_rate": (round(ood_rej_num / ood_n, 4)
                            if ood_n else None)}
    meta = {"mode": "no_ood_gt",
            "note": ("no OOD ground truth provided; overall and OOD "
                     "accuracy metrics are omitted (not null) by design. "
                     "per_task_ood reports behavior only; review "
                     "top_routes to author a task-level gt file, then "
                     "rerun with evaluation.ood_groundtruth set."),
            "denominators": {"id": n_id, "ood": ood_n}}
    return {"micro": micro, "per_task_id": per_id,
            "per_task_ood": per_ood, "metadata": meta}

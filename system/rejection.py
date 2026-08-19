# -*- coding: utf-8 -*-
"""
system/rejection.py — Rejection Runtime 的唯一接入點

Router 拒絕一筆請求後，互動模式與批次模式都經由本模組作答，不各自呼叫
InferenceEngine——「Rejection Runtime 只有一條路徑」是本專案的不變式
（見 CONTEXT.md）。

實際使用哪一種方法由 system.rejection_method 在啟動時明確指定並驗證：
base、merged artifact、Direct Arrow 或 Taskwise-K16 Arrow。本模組不掃描
runs、不自動挑選最新 run，也不在 query 時執行 merging 或 training。
"""


def run_rejection(engine, prompts, batch_size=1):
    """啟用選定的 rejection method 並作答。

    engine     system.inference.InferenceEngine
    prompts    完整且不含本題答案的請求全文清單
    batch_size 生成批次大小（互動模式為 1）

    回傳 (outputs, identity)：outputs 與 prompts 等長且同序；identity 為
    本次採用方法的身分（method / condition_id / run_id / format），需寫入
    批次輸出並在互動模式顯示，確保每筆答案都可追溯到來源 artifact。
    """
    identity = engine.ensure_rejection()
    step = max(1, int(batch_size))
    outputs = []
    for start in range(0, len(prompts), step):
        outputs.extend(engine.generate(prompts[start:start + step]))
    return outputs, identity

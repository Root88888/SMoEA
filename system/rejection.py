# -*- coding: utf-8 -*-
"""
system/rejection.py — 拒絕分支（Model Merging）的接入點

主程式啟動時把 base model 載進 GPU、包成一個物件叫 engine；每次有 query 被拒絕，
主程式就呼叫 handle_rejection(query, engine)，並把回傳的字串當作最終輸出
（互動模式直接印出；批次模式寫入 results/main_batch_outputs.jsonl）。
實作本函式即完成接入，不需改動其他任何檔案。

輸入  query: str   查詢原文
輸出  str          模型輸出文本


engine 內部是 base model＋可切換的 task／merged updates 和 tokenizer。
merged model 在啟動時由 system.merged_model_dir 明確指定及驗證；本模組不自行
挑選方法、掃描 runs 或即時合併。

engine（system.inference.InferenceEngine，base model 已載妥）：
  engine.ensure_merged()
      停用目前 task adapter，啟用啟動時已驗證的 merged model。
  engine.generate([prompt, ...]) -> [output, ...]
      使用完整且不含本題答案的 prompt 生成。

adapter 權重檔請放在 adapter/{task_key}/（目錄內直接放 checkpoint，
多個 checkpoint-*/ 自動取最新）。

測試：python main.py --mode interactive 輸入應拒絕 OOD query 即觸發本函式；
批次 python main.py --mode batch 後拒絕樣本的輸出在
results/main_batch_outputs.jsonl。
"""


def handle_rejection(query: str, engine) -> str:
    """使用啟動時選定的 Merged Inference Artifact 回答完整題目。"""

    engine.ensure_merged()
    return engine.generate([query])[0]

# -*- coding: utf-8 -*-
"""
system/rejection.py — 拒絕分支（Adapter Merging）的接入點

Router 拒絕的每一筆 query，主程式都會呼叫本檔的
handle_rejection(query, engine)，並把回傳的字串當作最終輸出
（互動模式直接印出；批次模式寫入 results/main_batch_outputs.jsonl）。
實作本函式即完成接入，不需改動其他任何檔案。

輸入  query: str   查詢原文
輸出  str          模型輸出文本

engine（system.inference.InferenceEngine，base model 已載妥）：
  engine.load_adapters_merged({"task23": 0.6, "task10": 0.4})
      依 {task_key: weight} 線性合成各任務 adapter 並切換生效；
      需要其他合成方式可自行擴充 InferenceEngine 或直接操作
      engine.model（PeftModel）。
  engine.ensure_adapter("task23")
      切換至單一任務 adapter。
  engine.generate([prompt, ...]) -> [output, ...]
      生成（解碼設定與 ID 路徑一致，見 system/inference.py）。

adapter 權重檔請放在 adapter/{task_key}/（目錄內直接放 checkpoint，
多個 checkpoint-*/ 自動取最新）。

測試：python main.py --mode interactive 輸入域外 query 即觸發本函式；
批次 python main.py --mode batch 後拒絕樣本的輸出在
results/main_batch_outputs.jsonl。環境照 README，
scripts/check_env.py 全 PASS 後開工。
"""


def handle_rejection(query: str, engine) -> str:
    raise NotImplementedError("Adapter Merging 分支尚未實作。")

# -*- coding: utf-8 -*-
"""
system/rejection.py — 拒絕分支（Adapter Merging）的接入點

主程式啟動時把 base model 載進 GPU、包成一個物件叫 engine；每次有 query 被拒絕，
主程式就呼叫 handle_rejection(query, engine)，並把回傳的字串當作最終輸出
（互動模式直接印出；批次模式寫入 results/main_batch_outputs.jsonl）。
實作本函式即完成接入，不需改動其他任何檔案。

輸入  query: str   查詢原文
輸出  str          模型輸出文本


engine 內部是 base model＋可掛卸的 adapter 們和 tokenizer，已實作以下三個現成方法

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

測試：python main.py --mode interactive 輸入應拒絕 OOD query 即觸發本函式；
批次 python main.py --mode batch 後拒絕樣本的輸出在
results/main_batch_outputs.jsonl。
"""


def handle_rejection(query: str, engine) -> str:
    raise NotImplementedError("Adapter Merging 分支尚未實作。")




"""
使用範例，以下可刪
1. 什麼都不做，用 base model
def handle_rejection(query, engine):
    return engine.generate([query])[0]

2. 固定比例合成
def handle_rejection(query, engine):
    engine.load_adapters_merged({"task23": 0.5, "task10": 0.5})
    return engine.generate([query])[0]

3. 其他演算法決定線性比例
def handle_rejection(query, engine):
    weights = my_merging_algorithm(query) # 例如 weights = {"task23": 0.6, "task10": 0.4}
    engine.load_adapters_merged(weights)
    return engine.generate([query])[0]

4. 在這裡寫自己的函式
def handle_rejection(query, engine):
    my_method(query, engine)         # 呼叫自己的函式
    return engine.generate([query])[0]

5. 先在 system.inference.InferenceEngine 實作新方法
def handle_rejection(query, engine):
    engine.new_method()
    return engine.generate([query])[0]

6. 直接動模型
def handle_rejection(query, engine):
    model = engine.model            # 標準 PeftModel，adapter 檔在 adapter/task{N}/
    ...                             # 合成術：讀權重檔、做任何數學、改 model
    return engine.generate([query])[0]
"""

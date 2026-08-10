# -*- coding: utf-8 -*-
"""
system/rejection.py

【拒絕分支（Adapter Merging）——本檔為此分支的唯一接入點】

■ 主程式如何呼叫
  main.py 對 Router 判定拒絕的每一筆 query 呼叫本檔的
  handle_rejection(query, diagnosis, engine)：
  - 互動模式：拒絕當下即呼叫，回傳值直接印為 [Output]；
  - 批次模式：拒絕樣本逐筆呼叫，回傳值寫入
    results/main_batch_outputs.jsonl 的 output 欄（目前未實作，
    該欄為 null）。
  本函式實作完成後不需改動 main.py 或其他任何檔案。

■ 輸入
  query: str
      查詢原文（同時就是生成用 prompt；批次模式傳入的是樣本的
      full_prompt）。
  diagnosis: dict
      Router 對該筆的判定素材：
        zone        "red"（紅區拒絕）| "esc_rej"（送審後拒絕）
        top_units   [int, int, int]        全域 top-3 路由單位編號
        top_tasks   ["task23", ...]        各單位內最相似任務（同長度）
        top_sims    [float, ...]           對應的嵌入相似度
        margin      float                  top1−top2 單位相似度差
        pval        float                  Mondrian 共形 p 值
        esc_p_yes   [float, float, float]  送審三候選的 LLM 裁決分數，
                    | None                 順序對應 top_tasks；紅區為 None
  engine: system.inference.InferenceEngine
      已載妥 base model（unsloth/Meta-Llama-3.1-8B）的生成引擎。

■ 輸出
  str——模型輸出文本，主程式原樣使用。

■ engine 提供的方法
  engine.load_adapters_merged(weights: dict[str, float]) -> str
      以 {task_key: weight} 對各任務 adapter 做線性合成
      （PEFT add_weighted_adapter, combination_type="linear"）並切換
      生效；成員 adapter 未載入者自動從 adapter/{task_key}/ 載入。
      需要其他合成方式時可自行擴充 InferenceEngine 或直接操作
      engine.model（PeftModel）。
  engine.ensure_adapter(task_key: str)
      切換至單一任務 adapter。
  engine.generate(prompts: list[str]) -> list[str]
      生成（貪婪解碼、max_new_tokens=512、stop_strings 等設定與
      ID 路徑一致，見 system/inference.py）。
  adapter 檔案位置與解析規則：adapter/{task_key}/ 直含 adapter 檔，
  或多個 checkpoint-*/ 取最新（system.inference.resolve_adapter_path）。

■ 執行與驗收
  互動：python main.py --mode interactive，輸入域外 query 即觸發本函式。
  批次：python main.py --mode batch [--tasks ...]，拒絕樣本的輸出
        （含完整 diagnosis）落在 results/main_batch_outputs.jsonl，
        可直接接下游評分。
  環境：照 README 環境建置節；scripts/check_env.py 全 PASS 後開工。
"""


def handle_rejection(query: str, diagnosis: dict, engine) -> str:
    raise NotImplementedError("Adapter Merging 分支尚未實作。")

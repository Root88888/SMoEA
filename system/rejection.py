# -*- coding: utf-8 -*-
"""
system/rejection.py

【拒絕分支：Adapter Merging → Inference】——本檔是拒絕分支負責人的
實作起點。架構圖中 Router 判「Reject」的查詢會進到 handle_rejection，
目標是：從 router 診斷資訊決定「合成哪些任務的 adapter、各配多少
權重」，合成後生成輸出。

■ 已備妥、不需要重做的部分
  - 合成與載入機制：InferenceEngine.load_adapters_merged(weights)
    （PEFT add_weighted_adapter 線性合成＋切換生效），合完直接
    engine.generate([query]) 即得輸出——不需要碰 PEFT/模型細節。
  - 診斷素材：diagnosis dict（見下）包含決定權重所需的全部訊號。

■ 待實作的部分（本函式）
  由 diagnosis 決定 weights，例如（僅供發想，非指定作法）：
  - 以 top-3 單位相似度 softmax 當權重；
  - 送審樣本可再利用三候選的 p_yes 加權；
  - 或先做實驗決定固定 top-k 與溫度係數。

■ diagnosis 內容
  {
    "zone":       "red" | "esc_rej"        # 哪種拒絕
    "top_units":  [u1, u2, u3],            # 全域 top-3 路由單位
    "top_tasks":  ["task23", ...],         # 各單位內最相似任務（同長度）
    "top_sims":   [0.71, 0.66, 0.60],      # 對應單位相似度
    "margin":     float,                   # top1-top2 單位相似度差
    "pval":       float,                   # Mondrian 共形 p 值
    "esc_p_yes":  [0.31, 0.22, 0.08] | None  # 送審三候選的裁決分數
                                             # （紅區拒絕為 None）
  }

■ 驗收建議
  批次模式（main.py --mode batch）對 OOD 測試檔會把拒絕樣本逐筆
  送進本函式，輸出 jsonl 可直接接 LLM judge 評分，與「純 base
  model」「最相似單一 adapter」兩條 baseline 比較。
"""


def handle_rejection(query: str, diagnosis: dict, engine) -> str:
    """拒絕分支入口。回傳模型輸出字串。

    參數
    ----
    query:     使用者查詢原文（也是生成 prompt）
    diagnosis: 見檔頭說明
    engine:    system.inference.InferenceEngine（base model 已載）

    範例骨架（決定 weights 後只需兩行）：
        weights = {...task_key -> float...}   # ← 本函式的核心工作
        engine.load_adapters_merged(weights)
        return engine.generate([query])[0]
    """
    raise NotImplementedError(
        "Adapter Merging 分支尚未實作——由拒絕分支負責人從本檔繼續。"
        "合成/載入/生成機制已備妥（見檔頭說明），只需實作權重決策。")

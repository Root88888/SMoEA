# SMoEA Router

Conformal task routing with selective rejection for a scalable
mixture-of-experts adapter system（SMoEA 的任務路由器：嵌入相似度
margin ＋ Mondrian 共形校準 ＋ 詞彙一致性檢查 ＋ LLM 送審裁決，
對已知任務路由至對應 adapter、對域外查詢選擇性拒絕）。

## 系統流程（單筆查詢）

1. 嵌入查詢，對每個「路由單位」算相似度（多質心取 max），
   取全域 unit margin = top1 − top2 相似度。
2. **直判**：margin > `transfer_floor` → 直接路由 top-1 單位。
3. 否則算 Mondrian 共形 p 值（按預測 top-1 單位分組，對照該單位
   校準樣本的 margin 排名）：
   - p < `p_lo` → **紅區拒絕**（可交換前提下誤拒率 ≤ p_lo，
     有限樣本精確保證）；
   - p ∈ [`p_lo`, `p_hi`)，或 p ≥ `p_hi` 但 TF-IDF 詞彙 top-1 與
     嵌入 top-1 不一致 → **送審**；
   - 其餘 → **綠區路由**。
4. **送審裁決**：對全域 top-3 候選單位各問一題 LLM 是非題
   （單位說明書＋示例＋查詢），max p_yes ≥ `theta_verify` 才路由、
   否則拒絕。
5. 路由單位 → 具體任務：單位內任務相似度 argmax 還原。

路由單位（route unit）：訓練指紋 cosine > 0.97 的攣生任務以
union-find 併組——嵌入空間無法區分的任務不強行區分，margin 與
校準都在單位層進行。

## 目錄

```
configs/default.yaml        全部超參數（任務集合由 dataset 掃描自動推導）
router/                     核心套件（config/data_io/embedding/fingerprint/
                            units/lexical/conformal/verifier/core/metrics）
scripts/
  build_router_assets.py    離線建置一條龍（嵌入→指紋→單位→校準庫→落地）
  eval_router.py            主評測三段（decide → score → run）
  eval_baseline_mean_embedding.py   Baseline: Pure Embedding
  eval_baseline_bm25_voting.py      Baseline: BM25 Voting
  verify_flow_table.py      六流向表獨立重放驗證
  plot_centroids.py         多質心結構圖
  selftest_core_modules.py  底層單元自測（無資料）
  selftest_end_to_end.py    端到端管線自測（合成資料、零 GPU）
dataset/train_data/task{t}_train.json
dataset/test_data/task{t}_test.json     （ID 與 OOD 測試檔同放；
                                          只有測試檔、無訓練檔的任務即 OOD）
assets/                     建置產物＋ unit_descriptions.json（單位說明書）
adapter/                    LoRA adapters（router 不讀，供下游載入）
results/                    評測輸出
```

## Reproduce（Linux、Python ≥ 3.10）

```bash
pip install -r requirements.txt

# 0) 環境驗收（零資料、零 GPU；全部 PASS 代表依賴與管線就緒）
python scripts/selftest_core_modules.py
python scripts/selftest_end_to_end.py

# 1) 放資料：dataset/train_data/、dataset/test_data/（格式見下）
#    放單位說明書：assets/unit_descriptions.json

# 2) 離線建置（首次需 GPU＋網路下載嵌入模型；嵌入快取齊備後純 CPU）
python scripts/build_router_assets.py 2>&1 | tee results/build_log.txt

# 3) 主評測三段（score 段需 GPU 與裁決 LLM）
python scripts/eval_router.py --mode decide 2>&1 | tee results/eval_decide_log.txt
python scripts/eval_router.py --mode score  2>&1 | tee results/eval_score_log.txt
python scripts/eval_router.py --mode run    2>&1 | tee results/eval_run_log.txt

# 4) 獨立驗證（雙路徑逐格對帳，PASS 即數字可信）
python scripts/verify_flow_table.py 2>&1 | tee results/verify_flow_log.txt

# 5) Baselines（純 CPU；先 score_dist 看分布、sweep 掃門檻、eval 定稿）
python scripts/eval_baseline_mean_embedding.py --mode score_dist
python scripts/eval_baseline_mean_embedding.py --mode sweep
python scripts/eval_baseline_mean_embedding.py --mode eval --tau 0.72
python scripts/eval_baseline_bm25_voting.py --mode score_dist
python scripts/eval_baseline_bm25_voting.py --mode sweep
python scripts/eval_baseline_bm25_voting.py --mode eval --ratio_tau 0.5
```

任何設定可由指令列覆蓋：`--set thresholds.p_lo=0.02`。

## 指標（全樣本平均 micro；分母寫在輸出 metadata）

有 OOD 任務級標記（`evaluation.ood_groundtruth` 設路徑）時輸出七鍵：
`overall_acc`（全體行為正確率：ID 以單位級計、OOD 以行為正確計）、
`id_acc_task`、`id_acc_unit`、`id_reject_rate`、`ood_acc`、
`ood_rej_acc`（應拒 OOD 拒絕率）、`ood_route_acc`（應路由 OOD 正確
路由率）。

無標記（預設 `null`）時 overall 與 OOD acc **取消**（不留空欄）：
輸出 ID 三鍵＋`ood_reject_rate`，且逐 OOD 任務給行為描述——拒絕率、
top-k 路由去向與占比、集中度。人工只需覆核 top 去向是否合理近親；
覆核結論寫成標記檔（格式如下）重跑即得完整指標：

```json
{"results": {"9013": {"standard_answer": "reject"},
             "149":  {"standard_answer": "route_to_task9"}}}
```

## 資料格式

樣本檔支援三種：JSON array（`[{"input": ...}, ...]`）、dict 包裹
（`{"instances": [...]}` 等）、JSONL。查詢文本取 `data.field`
（預設 `input`）。

## 統計保證的適用範圍

紅區誤拒率 ≤ `p_lo` 是有限樣本精確定理，前提是「查詢與其 top-1
單位的校準樣本可交換」——評測內由建置程序構造性成立（內容盲隨機
切分＋校準/測試對打分函數同為建置局外人）；上線流量不在此構造內，
對應做法是監測紅區占比並定期以新流量重建校準庫。

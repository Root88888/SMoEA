# SMoEA — Scalable Mixture-of-Experts Adapters

![architecture](docs/architecture.png)

## 目錄結構

```
main.py                     系統入口：interactive / batch 兩模式
configs/default.yaml        全部設定唯一定義處（路徑、門檻、模型、生成參數；
                            任何設定可用 --set key=value 臨時覆蓋）
router/                     路由決策層（圖中 Router 方塊）
  core.py                   Router 類別：build / save / load / decide /
                            escalate / finalize——路由邏輯唯一所在
  config.py data_io.py      設定載入、資料與嵌入快取 I/O
  embedding.py              查詢嵌入（bge）
  fingerprint.py units.py   任務指紋、多質心、路由單位
  lexical.py conformal.py   詞彙一致性訊號、共形校準與四區判定
  verifier.py               送審 LLM 是非題裁決
  metrics.py                評測計分（指標定義見檔頭）
system/                     路由之後的執行層
  inference.py              InferenceEngine：base model 常駐、
                            per-task adapter 熱切換、生成
  rejection.py              拒絕分支（Model Merging）介面
scripts/
  check_env.py              環境體檢（任何異常先跑這支）
  selftest_*.py             三支自測（零資料零 GPU）
  build_router_assets.py    路由資產離線建置（掃 dataset 自動推導任務集合）
  eval_router.py            路由評測三段（decide → score → run）
  eval_baseline_*.py        兩支 baseline
  verify_flow_table.py      評測結果獨立重放驗證
  plot_centroids.py         質心結構圖
dataset/                    任務樣本（不進 git）
  train_data/task{N}_train.json
  test_data/task{N}_test.json     ID 與 OOD 測試檔同放；只有測試檔、
                                  沒有訓練檔的任務自動視為 OOD
adapter/task{N}/            LoRA adapters（不進 git）：目錄內直接放
                            adapter 檔，或多個 checkpoint-*/ 自動取最新
assets/                     路由建置產物；unit_descriptions.json 為
                            人工校訂的單位說明書（唯一進 git 的資產）
results/                    評測與批次輸出（不進 git）
docs/                       架構圖與文件
```

## 環境建置

```bash
# 通用環境
pip install -r requirements.txt
# TWCC（揮發容器）：用鎖定檔建自包含 conda env，流程見部署指南
pip install -r requirements-lock-twcc.txt

python scripts/check_env.py            # 環境體檢（PASS 才開工）
python scripts/selftest_core_modules.py
python scripts/selftest_end_to_end.py
python scripts/selftest_main_pipeline.py
```

## 使用

```bash
# 資料與 adapter 就位後，先建路由資產（任務集合自動掃描）：
python scripts/build_router_assets.py

# 互動：單筆 query 跑完整流程，逐步顯示 Router 判定
python main.py --mode interactive

# 批次：跑 dataset 測試檔（--tasks 3,7 限任務、--limit 50 試跑）
#       輸出 results/main_batch_outputs.jsonl（含逐筆診斷）
python main.py --mode batch
```

互動模式輸出範例：

```
> <query>
[Router] margin=0.183  p=0.42  詞彙一致✓
[Router] top-3：task23(sim 0.87)  task10(sim 0.71)  task24(sim 0.66)
[Router] 判定：綠區路由 → task23
[Output] <模型輸出>
```

## 從哪裡下手

- **拒絕分支（Model Merging）**：入口 `system/rejection.py`——
  介面、可用的診斷素材、與已備妥的合成/載入/生成機制
  （`InferenceEngine.load_adapters_merged`）全寫在該檔檔頭；
  只需實作「用哪些 adapter、各配多少權重」的決策。
- **生成行為**（prompt、解碼參數、adapter 解析）：`system/inference.py`。
- **路由邏輯**（分數、校準、判定規則）：`router/core.py` 起，
  各訊號在 router/ 內各自的模組。
- **評測**：`scripts/eval_router.py` 檔頭有指標定義與執行方式；
  改動路由後用它加 `scripts/verify_flow_table.py` 驗證。
- **新增任務**：樣本放 `dataset/`、adapter 放 `adapter/task{N}/`、
  重跑 `build_router_assets.py` 即完成擴充（送審裁決另需在
  `assets/unit_descriptions.json` 補該任務所屬單位的說明）。

## 資料格式

樣本檔 `{"task_key", "task_name", "definition", "instances": [...]}`，
每筆 instance 含 `input`（路由用）、`full_prompt`（生成 prompt）、
`output`、`instance_id`；亦相容純 array 與 JSONL。

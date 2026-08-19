# SMoEA — Scalable Mixture-of-Experts Adapters

![architecture](docs/architecture.jpg)

完整的 MoEA-Trainer → merged artifact → SMoEA 運作方式、公司操作指令與目前驗收
範圍，見 [`docs/DELIVERY_ARCHITECTURE_RUNBOOK.md`](docs/DELIVERY_ARCHITECTURE_RUNBOOK.md)。

## 目錄結構

```
main.py                     系統入口：interactive / batch 兩模式
configs/default.yaml        全部設定唯一定義處（路徑、門檻、模型、生成參數；任何設定可用 --set key=value 臨時覆蓋）
router/                     路由決策層
  core.py                   Router 類別：build / save / load / decide / escalate / finalize——路由邏輯唯一所在
  config.py data_io.py      設定載入、資料與嵌入快取 I/O
  embedding.py              查詢嵌入（bge）
  fingerprint.py units.py   任務指紋、多質心、路由單位
  lexical.py conformal.py   詞彙一致性訊號、共形校準與四區判定
  verifier.py               送審 LLM 是非題裁決
  metrics.py                評測計分
system/                     路由之後的執行層
  inference.py              InferenceEngine：base model 常駐、per-task adapter 熱切換、生成
  merged_model.py           delivery artifact 驗證、exact dense delta 非破壞切換
  rejection.py              拒絕分支：使用啟動時明確選定的 merged model
scripts/
  check_env.py              環境體檢
  selftest_*.py             三支自測（零資料零 GPU）
  build_router_assets.py    路由資產離線建置（掃 dataset 自動推導任務集合）
  eval_router.py            路由評測三段
  eval_baseline_*.py        兩支 baseline
  verify_flow_table.py      評測結果獨立重放驗證
  plot_centroids.py         質心結構圖
dataset/                    資料（不進 git）
  train_data/task{N}_train.json    線上 router 建置需要
  test_data/task{N}_test.json      批次評測才需要
  ood_test_data/task149_test.json  批次評測用的原始 OOD task149
adapter/task{N}/            LoRA adapters（不進 git）：請建立 adapter 目錄，將 task{N} 直接放在 adapter/ 下，task 內如有多個 checkpoint-*/ 自動取最新
<external>/merged_model/    MoEA-Trainer delivery 產生的 selected merging artifact（不進 git）
assets/                     路由建置產物；unit_descriptions.json 為人工校訂的單位說明書
results/                    評測與批次輸出
docs/                       架構圖與文件
```

## 上手流程

1. git clone
```bash
   git clone https://github.com/Root88888/SMoEA.git && cd SMoEA
```
 
2. 放置檔案（手動步驟）
   - Router 訓練樣本：放到 `dataset/train_data/`，檔名
     `task{N}_train.json`。互動與線上服務建置只需要這一份資料
   - 批次評測資料：要跑 benchmark 時才放到 `dataset/test_data/`，檔名
     `task{N}_test.json`
   - adapters：每任務一個目錄，放成 `adapter/task{N}/`；解壓後若外層
     多包一層目錄，將其中的 `task*` 移出攤平
   - OOD Natural Instructions `task149`：保留原始檔名，放到
     `dataset/ood_test_data/task149_test.json`；source Adapter Slot `task149`
     的測試檔仍可留在 `dataset/test_data/task149_test.json`
   - selected merge：將 MoEA-Trainer delivery 的完整
     `prepare/merged_model/` 放在共享儲存；不要只複製 weight file

要跑完整 benchmark 時，先執行
`python scripts/map_ood_aliases.py --dataset-dir dataset`，為 OOD `task149`
建立內部 `task9149` symlink，不需手動改名。Batch output 的 `source_task`
仍是原始 `task149`，並另以 `internal_task_id: task9149` 保留除錯資訊。

3. 一鍵建置
   
   已在 dataset/ood_tasks.txt 定義 OOD 任務有哪些，如果編號方式不同請修改

```bash
   bash scripts/setup_workspace.sh
```

   setup 預設建立互動／線上服務需要的 router 資產，不要求 benchmark
   test data。它會自動完成：檔案檢查（缺漏會明確提示）、conda 環境建置與依賴安裝
   （首次 10-20 分鐘）、環境體檢、查詢嵌入計算與路由資產建置
   （首次 GPU 數分鐘）。結尾印出「全部就緒」即完成；中途停止時
   依提示處理後重跑即可（已完成步驟自動跳過）。
   
4. 單筆執行互動
```bash
   conda activate smoea
   python main.py --mode interactive
```

   首次執行自動下載生成與裁決模型（各約 16GB）。請輸入完整任務要求與內容
   （相當於不含答案的 `full_prompt`）。多行內容先輸入 `:paste`，貼完後以
   單獨一行 `:send` 送出。應看到 Router 判定與模型回答；
   輸入無關文字應看到進入
   selected merged model 的回答。Production 啟動請明確指定 artifact：

```bash
   python main.py --mode interactive \
     --set system.merged_model_dir=/shared/run/prepare/merged_model \
     --set system.merged_model_required=true \
     --set system.dtype=bfloat16
```

`merged_model_dir: null` 僅供不觸發拒絕分支的開發／router 測試。Production
設為 required 後，路徑未設定、schema 不符、base model 不同、檔案大小或 checksum
錯誤都會在接受輸入前停止，不會改用 base model。
 
5. 批次評測

   批次模式另外需要完整 `dataset/test_data/`。若資料尚未備齊，互動模式
   仍可正常使用，但不要把部分 test data 的結果當成完整 benchmark。

```bash
   python main.py --mode batch          # --tasks a,b 限任務、--limit n 每個任務test set只取前n筆
```
 
   路由命中使用 task adapter；router 拒絕則按 batch 集中使用 selected merged
   model，不再寫 `output=null`。逐筆結果（含 Router 診斷、實際 model source、
   condition/run identity 與模型輸出）落於
   `results/main_batch_outputs.jsonl`。
 
之後每次開機僅需 `conda activate smoea`；步驟 2、3 為一次性作業。

## 使用

```bash
# 互動：單筆完整請求跑完整流程，逐步顯示 Router 判定
python main.py --mode interactive

# 批次：跑 dataset 測試檔（--tasks 3,7 限任務、--limit 50 試跑）
#       輸出 results/main_batch_outputs.jsonl（含逐筆診斷）
python main.py --mode batch
```

互動模式輸出範例：

```
> <完整任務要求與內容，不含答案>
[Router] margin=0.183  p=0.42  詞彙一致✓
[Router] top-3：task23(sim 0.87)  task10(sim 0.71)  task24(sim 0.66)
[Router] 判定：綠區路由 → task23
[Output] <模型輸出>
```

## 從哪裡下手

- **拒絕分支（Model Merging）**：入口 `system/rejection.py`；artifact 契約與
  checksum 驗證在 `system/merged_model.py`，task／merged 切換在
  `system/inference.py`。SMoEA 不會在 query 時重新 merge，也不會自動選最新 run。
- **生成行為**（prompt、解碼參數、adapter 解析）：`system/inference.py`。
- **新增任務**：樣本放 `dataset/`、adapter 放 `adapter/task{N}/`、
  線上服務重跑 `python scripts/build_router_assets.py --serving-only`；
  要連同 benchmark test cache 一起建才改跑不帶此旗標的完整指令。
  送審裁決另需在
  `assets/unit_descriptions.json` 補該任務所屬單位的說明）。

## 資料格式

樣本檔 `{"task_key", "task_name", "definition", "instances": [...]}`，
每筆 instance 含 `input`、`full_prompt`（完整任務說明與使用者輸入）、
`output`、`instance_id`；亦相容純 array 與 JSONL。

交付預設 `data.routing_text=answer_free_full_prompt` 使用完整、但不含標準答案的
請求建 router，因為本專案 task0–14 的 `input` 沒有完整任務說明：

```bash
python scripts/build_router_assets.py --serving-only
```

訓練資料的 `full_prompt` 尾端必須精確等於 `output`；建置時只移除這段答案。
測試資料與互動輸入本來不含答案，會直接使用完整請求。建置出的
`router_assets_meta.json` 會記錄 `routing_text`，可避免混淆兩套路由資產。
若要重現上游直接使用短 `input` 的 router，才顯式加上
`--set data.routing_text=input`，並使用另一個 assets 目錄。

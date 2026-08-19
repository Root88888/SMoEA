# SMoEA — Scalable Mixture-of-Experts Adapters

![architecture](docs/architecture.jpg)

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
  rejection.py              拒絕分支（Model Merging）介面
scripts/
  check_env.py              環境體檢
  selftest_*.py             三支自測（零資料零 GPU）
  build_router_assets.py    路由資產離線建置（掃 dataset 自動推導任務集合）
  eval_router.py            路由評測三段
  eval_baseline_*.py        兩支 baseline
  verify_flow_table.py      評測結果獨立重放驗證
  plot_centroids.py         質心結構圖
dataset/                    資料（不進 git）；請建立 dataset 目錄以及 dataset/train_data/ 和 dataset/test_data/
  train_data/task{N}_train.json
  test_data/task{N}_test.json
adapter/task{N}/            LoRA adapters（不進 git）：請建立 adapter 目錄，將 task{N} 直接放在 adapter/ 下，task 內如有多個 checkpoint-*/ 自動取最新
assets/                     路由建置產物；unit_descriptions.json 為人工校訂的單位說明書
results/                    評測與批次輸出
docs/                       架構圖與文件
```

## 上手流程

1. git clone
```bash
   git clone https://github.com/Root88888/SMoEA.git && cd SMoEA
```
 
2. 放置檔案
   - 任務樣本：train 檔放 `dataset/train_data/`、test 檔放
     `dataset/test_data/`（檔名 `task{N}_train.json` / `task{N}_test.json`）
     ```bash
        cd ./
        pip install gdown
        
        mkdir -p dataset/train_data dataset/test_data
        
        gdown 1AsJwaqQ3AXmPT8TpAxOyvCPbyHtCi1lG -O dataset/train_data/train_data.zip
        gdown 1aiT9r9v2tyH-0cdf_F6zhfEvYF0mZ2tM -O dataset/test_data/test_data.zip
        
        python3 -m zipfile -e dataset/train_data/train_data.zip dataset/train_data/
        python3 -m zipfile -e dataset/test_data/test_data.zip  dataset/test_data/
     ```
   - adapters：每任務一個目錄，放成 `adapter/task{N}/`；解壓後若外層
     多包一層目錄，將其中的 `task*` 移出攤平

重要說明: 我在這個系統把原本的 OOD task149 稱為 task9149，以便跟 ID task149 區分，麻煩檔案就位後手動把 OOD task149 檔名改為 task9149_test.json

3. 一鍵建置
   
   已在 dataset/ood_tasks.txt 定義 OOD 任務有哪些，如果編號方式不同請修改

```bash
   bash scripts/setup_workspace.sh
```

   自動完成：檔案檢查（缺漏會明確提示）、conda 環境建置與依賴安裝
   （首次 10-20 分鐘）、環境體檢、查詢嵌入計算與路由資產建置
   （首次 GPU 數分鐘）。結尾印出「全部就緒」即完成；中途停止時
   依提示處理後重跑即可（已完成步驟自動跳過）。
   
4. 單筆執行互動
```bash
   conda activate smoea
   python main.py --mode interactive
```

   首次執行自動下載生成與裁決模型（各約 16GB）。輸入任務內 query
   應看到 Router 判定與模型回答；輸入無關文字應看到進入
   Model Merging 分支的訊息（即 `system/rejection.py` 的呼叫點）。
 
5. 批次執行
```bash
   python main.py --mode batch          # --tasks a,b 只跑任務a,b、--limit n 每個任務test set只取前n筆，這些沒加就是全跑
```
 
   逐筆結果（含 Router 診斷與模型輸出）落於
   `results/main_batch_outputs.jsonl`。
 
之後每次開機僅需 `conda activate smoea`；步驟 2、3 為一次性作業。

## 批次推論結果評測 (LLM-as-a-judge)

對批次推論的輸出以 OpenAI 模型閱卷：每筆將題目、標準答案、模型輸出
交給 LLM 評分——score 0–5（5=完全正確）、score≥4 計為正確
（is_correct），並附簡短評語。需自備 OpenAI API key。

要先產生批次推論結果 results/main_batch_outputs.jsonl or results/main_batch_outputs_{時間戳}.jsonl

```bash
# key 僅存在當前終端機，不要寫進任何檔案
export OPENAI_API_KEY=你的OPENAI_API_KEY

# 評測最新的 batch_output 檔
python scripts/eval_outputs_llm_judge.py

or
# 評測某個歷史 batch_output 檔
python scripts/eval_outputs_llm_judge.py --batch results/main_batch_outputs_{時間戳}.jsonl
```

選用參數，可組合：

- `--tasks 3,7`　只評這些來源任務（預設全部）
- `--limit 5`　每任務最多評幾筆（少量測試用）
- `--batch results/main_batch_outputs_{時間戳}.jsonl`
  指定評哪份推論結果（預設評主檔 `results/main_batch_outputs.jsonl`）
- `--model gpt-5-mini`　Judge 模型（預設 gpt-5-mini）
- `--resume`　斷點續評（跳過已成功評分的樣本）
- `--workers 8`　併發請求數

輸出 `results/llm_judge_{時間戳}.json`，時間戳繼承所評 batch 檔的
產出時間。

## 從哪裡下手

- **拒絕分支（Model Merging）**：入口 `system/rejection.py`——
  介面、可用的診斷素材、與已備妥的合成/載入/生成機制
  （`InferenceEngine.load_adapters_merged`）全寫在該檔檔頭；
  只需實作「用哪些 adapter、各配多少權重」的決策。
- **生成行為**（prompt、解碼參數、adapter 解析）：`system/inference.py`。
- **新增任務**：樣本放 `dataset/`、adapter 放 `adapter/task{N}/`、
  重跑 `build_router_assets.py` 即完成擴充（送審裁決另需在
  `assets/unit_descriptions.json` 補該任務所屬單位的說明）。

## 資料格式

樣本檔 `{"task_key", "task_name", "definition", "instances": [...]}`，
每筆 instance 含 `input`（路由用）、`full_prompt`（生成 prompt）、
`output`、`instance_id`；亦相容純 array 與 JSONL。

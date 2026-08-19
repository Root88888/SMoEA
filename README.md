# SMoEA — Scalable Mixture-of-Experts Adapters

![architecture](docs/architecture.jpg)

完整的 rejection runtime、外部 assets、benchmark 操作指令與目前驗收
範圍，見 [`docs/DELIVERY_ARCHITECTURE_RUNBOOK.md`](docs/DELIVERY_ARCHITECTURE_RUNBOOK.md)。

## 這個系統在做什麼

收到一段文字請求後，SMoEA 先判斷「這是不是我學過的任務」：

- 認得 → 交給那個任務專用的小型權重檔（adapter）來回答。
- 不認得 → 走「拒絕分支」，改用另一組權重回答。

負責判斷的那一層叫 Router。它只看請求本身、不看答案，結果分成四種：直接路由、
一般路由、送第二個模型複判、直接拒絕。前三種有機會命中某個任務，最後一種一定
進拒絕分支。複判沒通過的也會進拒絕分支。

拒絕分支要用哪一組權重，由你指定。以下是目前可選的五種。

### 拒絕之後可以用哪些方法

五種方法共用同一個底層大模型（Llama-3.1-8B），差別只在額外疊上去的東西。疊加
不會改動底層模型本身，所以互相切換不需要重新載入模型。

| 名稱 | 疊了什麼 | 實際做的事 |
|---|---|---|
| `base` | 不疊任何東西 | 直接用原始大模型回答。不需要額外檔案，是預設值。 |
| `ta` | 一組固定權重 | 把 150 個任務 adapter 直接平均成一組權重。所有任務的調整混在一起，不做取捨。 |
| `ties` | 一組固定權重 | 也是把 150 個混起來，但先丟掉每個 adapter 裡數值偏小的部分（只留最大的兩成），再讓剩下的部分投票決定每個位置該往哪個方向調整，方向和多數不一致的就不採用，最後整體縮到約三成。目的是減少不同任務互相拉扯。 |
| `dare-ties` | 一組固定權重 | 流程是「隨機丟掉一部分數值 → 把剩下的放大補回總量 → 投票」。交付設定的丟棄比例是 0，也就是隨機丟棄實際上關閉了，所以真正在做的是「全部保留 → 投票 → 整體縮小到四分之一」。它沒有 `ties` 那個「只留最大兩成」的步驟。 |
| `arrow` | 150 組權重加一份索引 | 不預先混合。回答時逐個 token（大致是一個字或詞的片段）判斷「這一段最接近哪個任務」，當場只套用那一個 adapter。同一句話裡不同位置可能用到不同 adapter，每一層也各自判斷。 |

前四種在開始回答之前權重就固定了，同樣的輸入會得到同樣的輸出。`arrow` 的權重
組合是回答過程中決定的，所以不同請求走的路徑不一樣。

需要準備什麼：

- `base` 不需要任何額外檔案。
- `ta`、`ties`、`dare-ties` 各需要一個約 3.76 GB 的權重檔，可以下載現成的，也可以
  用手上的 150 個 adapter 自己算（見〈準備拒絕分支要用的檔案〉）。
- `arrow` 需要那 150 個 adapter 本身，加上一份事先算好的索引檔。

不指定的話就是 `base`。系統不會自己去找或下載任何東西——沒準備就是沒有，
不會靜悄悄改用別的。

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
  arrow_runtime.py          Direct Arrow／Taskwise-K16 assets 驗證與動態 token routing
  rejection.py              拒絕分支：統一啟用 base、artifact 或 Arrow
  benchmark.py              15-OOD loader 與本地評分，共用 production rejection runtime
scripts/
  check_env.py              環境體檢
  selftest_*.py             三支自測（零資料零 GPU）
  build_router_assets.py    路由資產離線建置（掃 dataset 自動推導任務集合）
  eval_router.py            路由評測三段
  eval_baseline_*.py        兩支 baseline
  run_rejection_benchmark.py 直接測試 rejection method 的固定 15-OOD benchmark
  verify_flow_table.py      評測結果獨立重放驗證
  plot_centroids.py         質心結構圖
dataset/                    資料（不進 git）
  train_data/task{N}_train.json    線上 router 建置需要
  test_data/task{N}_test.json      批次評測才需要
  ood_test_data/task149_test.json  批次評測用的原始 OOD task149
adapter/task{N}/            LoRA adapters（不進 git）：請建立 adapter 目錄，將 task{N} 直接放在 adapter/ 下，task 內如有多個 checkpoint-*/ 自動取最新
<external>/merged_model/    靜態 merging condition 的 serving artifact（不進 git）
<external>/arrow/prepare/   Arrow prototypes／Taskwise 代表 adapters（不進 git）
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

     上游資料可用以下指令下載；只跑互動／線上服務時可略過 test data：

     ```bash
     pip install gdown
     mkdir -p dataset/train_data dataset/test_data

     gdown 1AsJwaqQ3AXmPT8TpAxOyvCPbyHtCi1lG -O dataset/train_data/train_data.zip
     gdown 1aiT9r9v2tyH-0cdf_F6zhfEvYF0mZ2tM -O dataset/test_data/test_data.zip

     python3 -m zipfile -e dataset/train_data/train_data.zip dataset/train_data/
     python3 -m zipfile -e dataset/test_data/test_data.zip dataset/test_data/
     ```
   - adapters：每任務一個目錄，放成 `adapter/task{N}/`；解壓後若外層
     多包一層目錄，將其中的 `task*` 移出攤平
   - OOD Natural Instructions `task149`：保留原始檔名，放到
     `dataset/ood_test_data/task149_test.json`；source Adapter Slot `task149`
     的測試檔仍可留在 `dataset/test_data/task149_test.json`
   - rejection method 所需檔案：artifact 方法提供完整 `prepare/merged_model/`；
     Arrow 提供 ordered adapter manifest，並可另提供 `prepare/` routing assets

要跑完整 benchmark 時，先執行
`python scripts/map_ood_aliases.py --dataset-dir dataset`，為 OOD `task149`
建立內部 `task9149` symlink，不需手動改名。Batch output 的 `source_task`
仍是原始 `task149`，並另以 `internal_task_id: task9149` 保留除錯資訊。

3. 一鍵建置
   
   已在 dataset/ood_tasks.txt 定義 OOD 任務有哪些，如果編號方式不同請修改

```bash
   bash scripts/setup_workspace.sh                       # 拒絕分支只有 base
   bash scripts/setup_workspace.sh --artifacts merge     # 另以本機 adapter 合成
   bash scripts/setup_workspace.sh --artifacts fetch --hf-repo <org>/<repo>
```

   拒絕分支要用的檔案在這一步準備好，不是等到真的拒絕時才去拿——服務執行中
   不會對外連線。`--artifacts merge` 以 `adapter/` 線上合成
   `ta`、`ties`、`dare-ties`（每份約 3.76 GB、需 GPU，已存在者自動跳過）；
   `--artifacts fetch` 改為自 Hugging Face repo 取得現成的。兩者都會登記進
   `<artifact-root>/registry.json`，執行期以 id 選用。不加這個參數時拒絕分支
   只有 base，之後隨時可以單獨補跑。

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
   輸入無關文字應看到進入 rejection inference。預設 `base`；使用 merged
   artifact 時明確指定：

```bash
   python main.py --mode interactive \
     --set system.rejection_method=artifact \
     --set system.rejection_artifact_dir=/shared/run/prepare/merged_model \
     --set system.dtype=bfloat16
```

`artifact` 路徑未設定、schema 不符、base model 不同、檔案大小或 checksum
錯誤都會在接受輸入前停止，不會偷偷改用 base model。

   其他方法怎麼準備、怎麼在執行中切換，見下方〈準備拒絕分支要用的檔案〉。

5. 批次評測

   批次模式另外需要完整 `dataset/test_data/`。若資料尚未備齊，互動模式
   仍可正常使用，但不要把部分 test data 的結果當成完整 benchmark。

```bash
   python main.py --mode batch          # --tasks a,b 只跑任務a,b、--limit n 每個任務test set只取前n筆，這些沒加就是全跑
```
 
   路由命中使用 task adapter；router 拒絕則按 batch 集中使用 selected rejection
   method，不再寫 `output=null`。逐筆結果（含 Router 診斷、實際 model source、
   condition/run identity 與模型輸出）落於
   `results/main_batch_outputs.jsonl`。
 
之後每次開機僅需 `conda activate smoea`；步驟 2、3 為一次性作業。

互動模式輸出範例：

```text
> <完整任務要求與內容，不含答案>
[Router] margin=0.183  p=0.42  詞彙一致✓
[Router] top-3：task23(sim 0.87)  task10(sim 0.71)  task24(sim 0.66)
[Router] 判定：綠區路由 → task23
[Output] <模型輸出>
```

## 準備拒絕分支要用的檔案

`ta`、`ties`、`dare-ties` 這三個方法各需要一個約 3.76 GB 的權重檔。**取得方式有兩種，
選一種就好，結果完全一樣**（同一份權重，或差異在儲存格式的最小間隔之內）：

| | 需要什麼 | 花多久 |
|---|---|---|
| **方式一：下載** | 網路、Hugging Face 帳號與存取權 | 看網路速度，約 11 GB |
| **方式二：自己算** | 本機那 150 個 adapter（約 2.8 GB） | GPU 約 1 小時／CPU 約 3 小時（三個方法合計） |

`base` 兩種都不需要，直接可用。`arrow` 與 `taskwise_k16_arrow` 不走這裡——它們用的是
另一套檔案，見〈其他兩種動態方法〉。

一鍵處理（兩種方式都可以在建置時一起做完）：

```bash
bash scripts/setup_workspace.sh --artifacts fetch --hf-repo <org>/<repo>   # 方式一
bash scripts/setup_workspace.sh --artifacts merge                          # 方式二
```

也可以事後單獨補跑，見下面兩節。

### 方式一：從 Hugging Face 下載

先登入（一次就好，權杖會存在本機）：

```bash
hf auth login
```

下載並登記：

```bash
python scripts/fetch_artifact.py --repo <org>/<repo> --list --condition ties
python scripts/fetch_artifact.py --repo <org>/<repo> --condition ties \
  --artifact-root <本機路徑> --registry <本機路徑>/registry.json
```

`--list` 先看遠端有哪些版本。遠端同一個方法有多個版本時，必須用 `--run-id` 明講要哪一個
——系統不會自己挑最新的。

下載完會逐檔核對檔案大小與雜湊值，對不上就拒絕，不會把壞掉的檔案留下來。

### 方式二：用本機的 150 個 adapter 自己算

不需要網路。把那 150 個 adapter 放在 `adapter/task{N}/` 之下即可：

```bash
python scripts/merge_pool150.py --method ties --adapter-dir adapter \
  --artifact-root <本機路徑> \
  --registry <本機路徑>/registry.json --register-as ties \
  --set system.dtype=bfloat16
```

`--method` 三選一：`ta`、`ties`、`dare-ties`。三個都要就跑三次。

**選 CPU 還是 GPU**：預設走 GPU（`--device cuda`）。沒有顯卡就加 `--device cpu`，
結果一樣，只是慢很多。`ties` 另外需要約 7.5 GB 顯示記憶體，顯卡至少要 12 GB。

實測時間（RTX4000SFF Ada 20GB／CPU 8 執行緒）：

| 方法 | GPU | CPU |
|---|---|---|
| `ta` | 約 1 分鐘 | 約 5 分鐘 |
| `dare-ties` | 約 3 分鐘 | 約 40 分鐘 |
| `ties` | 約 36 分鐘 | 約 2.4 小時 |

`ties` 特別慢是因為它要幫每一個任務算一個門檻（「數值前 20% 大的分界線在哪」），
而要知道這條線就得把該任務的 18.8 億個數值排過一遍，150 個任務就排 150 次。排序吃的是
記憶體頻寬不是算力，所以 GPU 相對 CPU 只快約兩倍。

這是產生權重檔的一次性成本。之後回答請求時不會再做這件事。

已經算過的不會重算：程式會比對 adapter 的內容，同一批 adapter 算過就直接沿用。

**關於編號**：每個權重檔的編號取自檔案本身的雜湊值，所以編號相同就保證內容相同。
同一批 adapter 在同一台機器上算兩次會得到完全一樣的檔案；換一款顯卡則可能有極少數
數值差最後一位（浮點加法換個順序算就差一位），那會是另一個編號——這是正確的，因為
它們確實是兩份不同的檔案。差異程度可以用 `scripts/verify_against_producer.py` 對照。

### 準備好之後：怎麼選用

把設定指向產出的清單檔：

```bash
python main.py --mode interactive \
  --set system.artifact_registry=<本機路徑>/registry.json \
  --set system.dtype=bfloat16
```

互動模式中：

```text
:rejection              顯示目前使用哪一個
:rejection list         列出可選的項目（* 標示目前生效者）
:rejection use ties     切換；底層大模型不重載，幾秒完成
```

批次模式整批共用同一個：

```bash
python main.py --mode batch --artifact ties \
  --set system.artifact_registry=<本機路徑>/registry.json
```

只有列在清單檔裡的項目可以選——系統不掃描目錄，也不會自動挑最新的一份。切換前會完整
檢查要換過去的那一份（格式、底層模型是否相符、檔案大小與雜湊值）；檢查沒過就維持原本
生效的那一個，不會退回 `base`。

清單檔的格式見
[`examples/artifact_registry.example.json`](examples/artifact_registry.example.json)。

### 其他兩種動態方法

`arrow` 與 `taskwise_k16_arrow` 不需要上面那個 3.76 GB 的權重檔，但各自需要別的東西：

- `arrow`：那 150 個 adapter 本身，加上一份事先算好的索引檔（`prepare/` 目錄）。
- `taskwise_k16_arrow`：16 個代表 adapter 與索引檔（約 275 MB 的 `prepare/` 目錄）。
  哪 16 個當代表是離線分群決定的，本系統不做分群，只讀現成的。

兩者都在清單檔裡登記後即可用 `:rejection use` 切換，用法與上面相同。

### 舊版權重檔的相容處理

舊版工具產生的權重檔，說明檔（`result.json`）可能少了幾個欄位，本系統會拒絕載入。
先跑一次補寫，只改說明檔、不動權重本身。第一行的 `--dry-run` 是「空跑」——把會做的事
印出來但不真的改檔案，先確認對象沒選錯，確認後再跑第二行：

```bash
python scripts/migrate_artifact_manifest.py --scan <權重檔所在目錄> --dry-run
python scripts/migrate_artifact_manifest.py --scan <權重檔所在目錄>
```

### 把權重檔分享給別人

自己算好的權重檔要給其他人用時，可以上傳到私有的 Hugging Face repo。上傳與下載都只在
準備階段使用——服務執行中不會對外連線。

```bash
python scripts/push_artifact.py --repo <org>/<repo> --dry-run \
  --artifact <本機路徑>/ties/<編號>/prepare/merged_model
python scripts/push_artifact.py --repo <org>/<repo> \
  --artifact <本機路徑>/ties/<編號>/prepare/merged_model
```

只會建立私有 repo。加上 `--dry-run` 是空跑：檢查檔案、算出會傳到哪個路徑、加總大小，
然後停住，不建 repo 也不上傳任何東西。11 GB 傳完才發現弄錯很花時間，空跑幾秒就能先確認
清單。這個模式不需要登入。

權重檔是 Llama-3.1-8B 的衍生物（只含 `down_proj` 的差值，不含底層模型本身）。公開散布前
須確認 Llama 3.1 Community License 的附隨條款與上游資料授權。

## 重現路由結果 Routing Zone Outcome and Accuracy/Ablation/Baseline Comparison

### 1. 主評測

```bash
# 1a. 分區（CPU 數分鐘）：全部測試樣本分四區、產送審佇列
python scripts/eval_router.py --mode decide 2>&1 | tee results/eval_decide.txt

# 1b. 送審打分（GPU 數小時；中斷重跑自動續）：裁決 LLM 對佇列逐筆三題是非
python scripts/eval_router.py --mode score 2>&1 | tee results/eval_score.txt

# 1c. 結算
python scripts/eval_router.py --mode run 2>&1 | tee results/eval_run.txt
```

### 2. Ablation 變體資產準備（無多質心版；一次性）

```bash
mkdir -p assets_ablate_nomc
cp assets/emb_*.npz assets/unit_descriptions.json assets_ablate_nomc/
python scripts/build_router_assets.py \
    --set paths.assets_dir=assets_ablate_nomc --set fingerprint.k_max=1
```

### 3. 五個 Ablation 變體

```bash
for AB in gray_reject gray_route no_lexical no_direct no_multicentroid; do
  python scripts/eval_router.py --mode decide --ablate $AB
  python scripts/eval_router.py --mode score  --ablate $AB
  python scripts/eval_router.py --mode run    --ablate $AB
done
```

### 4. 兩支 Baseline

```bash
python scripts/eval_baseline_mean_embedding.py --mode eval --tau 0.72
python scripts/eval_baseline_bm25_voting.py    --mode eval --ratio_tau 0.5
```

### 5. 匯總數據

```bash
python scripts/export_report_data.py        # 輸出 results/report_data.json
```

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
- `--ood_dataset_dir dataset/ood_test_data`　OOD 標準答案目錄

輸出 `results/llm_judge_{時間戳}.json`，時間戳繼承所評 batch 檔的產出時間。
`per_path` 會把 routed 與各 rejection method（base、artifact、Arrow、
Taskwise-K16）分開統計；這支 script 評的是完整 `main.py --mode batch`
輸出，和只測 rejection condition 的 `run_rejection_benchmark.py` 用途不同。

## 從哪裡下手

- **拒絕分支**：入口 `system/rejection.py`；artifact 契約在
  `system/merged_model.py`，Arrow 契約在 `system/arrow_runtime.py`，所有切換集中於
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

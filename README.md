# SMoEA — Scalable Mixture-of-Experts Adapters

### 系統概覽
- Router把每個query路由到150個任務專用LoRA adapter的其中一個
- 若router對於query屬於哪個任務沒有把握而reject時，則改走merging 分支
- merging分支分為動態和靜態，靜態支援直接載入現有權重（artifacts）、動態則是在infernce time即時merging。在下方Merging分支的地方會詳細說明

---

## 專案架構

![architecture](docs/architecture.jpg)

### Router

Router 只讀請求本身、不看答案，決定兩件事之一：

- **Accept（命中）** —— 請求對應到某個已知任務，交給 `adapter/task{N}/` 底下該任務的
  adapter 回答。
- **Reject（拒絕）** —— 沒有任務有把握對應，請求轉入 **merging 分支**。

判定內部分成四區：直判路由、綠區路由、送審（交第二個模型複判）、紅區拒絕。前三種有
機會產生命中；紅區一定拒絕，送審沒通過門檻的也會拒絕。

路由資產由 `scripts/build_router_assets.py` 離線建置，放在 `assets/`。

### Merging 分支

Router Reject之後，會進入設定好的merging conditions(即不同的methods)，並且所有 condition共用同一個 **base model**
（`unsloth/Meta-Llama-3.1-8B`）。

#### Conditions

**Baselines** —— 權重在開始生成之前就固定了，同樣的輸入得到同樣的輸出。

| `condition_id` | 做什麼 | 可當線上拒絕選項 | 可本機自行建置 |
|---|---|---|---|
| `base` | 什麼都不疊，直接由 base model 回答。不需要額外檔案。 | 可 | 不適用 |
| `ta` | Task Arithmetic。把 150 個任務 adapter 平均成一組權重。 | 可 | 可 |
| `pico_ta` | 在 Task Arithmetic 之前先做一層低秩處理。 | 可 | 可 |
| `ties_only` | 先把每個 adapter 修剪成數值最大的那部分座標，再逐座標選出方向，只保留與該方向一致的貢獻。名稱裡的 `only` 表示 TIES 之後沒有再接最佳化階段——用來與 `adamerging_pp` 區分。 | 可 | 可 |
| `dare_ties_ta` | 隨機丟棄、放大補回、取號投票，再接 Task Arithmetic。封板的丟棄比例是 0，所以隨機那一階段實際上關閉了。 | 可 | 可 |
| `lora_lego` | LoRA-Lego 的合併方式，把整個池的逐 rank 單元分群。 | 可 | 可 |
| `adamerging_pp` | 以 TIES 當前處理，再對合併係數做最佳化。學出的是「逐層 × 逐任務」的係數，需要載入模型、對校準資料跑迭代，不是純權重運算。 | 可 | 可（需 GPU 與 `dataset/train_data/`） |
| `lorahub` | LoRAHub。從 150 個中隨機挑 20 個，用 CMA-ES 搜尋權重讓它們的加權和在幾筆示範樣本上損失最低。**只能當 benchmark 的受測對象** —— 係數是針對特定示範樣本擬合的，沒有一組權重能對應任意的線上請求。 | **不可** | 可（需 GPU 與示範樣本） |

除了 `base` 之外，每個 baseline 各需要一個權重檔。前六個是 dense delta，約 3.76 GB；
`lorahub` 的產物是 LoRA（與來源同 rank），小很多。

**Arrow routing** —— 不預先合併權重。生成過程中逐 token 與各 expert 的原型比對，
只套用最接近的那一個，而且每一層各自判斷。**不同請求走的路徑不一樣。**

| `condition_id` | 候選數 | 需要的檔案 |
|---|---|---|
| `arrow` | 全部 150 個 adapter | 那 150 個 adapter，加一份事先算好的原型索引 |
| `taskwise_k16_arrow` | 16 個群代表 | 16 個代表 adapter 與索引（約 275 MB） |

**這兩個是為「像訓練任務但沒見過」的請求設計的。** 面對與訓練分布差距很大的自由形式
提問（寫詩、閒聊之類），逐層獨立的路由可能各層挑到互不相關的 expert，輸出品質會明顯
下降。這是方法本身的性質，不是設定錯誤。`taskwise_k16_arrow` 只有 16 個候選，通常比
150 個候選的 `arrow` 穩定。要評估這兩個方法，用貼近任務型態的輸入，或直接跑 15-OOD
benchmark。

### 執行期實際能選哪些

引擎能服務四種型態：`base`、`artifact`、`arrow`、`taskwise_k16_arrow`。其中
**`artifact` 是通用的** —— 任何合規的 `prepare/merged_model/` 目錄都能載入，不管它是
哪種合併方法產生的。（包括在本地生成的lorahub, adadmerging++, lego等）

可選的項目宣告在一份**清單檔（registry）**裡，預設是 `artifacts/registry.json`。
只有宣告過的項目才能選。

### 目錄架構

```
main.py                     程式入口：interactive 與 batch 兩種模式
configs/default.yaml        所有設定的唯一定義處；可用 --set key=value 覆蓋單項

router/                     路由決策層
  core.py                     Router 類別：build / save / load / decide / escalate /
                              finalize —— 路由邏輯唯一所在
  config.py data_io.py        設定載入、資料與嵌入快取 I/O
  embedding.py                查詢嵌入（bge）
  fingerprint.py units.py     任務指紋、多質心、路由單位
  lexical.py conformal.py     詞彙一致性訊號、共形校準、四區判定
  verifier.py                 送審裁決（對第二個 LLM 問是非題）
  metrics.py                  評測計分

system/                     執行層，路由決定之後的一切
  inference.py                InferenceEngine：base model 常駐、per-task adapter
                              熱切換、生成、拒絕 condition 的選擇
  registry.py                 清單檔：解析與驗證宣告過的可選 condition
  rejection.py                merging 分支的唯一入口，互動／批次／benchmark 共用
  merged_model.py             權重檔契約：writer 與 validator 放在一起，不會各自演進
  adapter_pool.py             pool150 清單解析、LoRA 載入、adapter 池指紋
  merging.py                  本機建置 ta / ties_only / dare_ties_ta，超參數封板
  arrow_runtime.py            Arrow 與 Taskwise-K16 的資產驗證與 token 路由
  benchmark.py                15-OOD 載入與本地計分

scripts/
  setup_workspace.sh          一鍵前置：環境、路由資產、artifact
  check_env.py                環境體檢
  build_router_assets.py      路由資產離線建置
  selftest_*.py               三支自測，不需要 GPU 也不需要真實資料
  merge_pool150.py            用本機的 150 個 adapter 自行建置 artifact
  fetch_artifact.py           自 Hugging Face repo 下載 artifact
  push_artifact.py            上傳 artifact 到私有 Hugging Face repo
  migrate_artifact_manifest.py  修補舊版 producer 寫出的說明檔
  verify_against_producer.py  比對本機建置的 artifact 與參考版本
  rename_remote_condition.py  在遠端 repo 內改 condition 名稱，伺服器端執行
  run_rejection_benchmark.py  固定 15-OOD benchmark，跳過 Router
  eval_router.py              路由評測，三個階段
  eval_baseline_*.py          兩支路由 baseline
  eval_outputs_llm_judge.py   以 LLM 為評分者，評批次輸出
  verify_flow_table.py        評測結果的獨立重放驗證

adapter/task{N}/            LoRA adapters（不進 git）
dataset/                    資料集（不進 git）
  train_data/task{N}_train.json     建置路由資產需要
  test_data/task{N}_test.json       批次評測需要
  ood_test_data/task149_test.json   原始的 OOD task149
assets/                     路由建置產物；unit_descriptions.json 為人工校訂
artifacts/                  merging 分支的權重檔與 registry.json（不進 git）
results/                    批次輸出與評測結果
docs/                       架構圖、交付說明書、架構決策紀錄
```

**Artifact** 指的是一個目錄，裡面放一組權重，加上一份 `result.json` 契約
——記錄 base model 指紋、dtype、檔案大小與雜湊值，以及它由哪一批 adapter 產生。它的
`run_id` 取權重檔本身 SHA-256 的前 16 碼，所以**編號相同就保證內容相同**。

---

## 設置方法

Clone 之後，手動放三樣東西：

| 什麼 | 放哪 | 什麼時候需要 |
|---|---|---|
| Router 訓練樣本 | `dataset/train_data/task{N}_train.json` | 建置路由資產 |
| LoRA adapters | `adapter/task{N}/` | 路由與本機合併 |
| 測試資料 | `dataset/test_data/task{N}_test.json` | 只有批次評測需要 |
| OOD 測試資料 | `dataset/ood_test_data/task149_test.json` | 只有完整 benchmark 需要 |

adapters 每個任務一個目錄。解壓後若外層多包一層目錄，把裡面的 `task*` 移出攤平。

上游資料可以這樣下載（只跑互動／線上服務的話，test data 可以略過）：

```bash
pip install gdown
mkdir -p dataset/train_data dataset/test_data

gdown 1AsJwaqQ3AXmPT8TpAxOyvCPbyHtCi1lG -O dataset/train_data/train_data.zip
gdown 1aiT9r9v2tyH-0cdf_F6zhfEvYF0mZ2tM -O dataset/test_data/test_data.zip

python3 -m zipfile -e dataset/train_data/train_data.zip dataset/train_data/
python3 -m zipfile -e dataset/test_data/test_data.zip dataset/test_data/
```

**哪些任務算 OOD** 由 `dataset/ood_tasks.txt` 宣告（repo 自帶）。宣告即真相——
列在裡面的就是 OOD，不管有沒有訓練檔。編號方式不同的話改這個檔。

**OOD 的 task149**：保留原始檔名放進 `dataset/ood_test_data/`。要跑完整 benchmark
之前先執行一次：

```bash
python scripts/map_ood_aliases.py --dataset-dir dataset
```

它會為 OOD `task149` 建立內部用的 `task9149` symlink，不需要手動改名。批次輸出的
`source_task` 仍然是原始的 `task149`，另以 `internal_task_id` 保留內部編號供除錯。

然後執行一次前置，可以選擇需不需要下載權重＆本地生成靜態權重。

```bash
bash scripts/setup_workspace.sh                                            # 只有 base
bash scripts/setup_workspace.sh --artifacts fetch --hf-repo <org>/<repo>   # 下載權重
bash scripts/setup_workspace.sh --artifacts merge                          # 本地生成靜態權重
```

| 參數 | 意思 |
|---|---|
| `--artifacts fetch` | 從 Hugging Face repo 下載備好的 artifact |
| `--artifacts merge` | 用本機的 adapter 建置 `ta`、`ties_only`、`dare_ties_ta`；其他用 `--methods` 指定 |
| `--hf-repo <org>/<repo>` | 來源 repo，搭配 `fetch` 使用時必填 |
| `--artifact-root <路徑>` | artifact 落地位置（預設 `artifacts/`） |
| `--methods ta,ties_only` | 限定要準備哪幾個 condition |

不加 `--artifacts` 的話，merging 分支只有 `base`；之後隨時可以用 `merge_pool150.py`
或 `fetch_artifact.py` 補上。

base model 與送審裁決模型（各約 16 GB）會在首次執行 `main.py` 時自動下載。

前置可以重複執行：已完成的步驟會自動跳過。

---

## Inference 方法

### Interactive mode

```bash
conda activate smoea
python main.py --mode interactive
```

不需要任何路徑參數 —— 清單檔預設就是 `artifacts/registry.json`。

| 參數 | 意思 |
|---|---|
| `--set system.dtype=bfloat16` | 生成用的 dtype；必須與 artifact 宣告的一致 |
| `--set system.rejection_method=<id>` | 沒有清單檔時，用哪一個 condition |
| `--set system.verifier_mode=<模式>` | 送審裁決：`resident_4bit`（預設，裁決模型 4bit 常駐、與生成模型共存）、`swap`（分時載卸，較慢但精度同封板評測）、`off`（不載裁決，送審一律拒絕） |
| `--no_preload` | 延後到第一筆請求才載入模型 |

session 內的指令：

| 指令 | 作用 |
|---|---|
| `:rejection` | 顯示目前生效的 condition |
| `:rejection list` | 列出清單檔的項目，`*` 標示目前生效者 |
| `:rejection use <id>` | 切換 condition。base model 不會重新載入 |
| `:paste` … `:send` | 輸入多行請求 |
| `exit` | 離開 |

切換時會**先完整驗證**要換過去的那一個 —— schema、base model 指紋、dtype、檔案大小
與雜湊值 —— 通過了才生效。驗證失敗就維持原本的 condition；系統絕不會靜默退回 `base`。

範例：

```text
[Router] 資產已載：150 任務、N 路由單位
拒絕分支目前使用 base；:rejection list 看可選項目、:rejection use <id> 切換

> <完整的任務要求與內容，不含本題答案>
[Router] margin=0.183  p=0.42  詞彙一致✓
[Router] top-3：task23(sim 0.87)  task10(sim 0.71)  task24(sim 0.66)
[Router] 判定：綠區路由 → task23
[Output] (2.4s)
<答案>

> :rejection use ties_only
[Rejection] 已切換到 ties_only（ties_only:e3de085e3caeaf23）

> <與任何已知任務都無關的內容>
[Router] 判定：紅區拒絕 → 進入 rejection inference 分支
[Rejection] ties_only:e3de085e3caeaf23
[Output]
<答案>
```

輸入必須是完整的任務要求與內容，而且不能包含本題的答案；系統會擋掉 prompt 尾端等於
自己標準答案的樣本。

### Batch mode

```bash
python main.py --mode batch --artifact ties_only
```

| 參數 | 意思 |
|---|---|
| `--artifact <id>` | 這一批所有拒絕樣本共用的 condition |
| `--tasks 3,7,10` | 限定這些任務（預設全部） |
| `--limit 50` | 每個任務最多取幾筆 |

整批共用一個 condition，結果才能互相比較。批次模式需要 `dataset/test_data/` 就位。

結果落在 `results/main_batch_outputs.jsonl`，一筆一個 JSON 物件：

```json
{"source_task": "task23",
 "instance_id": "task23-0007",
 "routed_to": null,
 "diagnosis": {"zone": 3, "margin": 0.041, "pval": 0.01, "...": "..."},
 "model_source": "rejection",
 "rejection_method": "artifact",
 "rejection_condition_id": "ties_only",
 "rejection_run_id": "e3de085e3caeaf23",
 "output": "..."}
```

命中時 `routed_to` 是用的 task adapter，拒絕時為 `null`。三個 `rejection_*` 欄位記錄
究竟是哪一個 condition 回答的，所以任何一筆輸出都追溯得到一組具體的權重。同時會另外
寫一份帶時間戳的副本。

### Merging 分支的 artifact 怎麼取得

兩種方式，結果等價。

**方式一：放在 Hugging Face 上，下載取用。** 先登入一次：

```bash
hf auth login
```

```bash
python scripts/fetch_artifact.py --repo <org>/<repo> --condition ties_only --list
python scripts/fetch_artifact.py --repo <org>/<repo> --condition ties_only
```

| 參數 | 意思 |
|---|---|
| `--condition <id>` | 要取哪一個 condition |
| `--list` | 只列出遠端有哪些版本，不下載 |
| `--run-id <id>` | 遠端同一 condition 有多個版本時必填 |
| `--artifact-root <路徑>` | 放到哪（預設 `artifacts/`） |
| `--register-as <id>` | 登記進清單檔時用的 id |

下載完會逐檔核對 `result.json` 記錄的大小與雜湊值。同一個 condition 有多個版本時
必須明講要哪一個 —— 系統不會自己挑最新的。

**方式二：離線用本機的 adapter 自己跑。** 不需要網路。

```bash
python scripts/merge_pool150.py --method ties_only
```

| 參數 | 意思 |
|---|---|
| `--method {ta,ties_only,dare_ties_ta,pico_ta,lora_lego,adamerging_pp,lorahub}` | 要建置哪一個 condition |
| `--train-root <路徑>` | `adamerging_pp` 的校準資料；預設 `dataset/train_data` |
| `--examples <檔案>` | `lorahub` 的示範樣本 JSON；`lorahub` 必填 |
| `--run-seed N` | `lorahub` 的 run seed（封板為 1、2、3） |
| `--adapter-dir <路徑>` | adapter 放在別處時才要指定；預設取 `system.adapter_dir`，由 `adapter/task{N}/` 慣例推導有序清單 |
| `--manifest <檔案>` | 改用明確指定的有序清單 |
| `--device cpu` | 用 CPU 計算（預設 `cuda`） |
| `--artifact-root <路徑>` | 產物落地根目錄（預設取 `system.artifact_root`） |
| `--registry <檔案>` | 要登記進哪份清單檔；傳 `none` 可略過 |

超參數是封板的、不開放調整 —— 你選方法，不選設定。三者之中 `ties_only` 明顯比其他
兩個久，因為它要為每個任務算一個數值門檻，跑之前先預留時間。`ties_only` 另外需要約
7.5 GB 顯示記憶體，不用 `--device cpu` 的話顯卡至少要 12 GB。這是一次性成本，回答
請求時不會再做這件事。

同一批 adapter 已經建置過就不會重算。

**七個方法分成兩種性質:**

前五個（`ta`、`ties_only`、`dare_ties_ta`、`pico_ta`、`lora_lego`）是純權重運算，
只吃 adapter、不需要任何資料，CPU 也能跑。

後兩個要對資料做最佳化，只能用 GPU：

```bash
# adamerging_pp：對 dataset/train_data 跑 500 次迭代學出逐層係數，再走 TIES 合成
python scripts/merge_pool150.py --method adamerging_pp

# lorahub：對指定的示範樣本跑 CMA-ES（40 代 × 12 族群 = 480 次評估）
python scripts/merge_pool150.py --method lorahub \
  --examples <示範樣本.json> --run-seed 1
```

`lorahub` 的示範樣本格式是 `[{"instance_id", "prompt", "output"}, ...]`，封板設定是
5 筆。**產出的 manifest 會記錄示範樣本的指紋**，因為那份權重只對這組樣本有意義；
換一批樣本就要重跑，指紋不同、編號也不同。

---

## Benchmark：直接略過 Router，只測某一個 merging 方法

`scripts/run_rejection_benchmark.py` **完全跳過 Router**，把固定的 15-OOD 資料餵給
單一 condition。它衡量的是 merging 分支本身，不是路由準確率。

```bash
python scripts/run_rejection_benchmark.py --artifact ties_only \
  --benchmark-root <benchmark 資料根目錄> \
  --output-dir results/rejection-ties_only \
  --set system.dtype=bfloat16
```

| 參數 | 意思 |
|---|---|
| `--artifact <id>` | 受測的 condition；不給則用設定檔的 `system.rejection_method` |
| `--benchmark-root <路徑>` | benchmark 資料根目錄（必填） |
| `--output-dir <路徑>` | 結果輸出位置（必填） |
| `--group {all,ni,bbh,mmlu_pro}` | 限定某一個資料集家族 |
| `--batch-size N` | 生成批次大小 |
| `--smoke` | 每個家族只跑第一筆，用來確認流程通 |

資料是 5 個 Natural Instructions、5 個 BBH、5 個 MMLU-Pro，完整執行共 4,159 筆。
benchmark 用的是**和互動、批次完全相同的引擎**，只是換了資料來源並跳過 Router。

輸出：

```text
<output-dir>/ni_results.json
<output-dir>/bbh_results.json
<output-dir>/mmlu_pro_results.json
<output-dir>/metrics.json      "rejection" 欄位記錄受測 condition 的身分
```

生成固定使用 bfloat16、8,192 input tokens、1,024 new tokens。prompt 過長會中止而不是
靜默截斷。本地計分包含分類準確率、ROUGE-L 與 BLEU。GPT judge 不會被自動呼叫，
`metrics.json` 會記 `judge: not_run`。

`lorahub` 可以當這裡的受測對象（前提是有人為這份資料集擬合出 artifact），但它不能
服務任意的線上請求。

---

## Evaluation：如何評估

有三件不同的東西要評，不要混在一起。

### 一、路由品質

```bash
python scripts/eval_router.py --mode decide   # 分區，CPU，數分鐘
python scripts/eval_router.py --mode score    # 送審打分，GPU，可中斷續跑
python scripts/eval_router.py --mode run      # 結算
```

| 參數 | 意思 |
|---|---|
| `--mode {decide,score,run}` | 要跑哪一段；必須照這個順序 |
| `--ablate <變體>` | `no_multicentroid`、`no_direct`、`no_lexical`、`gray_reject`、`gray_route` |
| `--fake_verifier` | 不載真的裁決模型，用來確認流程 |

`no_multicentroid` 需要另一份資產目錄，以 `k_max=1` 建置一次：

```bash
mkdir -p assets_ablate_nomc
cp assets/emb_*.npz assets/unit_descriptions.json assets_ablate_nomc/
python scripts/build_router_assets.py \
    --set paths.assets_dir=assets_ablate_nomc --set fingerprint.k_max=1
```

其他四個變體直接用主要資產。每個變體都要跑完三個階段，順序不能變：

```bash
for AB in gray_reject gray_route no_lexical no_direct no_multicentroid; do
  python scripts/eval_router.py --mode decide --ablate $AB
  python scripts/eval_router.py --mode score  --ablate $AB
  python scripts/eval_router.py --mode run    --ablate $AB
done
```

另有兩支路由 baseline 可供對照：

```bash
python scripts/eval_baseline_mean_embedding.py --mode eval --tau 0.72
python scripts/eval_baseline_bm25_voting.py    --mode eval --ratio_tau 0.5
python scripts/export_report_data.py           # 匯總成 results/report_data.json
```

`scripts/verify_flow_table.py` 會獨立重放一次評測，與主結果逐格比對。

### 二、輸出品質（LLM as a judge）

評的是 `main.py --mode batch` 產生的答案。每一筆把題目、標準答案、模型輸出交給
OpenAI 模型評 0–5 分，4 分以上算正確。需要自備 API key。

```bash
export OPENAI_API_KEY=<你的 key>          # 不要寫進任何檔案
python scripts/eval_outputs_llm_judge.py
```

| 參數 | 意思 |
|---|---|
| `--batch <檔案>` | 要評哪一份批次輸出（預設評主檔） |
| `--tasks 3,7` | 只評這些來源任務 |
| `--limit 5` | 每任務最多評幾筆 |
| `--model gpt-5-mini` | 評分用的模型 |
| `--resume` | 跳過已評分的樣本 |
| `--workers 8` | 併發請求數 |
| `--dry_run` | 不呼叫 API，只驗資料對齊 |

結果落在 `results/llm_judge_{時間戳}.json`，時間戳繼承所評的批次檔。`per_path` 會把
路由樣本與各個拒絕 condition 分開統計。

### 三、merging 分支的品質

用上面的 benchmark，它在本地計分，不需要 API key。

要確認本機建置的 artifact 與參考版本一致：

```bash
python scripts/verify_against_producer.py --method ties_only \
  --producer <參考的 merged_model 目錄> --work-dir <暫存目錄> --device cuda
```

**不要求雜湊值完全相同。** 浮點加法不滿足結合律，不同型號的顯卡累加順序不同，因此
極少數數值可能在儲存格式上差一格。報告會統計有多少個數值不同、各差幾格；超過一格就
不是捨入能解釋的，應當成缺陷處理。

---

## 測試

### 快速確認每個方法都能服務

```bash
python scripts/smoke_rejection_methods.py --set system.dtype=bfloat16
```

逐一切換清單檔裡的每個 condition，各生成一次，最後給總結表。**完全跳過 Router**，
因此不受路由資產設定影響；base model 只載一次、方法之間熱切換。

`--only base,ties_only` 只測其中幾個；`--prompt "..."` 換成自己的提示。任何一個方法
失敗會標示原因但不中斷，其餘照樣測完。


四層，由便宜到昂貴。

```bash
python -m unittest discover -s tests      # 74 個測試，不需要 GPU 也不需要資料
python scripts/selftest_core_modules.py   # 四區判定、conformal p 值、計分對帳
python scripts/selftest_main_pipeline.py  # 合成資料上的完整批次流程
python scripts/selftest_end_to_end.py     # 資產建置到評測
python scripts/check_env.py               # 套件版本、CUDA、磁碟
```

前四項不需要 GPU 也不需要真實資料，一分鐘內跑完。它們證明的是「接線正確」，不是
「答案好」—— 輸出品質只有用真實權重才看得到。

真實硬體上的驗收包含：互動模式跑一次拒絕、benchmark 的 `--smoke`、以及完整 4,159 筆。

---

## 從哪裡下手

接手這份程式時，依你要改的東西找對應的檔案：

| 想改什麼 | 從哪裡看 |
|---|---|
| **拒絕分支的行為** | `system/rejection.py` —— 互動、批次、benchmark 共用的唯一入口，只有一個 `run_rejection()` |
| **可選哪些方法** | `system/registry.py` —— 清單檔的解析與驗證；`system/inference.py` 的 `select_rejection()` 負責切換 |
| **權重檔的契約** | `system/merged_model.py` —— writer 與 validator 放在同一個模組，兩者不會各自演進 |
| **本機建置某個方法** | `system/merging.py`（五種純權重運算）、`system/adamerging.py`（係數最佳化）、`system/lorahub.py`（CMA-ES） |
| **Arrow 的逐 token 路由** | `system/arrow_runtime.py` —— 資產驗證與 forward hook |
| **生成行為**（prompt、解碼參數、adapter 解析） | `system/inference.py` |
| **路由決策** | `router/core.py` —— Router 類別，路由邏輯唯一所在 |

**新增任務**：樣本放 `dataset/train_data/task{N}_train.json`、adapter 放
`adapter/task{N}/`、重跑 `python scripts/build_router_assets.py --serving-only`。
送審裁決另需在 `assets/unit_descriptions.json` 補該任務所屬單位的說明。

**改動之後**：跑 `python -m unittest discover -s tests` 與三支 `scripts/selftest_*.py`，
全部不需要 GPU 也不需要真實資料。

## 資料格式

樣本檔是 `{"task_key", "task_name", "definition", "instances": [...]}`，每筆 instance
含 `input`、`full_prompt`（完整的任務說明與使用者輸入）、`output`、`instance_id`。
純 array 與 JSONL 也接受。

交付預設是 `data.routing_text=input`，與隨附的路由資產一致。

另一個選項是 `answer_free_full_prompt`：改用移除答案後的完整請求建置 router。本專案
task0–14 的 `input` 欄位沒有完整的任務說明，用完整請求可以補上這段脈絡。訓練樣本的
`full_prompt` 尾端必須精確等於它的 `output`，建置時只移除這一段；測試資料與互動輸入
本來就不含答案，直接使用。

`router_assets_meta.json` 會記錄用的是哪一種 `routing_text`，載入時對不上會直接報錯，
避免兩套路由資產混淆。所以切換模式必須建置到另一個 assets 目錄：

```bash
python scripts/build_router_assets.py \
    --set data.routing_text=answer_free_full_prompt \
    --set paths.assets_dir=assets_afp
```

---

## 延伸閱讀

- [`docs/DELIVERY_ARCHITECTURE_RUNBOOK.md`](docs/DELIVERY_ARCHITECTURE_RUNBOOK.md)
  —— 交付架構、condition 清單、驗收步驟
- [`CONTEXT.md`](CONTEXT.md) —— 共用詞彙與本系統守住的不變式
- [`docs/adr/`](docs/adr/) —— 架構決策與其取捨

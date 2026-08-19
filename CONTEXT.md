# CONTEXT — SMoEA 領域模型

本檔定義 SMoEA 這個 repo 的**共用詞彙與邊界**。程式碼、issue、commit message、
文件在指涉這些概念時，一律使用本檔的用語，不要換同義詞。

架構決策記錄在 `docs/adr/`；操作步驟見 `README.md` 與
`docs/DELIVERY_ARCHITECTURE_RUNBOOK.md`。

---

## 1. 系統一句話

SMoEA 是**線上服務端（consumer）**：收到一筆請求後由 Router 決定要不要交給某個
任務專用 adapter；Router 拒絕時，改由 **Rejection Runtime** 以指定的方法作答。
權重的產生端（producer）是內部的 MoEA-Trainer，**不是線上依賴**——SMoEA 不
import、不 clone、不安裝它，只掛載它的輸出。

```text
answer-free full prompt → Router ──命中(Accept)──→ Task Adapter → 生成
                                └──拒絕(Reject)──→ Rejection Runtime → 生成
```

---

## 2. 詞彙表（Glossary）

### 請求與資料

| 用語 | 定義 | 不要說成 |
|---|---|---|
| **Request / Query** | 使用者送進來的一筆完整任務要求與內容 | prompt（太籠統） |
| **Answer-free full prompt** | 含完整任務說明、**不含本題標準答案**的請求全文。這是 Router 與生成兩端共用的唯一輸入形式 | full_prompt（僅指欄位名） |
| **Routing text** | 建置路由資產時採用的文字來源，`answer_free_full_prompt`（交付預設）或 `input`（重現上游）。資產與設定不一致時必須報錯 | — |
| **Instance** | 資料檔中的一筆樣本，含 `input` / `full_prompt` / `output` / `instance_id` | sample（可，但以 instance 為準） |

### 路由層（`router/`）

| 用語 | 定義 |
|---|---|
| **Router** | 決策層。唯一決策邏輯在 `router/core.py` 的 `Router` 類別（build / save / load / decide / escalate / finalize） |
| **Adapter Slot（task{N}）** | 一個任務身分，對應 `dataset/*/task{N}_*.json` 與 `adapter/task{N}/` |
| **Routing Unit（路由單位）** | 指紋相似度超過 `units.sim_threshold` 而合併的一組 Adapter Slot。Router 先選單位、再選單位內任務 |
| **Fingerprint / Multi-centroid** | 任務的嵌入指紋；一個任務可有多個質心 |
| **四區（Zone）** | `直判路由` / `綠區路由` / `送審` / `紅區拒絕`，由 margin、conformal p 值與詞彙一致性決定 |
| **Escalation（送審）** | 灰區樣本交由裁決 LLM 作是非題判定 |
| **Accept / Reject** | Router 的最終二分結果。**Accept** = 指定某個 Adapter Slot；**Reject** = 沒有可信任務，交給 Rejection Runtime |

### 執行層（`system/`）

| 用語 | 定義 |
|---|---|
| **InferenceEngine** | 唯一的生成執行者（`system/inference.py`）。base model 常駐、權重狀態熱切換 |
| **Base model** | 未套任何 update 的原始模型（預設 `unsloth/Meta-Llama-3.1-8B`） |
| **Task Adapter** | `adapter/task{N}/` 下的 LoRA 權重。Accept 時使用 |
| **Rejection Runtime** | Reject 之後的執行路徑總稱。互動、批次、rejection benchmark **共用同一條路徑**，不維護第二套推論程式 |
| **Rejection Method** | Rejection Runtime 實際採用的方法：`base` / `artifact` / `arrow` / `taskwise_k16_arrow` |
| **非破壞切換（non-destructive switching）** | 切換權重狀態不重載 base model，也不就地改寫 base 權重（dense delta 用 forward hook、LoRA 用 PEFT enable/disable） |

### Artifact（權重產物）

「Artifact」在本 repo 專指**已經產生、可直接掛載使用的權重產物**，不是指任何一組模型檔。

| 用語 | 定義 |
|---|---|
| **Artifact** | 外部掛載的權重產物總稱。Git 不收；由外部路徑提供 |
| **Merged Artifact** | 一個 `prepare/merged_model/` 目錄，內含 `result.json` 契約（schema_version、format、condition_id、run_id、base_model 指紋、inference 設定、weights 的大小與 sha256、modules 清單）。`rejection_method=artifact` 使用它 |
| **Artifact Format** | Merged Artifact 的兩種格式：`dense_delta_v1`（整層 dense 差值，forward hook 疊加）與 `peft_adapter_v1`（LoRA A/B 成對） |
| **Arrow Assets** | Direct Arrow / Taskwise-K16 用的 `prepare/` 目錄（`method.json` + `prototypes.safetensors`，K16 另有 16 個代表 adapter），加上 Direct Arrow 需要的 ordered adapter manifest |
| **Condition** | 產生 artifact 的方法身分（`ta`、`ties_only`、`dare_ties_ta`、`arrow`…），記為 `condition_id` |
| **Run** | 同一 condition 的一次具體產出，記為 `run_id`。**（condition_id, run_id）是 artifact 的身分**，會寫進每筆批次輸出 |
| **Prepared（離線預備）** | artifact 由 producer 事先產生並帶 checksum |
| **Runtime-computed（線上現算）** | 啟動時由原始 adapters 現場推導出來的產物，`run_id` 記為 `runtime-computed`。Direct Arrow 的 prototypes 走這條 |
| **Artifact Registry** | 一份明確宣告可用 artifact 的清單檔（`system.artifact_registry`）。**列舉的唯一來源**——沒被列出的目錄即使存在也不可選。見 [ADR-0001](docs/adr/0001-runtime-artifact-registry.md) |
| **Runtime Merge（線上合成）** | 由使用者明確觸發、以完整 150-adapter 池線上跑 `ta`／`ties`／`dare-ties` 產生新 artifact 的動作。超參數鎖成封板值，產物落地後才可被選用。見 [ADR-0002](docs/adr/0002-online-merge-and-artifact-delivery.md) |
| **Run ID** | artifact 的編號，取產出權重檔 sha256 的前 16 碼。**編號相同保證內容相同**。不從輸入推算——同樣的輸入在不同顯卡上會差 1 ulp（ADR-0002 第 8 節） |
| **1 ulp** | bfloat16 在某個數值量級下能表示的最小間隔。跨硬體重現的驗收標準是「差異不超過 1 ulp」，不是「位元完全相同」 |
| **Adapter Pool** | 一次 merge 的完整輸入來源。本專案只有一個：**pool150**（`pool150_in_domain`，adapter slot task0–task48 與 task50–task150，task49 未指派）。**一律寫 pool150，不要出現 pool50** —— 舊的 50-adapter 池是歷史實驗，不是交付範圍 |
| **Contract Drift（契約分歧）** | producer 寫出的 artifact manifest 與 SMoEA 驗證器要求的格式不一致。成因是同一份契約由兩端各寫一半（producer 寫、SMoEA 讀）而各自演進。解法是讓契約只有一個定義處：writer 與 validator 都放在 SMoEA（ADR-0002） |

### 執行模式

| 用語 | 定義 |
|---|---|
| **Interactive mode** | `main.py --mode interactive`。單筆、逐步印出 Router 判定 |
| **Batch mode** | `main.py --mode batch`。跑資料集測試檔，分三段（全量分區 → 送審打分 → 按任務分組生成），輸出 `results/main_batch_outputs.jsonl`。一次 batch 的所有拒絕樣本使用**同一個** artifact |
| **Rejection benchmark** | `scripts/run_rejection_benchmark.py`。**跳過外層 Router**，直接比較各 rejection method 在固定 15-OOD 上的表現。它不衡量 Router 的 accept/reject 準確率 |

---

## 3. 目前的邊界與不變式（Invariants）

改動時若要違反其中任何一條，必須先寫 ADR。

1. **請求絕不含本題答案。** 生成端看到的 prompt 尾端等於標準答案時直接報錯，不是靜默通過。
2. **Rejection Runtime 只有一條路徑。** 互動、批次、benchmark 都經由
   `InferenceEngine.ensure_rejection() + generate()`。不允許為某個模式另寫一套推論。
3. **Artifact 在接受第一筆 query 之前完成驗證。** schema、base model 指紋、dtype／量化設定、
   檔案大小與 checksum 有任一不符即停止，**不得偷偷退回 base model**。
4. **不自動挑選 artifact。** 系統不掃描 runs 目錄、不自動選「最新的 run」。可選項目來自
   明確宣告的 registry，實際使用哪一個由設定或使用者明確指定（ADR-0001）。
5. **不在 query 時做 merging 或訓練。** 線上合成是使用者明確觸發的獨立動作，產物落地並
   通過驗證後才可被選用；它永遠不發生在回答某一筆 query 的路徑上（ADR-0002）。
6. **artifact 的取得與生成都屬於 setup 階段。** 下載與合成由 `setup_workspace.sh`
   或其底下的腳本執行。**服務執行期不對任何外部服務發出請求**——斷網或遠端故障
   不會變成服務故障，啟動延遲也才可控。
7. **切換權重不重載 base model，也不破壞 base 權重。**
8. **設定只有一處定義。** `configs/default.yaml` 是唯一定義處，指令列以 `--set key=value` 覆蓋單項。
9. **大檔不進 git。** base model、adapters、datasets、merged weights、評測結果都由外部路徑掛載。
10. **OOD alias 只是內部細節。** 內部用 `task9149`，對外輸出的 `source_task` 一律還原成原始 `task149`。

---

## 4. 交付立場（Delivery posture）

這個 repo 會整份交付給公司方。因此：

- 文件、程式碼與設定只描述**交付後成立的事實**，不留實驗過程、暫時性 workaround
  或個人環境路徑。
- 保留給「舊部署相容」的設定或輸出欄位，必須確實有已交付的舊版本在用；否則就是該刪的殘留。
- 每個對外行為都要能由 `python -m unittest discover -s tests` 加上文件覆蓋到。

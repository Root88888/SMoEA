# ADR-0002：線上 Merge 與 Artifact 的產生、落地與交付

日期：2026-08-20
狀態：已接受

## 背景

ADR-0001 讓使用者能在執行期**選定既有的** artifact。第二個需求是**當場生成**一個：
使用者手上沒有預先備好的 merged model 時，SMoEA 應能自行合成。

交付前的立場是「不在 query 時做 merging 或 training」，唯一的線上推導是 Direct Arrow
啟動時算一次 prototypes。同時 `docs/DELIVERY_ARCHITECTURE_RUNBOOK.md` 寫明「SMoEA 不
import、clone 或安裝 MoEA-Trainer」——merge 演算法只存在於 producer 端。

## 決策

### 1. 線上支援的方法

Registry 可選的 rejection method 為 **base / ta / ties / dare-ties / arrow**。其中
只有 **ta、ties、dare-ties** 需要線上 merge；base 與 arrow 沿用既有路徑不變。
需要最佳化的 `adamerging_pp`、需要 dataset-specific adaptation 的 `lorahub` 不做線上生成，
只接受離線預備的 artifact。

### 2. Merge 的輸入是完整的 150-adapter 池

與離線各 condition 的做法一致（全池 merge），產出可直接與離線 artifact 逐位元比對。
不採用「以 Router top-k 候選動態合成」——那會讓每筆 query 的產物都不同，與
ADR-0001 的「整批單一 artifact」相衝突，也讓結果無法重現。

### 3. 超參數鎖成封板值，不開放調整

使用者選方法，不調參。封板值取自 `MoEA-Trainer-delivery/src/moea_repro/conditions.py`：

| condition | 封板超參數 |
|---|---|
| `ta` | `lambda=1.0`、`reduction=mean`、`exact=True`、`sparsification=None` |
| `ties` | `lambda=0.3`、`density=0.2`、`sparsification=magnitude`、`consensus=sum`、`reduction=disjoint_mean`、`tie_policy=zero`、`tile_rows=4` |
| `dare-ties` | `density=1.0`、`lambda=0.25`、`sign_method=total`、`reduction=disjoint_mean`、`tie_policy=zero`、`weights=uniform`、`rescale=True`、`tile_rows=16`、`mask_block_rows=8`、`selection_reference=d1p0_l0p25` |

全部使用 `CANONICAL_SEED = 42`。

**決定性只在同一台機器上成立。** 2026-08-20 的等價驗證（見下節）證明：同一張顯卡跑
兩次得到位元完全相同的結果，但換一款顯卡就會有極少數元素差一個 ulp——浮點加法不
滿足結合律，而 GPU 依卡片型號選用不同的計算核心，累加順序因此不同。

這推翻了「用輸入推算 `run_id`」的做法：那會讓兩份內容不同的檔案拿到同一個編號，
正是編號要防的事。**`run_id` 改取產出檔案本身的 sha256 前 16 碼**——編號相同就保證
內容相同。代價是算完才知道編號；重算的避免改以「比對 adapter 池指紋」達成。

### 4. 只搬 `materialize_*`，不搬實驗生命週期

從 `MoEA-Trainer-delivery` 搬入：

- `src/task_arithmetic/runtime.py::materialize_task_arithmetic`
- `src/adamerging/exact_ties.py`（`trim_elect_weighted_sum`、`compute_global_thresholds`、
  `materialize_exact_ties_sum`）
- `src/dare_ties/runtime.py::materialize_dare_ties`
- `src/adamerging/pool150.py` 的 adapter pool 載入部分
- `src/moea_repro/merged_model.py::write_dense_delta_artifact`

搬入時**一律改用 pool150 命名**：producer 的 `src/dare_ties/pool50.py` 是 50-adapter
池時代留下的檔名，其中的 `dare_ties_tile` 現在服務的是 pool150。交付的程式碼裡不應
出現 `pool50` —— 那個池不在交付範圍內，留著只會讓公司方誤以為有兩種池。

**不搬** 各 runtime 的 `prepare(ctx)` / `infer(ctx)`：那一層牽連 15-OOD 評測、sampling、
run binding 與 `RunContext`，屬於實驗框架。`materialize_*`（純權重運算）與生命週期之間
本來就是乾淨的分界，沿著它切。

**例外：artifact writer 不從 producer 搬，在 SMoEA 內自己寫。**
盤點時發現 `MoEA-Trainer-delivery` 的 `write_dense_delta_artifact` **不寫 `inference`
物件**，而 SMoEA 的 `load_merged_model_artifact` 要求它——因此該版本產出的 artifact
目前一律會被 SMoEA 拒絕。唯一會寫 `inference` 的是 `MoEA-Trainer-smoea-export`，
但那一份缺 `task_arithmetic/` 與 `dare_ties/runtime.py`。兩份都不是超集。

這正是契約由兩端各寫一半所導致的漂移。因此：**SMoEA 已經擁有 validator
（`system/merged_model.py`），writer 就寫在它旁邊、與 validator 對稱**，並以
round-trip 測試（write → load → 驗證通過）鎖死。契約從此只有一個定義處，不可能再漂移。

搬入後 SMoEA 仍**不 import、不 clone、不安裝** MoEA-Trainer：搬的是程式碼副本，不是依賴。
副本必須記錄來源版本，並以「線上產物 == 離線同 condition 產物」的比對測試守住數值一致性。

### 5. 成本（實測參數推算，非估計）

adapter 為 `r=8`、`target_modules=["down_proj"]`、每個 18.8 MB：

- 輸入：150 × 18.8 MB ≈ **2.8 GB** LoRA 因子
- 輸出：down_proj 每層 4096×14336 = 58.7M 參數 × 32 層 = 1.88B → bf16 **每個 artifact 約 3.76 GB**
- 峰值記憶體：producer 的 `materialize_*` 已內建 `tile_rows` 分塊，`tile_rows=16` 時
  每塊約 137 MB。線上 merge 不會爆記憶體。

### 6. 產物落地與交付通道

**身分與取得方式分開處理。**

- **線上只認本地目錄契約。** SMoEA 服務啟動與切換 artifact 時，只讀本地檔案系統上的
  `prepare/merged_model/` 並驗證 checksum。它**不會**在服務期間對外部服務發出請求。
- **落地位置**：新增 `system.artifact_root` 指向 repo 之外的掛載路徑，結構為
  `<artifact_root>/<condition>/<run_id>/prepare/merged_model/`，registry 與其同層。
  大檔不進 git 的不變式維持不變。
- **交付通道**：交付給公司方時，artifact 走**私有 Hugging Face repo**（或公司自有的
  內部物件儲存）傳輸——3.76 GB × N 用 scp 或隨身碟不可行，而 HF 提供版本、checksum
  與存取控制。另備一支 `scripts/fetch_artifact.py`，把遠端 repo 下載成標準的
  `prepare/merged_model/` 目錄並登記進 registry；下載完成後所有驗證與載入路徑完全不變。
- **授權**：dense delta 是 Llama-3.1-8B 的衍生物（只含 down_proj 差值，不含 base model
  本身，沒有 base 無法使用）。散布前需確認 Llama 3.1 Community License 的附隨條款
  （附上授權條文、標示 "Built with Llama"、命名規範）以及上游 Natural Instructions
  資料的授權。**這些是交付前要與公司方確認的項目，不是本 ADR 能單方面決定的。**
  在確認之前，HF repo 一律設為私有。

### 7. 既有 artifact 的相容性遷移

盤點本機發現 15 個由 producer 產生的真實 pool150 artifact（`ta`／`ties_only`／
`dare_ties_ta`／`pico_ta`／`lora_lego` × 3 個 run 集合），全部 150 adapters、32 modules、
bfloat16、每個 3.76 GB。它們**只缺 `inference` 這一個物件**，權重本身完全有效。

遷移不需要猜測：`torch_dtype` 直接取自 manifest 既有的 `modules[].dtype`（即
safetensors 的實際 dtype），`quantization` 固定為 `"none"`（dense delta 不量化）。
以 `scripts/migrate_artifact_manifest.py` 補寫，只改 `result.json`、不動權重檔，
因此 weights 的 sha256 保持不變。dtype 不一致或無法從 manifest 推得時拒絕遷移，
不做預設值填補。

### 8. 等價驗證的結果（2026-08-20）

以完整的 150-adapter 池、RTX4000SFF Ada 顯卡，對 producer 的封板產物逐一比對：

| 方法 | 結果 | 耗時 |
|---|---|---|
| `ta` | 位元完全相同 | 59 秒 |
| `dare-ties` | 位元完全相同 | 3 分鐘 |
| `ties` | 相異 17,811 / 1,879,048,192（約十萬分之一），**全部只差 1 ulp** | 36 分鐘 |

`ties` 的差異分布：17,512 個差 1 ulp、299 個差不到 1 ulp、**0 個差 2 ulp 以上**。
同一張卡重跑兩次得到相同的 sha256，證明本地實作本身是決定性的。

差異只出現在 `ties` 而非 `dare-ties`，指向具體位置：`ties` 用 `einsum` 做 150 路加總
（PyTorch 會轉成矩陣乘法核心，核心選擇與卡片型號有關），`dare-ties` 用 `.sum(dim=0)`。

**驗收標準因此定為「差異不超過 1 ulp」**，而不是「位元完全相同」。1 ulp 是 bfloat16
在該量級能表示的最小間隔，再小的差異這個格式存不下來；超過 1 ulp 就不能用捨入解釋，
應當成搬移錯誤處理。`scripts/publish_artifacts.sh` 與 `scripts/stamp_content_id.py`
都以此為閘門。

（驗證過程中抓到一個真實的搬移錯誤：`ta` 的 `reduction=mean` 係數要除以池大小，
producer 在**呼叫端**做這個換算，只搬 `materialize_*` 會漏掉，導致輸出放大 150 倍。
這說明「呼叫端的參數換算也是實作的一部分」。）

## 後果

- 不變式「不在 query 時做 merging」正式放寬為：**不在 query 時做 merging；生成是使用者
  明確觸發的獨立動作，產物落地後才可被選用。** 生成不會發生在回答某一筆 query 的路徑上。
- SMoEA 的程式碼量顯著增加（約 800–1000 行的權重運算），需要對應的數值一致性測試。
- 公司方仍然只需要這一個 repository。

## 替代方案

- **執行期呼叫 `moea-repro prepare`**：實作最少，但讓 SMoEA 依賴 producer，違反
  「公司只需要這一個 repository」的交付前提。
- **在 SMoEA 內重寫精簡版**：交付邊界一樣乾淨，但數值必須自行驗證與離線一致，
  風險高於直接搬移經過驗證的實作。
- **不落地、只在記憶體使用**：省下每次 3.76 GB 磁碟，但交付給公司方時無法以同一份
  權重重現結果，可追溯性斷裂。

# ADR-0001：Rejection Artifact Registry 與執行期選擇

日期：2026-08-20
狀態：已接受

## 背景

交付前的 SMoEA，拒絕分支是**啟動期一次定案**的：`InferenceEngine.__init__` 讀
`system.rejection_method` 與 `system.rejection_artifact_dir`，把 artifact 讀完並全部
驗證，`_rejection_method` 與 `_merged_artifact` 都是單值。要換一個 artifact 只能重啟
整個程式，連帶重載 8B base model 與路由資產。

這個設計的動機是安全：不掃描 runs 目錄、不自動選「最新的 run」、不在 query 時做
merging，因此每一筆輸出都能追溯到一組明確的 (condition_id, run_id)。

需求是讓使用者能**在執行期選定要用哪個 artifact 作答**，且互動模式與批次模式都適用。
這與上述立場正面衝突：能「選」就意味著要能「列舉」，而列舉正是原設計刻意排除的。

## 決策

引入 **Artifact Registry**：一份明確宣告的清單檔，取代「掃目錄」作為列舉的來源。

1. **Registry 是宣告，不是掃描結果。** 新增設定 `system.artifact_registry`，指向一份
   JSON。每一筆逐項寫明 `id`、`condition_id`、`run_id`、`method`、artifact 目錄路徑。
   SMoEA **只認得 registry 裡列出的項目**；沒被列出的目錄即使存在也不可選。
   「不自動挑選最新 run」的不變式因此完整保留——registry 由人（或 merge 流程）明確寫入。

2. **選擇介面在兩個模式對稱：**
   - 互動模式：`:artifact list` 列出 registry；`:artifact use <id>` 當場切換；
     `:artifact` 顯示目前生效者。
   - 批次模式：`--artifact <id>`，整批固定使用同一個。
   - 兩者都不改變「沒指定時用 `system.rejection_method` 的設定值」這個預設行為。

3. **批次模式的粒度是「整批單一 artifact」。** 一次 batch 的所有拒絕樣本用同一個
   artifact。逐筆切換會讓結果無法互相比較，也會付出反覆載入 3.76 GB dense delta 的代價。

4. **引擎狀態模型改為「一個已驗證的 selection，可換」。** `InferenceEngine` 持有一個
   `RejectionSelection`（已完整驗證、可直接啟用），切換時建新的、成功後才換掉舊的，
   base model 全程不重載。

   **權重一次只掛一組**：dense delta 每個 3.76 GB，同時常駐多組是純浪費，
   因此採 `attach` / `detach` 進出而非多組 hook 共存。切換的成本是重掛一次權重
   （秒級），遠低於重載 8B base model。

5. **驗證時機：** artifact 在**被選用之前**完成完整驗證（schema、base model 指紋、
   dtype/量化、大小與 checksum）。驗證失敗就拒絕切換並保持原本生效的 artifact，
   絕不靜默退回 base model。啟動時是否預先驗證 registry 全部項目，由
   `system.artifact_registry_preload` 決定，預設只驗證即將使用的那一個。

6. **每筆輸出都帶身分。** 互動模式印 `[Rejection] <condition_id>:<run_id>`；批次模式
   的 `rejection_condition_id` / `rejection_run_id` 欄位維持不變。沒有身分顯示的選擇
   功能是不可驗證的。

## 後果

- 放寬的是「不列舉」，**不是**「不驗證」或「自動挑選」。安全立場的核心保持完整。
- `system/rejection.py` 成為兩個模式共用的唯一入口（`run_rejection`），選擇邏輯集中在此。
- Registry 檔本身成為交付物的一部分：公司方拿到的是「有哪些 artifact 可用」的明確宣告。
- 代價：`InferenceEngine` 的狀態機變複雜，dense delta 的 hook 管理需要重寫。

## 替代方案

- **啟動時選一次**：改動最小，但互動模式中途換不了，等於沒解決問題。
- **掃描 artifact 根目錄**：使用上最方便，但直接違反「不自動挑選」的立場，且無法防止
  半成品目錄被誤選。Registry 用一層明確宣告換取這個保證。

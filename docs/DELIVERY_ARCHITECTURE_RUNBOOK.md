# SMoEA rejection runtime 與 benchmark 操作指南

更新日期：2026-08-19

## 1. 交付架構

公司只需要這一個 SMoEA repository。MoEA-Trainer 可以在內部產生 merge／routing
assets，但不是線上依賴；模型、adapters、datasets 與衍生 weights 都由外部路徑掛載。

```text
answer-free full prompt → SMoEA Router
                         ├── 命中 → task adapter
                         └── 拒絕 → Rejection Runtime
                                      ├── base model
                                      ├── selected merged artifact
                                      ├── Direct Arrow (150 experts)
                                      └── Taskwise K16 Arrow (16 experts)
```

互動、SMoEA batch 與 rejection benchmark 都呼叫同一個
`InferenceEngine.ensure_rejection() + generate()`，不維護第二套推論程式。

## 2. 十個方法現在如何使用

| condition | 線上 rejection | 所需外部檔案 |
|---|---|---|
| `base` | 支援 | base model |
| `ta` | 支援 | `prepare/merged_model/` |
| `pico_ta` | 支援 | `prepare/merged_model/` |
| `ties_only` | 支援 | `prepare/merged_model/` |
| `dare_ties_ta` | 支援 | `prepare/merged_model/` |
| `adamerging_pp` | 支援 | 最佳化後的 `prepare/merged_model/` |
| `lora_lego` | 支援 | `prepare/merged_model/` |
| `arrow` | 支援 | ordered adapter manifest + 150 adapters；prepared prototypes 選填 |
| `taskwise_k16_arrow` | 支援 | 完整 `prepare/`（16 代表 adapters + prototypes） |
| `lorahub` | benchmark only | dataset／seed-specific adaptations，沒有通用線上狀態 |

Direct Arrow 不做 merging 或訓練。未提供 `rejection_router_dir` 時，SMoEA 會在第一次
reject 前從 adapters 計算一次 prototypes；正式部署建議提供 `moea-repro prepare arrow`
產生的 `prepare/`，啟動更快且有 checksum。Taskwise K16 的分群與代表 adapters 必須
離線準備，不會在 query 時建置。

## 3. 線上／互動指令

Base model：

```bash
python main.py --mode interactive \
  --set system.rejection_method=base
```

靜態 merged artifact（六個 condition 共用同一介面）：

```bash
python main.py --mode interactive \
  --set system.rejection_method=artifact \
  --set system.rejection_artifact_dir=/data/runs/ties_only/RUN_ID/prepare/merged_model \
  --set system.dtype=bfloat16
```

Direct Arrow，使用已準備的 prototypes：

```bash
python main.py --mode interactive \
  --set system.rejection_method=arrow \
  --set system.rejection_router_dir=/data/runs/arrow/RUN_ID/prepare \
  --set system.rejection_adapter_manifest=/data/pool150/manifest.json \
  --set system.rejection_adapter_root=/data/pool150
```

Direct Arrow，不預先提供 prototypes：

```bash
python main.py --mode interactive \
  --set system.rejection_method=arrow \
  --set system.rejection_adapter_manifest=/data/pool150/manifest.json \
  --set system.rejection_adapter_root=/data/pool150
```

Taskwise K16 Arrow：

```bash
python main.py --mode interactive \
  --set system.rejection_method=taskwise_k16_arrow \
  --set system.rejection_router_dir=/data/runs/taskwise_k16_arrow/RUN_ID/prepare
```

輸入必須是完整任務要求與內容，但不能包含本題答案。多行輸入先輸入 `:paste`，最後以
`:send` 送出。

## 4. Rejection benchmark 的範圍

這個入口跳過 SMoEA 外層 router，直接比較「router reject 後的指定方法」。它讀取固定
15-OOD：5 個 Natural Instructions、5 個 BBH、5 個 MMLU-Pro，完整執行為 4,159 筆。

```bash
python scripts/run_rejection_benchmark.py \
  --benchmark-root /data/moea-benchmark \
  --output-dir results/rejection-base \
  --set system.rejection_method=base \
  --set system.dtype=bfloat16 \
  --batch-size 8
```

先跑三筆 GPU smoke（NI／BBH／MMLU-Pro 各第一筆）：

```bash
python scripts/run_rejection_benchmark.py \
  --benchmark-root /data/moea-benchmark \
  --output-dir results/rejection-smoke \
  --set system.rejection_method=base \
  --set system.dtype=bfloat16 \
  --smoke
```

同一指令可改成上節任一 artifact／Arrow 設定。輸出：

```text
<output-dir>/ni_results.json
<output-dir>/bbh_results.json
<output-dir>/mmlu_pro_results.json
<output-dir>/metrics.json
```

Benchmark 固定使用 bfloat16、8,192 input tokens、1,024 new tokens，超長 prompt 會停止
而不會靜默截斷。本地評分包含 classification accuracy、generation ROUGE-L 與 BLEU。GPT judge 不會被
自動呼叫，`metrics.json` 會明確記錄 `judge: not_run`。

這個 benchmark 不包含：SMoEA accept/reject 準確率、source adapter 訓練、merge 方法的
prepare 成本、LoRAHub adaptation 或 Docker 建置。要驗證完整外層 router，使用原本的
`python main.py --mode batch`。

## 5. 資產產生與 repo 關係

靜態 merge、Arrow prototypes 與 Taskwise K16 assets 仍可由內部 MoEA-Trainer 的
`moea-repro prepare <condition>` 產生。交付時只把其輸出掛載給 SMoEA：SMoEA 不 import、
clone 或安裝 MoEA-Trainer。

Git 不包含 base model、150 adapters、benchmark dataset、merged weights 或正式結果。
`artifact` 與 prepared Arrow 會在接受 query 前驗證檔案大小／checksum；Taskwise K16
也會驗證 16 個代表 adapter。

## 6. 驗收

```bash
python -m unittest discover -s tests -v
python scripts/selftest_main_pipeline.py
```

有正式 GPU 與 assets 時，再分別執行互動 reject、`run_rejection_benchmark.py --smoke`
及完整 4,159 筆 benchmark。單元測試證明切換、asset 驗證與完整 prompt 傳遞；它不取代
真實 8B model 的 GPU 驗收。

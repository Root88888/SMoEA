# SMoEA／MoEA-Trainer 交付架構與操作指令

更新日期：2026-08-19

## 1. 目前架構

目前是兩個獨立 repository，透過一個可攜式 `merged_model/` 資料夾交接；
SMoEA 不 import MoEA-Trainer，也不會在收到 query 時重新 merge。

```text
150 source adapters
        │
        ▼
MoEA-Trainer：moea-repro prepare <condition>
        │
        └── runs/<condition>/<run-id>/prepare/merged_model/
              ├── result.json
              └── dense_delta.safetensors
                            │
                            ▼
answer-free full prompt → SMoEA Router
                            ├── 命中：task adapter
                            └── 拒絕：selected merged model
                                         │
                                         ▼
                              interactive／batch output
```

版本基準：

- `Tincan0325/SMoEA`：`main`，整合基準 `4475923a`。
- `Tincan0325/MoEA-Trainer`：`feature/taskwise-k16-arrow`，`3870f3a9`。

## 2. 方法與 serving 關係

以下六種方法會從 150 adapters 建立可供 SMoEA 載入的
`dense_delta_v1` artifact：

```text
ta  pico_ta  ties_only  dare_ties_ta  adamerging_pp  lora_lego
```

其中 `adamerging_pp` 需要 source calibration prompts，執行固定 500 iterations
的係數最佳化；其餘五種不做梯度訓練，但仍需執行各自的合併計算。

以下四種只參與 Benchmark Harness，沒有單一 SMoEA serving artifact：

```text
base  arrow  taskwise_k16_arrow  lorahub
```

`arrow` 使用 150 adapters routing；`taskwise_k16_arrow` 建立 16 個代表 adapters
後 routing；LoRAHub 產生 dataset／seed-specific adaptations。SMoEA 目前不接受這三種
結果作為 rejection path 的 selected merged model。

## 3. 建立 merged model

```bash
git clone -b feature/taskwise-k16-arrow \
  https://github.com/Tincan0325/MoEA-Trainer.git
cd MoEA-Trainer

python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

先驗證輸入，再建立選定方法：

```bash
moea-repro validate ties_only \
  --adapter-root /data/pool150 \
  --manifest /data/pool150_manifest.json \
  --output-root /data/artifacts

moea-repro prepare ties_only \
  --adapter-root /data/pool150 \
  --manifest /data/pool150_manifest.json \
  --output-root /data/artifacts \
  --cache-root /data/huggingface-cache \
  --device cuda

moea-repro status ties_only --output-root /data/artifacts
```

AdaMerging++ 另加 calibration data：

```bash
moea-repro prepare adamerging_pp \
  --adapter-root /data/pool150 \
  --manifest /data/pool150_manifest.json \
  --source-data-root /data/source-calibration \
  --output-root /data/artifacts \
  --cache-root /data/huggingface-cache \
  --device cuda
```

要連同 NI／BBH／MMLU-Pro benchmark 一起跑，改用 `moea-repro run`，並加上
`--benchmark-root`；`--smoke` 代表每個 suite 只跑第一筆。

## 4. 啟動 SMoEA

```bash
git clone https://github.com/Tincan0325/SMoEA.git
cd SMoEA
```

準備資料：

```text
dataset/train_data/task{N}_train.json   router 建置
dataset/test_data/task{N}_test.json     batch 才需要
adapter/task{N}/                        routed task adapters
/data/artifacts/.../merged_model/       rejection 使用的 selected merge
```

建立環境與 serving router assets：

```bash
bash scripts/setup_workspace.sh
```

完成後先啟用腳本最後印出的環境（預設為此 repo 下的 `.conda/smoea`），再執行下列
interactive／batch 指令。

Production interactive：

```bash
python main.py --mode interactive \
  --set system.merged_model_dir=/data/artifacts/runs/ties_only/<run-id>/prepare/merged_model \
  --set system.merged_model_required=true \
  --set system.dtype=bfloat16
```

輸入必須是完整任務要求與內容，但不能包含答案；多行輸入使用 `:paste`，最後輸入
`:send`。

Batch：

```bash
python main.py --mode batch \
  --tasks 0,1 \
  --limit 5 \
  --set system.merged_model_dir=/data/artifacts/runs/ties_only/<run-id>/prepare/merged_model \
  --set system.merged_model_required=true \
  --set system.dtype=bfloat16
```

結果寫入 `results/main_batch_outputs.jsonl`。每筆會標示 `task_adapter` 或
`merged_model`、condition ID、run ID 與 router 診斷。

## 5. 驗收指令與目前證據

```bash
python -m unittest discover -v
python scripts/selftest_main_pipeline.py
```

- SMoEA 單元測試：21/21 通過。
- 合成 batch 主流程：320 筆完成；311 筆使用 task adapter，9 筆 rejection 使用
  merged model，沒有 `output=null`。
- 真實 GPU 已驗證 `task0 → dense merged → task1 → task0`，dense layer equation 與
  切回 task0 的 logits 誤差皆為 0。
- 上述 batch 是完整主流程的合成模型測試；尚未以正式 selected merge 跑完整真實
  dataset batch。舊 AdaMerging smoke weight 曾產生大量空白文字，不可當成 production
  selected model；正式交付需以最終選定 artifact 重跑 non-empty generation gate。

真實 artifact 切換 smoke：

```bash
python scripts/smoke_merged_model_gpu.py \
  --artifact /data/artifacts/runs/<condition>/<run-id>/prepare/merged_model \
  --adapter-dir adapter \
  --task-a task0 \
  --task-b task1 \
  --prompt-file examples/merged_model_smoke/task2_title_prompt.txt \
  --output results/merged_model_gpu_smoke.json
```

## 6. 交付限制

- Git 不包含 base model、150 adapters、datasets、merged weights 或 benchmark results。
- 公司若自行執行 `prepare`，不需預先取得 derived merged weights；若只執行 SMoEA，
  則需提供選定方法的完整 `merged_model/` 資料夾，不能只複製 weight file。
- 目前 single-repo 與 Docker／OCI image 尚未實作；現階段仍需依兩個 lock 建立兩個
  Python environments。不可把規劃中的 `docker compose` 指令當成已完成入口。

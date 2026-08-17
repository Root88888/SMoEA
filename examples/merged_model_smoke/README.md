# Merged model smoke

程式入口是 `scripts/smoke_merged_model_gpu.py`。它不需要 router assets，會在同一個
base model process 中執行：

```text
task0 adapter → selected merged model → task1 adapter → task0 adapter
```

`task2_title_prompt.txt` 是不含答案的完整 Natural Instructions 測試題；參考答案是
`Llama`。Production acceptance 要求 merged generation 不可為空白。

測試時需另外提供：

- `--artifact`：含 `result.json` 與 weight files 的完整 `merged_model/`。
- `--adapter-dir`：其下可找到 `task0/`、`task1/` 的 adapter root。
- `--output`：測試結果 JSON 的寫入位置。

只有在診斷已知品質不合格的模型、單純驗證 dense loader 數值時，才可加
`--allow-empty-generation`；正式驗收不可加。

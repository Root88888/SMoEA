# SMoEA — Scalable Mixture-of-Experts Adapters

### Overview

- The Router sends each query to one of 150 task-specific LoRA adapters.
- When the Router is not confident which task a query belongs to, it rejects, and the
  query goes to the **merging branch** instead.
- The merging branch has a static side and a dynamic side. The static side loads
  prepared weights (artifacts); the dynamic side merges at inference time. Both are
  described under Merging branch below.

[繁體中文版](README.zh-TW.md)

---

## Architecture

![architecture](docs/architecture.jpg)

### Router

The Router reads the request text only — never the answer — and decides one of two
things:

- **Accept** — the request matches a known task. It is handed to that task's adapter
  under `adapter/task{N}/`.
- **Reject** — no task is a confident match. The request goes to the **merging branch**.

Internally the decision falls into four zones: direct routing, green-zone routing,
escalation to a second model, and red-zone rejection. The first three can produce an
accept; red-zone always rejects, and an escalation that fails to clear its threshold
rejects as well.

Router assets are built offline by `scripts/build_router_assets.py` and live in
`assets/`.

### Merging branch

After a Reject the query goes to the configured merging condition (that is, one of
the methods below). Every condition shares the same **base model**
(`unsloth/Meta-Llama-3.1-8B`).

#### Conditions

**Baselines** — weights are fixed before generation starts. Same input, same output.

| `condition_id` | What it does | Online rejection | Can be built locally |
|---|---|---|---|
| `base` | Nothing layered on top; the base model answers directly. Needs no extra files. | yes | n/a |
| `ta` | Task Arithmetic. Averages all 150 task adapters into one set of weights. | yes | yes |
| `pico_ta` | A low-rank pre-step before Task Arithmetic. | yes | yes |
| `ties_only` | Trims each adapter to its largest-magnitude coordinates, elects a sign per coordinate, and keeps only the contributions that agree. `only` means TIES with no optimisation stage on top — as opposed to `adamerging_pp`. | yes | yes |
| `dare_ties_ta` | Random drop, rescale, sign election, then Task Arithmetic. The sealed drop rate is 0, so the random stage is effectively disabled. | yes | yes |
| `lora_lego` | LoRA-Lego merging. Clusters rank-wise units across the pool. | yes | yes |
| `adamerging_pp` | TIES as a pre-step, then optimisation of the merge coefficients. Learns one coefficient per layer per task, which needs the model loaded and iterations over calibration data — not a pure weight operation. | yes | yes (GPU and `dataset/train_data/`) |
| `lorahub` | LoRAHub. Picks 20 of the 150 adapters and searches their weights with CMA-ES so their weighted sum scores lowest on a few demonstration examples. **Benchmark subject only** — the coefficients are fitted to those specific examples, so no single set of weights is valid for arbitrary online requests. | **no** | yes (GPU and examples) |

Each baseline except `base` needs one weight file. The first six are dense deltas of
roughly 3.76 GB; `lorahub` produces a LoRA at the source rank, which is far smaller.

**Arrow routing** — weights are not merged ahead of time. During generation, each
token is matched against per-expert prototypes and only the closest expert is applied,
independently at every layer. **Different requests take different paths.**

| `condition_id` | Candidates | Files needed |
|---|---|---|
| `arrow` | all 150 adapters | the 150 adapters plus a prepared prototype index |
| `taskwise_k16_arrow` | 16 cluster representatives | 16 representative adapters plus the index (~275 MB) |

`taskwise_k16_arrow` clusters the 150 adapters into 16 groups offline and keeps one
representative per group. It uses far less memory than `arrow` at a coarser routing
granularity. The clustering itself is not performed by this system; it consumes the
prepared assets.

**Both are built for requests that resemble the training tasks without having been
seen.** On free-form input far from that distribution — writing a poem, casual
conversation — the per-layer routing can land on unrelated experts layer after layer,
and output quality drops noticeably. That is a property of the method, not a
misconfiguration. `taskwise_k16_arrow`, with 16 candidates instead of 150, is usually
steadier. Evaluate these two with input shaped like the tasks, or through the 15-OOD
benchmark.

### Which conditions are selectable at runtime

The engine serves four shapes: `base`, `artifact`, `arrow` and `taskwise_k16_arrow`.
`artifact` is generic — it loads any conforming `prepare/merged_model/` directory,
regardless of which merging method produced it. That is why every baseline above is
served through the same code path.

Selectable entries are declared in a **registry** (`artifacts/registry.json` by
default). Only declared entries can be selected; the system never scans directories
and never picks "the latest run" on its own.

### Directory layout

```
main.py                     Entry point: interactive and batch modes
configs/default.yaml        Single source of truth for every setting; override one
                            key at a time with --set key=value

router/                     Routing decision layer
  core.py                     Router class: build / save / load / decide / escalate /
                              finalize — the only place routing logic lives
  config.py data_io.py        Config loading, dataset and embedding-cache I/O
  embedding.py                Query embedding (bge)
  fingerprint.py units.py     Task fingerprints, multi-centroid, routing units
  lexical.py conformal.py     Lexical-agreement signal, conformal calibration, zones
  verifier.py                 Escalation judge (yes/no questions to a second LLM)
  metrics.py                  Scoring for evaluation

system/                     Execution layer, everything after the routing decision
  inference.py                InferenceEngine: resident base model, per-task adapter
                              hot-swap, generation, rejection-condition selection
  registry.py                 Artifact registry: parses and validates the declared
                              list of selectable conditions
  rejection.py                The single entry point into the merging branch, shared
                              by interactive, batch and benchmark
  merged_model.py             The merged-model contract: writer and validator live
                              together so they cannot drift apart
  adapter_pool.py             pool150 manifest parsing, LoRA loading, pool fingerprint
  merging.py                  Local merging for ta / ties_only / dare_ties_ta with
                              sealed hyperparameters
  arrow_runtime.py            Arrow and Taskwise-K16 asset validation and token routing
  benchmark.py                15-OOD loader and local scoring

scripts/
  setup_workspace.sh          One-shot setup: environment, router assets, artifacts
  check_env.py                Environment health check
  build_router_assets.py      Offline router asset build
  selftest_*.py               Three self-tests, no GPU and no real data required
  merge_pool150.py            Build an artifact locally from the 150 adapters
  fetch_artifact.py           Download an artifact from a Hugging Face repo
  push_artifact.py            Upload an artifact to a private Hugging Face repo
  migrate_artifact_manifest.py  Repair manifests written by older producer versions
  verify_against_producer.py  Check a locally built artifact against a reference one
  rename_remote_condition.py  Rename conditions in a remote repo, server-side
  run_rejection_benchmark.py  Fixed 15-OOD benchmark, Router bypassed
  eval_router.py              Router evaluation, three stages
  eval_baseline_*.py          Two routing baselines
  eval_outputs_llm_judge.py   LLM-as-a-judge scoring of batch outputs
  verify_flow_table.py        Independent replay of evaluation results

adapter/task{N}/            LoRA adapters (not in git)
dataset/                    Datasets (not in git)
  train_data/task{N}_train.json     required to build router assets
  test_data/task{N}_test.json       required for batch evaluation
  ood_test_data/task149_test.json   the original OOD task149
assets/                     Router build output; unit_descriptions.json is hand-checked
artifacts/                  Merging-branch weight files and registry.json (not in git)
results/                    Batch outputs and evaluation results
docs/                       Architecture diagram, delivery runbook, ADRs
```

**Artifact** means a directory holding one set of ready-to-serve weights plus a
`result.json` contract recording the base model fingerprint, dtype, file sizes and
checksums, and the adapter pool it was derived from. Its `run_id` is the first 16
characters of the weight file's own SHA-256, so an identical `run_id` guarantees
identical content.

---

## Setup

Clone the repository, then place three things by hand:

| What | Where | Needed for |
|---|---|---|
| Router training samples | `dataset/train_data/task{N}_train.json` | building router assets |
| LoRA adapters | `adapter/task{N}/` | routing and local merging |
| Test data | `dataset/test_data/task{N}_test.json` | batch evaluation only |

Then run setup once:

```bash
bash scripts/setup_workspace.sh                                            # base only
bash scripts/setup_workspace.sh --artifacts fetch --hf-repo <org>/<repo>   # download
bash scripts/setup_workspace.sh --artifacts merge                          # build locally
```

Setup builds the conda environment, installs pinned dependencies, runs the health
check, computes query embeddings, builds router assets, and — when `--artifacts` is
given — prepares the merging-branch weight files under `artifacts/`.

| Flag | Meaning |
|---|---|
| `--artifacts fetch` | Download prepared artifacts from a Hugging Face repo |
| `--artifacts merge` | Build the faster three (`ta`, `ties_only`, `dare_ties_ta`) from the local adapters; use `--methods` for others |
| `--hf-repo <org>/<repo>` | Source repository, required with `fetch` |
| `--artifact-root <path>` | Where artifacts land (default: `artifacts/`) |
| `--methods ta,ties_only` | Restrict which conditions to prepare |

Without `--artifacts` the merging branch offers only `base`; you can add the rest
later with `merge_pool150.py` or `fetch_artifact.py`.

The base model and the escalation judge (~16 GB each) download automatically the
first time `main.py` runs.

Re-running setup is safe: completed steps are skipped.

---

## Inference

### Interactive mode

```bash
conda activate smoea
python main.py --mode interactive
```

No path arguments are needed — the registry defaults to `artifacts/registry.json`.

| Flag | Meaning |
|---|---|
| `--set system.dtype=bfloat16` | Generation dtype; must match the artifact's declared dtype |
| `--set system.rejection_method=<id>` | Condition to use when no registry is present |
| `--set system.verifier_mode=<mode>` | Escalation judge: `resident_4bit` (default, judge resident in 4-bit alongside the generator), `swap` (load and unload in turn — slower, matches the sealed evaluation), `off` (no judge; every escalation rejects) |
| `--no_preload` | Delay model loading until the first request |

In-session commands:

| Command | Effect |
|---|---|
| `:rejection` | Show the condition currently in effect |
| `:rejection list` | List registry entries; `*` marks the active one |
| `:rejection use <id>` | Switch condition. The base model is not reloaded |
| `:paste` … `:send` | Enter a multi-line request |
| `exit` | Leave |

Switching validates the new condition in full — schema, base model fingerprint,
dtype, file size and checksum — **before** it takes effect. If validation fails the
current condition stays in place; the system never silently falls back to `base`.

Example session. **The program prints in Traditional Chinese**; the annotations
below are for readers of this document only.

```text
[Router] 資產已載：150 任務、N 路由單位          ← assets loaded
拒絕分支目前使用 base；:rejection list 看可選項目  ← current condition

> <a complete task instruction and its input, without the answer>
[Router] margin=0.183  p=0.42  詞彙一致✓         ← margin / p-value / lexical agreement
[Router] top-3：task23(sim 0.87)  task10(sim 0.71)  task24(sim 0.66)
[Router] 判定：綠區路由 → task23                  ← decision: green-zone routing
[Output] (2.4s)
<answer>

> :rejection use ties_only
[Rejection] 已切換到 ties_only（ties_only:e3de085e3caeaf23）   ← switched

> <something unrelated to any known task>
[Router] 判定：紅區拒絕 → 進入 rejection inference 分支   ← red-zone rejection
[Rejection] ties_only:e3de085e3caeaf23
[Output]
<answer>
```

The four zone labels are `直判路由` (direct routing), `綠區路由` (green-zone routing),
`送審` (escalation) and `紅區拒絕` (red-zone rejection).

The request must be the complete instruction and input, and must not contain the
answer to itself; the system rejects records whose prompt ends with their own target.

### Batch mode

```bash
python main.py --mode batch --artifact ties_only
```

| Flag | Meaning |
|---|---|
| `--artifact <id>` | Condition for all rejected samples in this run |
| `--tasks 3,7,10` | Restrict to these tasks (default: all) |
| `--limit 50` | At most this many records per task |

One condition is used for the whole run, so results stay comparable. Batch mode needs
`dataset/test_data/` in place.

Results land in `results/main_batch_outputs.jsonl`, one JSON object per record:

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

`routed_to` is the task adapter used on accept, or `null` on reject. The three
`rejection_*` fields record exactly which condition answered, so any output can be
traced back to a specific set of weights. A timestamped copy of the file is written
alongside it.

### Getting the merging-branch artifacts

Two ways, and they produce equivalent results.

**Download from Hugging Face.** Log in once:

```bash
hf auth login
```

```bash
python scripts/fetch_artifact.py --repo <org>/<repo> --condition ties_only --list
python scripts/fetch_artifact.py --repo <org>/<repo> --condition ties_only
```

| Flag | Meaning |
|---|---|
| `--condition <id>` | Which condition to fetch |
| `--list` | List available runs without downloading |
| `--run-id <id>` | Required when the remote holds more than one run |
| `--artifact-root <path>` | Where to place it (default: `artifacts/`) |
| `--register-as <id>` | Registry id to record it under |

Downloads are verified file by file against the sizes and checksums in `result.json`.
When several runs exist for a condition you must name one — the system does not pick
the newest.

**Build offline from the local adapters.** No network required.

```bash
python scripts/merge_pool150.py --method ties_only --adapter-dir adapter
```

| Flag | Meaning |
|---|---|
| `--method {ta,ties_only,dare_ties_ta,pico_ta,lora_lego,adamerging_pp,lorahub}` | Which condition to build |
| `--train-root <path>` | Calibration data for `adamerging_pp` (default `dataset/train_data`) |
| `--examples <file>` | Demonstration examples for `lorahub`; required for it |
| `--run-seed N` | Run seed for `lorahub` (sealed values are 1, 2, 3) |
| `--adapter-dir adapter` | Derive the ordered manifest from `adapter/task{N}/` |
| `--manifest <file>` | Use an explicit ordered manifest instead |
| `--device cpu` | Compute on CPU (default `cuda`) |
| `--artifact-root <path>` | Output root (default: `system.artifact_root`) |
| `--registry <file>` | Registry to record into; `none` to skip |

Hyperparameters are sealed and not adjustable — you choose the method, not the
settings. Of the three, `ties_only` takes markedly longer than the others because it
computes a magnitude threshold per task; budget time accordingly. `ties_only` also
needs about 7.5 GB of GPU memory, so a card of at least 12 GB if you are not using
`--device cpu`. This is a one-time cost; nothing of the sort happens while answering
requests.

Rebuilding is skipped when the same adapter pool has already produced an artifact.
**The seven methods split into two kinds.** The first five (`ta`, `ties_only`,
`dare_ties_ta`, `pico_ta`, `lora_lego`) are pure weight operations: they consume the
adapters and nothing else, and run on CPU. The last two optimise against data and
need a GPU:

```bash
# adamerging_pp: 500 iterations over dataset/train_data to learn per-layer
# coefficients, then the TIES materialisation path
python scripts/merge_pool150.py --method adamerging_pp --adapter-dir adapter

# lorahub: CMA-ES over the given examples (40 generations x 12 population)
python scripts/merge_pool150.py --method lorahub --adapter-dir adapter \
  --examples <examples.json> --run-seed 1
```

`lorahub` examples are `[{"instance_id", "prompt", "output"}, ...]`; the sealed
setting uses five. **The manifest records a fingerprint of those examples**, because
the weights are only meaningful for them: a different set means a different
fingerprint and a different `run_id`.

---

## Benchmark

`scripts/run_rejection_benchmark.py` **bypasses the Router entirely** and runs a
fixed 15-OOD set through one condition. It measures the merging branch alone, not
routing accuracy.

```bash
python scripts/run_rejection_benchmark.py --artifact ties_only \
  --benchmark-root <benchmark data root> \
  --output-dir results/rejection-ties_only \
  --set system.dtype=bfloat16
```

| Flag | Meaning |
|---|---|
| `--artifact <id>` | Condition under test; falls back to `system.rejection_method` |
| `--benchmark-root <path>` | Benchmark dataset root (required) |
| `--output-dir <path>` | Where results are written (required) |
| `--group {all,ni,bbh,mmlu_pro}` | Restrict to one dataset family |
| `--batch-size N` | Generation batch size |
| `--smoke` | Run only the first record of each family, to check the pipeline |

The set is 5 Natural Instructions, 5 BBH and 5 MMLU-Pro tasks, 4,159 records in full.
The benchmark uses the same engine as interactive and batch modes; only the data
source differs and the Router is skipped.

Outputs:

```text
<output-dir>/ni_results.json
<output-dir>/bbh_results.json
<output-dir>/mmlu_pro_results.json
<output-dir>/metrics.json      includes the condition identity under "rejection"
```

Generation is pinned to bfloat16, 8,192 input tokens and 1,024 new tokens. An
over-long prompt stops the run rather than being silently truncated. Local scoring
covers classification accuracy, ROUGE-L and BLEU. The GPT judge is never invoked
automatically; `metrics.json` records `judge: not_run`.

`lorahub` can be a benchmark subject if someone prepares an artifact fitted to this
dataset, but it cannot serve arbitrary online requests.

---

## Evaluation

Three separate things get evaluated. Keep them apart.

### Routing quality

```bash
python scripts/eval_router.py --mode decide   # zone assignment, CPU, minutes
python scripts/eval_router.py --mode score    # escalation judging, GPU, resumable
python scripts/eval_router.py --mode run      # settle
```

| Flag | Meaning |
|---|---|
| `--mode {decide,score,run}` | Stage to run; they must run in this order |
| `--ablate <variant>` | `no_multicentroid`, `no_direct`, `no_lexical`, `gray_reject`, `gray_route` |
| `--fake_verifier` | Skip the real judge model, for pipeline checks |

`no_multicentroid` needs its own asset directory, built once with `k_max=1`:

```bash
mkdir -p assets_ablate_nomc
cp assets/emb_*.npz assets/unit_descriptions.json assets_ablate_nomc/
python scripts/build_router_assets.py \
    --set paths.assets_dir=assets_ablate_nomc --set fingerprint.k_max=1
```

The other four variants run against the main assets. Each variant needs all three
stages, in order:

```bash
for AB in gray_reject gray_route no_lexical no_direct no_multicentroid; do
  python scripts/eval_router.py --mode decide --ablate $AB
  python scripts/eval_router.py --mode score  --ablate $AB
  python scripts/eval_router.py --mode run    --ablate $AB
done
```

Two routing baselines are available for comparison:

```bash
python scripts/eval_baseline_mean_embedding.py --mode eval --tau 0.72
python scripts/eval_baseline_bm25_voting.py    --mode eval --ratio_tau 0.5
python scripts/export_report_data.py           # collate into results/report_data.json
```

`scripts/verify_flow_table.py` replays the evaluation independently and checks it
cell by cell against the main result.

### Output quality (LLM as a judge)

Scores the answers produced by `main.py --mode batch`. Each record's question, gold
answer and model output go to an OpenAI model, which returns 0–5; 4 or above counts
as correct. Requires your own API key.

```bash
export OPENAI_API_KEY=<your key>          # keep it out of files
python scripts/eval_outputs_llm_judge.py
```

| Flag | Meaning |
|---|---|
| `--batch <file>` | Which batch output to score (default: the main file) |
| `--tasks 3,7` | Restrict to these source tasks |
| `--limit 5` | At most this many records per task |
| `--model gpt-5-mini` | Judge model |
| `--resume` | Skip records already scored |
| `--workers 8` | Concurrent requests |
| `--dry_run` | Validate data alignment without calling the API |

Results go to `results/llm_judge_{timestamp}.json`, with the timestamp inherited from
the batch file being scored. `per_path` reports routed samples and each rejection
condition separately.

### Merging-branch quality

Use the benchmark above. It scores locally without an API key.

To check that a locally built artifact matches a reference one:

```bash
python scripts/verify_against_producer.py --method ties_only --adapter-dir adapter \
  --producer <reference merged_model dir> --work-dir <scratch> --device cuda
```

Identical checksums are not required. Floating-point addition is not associative and
GPUs of different models sum in different orders, so a few values may land one step
apart in the storage format. The report counts how many values differ and by how many
steps; anything beyond one step is not rounding and should be treated as a defect.

---

## Testing

### Quickly confirm every method can serve

```bash
python scripts/smoke_rejection_methods.py --set system.dtype=bfloat16
```

Activates each condition in the registry in turn, generates once from each, and prints
a summary. **The Router is skipped entirely**, so this is unaffected by routing-asset
settings; the base model loads once and conditions hot-swap.

`--only base,ties_only` restricts the set; `--prompt "..."` supplies your own text. A
method that fails is reported with its reason but does not stop the others.


Four layers, cheapest first.

```bash
python -m unittest discover -s tests      # 74 tests, no GPU or data needed
python scripts/selftest_core_modules.py   # routing zones, conformal p-values, scoring
python scripts/selftest_main_pipeline.py  # full batch pipeline on synthetic data
python scripts/selftest_end_to_end.py     # asset build through evaluation
python scripts/check_env.py               # package versions, CUDA, disk
```

The first four need neither GPU nor real data and finish in under a minute. They
prove the wiring is correct, not that the answers are good — output quality is only
visible with real weights.

Acceptance on real hardware means: an interactive rejection, `--smoke` on the
benchmark, and the full 4,159-record run.

---

## Where to start reading

When picking this codebase up, find the file that owns what you want to change:

| To change | Look at |
|---|---|
| **Rejection-branch behaviour** | `system/rejection.py` — the single entry shared by interactive, batch and benchmark; one `run_rejection()` |
| **Which methods are selectable** | `system/registry.py` parses and validates the declared list; `select_rejection()` in `system/inference.py` performs the switch |
| **The weight-file contract** | `system/merged_model.py` — writer and validator live in one module so they cannot drift apart |
| **Building a method locally** | `system/merging.py` (five pure weight operations), `system/adamerging.py` (coefficient optimisation), `system/lorahub.py` (CMA-ES) |
| **Arrow's per-token routing** | `system/arrow_runtime.py` — asset validation and the forward hook |
| **Generation behaviour** (prompt, decoding, adapter resolution) | `system/inference.py` |
| **Routing decisions** | `router/core.py` — the Router class, the only place routing logic lives |

**Adding a task:** put samples in `dataset/train_data/task{N}_train.json`, the adapter
in `adapter/task{N}/`, and rerun
`python scripts/build_router_assets.py --serving-only`. Escalation judging also needs
a description for the task's routing unit in `assets/unit_descriptions.json`.

**After any change:** run `python -m unittest discover -s tests` and the three
`scripts/selftest_*.py`. None of them need a GPU or real data.

## Data format

Sample files are `{"task_key", "task_name", "definition", "instances": [...]}` where
each instance carries `input`, `full_prompt` (the complete instruction and user
input), `output` and `instance_id`. Plain arrays and JSONL are also accepted.

The delivered default is `data.routing_text=answer_free_full_prompt`: the router is
built from the complete request with the answer removed, because tasks 0–14 in this
project have no full instruction in their `input` field. A training record's
`full_prompt` must end exactly with its `output`; only that suffix is removed. Test
data and interactive input contain no answer to begin with and are used as-is.

`router_assets_meta.json` records which `routing_text` was used, so two sets of
router assets cannot be confused. To reproduce the upstream router that used the
short `input` field, pass `--set data.routing_text=input` explicitly and build into a
separate assets directory.

---

## Further reading

- [`docs/DELIVERY_ARCHITECTURE_RUNBOOK.md`](docs/DELIVERY_ARCHITECTURE_RUNBOOK.md) —
  delivery architecture, condition inventory, acceptance steps
- [`CONTEXT.md`](CONTEXT.md) — shared vocabulary and the invariants this system holds
- [`docs/adr/`](docs/adr/) — architecture decisions and their trade-offs

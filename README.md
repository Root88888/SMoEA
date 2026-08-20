# SMoEA — Scalable Mixture-of-Experts Adapters
[繁體中文](README.zh-TW.md)
![architecture](docs/architecture.jpg)

## Repository Layout

```
main.py                     System entry point: interactive / batch modes
configs/default.yaml        Single source of truth for all settings (paths, thresholds,
                            models, generation params; override any item with --set key=value)
router/                     Routing decision layer
  core.py                   Router class: build / save / load / decide / escalate / finalize
                            — the only place routing logic lives
  config.py data_io.py      Config loading, data and embedding-cache I/O
  embedding.py              Query embedding (bge)
  fingerprint.py units.py   Task fingerprints, multi-centroid, routing units
  lexical.py conformal.py   Lexical-agreement signal, conformal calibration, four-zone decision
  verifier.py               Escalation LLM yes/no adjudication
  metrics.py                Scoring for evaluation
system/                     Execution layer after routing
  inference.py              InferenceEngine: resident base model, per-task adapter
                            hot-swapping, generation
  rejection.py              Rejection branch (Model Merging) interface — the single entry
                            shared by interactive / batch / benchmark
  registry.py               Parsing and validation of the selectable-rejection-method registry
  merged_model.py           Artifact contract: writer and validator live together
  adapter_pool.py           pool150 manifest parsing, LoRA loading, pool fingerprint
  merging.py                Local builds of ta / ties_only / dare_ties_ta / pico_ta / lora_lego
  adamerging.py             Per-layer coefficient optimization for adamerging_pp
  lorahub.py                CMA-ES weight search for lorahub
  arrow_runtime.py          Asset validation and per-token routing for Arrow / Taskwise-K16
  benchmark.py              15-OOD loading and local scoring
scripts/
  check_env.py              Environment checkup
  selftest_*.py             Three self-tests (no data, no GPU)
  build_router_assets.py    Offline build of routing assets (scans dataset to derive the task set)
  eval_router.py            Router evaluation, three stages
  eval_baseline_*.py        Two baselines
  eval_outputs_llm_judge.py LLM scoring of batch outputs
  merge_pool150.py          Build one condition's artifact from local adapters
  fetch_adapter_pool.py     Fetch the 150-adapter pool from Hugging Face
  push_adapter_pool.py      Upload the adapter pool to Hugging Face
  fetch_artifact.py         Fetch artifacts from Hugging Face
  push_artifact.py          Upload artifacts to a Hugging Face repo
  migrate_artifact_manifest.py  Patch manifests written by older tooling
  verify_against_producer.py    Compare local builds against reference artifacts
  smoke_rejection_methods.py    Confirm every rejection method can serve, one by one
  run_rejection_benchmark.py    Fixed 15-OOD benchmark, bypassing the Router
  verify_flow_table.py      Independent replay verification of evaluation results
  plot_centroids.py         Centroid-structure plots
dataset/                    Data (not in git); create dataset/ with dataset/train_data/
                            and dataset/test_data/
  train_data/task{N}_train.json
  test_data/task{N}_test.json
adapter/task{N}/            LoRA adapters (not in git): downloaded by setup_workspace.sh
                            from Hugging Face. If you supply your own, put task{N}
                            directly under adapter/; if a task has several checkpoint-*/
                            dirs the latest is used automatically
assets/                     Routing build outputs; unit_descriptions.json is the
                            hand-curated unit description book
artifacts/                  Rejection-branch artifacts and registry.json (not in git)
results/                    Evaluation and batch outputs
docs/                       Architecture figures and documents
```

## Setup and Running

1. git clone
```bash
   git clone https://github.com/Root88888/SMoEA.git && cd SMoEA
```

2. Put the files in place
   - Task samples: train files go to `dataset/train_data/`, test files to
     `dataset/test_data/` (file names `task{N}_train.json` / `task{N}_test.json`)

     ```bash
        cd ./
        pip install gdown

        mkdir -p dataset/train_data dataset/test_data

        gdown 1AsJwaqQ3AXmPT8TpAxOyvCPbyHtCi1lG -O dataset/train_data/train_data.zip
        gdown 1aiT9r9v2tyH-0cdf_F6zhfEvYF0mZ2tM -O dataset/test_data/test_data.zip

        python3 -m zipfile -e dataset/train_data/train_data.zip dataset/train_data/
        python3 -m zipfile -e dataset/test_data/test_data.zip  dataset/test_data/
     ```
   - Adapters: nothing to do. Step 3 downloads the 150-adapter pool
     (~2.7 GB) from Hugging Face. To supply your own instead, put one
     directory per task as `adapter/task{N}/` and pass `--no-adapter-fetch`
     in step 3.

Important note: in this system the original OOD task149 is called task9149 to
distinguish it from the ID task149. After the files are in place, please rename
the OOD task149 file to `task9149_test.json` manually.

3. One-shot setup

   The OOD tasks are declared in `dataset/ood_tasks.txt`; edit it if your
   numbering differs.

```bash
   bash scripts/setup_workspace.sh                                          # base only
   bash scripts/setup_workspace.sh --artifacts fetch --hf-repo <org>/<repo> # also download artifacts
   bash scripts/setup_workspace.sh --artifacts merge                        # also build locally
   bash scripts/setup_workspace.sh --no-adapter-fetch                       # bring your own adapters
```

   `--artifacts` decides which rejection methods become available; without it
   only `base` is available. See "Rejection Branch" below.

   The script automatically performs: file checks (missing items are reported
   explicitly), conda environment creation and dependency install (10–20 min
   the first time), environment checkup, **adapter-pool download** (~2.7 GB,
   skipped when `adapter/` already holds 150 tasks), query-embedding
   computation and routing-asset build (a few GPU minutes the first time).
   It ends with an "all ready" message; if it stops midway, follow the hint
   and rerun — completed steps are skipped automatically.

   The pool comes from `Tincan0325/smoea-adapter-pool150`; `--adapter-repo`
   points elsewhere. Every file is checked against the sha256 recorded in the
   pool manifest, and files already present and matching are skipped, so an
   interrupted download only refetches what is missing. To run that step on
   its own:

```bash
   python scripts/fetch_adapter_pool.py --repo Tincan0325/smoea-adapter-pool150 --list
   python scripts/fetch_adapter_pool.py --repo Tincan0325/smoea-adapter-pool150
```

4. Interactive, one query at a time
```bash
   conda activate smoea
   python main.py --mode interactive
```

   The first run automatically downloads the generation and adjudication
   models (~16 GB each). Typing an in-task query should show the Router
   decision and the model answer; typing unrelated text should show the
   message for entering the Model Merging branch (i.e. the call site of
   `system/rejection.py`).

   The rejection method can be switched within a session without reloading
   the underlying model:

   ```text
   :rejection              show which one is currently in use
   :rejection list         list the selectable entries
   :rejection use ties_only  switch
   ```

5. Batch
```bash
   python main.py --mode batch                        # --tasks a,b runs only tasks a,b; --limit n takes the first n test samples per task; omit both to run everything
   python main.py --mode batch --artifact ties_only   # all rejected samples in the batch share the specified method
```

   Per-sample results (Router diagnosis and model output) land in
   `results/main_batch_outputs.jsonl`. Every rejected sample additionally
   records `rejection_condition_id` and `rejection_run_id`, traceable to the
   exact artifact.

From then on, each new shell only needs `conda activate smoea`; steps 2 and 3
are one-time work.

## The Adapter Pool (pool150)

One pool serves both halves of the system: the Router hot-swaps a single
adapter per routed query, and the rejection branch merges the whole pool.

| | |
|---|---|
| Adapters | 150 (`task0`–`task48`, `task50`–`task150`; slot 49 unassigned) |
| Base model | `unsloth/Meta-Llama-3.1-8B` |
| Rank (`r`) | 8, `lora_alpha` 16, `lora_dropout` 0.1 |
| `target_modules` | `down_proj` only — no attention modules |
| Per adapter | ~18 MiB (`adapter_model.safetensors`) |
| Whole pool | ~2.64 GiB |

The configuration above is identical across all 150; a pool with mixed LoRA
settings cannot be merged and is rejected at load time.

`scripts/fetch_adapter_pool.py` writes `adapter/pool150_manifest.json`
alongside the adapters. That manifest is the ordered pool definition —
**the order is by ascending task number and it affects merge results**, so
merges built from it are reproducible.

Distributing your own pool works the same way in reverse:

```bash
hf auth login
python scripts/push_adapter_pool.py --repo <org>/<repo> --dry-run
python scripts/push_adapter_pool.py --repo <org>/<repo>
```

Only `adapter_config.json` and `adapter_model.safetensors` are uploaded;
training residue (optimizer state, RNG state, per-checkpoint tokenizer
copies) is left behind — it is roughly half the size of a raw training
output directory and neither serving nor merging reads it. The manifest is
uploaded **last**, so a repo left behind by a failed upload has no manifest
and the download side refuses it outright rather than handing out half a pool.

## The Router

The Router's job: for every incoming query, pick which of the 150 task
adapters should answer it, or decide to reject (rejected queries go to the
rejection branch).

The core of the decision is the **conformal p-value** — the query's
similarity lead margin over tasks is ranked within its group in an offline
calibration score bank, converting it into a probability of "how rare this
lead is among known tasks of the same kind", and the zone then decides the
outcome:

| Order | Condition | Outcome |
|---|---|---|
| Direct | margin > 0.10 | route immediately |
| Red zone | p < 0.02 | reject |
| Green zone | p ≥ 0.10 and the embedding top-1 unit agrees with the TF-IDF lexical-fingerprint top-1 | route |
| Escalate | everything else (p in the gray band, or the two signals disagree) | slow path |

**Fast and slow paths**: the decision above (embedding + similarity +
p-value + lexical check) is the fast path, about 50 ms per query, and most
samples are settled there; escalated samples enter the slow path — the
adjudication LLM (Llama-3.1-8B) answers one yes/no question per top-3
candidate task, "is this query an instance of that task?". If the highest
confidence is ≥ 0.5 the query is routed to that candidate; if none of the
three looks right it is rejected. The LLM step takes about 220 ms, roughly
270 ms in total including the fast path.

**Routing units**: twin tasks with fingerprint similarity > 0.97 are bound
into one routing unit at build time (150 tasks → 146 units, only two triplet
groups), every other task is its own unit. Routing first picks a unit, and as
the final step restores to the highest-similarity member task within the
unit. Evaluation therefore reports two accuracies, task-level and unit-level:
when a twin task is sent to another task in the same unit it counts as
correct at unit level and wrong at task level.

**Offline build outputs**: task fingerprints and multi-centroids (k-means;
multi-modal tasks expand into several centroids), the routing-unit table,
the conformal calibration score bank (split 80/20, isolated from the
fingerprint pile), TF-IDF lexical fingerprints, and the task description
book (the adjudication basis for the slow path). Adding a task only requires
rerunning the build.

Full decision flowchart: [docs/router_flowchart.jpg](docs/router_flowchart.jpg)

## Router Evaluation on the Full Test Set

### 1. Main evaluation

```bash
# 1a. Zone assignment (CPU, minutes): all test samples into four zones, escalation queue produced
python scripts/eval_router.py --mode decide 2>&1 | tee results/eval_decide.txt

# 1b. Escalation scoring (GPU, hours; rerun resumes automatically): the adjudication LLM answers three yes/no questions per queued sample
python scripts/eval_router.py --mode score 2>&1 | tee results/eval_score.txt

# 1c. Settlement
python scripts/eval_router.py --mode run 2>&1 | tee results/eval_run.txt
```

### 2. Ablation asset preparation (no-multi-centroid variant; one-time)

```bash
mkdir -p assets_ablate_nomc
cp assets/emb_*.npz assets/unit_descriptions.json assets_ablate_nomc/
python scripts/build_router_assets.py \
    --set paths.assets_dir=assets_ablate_nomc --set fingerprint.k_max=1
```

### 3. Five ablation variants

```bash
for AB in gray_reject gray_route no_lexical no_direct no_multicentroid; do
  python scripts/eval_router.py --mode decide --ablate $AB
  python scripts/eval_router.py --mode score  --ablate $AB
  python scripts/eval_router.py --mode run    --ablate $AB
done
```

### 4. Two baselines

```bash
python scripts/eval_baseline_mean_embedding.py --mode eval --tau 0.72
python scripts/eval_baseline_bm25_voting.py    --mode eval --ratio_tau 0.5
```

### 5. Export

```bash
python scripts/export_report_data.py        # writes results/report_data.json
```

## Rejection Branch (Model Merging)

Queries rejected by the Router come here. All methods share the same base
model (`unsloth/Meta-Llama-3.1-8B`); they differ only in what is stacked on
top. Stacking never modifies the base model, so switching between methods
does not reload it.

> `base` is the name of one of the methods, meaning "stack nothing"; the
> base model is the shared Llama-3.1-8B. They are different things.

### Available methods

**Baselines** — weights are fixed before generation starts; the same input
gives the same output.

| `condition_id` | What it does | Selectable online | Local build |
|---|---|---|---|
| `base` | Stack nothing; the base model answers directly | yes | n/a |
| `ta` | Task Arithmetic: plain average of the 150 adapters | yes | yes |
| `pico_ta` | A low-rank pre-processing step before Task Arithmetic | yes | yes |
| `ties_only` | Trim to the largest-magnitude coordinates, pick a sign per coordinate, keep only agreeing contributions. `only` means no optimization follows, distinguishing it from `adamerging_pp` | yes | yes |
| `dare_ties_ta` | Random drop, rescale-and-compensate, sign voting, then Task Arithmetic. The sealed drop rate is 0 | yes | yes |
| `lora_lego` | LoRA-Lego: cluster the per-rank units of the whole pool | yes | yes |
| `adamerging_pp` | TIES as pre-processing, then optimize the merge coefficients | yes | yes (GPU and `dataset/train_data/` required) |
| `lorahub` | Pick 20 of the 150 and search weights with CMA-ES. The coefficients are bound to specific demonstration samples, so it can **only be a benchmark subject** | no | yes (GPU and demonstration samples required) |

**Arrow routing** — no pre-merging. At generation time each token is compared
against prototypes and only the closest expert is applied, independently per
layer, so different requests take different paths.

| `condition_id` | Candidates | Files needed |
|---|---|---|
| `arrow` | all 150 adapters | those 150 adapters plus a precomputed prototype index |
| `taskwise_k16_arrow` | 16 cluster representatives | the 16 representative adapters and index (~275 MB) |

These two are designed for requests that look like the training tasks but
were never seen. For free-form prompts far outside that distribution (poetry,
chit-chat and the like), per-layer independent routing may pick unrelated
experts at different layers and output quality degrades visibly — that is a
property of the method, not a misconfiguration. To evaluate them, use inputs
close to the task style or run the 15-OOD benchmark.

### How selection works

Selectable entries are declared in `artifacts/registry.json` (the default in
`configs/default.yaml`). **Only entries listed there can be selected** — the
system does not scan directories and never picks the latest run by itself.
See [`examples/artifact_registry.example.json`](examples/artifact_registry.example.json)
for the format.

Before switching, the target artifact is fully checked (format, base-model
fingerprint, dtype, file sizes and hashes); if the check fails, the currently
active one stays in effect — there is no silent fallback to `base`.

A missing registry file is not an error; there are simply no selectable
entries, and the rejection branch uses `system.rejection_method` (default
`base`).

The three entry points share the same usage:

```bash
python main.py --mode interactive                       # use :rejection use <id> within the session
python main.py --mode batch --artifact ties_only
python scripts/run_rejection_benchmark.py --artifact ties_only \
    --benchmark-root <data root> --output-dir results/rejection-ties_only
```

### Where the artifacts come from

Every method except `base` needs one artifact (a dense delta of about
3.76 GB; `lorahub` is LoRA and much smaller). Two equivalent routes, both
executed only during preparation — **the serving runtime never makes
outbound connections**.

**Download:**

```bash
hf auth login
python scripts/fetch_artifact.py --repo <org>/<repo> --condition ties_only --list
python scripts/fetch_artifact.py --repo <org>/<repo> --condition ties_only
```

`--list` only shows the available versions without downloading. When a
condition has several versions you must pick one with `--run-id`; the system
never picks the latest by itself. After download every file's size and hash
are verified.

**Local build:**

```bash
python scripts/merge_pool150.py --method ties_only
```

`--method` is one of the seven. The first five are pure weight arithmetic
and run on CPU; `adamerging_pp` and `lorahub` optimize against data and
require a GPU:

```bash
python scripts/merge_pool150.py --method adamerging_pp
python scripts/merge_pool150.py --method lorahub \
    --examples <demonstrations.json> --run-seed 1
```

Adapters are taken from `system.adapter_dir` in the config (the same value
the Router uses), so no extra flag is needed; use `--adapter-dir` or
`--manifest` only when the adapters live elsewhere. A local build needs the
complete pool of 150 — see "The Adapter Pool" above.

Hyperparameters are sealed; users pick a method, not knobs. A build over the
same adapter pool is never recomputed. An artifact's run id is the first 16
hex chars of its own content hash, so **equal ids guarantee equal content**.

If an artifact produced by older tooling lacks manifest fields, patch it
once (manifest only; weights untouched):

```bash
python scripts/migrate_artifact_manifest.py --scan <artifact dir> --dry-run
python scripts/migrate_artifact_manifest.py --scan <artifact dir>
```

### Confirm every method can serve

```bash
python scripts/smoke_rejection_methods.py --set system.dtype=bfloat16
```

Enables every condition in the registry one by one, generates once each, and
prints a summary table. **The Router is bypassed**, so routing-asset settings
have no effect. `--only base,ties_only` restricts to a subset.

## Benchmark: Bypass the Router, Test One Method

```bash
python scripts/run_rejection_benchmark.py --artifact ties_only \
  --benchmark-root <benchmark data root> \
  --output-dir results/rejection-ties_only \
  --set system.dtype=bfloat16 --smoke
```

Fixed 15-OOD: 5 Natural Instructions, 5 BBH, 5 MMLU-Pro; the full run is
4,159 samples. `--smoke` runs only the first sample per family, to confirm
the pipeline works. It measures the rejection branch itself, not routing
accuracy; it uses the same engine as interactive and batch modes, only with
a different data source and the Router bypassed.

Outputs are `ni_results.json`, `bbh_results.json`, `mmlu_pro_results.json`
and `metrics.json` (whose `rejection` field records the identity of the
method under test). Local scoring includes classification accuracy, ROUGE-L
and BLEU; the GPT judge is never called automatically.

Before a full benchmark run, execute
`python scripts/map_ood_aliases.py --dataset-dir dataset` once to create the
internal alias for OOD `task149`.

## Batch Output Evaluation (LLM-as-a-judge)

Batch inference outputs are graded by an OpenAI model: for each sample the
question, reference answer and model output are given to the LLM, which
returns a score of 0–5 (5 = fully correct), is_correct (score ≥ 4), and a
short comment. Bring your own OpenAI API key.

First produce a batch inference result:
`results/main_batch_outputs.jsonl` or
`results/main_batch_outputs_{timestamp}.jsonl`.

```bash
# the key lives only in the current shell; never write it into any file
export OPENAI_API_KEY=<your key>

# grade the latest batch output
python scripts/eval_outputs_llm_judge.py

# grade a specific historical batch output
python scripts/eval_outputs_llm_judge.py --batch results/main_batch_outputs_{timestamp}.jsonl
```

Optional flags, combinable:

- `--tasks 3,7`　grade only these source tasks (default: all)
- `--limit 5`　at most this many samples per task (for small tests)
- `--batch results/main_batch_outputs_{timestamp}.jsonl`
  which inference result to grade (default: the main file
  `results/main_batch_outputs.jsonl`)
- `--model gpt-5-mini`　judge model (default gpt-5-mini)
- `--resume`　resume from a checkpoint (skips samples already graded)
- `--workers 8`　concurrent requests

Output: `results/llm_judge_{timestamp}.json`; the timestamp inherits the
production time of the graded batch file.

## Where to Start

- **Rejection branch (Model Merging)**: entry point `system/rejection.py` —
  the single path shared by interactive, batch and benchmark. Which methods
  are selectable: `system/registry.py`; the artifact contract (writer and
  validator): `system/merged_model.py`; local builds:
  `system/merging.py`, `system/adamerging.py`, `system/lorahub.py`.
- **Generation behavior** (prompt, decoding params, adapter resolution):
  `system/inference.py`.
- **Adapter pool**: the contract (manifest parsing, LoRA loading, pool
  fingerprint) is `system/adapter_pool.py`; the delivery channel is
  `scripts/fetch_adapter_pool.py` / `scripts/push_adapter_pool.py`.
- **Adding a task**: put samples in `dataset/`, the adapter in
  `adapter/task{N}/`, rerun `build_router_assets.py` — done (escalation
  adjudication additionally needs the task's unit description added to
  `assets/unit_descriptions.json`).
- **After any change**: run `python -m unittest discover -s tests` and the
  three `scripts/selftest_*.py`; none of them needs a GPU or real data.
- **Design background**: `CONTEXT.md` holds the shared vocabulary and
  invariants; `docs/adr/` records the two major decisions and their
  trade-offs.

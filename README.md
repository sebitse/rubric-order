# Calibration-order dependence in LLM judges

The reproducibility-controlled v2/v3 workflow is documented in
[`docs/final_experiment_protocol.md`](docs/final_experiment_protocol.md). It starts with the
separate judge-validation gate; do not start `pilot-002` until its artifact says
`JUDGE_APPROVED`. Inference notebooks are saved with execution disabled and a zero call budget.

`pilot-002` completed rubric generation but stopped before judging after a manual gate found
task-specific thematic drift. Those rubrics are frozen evidence: they are neither regenerated nor
edited. Notebook 06 prepares their exploratory downstream audit, notebook 07 prepares the paired
fixed-inventory mitigation pilot as deferred future work, and notebook 08 describes the simplified
final primary design only. Their saved defaults are `EXECUTE_MODEL_CALLS=False` and
`MAX_CALLS=0`.

This project tests whether a frozen rubric updater produces different helpfulness rubrics when it
sees the same labeled examples in different orders, and whether those rubrics change scores or
within-prompt rankings from an independent frozen judge. No model training or weight updates occur.

The conditions are `static` (the fixed initial rubric), `batch` (all ten calibration examples in one
updater call), and `sequential` (one rubric update per example). Every block contains two examples at
each helpfulness score 0–4 in ascending, descending, random, and exact reverse-random orders.

## Notebook workflow

The v1 workflow remains in notebooks 00–04. V2 adds `judge_validation.ipynb` and
`05_readiness_and_main.ipynb`. V3 preparation adds notebooks 06-08. Each notebook reads saved run
artifacts and can be restarted independently; later notebooks do not rely on variables left by an
earlier kernel.

1. Activate the `da` Conda environment.
2. Install this package editable.
3. Start Jupyter from the repository root.
4. Run notebooks `00` through `04` in order.
5. Run the one-trajectory updater canary before expanding to the complete pilot.
6. Inspect the stored rubrics, calls, predictions, tables, and figures under `runs/<run_id>/`.

On this Windows machine the active interpreter is
`C:\Users\Sebastian\miniconda3\envs\da\python.exe`. It is not hard-coded in the package or
notebooks.

```powershell
conda activate da
python -m pip install --no-deps --no-build-isolation -e .
python -m jupyter lab
```

Open `notebooks/00_setup_and_health.ipynb` first. Jupyter may be started from either the repository
root or `notebooks/`; the notebooks locate the project with `pathlib.Path`.

### Notebook sequence

- `00_setup_and_health.ipynb` loads the configuration, reports dependency versions, and optionally
  lists local Ollama models without pulling or running them.
- `01_data_and_protocol.ipynb` validates the data, creates grouped partitions and balanced blocks,
  verifies order invariants, writes the manifest, and estimates calls.
- `02_rubric_generation.ipynb` plans updater work, runs the first sequential canary when explicitly
  enabled, and inspects rubric trajectories and call records.
- `03_judging.ipynb` validates stored rubrics, plans held-out judging, and reports incremental,
  cached, failed, and remaining work.
- `04_analysis.ipynb` refuses incomplete experiments by default, then exports metrics, grouped
  bootstrap intervals, text diagnostics, and figures without contacting Ollama.

## Execution safeguards

The two inference notebooks have editable controls near the top:

```python
EXECUTE_MODEL_CALLS = False
MAX_CALLS = 0
SELECTED_METHODS = ["sequential"]
SELECTED_BLOCKS = [0]
SELECTED_ORDERS = ["random"]
SELECTED_REPLICATES = [0]
```

Committed defaults make zero Qwen or Llama calls, including after “Restart Kernel and Run All.” A
real stage requires `EXECUTE_MODEL_CALLS = True` and `MAX_CALLS` at least as large as the displayed
plan. The first updater canary is one sequential trajectory: block 0, random order, replicate 0, and
10 calls. The API refuses a smaller call limit. Run the matching judge canary only after inspecting
the final canary rubric.

Successful responses are content-addressed in SQLite and reused after interruption. Repeat controls
have distinct replicate IDs and cache namespaces. Schema failures may be retried at most twice.
Qwen thinking is disabled where supported, updater output is a validated complete rubric, and judge
output is limited to `{"score": n}`. Actual Ollama input/output token counts and latency are stored.

## Dataset configuration

`configs/pilot.yaml` and `configs/main.yaml` remain the preserved v1 definitions.
`configs/judge_validation.yaml`, `configs/pilot_v2.yaml`, and `configs/main_v2.yaml` define the
gated v2 workflow. `configs/pilot_002_downstream_audit.yaml`, `configs/pilot_v3.yaml`, and
`configs/main_v3.yaml` preserve the frozen-rubric audit and constrained extensions. The recommended
current-paper configuration is `configs/main_final_primary.yaml`; pilot-003, inventory-v3, and the
constrained ablation are optional deferred work and are not part of current primary results.
The exact launch and resume commands are in
[`docs/final_primary_launch.md`](docs/final_primary_launch.md).
The obvious settings
cell in every notebook selects the configuration, local dataset, and run ID. Small dotted-key runtime
overrides may be supplied in `RUNTIME_OVERRIDES`; effective values and overrides are recorded in the
resolved configuration and run manifest.

Supported inputs are a Hugging Face dataset saved with `save_to_disk`, a directory containing
`train` and optional `validation` files, or a Parquet, JSON, JSONL, or CSV file. Gzip-compressed
JSON/JSONL/CSV is accepted. Source fields are mapped in YAML to `prompt`, `response`, and an integer
helpfulness label from 0 through 4. An official validation split is used when present; otherwise the
split is deterministic and prompt-grouped.

The standard-library core handles JSON, JSONL, CSV, sampling, caching, and metrics. Optional adapters
and plotting dependencies can be installed when absent:

```powershell
python -m pip install -e ".[parquet,plots]"
python -m pip install -e ".[huggingface,plots]"
python -m pip install -e ".[dev,plots]"
```

## Outputs and metrics

Each run stores its manifest and resolved configuration, prepared rows and blocks, intermediate and
final rubric JSON, incremental predictions, the SQLite call store, dry-run plans, CSV metric tables,
JSON summaries, and figures under `runs/<run_id>/`. Existing runs are not silently replaced.

Primary outcomes are Calibration-Order Instability, Score Change Rate, Mean Score Range, Pairwise
Ranking Flip Rate, and tie-change rate. Agreement uses quadratic weighted kappa, Spearman
correlation, and mean absolute error. Sequential-minus-batch differences are paired on blocks and
held-out examples. Confidence intervals jointly resample calibration blocks and prompt groups.
Rubric length, criterion count, criterion-name Jaccard similarity, and TF-IDF cosine similarity are
secondary diagnostics.

Preserve the resolved configuration, dataset fingerprint, prompt-template hashes, model settings,
rubric hashes, and SQLite call store for the paper.

## Optional CLI compatibility

The CLI remains as a thin wrapper over `rubric_order.api` for existing scripts. It is not required by
the notebooks. Model commands are still dry runs unless `--execute` and `--max-calls` are supplied.

```powershell
python -m rubric_order.cli estimate --config configs/pilot.yaml --data-path HelpSteer2
python -m rubric_order.cli smoke-test --output runs/smoke
```

The preparation CLI also exposes `provenance-gate`, `estimate-downstream`, `plan-downstream`,
`run-downstream`, `analyze-downstream`, `downstream-readiness`, `export-drift`, `validate-drift`,
`freeze-drift`, `estimate-pilot-v3`, `plan-pilot-v3`, `pilot-v3-readiness`,
`estimate-main-v3`, `plan-main-v3`, `main-v3-readiness`, `resolve-lineage`, `audit-lineage`,
`audit-downstream`, `estimate-final-primary`, `freeze-pre-main`, and `pre-main-readiness`. Only
commands explicitly supplied with
`--execute` can invoke a model. Downstream dry-runs are written outside the frozen source run.

## Verification

```powershell
ruff format --check .
ruff check .
python -m pytest
```

Notebook validation and execution tests use `nbformat` and `nbclient`. They execute synthetic data
with both inference switches disabled and verify that no Ollama call database is created.
#   r u b r i c - o r d e r  
 
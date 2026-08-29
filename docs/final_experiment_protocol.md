# Final experiment protocol

## Research design

Question: when calibration evidence is fixed, how much does its presentation order change a rubric induced by a local LLM and the scores subsequently produced by a frozen judge?

Hypotheses:

- Sequential rubric updating exhibits measurable calibration-order variation.
- Batch induction may exhibit different order variation despite receiving the same evidence.
- Observed between-order variation should be interpreted relative to same-order repeat variation, not assumed to be deterministic.

The causal variable is calibration order. Each block contains the same ten processed examples in four orders: ascending label, descending label, seeded random, and the exact reverse of that random order. Text preprocessing happens once before permutation. The static condition uses the immutable initial rubric; batch sees all ten items in one call; sequential updates a canonical rubric after each item.

The analysis unit for order metrics is a held-out response within a calibration block. The two responses associated with one held-out prompt remain one cluster for resampling and ranking comparisons. Blocks are the calibration-evidence clusters.

## Frozen v2 contract

The rubric always retains, byte-for-byte at the object-field level, the three core criteria `Task fulfillment`, `Correctness`, and `Clarity and relevance`, the title, schema, and the five 0–4 anchors. At most two domain-general adaptive criteria are allowed. Adaptive criteria are sorted by normalized name before reuse or judging. Structural validation is distinct from the lexical drift audit; drift flags require inspection and are not semantic verdicts.

Qwen updater settings are `num_ctx=16384`, `num_predict=650`, temperature 0, seed 42, and `think=false`. Calibration prompt and response limits are 1,200 and 2,400 characters, respectively. Truncation is deterministic, recorded, and identical in every condition. Input tokens are conservatively estimated as `ceil(characters / 3)` unless an explicit local tokenizer is supplied. Preflight reserves 650 output tokens and a 1,024-token safety margin. Postflight rejects exact context-limit contact or observed headroom below 256 tokens. Changing context settings requires a new run ID.

## Judge validation

Judge validation is separate from rubric induction and always uses the canonical initial rubric. From HelpSteer2 validation, after excluding `pilot-001`, it selects:

- judge-dev: 50 responses, exactly 10 per human score;
- judge-validation: 100 responses, exactly 20 per human score.

Every selected response has a distinct prompt, and the two subsets are prompt-disjoint. Only ID and hash records are stored in concise subset artifacts; content is reloaded and hash-checked when needed.

On dev, Llama is evaluated with exactly `judge_absolute_v2` and `judge_anchor_first_v2`. Selection is descending QWK, descending Spearman, ascending MAE, then stable candidate name. Qwen may repeat the dev comparison only after an explicit user choice. Untouched validation evaluates only the frozen dev winner. It must never be used to search or edit prompts.

The internal approval gate is QWK ≥ 0.30, Spearman ≥ 0.30, and MAE ≤ 1.25. Undefined QWK/Spearman, constant predictions, missing predictions, or unresolved schema failures reject approval. These thresholds are project-specific operating criteria, not universal quality standards.

## Exclusions

Exclusions are applied before held-out selection or calibration sampling. Pilot v2 excludes all calibration examples and held-out prompts in `pilot-001`, plus both judge subsets. Main v2 additionally excludes all examples and prompts in `pilot-002`. A prompt exclusion removes every associated response. Each new manifest records source hashes, counts, cross-source collisions, and selected-subset disjointness.

## Metrics

Within each block:

- Calibration-Order Instability: mean across held-out responses of population SD (`ddof=0`) across the four order-derived scores.
- Score Change Rate: fraction of held-out responses with more than one score across orders.
- Mean Score Range: mean of `max(score) - min(score)` across responses.
- Strict ranking flip: reversal of a response pair's strict sign across two orders.
- Tie change: exactly one of two order-specific comparisons is tied.

Same-order repeat metrics apply the corresponding variation calculation to replicates 0, 1, and 2 for the random order. Reports include exact replicate agreement, between-order minus repeat variation, and a descriptive ratio. When repeat instability is zero, the ratio is omitted and `between_exceeds_zero_observed_noise` is reported instead.

Agreement is first computed for each non-static `block × order` condition and then averaged across conditions. Static is physically and analytically deduplicated. Pooled agreement is secondary and labeled `pooled_descriptive`. Exports include QWK, Spearman, MAE, mean signed error (`prediction - human`), distributions, and undefined flags.

TF-IDF fits one vocabulary and IDF weighting over every rubric in the method/block comparison set. Bootstrap resamples calibration blocks and held-out prompt groups, keeps paired responses together, uses shared draws for batch/sequential contrasts, and reports deterministic percentile 95% intervals. Pilot uses 1,000 repetitions and labels intervals exploratory because it has two blocks; main uses 10,000.

## Call counts

| Workflow | Base | Repeats/fallback | Normal total |
|---|---:|---:|---:|
| Judge validation | 100 dev + 100 untouched validation | optional 100 Qwen dev | 200 |
| Pilot v2 | 428 | 204 | 632 |
| Main v2 | 4,490 | 666 | 5,156 |

Counts exclude structured-output retries. Repeat namespaces include replicate IDs, so repeats do not reuse another replicate's cache; resume within the same replicate remains supported.

### Judge dry-run manifest

Judge-dev dry-runs write outside the real run tree under `dry_runs/<run-id>/`. They do not create a call database or contact Ollama. Each sanitized request record contains candidate/model and example/prompt IDs; full SHA-256 hashes for the request, prompt template, rubric, schema, and model options; the approximate v2 input estimate and reserved context fields; preflight status; and a read-only cache status (`fresh`, `hit`, or `unavailable`). Dataset prompt/response text and the human label are omitted. The terminal and notebook show only aggregate counts and hashes; the complete sanitized list remains in the external manifest.

`estimated_context_headroom` is `context_length - estimated_input_tokens - reserved_output_tokens - safety_margin_tokens`. `token_estimate_exact=false` identifies the conservative character-based v2 estimator. Any invalid context request or unavailable cache check makes `ready=false`.

## Controlled workflow

Run from the project root after installing the project locally. Commands without `--execute` do not perform inference.

```powershell
python -m rubric_order model-provenance --config configs/judge_validation.yaml
python -m rubric_order prepare-judge-validation --config configs/judge_validation.yaml --run-id judge-validation-v2
python -m rubric_order estimate-judge-validation --config configs/judge_validation.yaml
python -m rubric_order judge-validation-dev --config configs/judge_validation.yaml --run-id judge-validation-v2 --candidate judge_absolute
python -m rubric_order judge-validation-dev --config configs/judge_validation.yaml --run-id judge-validation-v2 --candidate judge_absolute --execute --max-calls 10
```

After the first batch, restore notebook switches to `EXECUTE_MODEL_CALLS=False` and `MAX_CALLS=0`, inspect SQLite/context metadata, then resume both Llama candidates. Run untouched validation only after dev produces a frozen gate-passing candidate:

```powershell
python -m rubric_order judge-validation-holdout --config configs/judge_validation.yaml --run-id judge-validation-v2
python -m rubric_order judge-validation-holdout --config configs/judge_validation.yaml --run-id judge-validation-v2 --execute --max-calls 100
```

Only after `JUDGE_APPROVED`:

```powershell
python -m rubric_order prepare --config configs/pilot_v2.yaml --run-id pilot-002
python -m rubric_order estimate --config configs/pilot_v2.yaml --run-id pilot-002
python -m rubric_order build-rubrics --config configs/pilot_v2.yaml --run-id pilot-002
python -m rubric_order judge --config configs/pilot_v2.yaml --run-id pilot-002
python -m rubric_order analyze --config configs/pilot_v2.yaml --run-id pilot-002
python -m rubric_order readiness --config configs/pilot_v2.yaml --run-id pilot-002
```

Do not prepare or run `main-v2` until readiness prints `READY_FOR_MAIN`. Then repeat prepare, estimate, updater, judge, analyze, and readiness with `configs/main_v2.yaml` and a unique explicit `main-v2` run ID. Notebook order is setup, `judge_validation.ipynb`, data/protocol, rubric generation, unload updater, judging, unload judge, analysis, and `05_readiness_and_main.ipynb`.

## Freeze before main

Freeze the approved judge summary and its prompt/model/options hashes; v2 prompts and JSON schemas; initial rubric; resolved main config; exclusion manifests; dataset fingerprint; preprocessing/context thresholds; seeds; model digests and quantization; Ollama/Python/package versions; and the Git commit plus dirty flag, or fallback relevant-file tree hash. No validation case may be used for subsequent prompt editing.

Required paper artifacts are resolved configs and manifests, subset ID/hash lists, call database, canonical rubric trajectories plus metadata sidecars, final rubric and prediction indexes, judge-dev and untouched-validation reports, drift reports, context reports, metric CSV/JSON files, bootstrap draws or deterministic seeds, figures, readiness report, and code provenance.

## Interpretation limits

An approved run can support claims about order sensitivity under the frozen local models, prompts, data, and settings. It can compare between-order variation with observed repeat noise and report judge agreement under the internal gate. It cannot establish semantic generality from lexical drift checks, universal judge validity, causal effects outside this protocol, model-family superiority, or state-of-the-art performance. The pilot validates measurement and execution; it is not required to make sequential more unstable than batch.

## Pilot-002 thematic-drift finding and downstream audit

`pilot-002` generated all 132 updater calls successfully: 12 batch finals, 12 sequential finals,
120 sequential steps, and eight materialized static artifacts. A manual pre-judging gate then found
criteria about extracted information and prior/next workflow steps. Judging was intentionally not
started. This is a possible scientific result about how an updater converts calibration particulars
into apparently general evaluation criteria; deleting or rewriting the rubrics would erase that
evidence.

The separate `pilot-002-downstream-audit` reads the original rubric index and files, verifies their
raw and canonical hashes and the complete frozen source inventory, reconstructs the stored held-out
judge plan, and makes no updater request. The approved judge remains `qwen3:14b` with
`judge_anchor_first_v2` at SHA-256
`75dd61241b143deabe1c9e9435f897c17874ea4cc7ed0b513c43cd35687cda7c`.
The exact plan is 640 logical predictions and 500 physical judge requests: 24 non-static rubric
artifacts times 20 held-out responses, plus one physically deduplicated static rubric times 20.

The manual gate defines `task_specific` versus `not_flagged_by_manual_gate`; the latter is not
silently promoted to a human-validated domain-general label. Outputs separate individual artifacts,
unique canonical rubrics, trajectories (`method × block × order`), and repeat artifacts nested in a
trajectory. The three random-order replicates that reproduce a thematic rubric are one trajectory,
not three independent scientific observations.

Agreement (QWK, Spearman, MAE, and mean signed error), score distributions, response score ranges,
strict flips, tie changes, repeat variation, block summaries, and prompt summaries are descriptive.
Bootstrap draws jointly cluster by calibration block and held-out prompt and keep paired responses
together. The thematic/not-flagged comparison is exploratory and cannot identify a causal effect of
drift.

## Fixed inventory v3

The constrained updater returns only
`{"adaptive_criterion_ids":["appropriate_level_of_detail"]}`: exactly one object key and zero,
one, or at most two unique active IDs. Unknown, inactive, duplicate, extra, or third IDs are errors.
The validator canonicalizes valid IDs into inventory order. Criterion names, definitions, and signal
lists are materialized server-side; model output cannot redefine them. Equal canonical selections
produce byte-identical canonical rubric JSON. Title, the three core criteria, and all score anchors
are copied unchanged from the frozen initial rubric.

| Criterion ID | Definition | Positive signals | Negative signals | Difference from core | Possible overlap | Decision |
|---|---|---|---|---|---|---|
| `completeness_and_coverage` | Addresses all material request parts. | Covers every part; no missing deliverable. | Leaves a part unanswered; omits deliverable. | None beyond Task fulfillment. | Direct near-duplicate. | Excluded from active whitelist. |
| `instruction_and_constraint_adherence` | Follows explicit instructions and constraints. | Follows constraints; uses required form. | Violates constraint; ignores form. | None beyond Task fulfillment. | Direct duplicate of its constraint clause. | Excluded from active whitelist. |
| `actionability_and_practical_utility` | When action is requested, gives feasible usable information. | Feasible next actions; operational recommendations. | Abstract advice; missing operational detail. | Conditional practical usability rather than request coverage. | Partial overlap with fulfillment. | Retained with conditional wording. |
| `appropriate_level_of_detail` | Detail is proportionate to request and stakes. | Enough detail; concise for simple tasks. | Too sparse; detail obscures answer. | Isolates granularity rather than organization or relevance. | Partial overlap with Clarity and relevance. | Retained for conservative review. |
| `uncertainty_and_limitation_handling` | Represents material uncertainty and limitations proportionately. | Calibrated uncertainty; relevant limitations. | Unsupported certainty; invented limitation. | Epistemic calibration rather than factual accuracy alone. | Partial overlap with Correctness. | Retained. |
| `supporting_justification_when_relevant` | Supplies support when a conclusion needs it. | Supports consequential claims; relevant reasons. | Unsupported conclusion; unrelated reasoning. | Inspectable support rather than correctness itself. | Partial overlap with Correctness and verbosity. | Retained with conditional wording. |

Batch selection sees all ten examples at once. Sequential selection carries only the current
canonical ID list and updates it one example at a time. Every materialized step records previous and
new state hashes, the rendered rubric hash, exact request hash, prompt/schema/options hashes,
context preflight, and cache status. The request builder is shared by planning and execution; later
sequential hashes remain phase-aware and deferred until their prior state exists.

## Paired pilot-003 and final main-v3

`pilot-003` intentionally reuses byte-identical `pilot-002` calibration blocks, example and prompt
IDs, score strata, order/permutation lists, held-out responses, and repeat controls. Source artifact
hash mismatch stops before inference; no resampling occurs. This is a paired development comparison
of free-form versus constrained batch/sequential trajectories, not a confirmatory independent
sample. It compares matched orders on drift, agreement, score instability, repeat noise,
distributions, strict flips, and tie changes. Its exact physical budget is 132 updater plus 500 judge
calls, or 632 total. Free-form predictions come only from the separate downstream run.

The final main-v3 retains the original 5,156-call main-v2 free-form experiment as its primary
analysis. Thematic drift is secondary and does not trigger post-hoc condition deletion. A
preregistered constrained ablation uses only blocks 0, 4, and 9, every one of the four orders, and no
extra constrained repeats: 12 batch updater + 120 sequential updater + 600 batch judge + 600
sequential judge = 1,332 calls. Total main-v3 cost is 6,488. Final sampling must exclude all examples
and prompts from pilot-001, judge development/untouched validation, pilot-002, and pilot-003; it is
not materialized during preparation.

## Blinded drift annotation

The annotation unit is a unique canonical rubric, while the separate blind mapping retains all
artifact and trajectory memberships. Stable blind IDs depend only on canonical rubric hashes. The
annotator export contains adaptive criterion content and blank labels but omits method, block,
order, replicate, downstream scores, and judge results. The mapping and blinded export are stored
separately and SHA-256 checked.

The concise codebook is in `docs/drift_annotation_codebook.md`. Imports require every blind ID
exactly once and strict labels. Two raters receive per-field Cohen's kappa and a disagreement list;
one rater produces an explicit methodological warning; an optional adjudicated file uses the same
validator. A freeze artifact records mapping, annotation, and adjudication hashes and time. The main
gate rejects changed files or annotations frozen after judging began. Automated lexical flags remain
inspectable heuristics, never semantic validators or replacements for human labels.

## V3 cost summary

| Workflow | Updater calls | Physical judge calls | Logical predictions | Total physical calls |
|---|---:|---:|---:|---:|
| Frozen pilot-002 downstream audit | 0 | 500 | 640 | 500 |
| Paired pilot-003 | 132 | 500 | 640 | 632 |
| Main-v3 primary | included | included | — | 5,156 |
| Main-v3 constrained ablation | 132 | 1,200 | — | 1,332 |
| Main-v3 complete | — | — | — | 6,488 |

All three v3 notebooks save with execution disabled. Preparation and estimator commands do not
construct an Ollama client. A real downstream run begins only with an explicit 10-call canary, then
resumes against the same content-addressed call store with a 500-call command limit. Pilot-003 and
main-v3 remain blocked until their saved prerequisite gates pass.

## Current short-paper primary route (superseding the required v3 route)

For the current 2–4 page paper, the required route is `configs/main_final_primary.yaml`: static,
free-form batch, and free-form sequential across ten blocks and all four orders, with the original
random-order repeat controls on blocks 0, 4, and 9. Its exact physical budget is 5,156 calls.

Pilot evidence shows semantic rubric drift while the independently audited downstream effect is
very small. A constrained mitigation study would introduce a second research question and dilute
the calibration-order contribution at this paper length. This scope decision was frozen before any
main execution. It does not imply that the constrained method fails; that method has not been run.
`pilot-003`, inventory-v3, and the 1,332-call constrained ablation remain implemented only as
deferred optional future work and are unused in the current paper's primary results.

The main analysis measures calibration-order dependence, rubric variation through the implemented
structural/textual diagnostics, and downstream effects through score instability and agreement.
Manual thematic drift is qualitative context from `pilot-002` only: main does not estimate a manual
semantic-drift rate, does not invent human labels, and does not treat lexical flags as ground truth.
Unique canonical rubrics may still be exported for optional post-hoc inspection, but neither export
nor annotation blocks judge planning or execution in `main-final-primary`. The annotation
infrastructure remains available for other configurations. Constrained mitigation remains future
work. This simplified protocol was fixed before the first main inference call.

The main execution uses a 5,156-call planned design and a 5,200-call hard physical ceiling. The
stage ceilings are 520 updater calls and 4,680 judge calls, including their one-request canaries.
Each request may be retried at most once for structural validation. Existing successful request
hashes are never relaunched; resume budgets apply only to fresh calls.

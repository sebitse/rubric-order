# Final primary launch

The current-paper route uses `configs/main_final_primary.yaml` and run ID
`main-final-primary`. Do not create this run until `pre-main-readiness` returns
`READY_TO_START_MAIN`.

```powershell
Set-Location E:\JUDGe
$PythonExe = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
& $PythonExe -m rubric_order pre-main-readiness `
  --config .\configs\main_final_primary.yaml `
  --freeze-manifest .\dry_runs\pre-main\pre_main_freeze_primary_no_annotation.json
& $PythonExe -m jupyter lab .\notebooks\08_final_main_experiment.ipynb
```

## Exact notebook order

1. Provenance.
2. Approved judge.
3. Exclusions.
4. Sampling.
5. Exact estimator.
6. Updater dry-run.
7. Updater canary.
8. Updater resume.
9. Optional blinded rubric export.
10. Non-blocking main annotation policy.
11. Judge dry-run.
12. Judge canary.
13. Judge resume.
14. Analysis.
15. Readiness.

The saved notebook always remains at:

```python
EXECUTE_MODEL_CALLS = False
MAX_CALLS = 0
```

The four inference cells use separate local gates: `RUN_UPDATER_CANARY`, `RUN_UPDATER_FULL`,
`RUN_JUDGE_CANARY`, and `RUN_JUDGE_FULL`. Never change the two globals.

## Sampling, estimator, and updater

These are the exact CLI equivalents of cells 4-8. They are commands for the later user-run
experiment; they were not executed while preparing this repository.

```powershell
& $PythonExe -m rubric_order prepare `
  --config .\configs\main_final_primary.yaml `
  --run-id main-final-primary

& $PythonExe -m rubric_order estimate-final-primary `
  --config .\configs\main_final_primary.yaml

& $PythonExe -m rubric_order build-rubrics `
  --config .\configs\main_final_primary.yaml `
  --run-id main-final-primary

& $PythonExe -m rubric_order build-rubrics `
  --config .\configs\main_final_primary.yaml `
  --run-id main-final-primary `
  --methods batch `
  --blocks 0 `
  --orders ascending `
  --replicates 0 `
  --execute `
  --max-calls 1

& $PythonExe -m rubric_order build-rubrics `
  --config .\configs\main_final_primary.yaml `
  --run-id main-final-primary `
  --execute `
  --max-calls 519
```

The first updater canary values are exactly:

- methods: `batch`;
- blocks: `0`;
- orders: `ascending`;
- replicates: `0`;
- `RUN_UPDATER_CANARY=True` only in an authorized execution copy;
- `UPDATER_CANARY_MAX_CALLS=1`.

The 519-call resume ceiling plus the canary gives the updater a 520-call hard ceiling. The normal
fresh resume is at most 505 calls.

## Optional blinded inspection

Export only unique canonical rubrics if post-hoc inspection is desired. This is non-blocking for
`main-final-primary`; no annotation or manual semantic-drift rate is required.

```powershell
& $PythonExe -m rubric_order export-drift `
  --config .\configs\main_final_primary.yaml `
  --source-run runs/main-final-primary `
  --output runs/main-final-primary/artifacts/drift_annotations/export

& $PythonExe -m rubric_order validate-drift `
  --config .\configs\main_final_primary.yaml `
  --annotations runs/main-final-primary/artifacts/drift_annotations/annotator_1.jsonl `
  --mapping runs/main-final-primary/artifacts/drift_annotations/export/blind_mapping.json

& $PythonExe -m rubric_order freeze-drift `
  --config .\configs\main_final_primary.yaml `
  --annotations runs/main-final-primary/artifacts/drift_annotations/annotator_1.jsonl `
  --mapping runs/main-final-primary/artifacts/drift_annotations/export/blind_mapping.json `
  --output runs/main-final-primary/artifacts/drift_annotations/annotation_freeze.json
```

## Judge and analysis

The judge canary is exactly one request. The explicit partial flag is required only for that
one-call canary; normal execution still refuses undersized fresh-call budgets.

```powershell
& $PythonExe -m rubric_order judge `
  --config .\configs\main_final_primary.yaml `
  --run-id main-final-primary

& $PythonExe -m rubric_order judge `
  --config .\configs\main_final_primary.yaml `
  --run-id main-final-primary `
  --methods static `
  --blocks 0 `
  --orders ascending `
  --replicates 0 `
  --execute `
  --allow-partial-execution `
  --max-calls 1

& $PythonExe -m rubric_order judge `
  --config .\configs\main_final_primary.yaml `
  --run-id main-final-primary `
  --execute `
  --max-calls 4679

& $PythonExe -m rubric_order analyze `
  --config .\configs\main_final_primary.yaml `
  --run-id main-final-primary

& $PythonExe -m rubric_order readiness `
  --config .\configs\main_final_primary.yaml `
  --run-id main-final-primary
```

The exact primary estimate is 5,156 physical calls: 506 updater-stage calls and 4,650 judge-stage
calls. Stage hard ceilings are 520 and 4,680, for a total ceiling of 5,200. After the one-call judge
canary, the full resume should make at most 4,649 fresh judge calls.

After the no-inference pre-main gates pass and the run is prepared, execute the notebook itself:

```powershell
& $PythonExe -m rubric_order execute-final-primary-notebook `
  --config .\configs\main_final_primary.yaml `
  --run-id main-final-primary `
  --output runs/main-final-primary/notebooks/08_final_main_experiment.executed.ipynb `
  --execute
```

# Calibration-order dependence in LLM judges

This repository studies whether the **order of calibration examples** changes the rubrics produced by a frozen LLM updater and, consequently, the scores and rankings assigned by an independent frozen judge.

The experiment compares three calibration strategies: **static**, **batch**, and **sequential**, using balanced example orders (ascending, descending, random, and reverse-random). No model training or weight updates are performed.

The recommended final configuration is `configs/main_final_primary.yaml`, with the main workflow implemented in the notebooks under `notebooks/`. Inference notebooks are committed with `EXECUTE_MODEL_CALLS=False` and `MAX_CALLS=0` for reproducibility and safety.

Main outputs, metrics, rubrics, predictions, and figures are stored under `runs/<run_id>/`. The primary metrics quantify **calibration-order instability, score changes, score range, and pairwise ranking flips**.

For the complete experimental protocol and execution details, see:
- [`docs/final_experiment_protocol.md`](docs/final_experiment_protocol.md)
- [`docs/final_primary_launch.md`](docs/final_primary_launch.md)

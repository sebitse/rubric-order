from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .api import (
    analyze_downstream_predictions,
    analyze_experiment,
    annotation_agreement,
    annotation_freeze_gate,
    audit_completed_downstream,
    audit_extreme_pair_lineage,
    build_rubrics,
    check_ollama,
    compact_judge_plan,
    create_pre_main_freeze,
    downstream_readiness,
    estimate_downstream_audit,
    estimate_experiment,
    estimate_final_primary,
    estimate_judge_validation,
    estimate_main_v3,
    estimate_pilot_v3,
    export_blinded_drift,
    freeze_annotations,
    installed_model_provenance,
    judge_responses,
    load_experiment_config,
    main_v3_readiness,
    no_inference_smoke_test,
    no_inference_v3_smoke_test,
    pilot_v3_readiness,
    plan_downstream_audit,
    plan_judge_validation,
    plan_judging,
    plan_rubric_generation,
    pre_main_readiness,
    prepare_experiment,
    prepare_judge_validation,
    readiness_report,
    resolve_dataset_lineage,
    run_downstream_audit,
    run_judge_dev,
    run_judge_holdout,
    validate_annotation_import,
    validate_dataset,
    verify_approved_judge,
    write_downstream_dry_run,
    write_main_v3_dry_run,
    write_pilot_v3_dry_run,
)
from .final_primary import final_main_readiness
from .notebook_runner import execute_final_primary_notebook

smoke_test = no_inference_smoke_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rubric-order")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="validate and summarize the dataset")
    _add_config(validate)

    prepare = subparsers.add_parser("prepare", help="sample and store experimental data")
    _add_config(prepare)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--resume", action="store_true")

    estimate = subparsers.add_parser("estimate", help="estimate inference calls and token ceilings")
    _add_config(estimate)
    estimate.add_argument("--run-id")
    _add_selection(estimate)

    build = subparsers.add_parser("build-rubrics", help="build and store rubric variants")
    _add_config(build)
    build.add_argument("--run-id", required=True)
    _add_execution(build)
    _add_selection(build)

    judge = subparsers.add_parser("judge", help="judge held-out responses with stored rubrics")
    _add_config(judge)
    judge.add_argument("--run-id", required=True)
    _add_execution(judge)
    judge.add_argument("--allow-partial-execution", action="store_true")
    _add_selection(judge)

    analyze = subparsers.add_parser("analyze", help="compute metrics, intervals, and figures")
    _add_config(analyze)
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--allow-partial-results", action="store_true")

    check = subparsers.add_parser("ollama-check", help="check Ollama health and installed models")
    _add_config(check)
    check.add_argument("--stage", choices=("updater", "judge", "both"), default="both")

    provenance = subparsers.add_parser(
        "model-provenance", help="read installed model metadata without inference or pulls"
    )
    _add_config(provenance)

    prepare_validation = subparsers.add_parser(
        "prepare-judge-validation", help="select disjoint judge-dev and validation IDs"
    )
    _add_config(prepare_validation)
    prepare_validation.add_argument("--run-id", default="judge-validation-v2")
    prepare_validation.add_argument("--resume", action="store_true")

    estimate_validation = subparsers.add_parser(
        "estimate-judge-validation", help="estimate the separate judge-validation workflow"
    )
    _add_config(estimate_validation)
    estimate_validation.add_argument("--include-qwen-fallback", action="store_true")

    dev = subparsers.add_parser("judge-validation-dev", help="run or dry-run judge-dev")
    _add_config(dev)
    dev.add_argument("--run-id", default="judge-validation-v2")
    dev.add_argument("--candidate", choices=("judge_absolute", "judge_anchor_first"))
    dev.add_argument("--model", choices=("llama", "qwen"), default="llama")
    dev.add_argument("--allow-qwen-fallback", action="store_true")
    _add_execution(dev)

    holdout = subparsers.add_parser(
        "judge-validation-holdout", help="run or dry-run the frozen untouched validation"
    )
    _add_config(holdout)
    holdout.add_argument("--run-id", default="judge-validation-v2")
    _add_execution(holdout)

    readiness = subparsers.add_parser("readiness", help="audit a completed pilot without inference")
    _add_config(readiness)
    readiness.add_argument("--run-id", required=True)

    gate = subparsers.add_parser(
        "provenance-gate", help="verify frozen source provenance and approved judge"
    )
    _add_config(gate)

    downstream_estimate = subparsers.add_parser(
        "estimate-downstream", help="estimate the frozen pilot-002 downstream audit"
    )
    _add_config(downstream_estimate)

    downstream_plan = subparsers.add_parser(
        "plan-downstream", help="write a read-only downstream audit plan"
    )
    _add_config(downstream_plan)

    downstream_run = subparsers.add_parser(
        "run-downstream", help="run or dry-run judging of frozen pilot-002 rubrics"
    )
    _add_config(downstream_run)
    downstream_run.add_argument("--run-id", required=True)
    _add_execution(downstream_run)

    downstream_analysis = subparsers.add_parser(
        "analyze-downstream", help="analyze completed exploratory downstream predictions"
    )
    _add_config(downstream_analysis)
    downstream_analysis.add_argument("--run-id", required=True)

    downstream_audit = subparsers.add_parser(
        "audit-downstream", help="independently close a completed downstream audit"
    )
    _add_config(downstream_audit)
    downstream_audit.add_argument("--run-id", required=True)
    downstream_audit.add_argument("--lineage-report", required=True)
    downstream_audit.add_argument("--output", required=True)

    downstream_ready = subparsers.add_parser(
        "downstream-readiness", help="check downstream completion without inference"
    )
    _add_config(downstream_ready)
    downstream_ready.add_argument("--run-id", required=True)

    lineage = subparsers.add_parser(
        "resolve-lineage", help="resolve internal prompt/example IDs without modifying data"
    )
    _add_config(lineage)
    lineage.add_argument("--prompt-id")
    lineage.add_argument("--example-id")
    lineage.add_argument("--predictions")

    lineage_audit = subparsers.add_parser(
        "audit-lineage", help="write a read-only lineage report for explicit internal IDs"
    )
    _add_config(lineage_audit)
    lineage_audit.add_argument("--prompt-ids", required=True)
    lineage_audit.add_argument("--example-ids", required=True)
    lineage_audit.add_argument("--predictions", required=True)
    lineage_audit.add_argument("--output", required=True)

    drift_export = subparsers.add_parser(
        "export-drift", help="export blinded unique rubrics for human annotation"
    )
    _add_config(drift_export)
    drift_export.add_argument("--source-run", required=True)
    drift_export.add_argument("--output", required=True)

    drift_validate = subparsers.add_parser(
        "validate-drift", help="validate drift annotations and optionally report agreement"
    )
    _add_config(drift_validate)
    drift_validate.add_argument("--annotations", nargs="+", required=True)
    drift_validate.add_argument("--mapping", required=True)
    drift_validate.add_argument("--mapping-sha256")
    drift_validate.add_argument("--adjudicated")

    drift_freeze = subparsers.add_parser(
        "freeze-drift", help="freeze validated annotations before judging"
    )
    _add_config(drift_freeze)
    drift_freeze.add_argument("--annotations", nargs="+", required=True)
    drift_freeze.add_argument("--mapping", required=True)
    drift_freeze.add_argument("--output", required=True)
    drift_freeze.add_argument("--adjudicated")

    pilot_estimate = subparsers.add_parser(
        "estimate-pilot-v3", help="estimate the paired inventory-constrained pilot"
    )
    _add_config(pilot_estimate)

    pilot_plan = subparsers.add_parser(
        "plan-pilot-v3", help="write a no-inference phase-aware pilot-v3 plan"
    )
    _add_config(pilot_plan)

    pilot_ready = subparsers.add_parser(
        "pilot-v3-readiness", help="check paired-pilot prerequisites without inference"
    )
    _add_config(pilot_ready)
    pilot_ready.add_argument("--downstream-config", required=True)
    pilot_ready.add_argument("--downstream-run", required=True)

    main_estimate = subparsers.add_parser(
        "estimate-main-v3", help="estimate the primary main design and constrained ablation"
    )
    _add_config(main_estimate)

    main_plan = subparsers.add_parser(
        "plan-main-v3", help="write the no-inference main-v3 estimate"
    )
    _add_config(main_plan)

    main_ready = subparsers.add_parser(
        "main-v3-readiness", help="check main-v3 prerequisites without inference"
    )
    _add_config(main_ready)
    main_ready.add_argument("--annotation-freeze")

    final_estimate = subparsers.add_parser(
        "estimate-final-primary", help="estimate the simplified final primary design"
    )
    _add_config(final_estimate)

    final_freeze = subparsers.add_parser(
        "freeze-pre-main", help="write the no-Git pre-main contract and readiness report"
    )
    _add_config(final_freeze)
    final_freeze.add_argument("--output", default="dry_runs/pre-main")

    final_ready = subparsers.add_parser(
        "pre-main-readiness", help="verify the frozen primary contract read-only"
    )
    _add_config(final_ready)
    final_ready.add_argument("--freeze-manifest")

    final_complete = subparsers.add_parser(
        "final-main-readiness", help="audit the completed final primary experiment"
    )
    _add_config(final_complete)
    final_complete.add_argument("--run-id", default="main-final-primary")
    final_complete.add_argument("--freeze-manifest")
    final_complete.add_argument("--executed-notebook")

    final_notebook = subparsers.add_parser(
        "execute-final-primary-notebook",
        help="execute an authorized copy of notebook 08 against the prepared final run",
    )
    _add_config(final_notebook)
    final_notebook.add_argument("--run-id", default="main-final-primary")
    final_notebook.add_argument("--output")
    final_notebook.add_argument("--execute", action="store_true")

    smoke = subparsers.add_parser("smoke-test", help="run a synthetic no-inference smoke test")
    smoke.add_argument("--output", default="runs/smoke")
    smoke_v3 = subparsers.add_parser(
        "smoke-test-v3", help="run the synthetic constrained no-inference smoke test"
    )
    smoke_v3.add_argument("--output", default="dry_runs/smoke-v3")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "smoke-test":
            result = no_inference_smoke_test(Path(args.output))
        elif args.command == "smoke-test-v3":
            result = no_inference_v3_smoke_test(Path(args.output))
        else:
            config = load_experiment_config(args.config, data_path=getattr(args, "data_path", None))
            result = _dispatch(args, config)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _dispatch(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if args.command == "validate-data":
        return validate_dataset(config)
    if args.command == "prepare":
        return prepare_experiment(config, args.run_id, resume=args.resume)
    if args.command == "estimate":
        return estimate_experiment(config, args.run_id, **_selection_kwargs(args))
    if args.command == "build-rubrics":
        plan = plan_rubric_generation(config, args.run_id, **_selection_kwargs(args))
        print(json.dumps(plan, sort_keys=True), file=sys.stderr)
        return build_rubrics(
            config,
            args.run_id,
            execute_model_calls=args.execute,
            max_calls=_max_calls(args),
            **_selection_kwargs(args),
        )
    if args.command == "judge":
        plan = plan_judging(config, args.run_id, **_selection_kwargs(args))
        print(json.dumps(plan, sort_keys=True), file=sys.stderr)
        return judge_responses(
            config,
            args.run_id,
            execute_model_calls=args.execute,
            max_calls=_max_calls(args),
            allow_partial_execution=args.allow_partial_execution,
            **_selection_kwargs(args),
        )
    if args.command == "analyze":
        return analyze_experiment(
            config, args.run_id, allow_partial_results=args.allow_partial_results
        )
    if args.command == "ollama-check":
        return check_ollama(config, args.stage)
    if args.command == "model-provenance":
        return installed_model_provenance(config)
    if args.command == "prepare-judge-validation":
        return prepare_judge_validation(config, args.run_id, resume=args.resume)
    if args.command == "estimate-judge-validation":
        return estimate_judge_validation(config, args.include_qwen_fallback)
    if args.command == "judge-validation-dev":
        if args.execute:
            plan = plan_judge_validation(
                config,
                args.run_id,
                candidate=args.candidate,
                model=args.model,
            )
            print(json.dumps(compact_judge_plan(plan), sort_keys=True), file=sys.stderr)
        result = run_judge_dev(
            config,
            args.run_id,
            execute_model_calls=args.execute,
            max_calls=_max_calls(args),
            candidate=args.candidate,
            model=args.model,
            allow_qwen_fallback=args.allow_qwen_fallback,
        )
        return compact_judge_plan(result)
    if args.command == "judge-validation-holdout":
        result = run_judge_holdout(
            config,
            args.run_id,
            execute_model_calls=args.execute,
            max_calls=_max_calls(args),
        )
        return result if args.execute else compact_judge_plan(result)
    if args.command == "readiness":
        return readiness_report(config, args.run_id)
    if args.command == "provenance-gate":
        plan = plan_downstream_audit(config, include_requests=False)
        return {
            "approved_judge": verify_approved_judge(config),
            "source_inventory": plan["source_inventory"],
            "rubric_integrity": plan["rubric_integrity"],
            "manual_gate": plan["manual_gate"],
            "ollama_contacted": False,
        }
    if args.command == "estimate-downstream":
        return estimate_downstream_audit(config)
    if args.command == "plan-downstream":
        return write_downstream_dry_run(config)
    if args.command == "run-downstream":
        return run_downstream_audit(
            config,
            args.run_id,
            execute_model_calls=args.execute,
            max_calls=_max_calls(args),
        )
    if args.command == "analyze-downstream":
        run_dir = Path(config["_project_root"]) / config["output_root"] / args.run_id
        return analyze_downstream_predictions(
            config,
            run_dir / "artifacts" / "predictions.jsonl",
            run_dir / "metrics",
        )
    if args.command == "audit-downstream":
        root = Path(config["_project_root"])
        return audit_completed_downstream(
            config,
            root / config["output_root"] / args.run_id,
            lineage_report=root / args.lineage_report,
            output_dir=root / args.output,
        )
    if args.command == "downstream-readiness":
        run_dir = Path(config["_project_root"]) / config["output_root"] / args.run_id
        return downstream_readiness(config, run_dir)
    if args.command == "resolve-lineage":
        return resolve_dataset_lineage(
            config,
            prompt_id=args.prompt_id,
            example_id=args.example_id,
            predictions_path=(
                Path(config["_project_root"]) / args.predictions if args.predictions else None
            ),
        )
    if args.command == "audit-lineage":
        root = Path(config["_project_root"])
        return audit_extreme_pair_lineage(
            config,
            prompt_ids=_parse_strings(args.prompt_ids) or [],
            example_ids=_parse_strings(args.example_ids) or [],
            predictions_path=root / args.predictions,
            output_dir=root / args.output,
        )
    if args.command == "export-drift":
        return export_blinded_drift(
            Path(config["_project_root"]) / args.source_run,
            Path(config["_project_root"]) / args.output,
        )
    if args.command == "validate-drift":
        root = Path(config["_project_root"])
        annotations = [root / path for path in args.annotations]
        mapping = root / args.mapping
        if args.mapping_sha256 is not None:
            validate_annotation_import(
                annotations[0],
                mapping,
                expected_mapping_sha256=args.mapping_sha256,
            )
        if len(annotations) == 1 and args.adjudicated is None:
            validated = validate_annotation_import(
                annotations[0],
                mapping,
                expected_mapping_sha256=args.mapping_sha256,
            )
            return {
                **validated,
                "methodological_warning": (
                    "Only one annotator is available; inter-rater reliability cannot be estimated."
                ),
            }
        return annotation_agreement(
            annotations,
            mapping,
            root / args.adjudicated if args.adjudicated else None,
        )
    if args.command == "freeze-drift":
        root = Path(config["_project_root"])
        return freeze_annotations(
            [root / path for path in args.annotations],
            root / args.mapping,
            root / args.output,
            adjudicated_path=root / args.adjudicated if args.adjudicated else None,
        )
    if args.command == "estimate-pilot-v3":
        return estimate_pilot_v3(config)
    if args.command == "plan-pilot-v3":
        return write_pilot_v3_dry_run(config)
    if args.command == "pilot-v3-readiness":
        downstream_config = load_experiment_config(args.downstream_config)
        return pilot_v3_readiness(
            config,
            args.downstream_run,
            downstream_config=downstream_config,
        )
    if args.command == "estimate-main-v3":
        return estimate_main_v3(config)
    if args.command == "plan-main-v3":
        return write_main_v3_dry_run(config)
    if args.command == "main-v3-readiness":
        freeze = (
            annotation_freeze_gate(Path(config["_project_root"]) / args.annotation_freeze)
            if args.annotation_freeze
            else None
        )
        return main_v3_readiness(config, annotation_freeze=freeze)
    if args.command == "estimate-final-primary":
        return estimate_final_primary(config)
    if args.command == "freeze-pre-main":
        return create_pre_main_freeze(config, Path(config["_project_root"]) / args.output)
    if args.command == "pre-main-readiness":
        return pre_main_readiness(
            config,
            Path(config["_project_root"]) / args.freeze_manifest if args.freeze_manifest else None,
        )
    if args.command == "final-main-readiness":
        root = Path(config["_project_root"])
        return final_main_readiness(
            config,
            args.run_id,
            freeze_manifest=(root / args.freeze_manifest if args.freeze_manifest else None),
            executed_notebook=(root / args.executed_notebook if args.executed_notebook else None),
        )
    if args.command == "execute-final-primary-notebook":
        if not args.execute:
            raise ValueError("notebook execution requires explicit --execute")
        root = Path(config["_project_root"])
        return execute_final_primary_notebook(
            config,
            args.run_id,
            output=(root / args.output if args.output else None),
        )
    raise AssertionError(f"unknown command: {args.command}")


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path")


def _add_execution(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-calls", type=int)


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--blocks", help="comma-separated zero-based block IDs")
    parser.add_argument("--orders", help="comma-separated order names")
    parser.add_argument("--methods", help="comma-separated method names")
    parser.add_argument("--replicates", help="comma-separated replicate IDs")


def _selection_kwargs(args: argparse.Namespace) -> dict[str, list[Any] | None]:
    return {
        "methods": _parse_strings(getattr(args, "methods", None)),
        "blocks": _parse_integers(getattr(args, "blocks", None), "blocks"),
        "orders": _parse_strings(getattr(args, "orders", None)),
        "replicates": _parse_integers(getattr(args, "replicates", None), "replicates"),
    }


def _parse_integers(value: str | None, name: str) -> list[int] | None:
    if value is None:
        return None
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not parsed or min(parsed) < 0:
        raise ValueError(f"{name} must contain non-negative integers")
    return parsed


def _parse_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("selection cannot be empty")
    return parsed


def _max_calls(args: argparse.Namespace) -> int:
    if not args.execute:
        return int(args.max_calls or 0)
    if args.max_calls is None or args.max_calls < 0:
        raise ValueError("--execute requires a non-negative --max-calls")
    return args.max_calls


if __name__ == "__main__":
    raise SystemExit(main())

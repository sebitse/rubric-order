from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rubric_order.cli import build_parser, main


def test_cli_estimate_remains_compatible(capsys) -> None:
    exit_code = main(
        [
            "estimate",
            "--config",
            "configs/pilot.yaml",
            "--data-path",
            "HelpSteer2",
        ]
    )
    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total_inference_calls"] == 428


def test_v3_estimator_commands_and_future_downstream_commands(capsys) -> None:
    for command, config, expected in (
        (
            "estimate-downstream",
            "configs/pilot_002_downstream_audit.yaml",
            ("total_inference_calls", 500),
        ),
        ("estimate-pilot-v3", "configs/pilot_v3.yaml", ("total_inference_calls", 632)),
        ("estimate-main-v3", "configs/main_v3.yaml", ("total_inference_calls", 6488)),
    ):
        assert main([command, "--config", config]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result[expected[0]] == expected[1]

    parser = build_parser()
    canary = parser.parse_args(
        [
            "run-downstream",
            "--config",
            "configs/pilot_002_downstream_audit.yaml",
            "--run-id",
            "pilot-002-downstream-audit",
            "--execute",
            "--max-calls",
            "10",
        ]
    )
    assert canary.execute and canary.max_calls == 10
    readiness = parser.parse_args(
        [
            "pilot-v3-readiness",
            "--config",
            "configs/pilot_v3.yaml",
            "--downstream-config",
            "configs/pilot_002_downstream_audit.yaml",
            "--downstream-run",
            "runs/pilot-002-downstream-audit",
        ]
    )
    assert readiness.command == "pilot-v3-readiness"


def test_v3_cli_smoke_is_no_inference(capsys) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "cli-smoke-v3"
        assert main(["smoke-test-v3", "--output", str(output)]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["ollama_contacted"] is False
        assert not list(output.rglob("calls.sqlite3"))

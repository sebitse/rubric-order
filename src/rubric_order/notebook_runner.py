from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

from .config import resolve_project_path
from .run import file_sha256, read_json, run_directory

LOCAL_EXECUTION_GATES = {
    "RUN_UPDATER_CANARY": True,
    "RUN_UPDATER_FULL": True,
    "RUN_JUDGE_CANARY": True,
    "RUN_JUDGE_FULL": True,
    "RUN_FINAL_ANALYSIS": True,
    "RUN_FINAL_READINESS": True,
}


def execute_final_primary_notebook(
    config: dict[str, Any],
    run_id: str = "main-final-primary",
    *,
    source: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Execute an authorized copy while leaving the source notebook inert and output-free."""
    source_path = (
        Path(source).resolve()
        if source is not None
        else resolve_project_path(config, config["pre_main"]["notebook"])
    )
    run_dir = run_directory(config, run_id)
    if not (run_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"final run is not prepared: {run_dir}")
    freeze_path = resolve_project_path(config, config["pre_main"]["freeze_manifest"])
    freeze = read_json(freeze_path)
    changed = [
        row["path"]
        for row in freeze.get("files", [])
        if not Path(row["path"]).is_file() or file_sha256(row["path"]) != row["sha256"]
    ]
    if changed:
        raise ValueError(f"frozen contract changed before notebook execution: {changed}")
    output_path = (
        Path(output).resolve()
        if output is not None
        else run_dir / "notebooks" / "08_final_main_experiment.executed.ipynb"
    )
    notebook = nbformat.read(source_path, as_version=4)
    _assert_saved_globals_are_safe(notebook)
    replacements = {f"{name} = False": f"{name} = True" for name in LOCAL_EXECUTION_GATES}
    replaced: set[str] = set()
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for old, new in replacements.items():
            if old in cell.source:
                cell.source = cell.source.replace(old, new)
                replaced.add(old.split(" =", maxsplit=1)[0])
    missing = sorted(set(LOCAL_EXECUTION_GATES) - replaced)
    if missing:
        raise ValueError(f"notebook lacks local execution gates: {missing}")
    notebook.metadata["final_primary_execution"] = {
        "authorized_local_gates": sorted(LOCAL_EXECUTION_GATES),
        "source_notebook": str(source_path),
        "source_sha256": file_sha256(source_path),
        "global_execute_model_calls": False,
        "global_max_calls": 0,
        "run_id": run_id,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path(config["_project_root"]))}},
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output_path)
    return {
        "status": "complete",
        "source_notebook": str(source_path),
        "source_sha256": file_sha256(source_path),
        "executed_notebook": str(output_path),
        "run_id": run_id,
        "local_gates": LOCAL_EXECUTION_GATES,
        "source_modified": False,
    }


def _assert_saved_globals_are_safe(notebook: Any) -> None:
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assignments: dict[str, Any] = {}
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                continue
    if assignments.get("EXECUTE_MODEL_CALLS") is not False or assignments.get("MAX_CALLS") != 0:
        raise ValueError("saved notebook has unsafe global inference defaults")
    unsafe = [name for name in LOCAL_EXECUTION_GATES if assignments.get(name) is not False]
    if unsafe:
        raise ValueError(f"saved notebook has enabled local gates: {unsafe}")

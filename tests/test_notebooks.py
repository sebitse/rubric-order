from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).parents[1].resolve()
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_NAMES = [
    "00_setup_and_health.ipynb",
    "01_data_and_protocol.ipynb",
    "02_rubric_generation.ipynb",
    "03_judging.ipynb",
    "04_analysis.ipynb",
    "05_readiness_and_main.ipynb",
    "06_pilot_002_downstream_audit.ipynb",
    "07_inventory_mitigation_pilot.ipynb",
    "08_final_main_experiment.ipynb",
    "judge_validation.ipynb",
]
FORBIDDEN = ("!python", "%run", "subprocess", "os.system")


class NotebookTests(unittest.TestCase):
    def test_notebooks_are_valid_and_do_not_shell_out(self) -> None:
        self.assertEqual(sorted(path.name for path in NOTEBOOK_DIR.glob("*.ipynb")), NOTEBOOK_NAMES)
        for name in NOTEBOOK_NAMES:
            with self.subTest(notebook=name):
                notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
                nbformat.validate(notebook)
                self.assertTrue(notebook.cells[0].source.startswith("# "))
                code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
                for forbidden in FORBIDDEN:
                    self.assertNotIn(forbidden, code)
                self.assertNotIn("E:\\JUDGe", code)
                self.assertTrue(
                    all(
                        not cell.get("outputs")
                        for cell in notebook.cells
                        if cell.cell_type == "code"
                    )
                )
                self.assertTrue(
                    all(
                        cell.get("execution_count") is None
                        for cell in notebook.cells
                        if cell.cell_type == "code"
                    )
                )

    def test_saved_execution_defaults_are_safe(self) -> None:
        expected = {
            "02_rubric_generation.ipynb": {"EXECUTE_MODEL_CALLS": False, "MAX_CALLS": 0},
            "03_judging.ipynb": {"EXECUTE_MODEL_CALLS": False, "MAX_CALLS": 0},
            "04_analysis.ipynb": {"ALLOW_PARTIAL_RESULTS": False},
            "judge_validation.ipynb": {
                "RUN_DRY_RUN": False,
                "EXECUTE_MODEL_CALLS": False,
                "MAX_CALLS": 0,
            },
            "05_readiness_and_main.ipynb": {"EXECUTE_MODEL_CALLS": False, "MAX_CALLS": 0},
            "06_pilot_002_downstream_audit.ipynb": {
                "EXECUTE_MODEL_CALLS": False,
                "MAX_CALLS": 0,
            },
            "07_inventory_mitigation_pilot.ipynb": {
                "EXECUTE_MODEL_CALLS": False,
                "MAX_CALLS": 0,
            },
            "08_final_main_experiment.ipynb": {
                "EXECUTE_MODEL_CALLS": False,
                "MAX_CALLS": 0,
            },
        }
        for name, values in expected.items():
            notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
            source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
            assignments = self._literal_assignments(source)
            with self.subTest(notebook=name):
                for variable, value in values.items():
                    self.assertEqual(assignments.get(variable), value)

    def test_all_notebooks_execute_with_synthetic_data_and_no_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "synthetic.jsonl"
            output_root = root / "runs"
            self._write_synthetic_data(data_path)
            replacements = {
                'DATA_PATH = PROJECT_ROOT / "HelpSteer2"': f"DATA_PATH = Path({str(data_path)!r})",
                'RUN_ID = "pilot-001"': 'RUN_ID = "notebook-smoke"',
                "RUNTIME_OVERRIDES = {}": (
                    f"RUNTIME_OVERRIDES = {{'output_root': {str(output_root)!r}}}"
                ),
            }
            for name in NOTEBOOK_NAMES:
                notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
                for cell in notebook.cells:
                    if cell.cell_type != "code":
                        continue
                    for old, new in replacements.items():
                        cell.source = cell.source.replace(old, new)
                client = NotebookClient(
                    notebook,
                    timeout=120,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(PROJECT_ROOT)}},
                )
                client.execute()
            run_dir = output_root / "notebook-smoke"
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertFalse((run_dir / "calls.sqlite3").exists())
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_calls_made"], 0)
            self.assertEqual(manifest["runtime_overrides"], {"output_root": str(output_root)})

    @staticmethod
    def _write_synthetic_data(path: Path) -> None:
        rows = [
            {
                "prompt": f"calibration-{score}-{index}",
                "response": f"response-{score}-{index}",
                "helpfulness": score,
            }
            for score in range(5)
            for index in range(8)
        ]
        rows.extend(
            {
                "prompt": f"paired-{prompt_index}",
                "response": f"choice-{response_index}",
                "helpfulness": (prompt_index + response_index) % 5,
            }
            for prompt_index in range(10)
            for response_index in range(2)
        )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    @staticmethod
    def _literal_assignments(source: str) -> dict[str, object]:
        result: dict[str, object] = {}
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                try:
                    result[node.targets[0].id] = ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    pass
        return result


if __name__ == "__main__":
    unittest.main()

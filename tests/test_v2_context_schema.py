from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from rubric_order.context import (
    ContextBudgetError,
    estimate_tokens,
    postflight_context,
    preflight_context,
    preprocess_examples,
)
from rubric_order.readiness import _check_call_context
from rubric_order.sampling import build_calibration_blocks, ordered_examples
from rubric_order.schemas import (
    canonicalize_rubric_v2,
    drift_audit,
    rubric_hashes,
    validate_rubric_v2,
)

ROOT = Path(__file__).parents[1]
INITIAL = json.loads((ROOT / "rubrics" / "initial_helpfulness_v2.json").read_text())


def adaptive(name: str, description: str) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "positive_signals": [f"positive {name}"],
        "negative_signals": [f"negative {name}"],
    }


class ContextAndRubricV2Tests(unittest.TestCase):
    def test_conservative_estimator_and_preflight(self) -> None:
        self.assertEqual(estimate_tokens("x" * 10), 4)
        accepted = preflight_context(
            "x" * 30,
            context_length=100,
            max_output_tokens=20,
            safety_margin_tokens=10,
            method="batch",
            block_id=0,
            order="random",
        )
        self.assertTrue(accepted["allowed"])
        with self.assertRaises(ContextBudgetError):
            preflight_context(
                "x" * 300,
                context_length=100,
                max_output_tokens=20,
                safety_margin_tokens=10,
                method="batch",
                block_id=0,
                order="random",
            )

    def test_postflight_saturation(self) -> None:
        self.assertTrue(
            postflight_context(100, context_length=100, minimum_headroom_tokens=10)["critical"]
        )
        self.assertTrue(
            postflight_context(95, context_length=100, minimum_headroom_tokens=10)[
                "context_saturation"
            ]
        )
        self.assertFalse(
            postflight_context(80, context_length=100, minimum_headroom_tokens=10)[
                "context_saturation"
            ]
        )

    def test_readiness_context_check_rejects_saturated_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calls.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                """
                CREATE TABLE calls (
                    status TEXT, schema_version TEXT, prompt_version TEXT,
                    estimated_input_tokens INTEGER, prompt_eval_count INTEGER,
                    observed_headroom_tokens INTEGER, context_saturation INTEGER,
                    backend_prompt_truncated INTEGER
                )
                """
            )
            connection.execute(
                "INSERT INTO calls VALUES ('success', '2.0', '2.0', 100, 16384, 0, 1, 1)"
            )
            connection.commit()
            connection.close()
            checks: dict[str, bool] = {}
            failures: list[str] = []
            _check_call_context(Path(temporary), checks, failures)
            self.assertFalse(checks["call_context_complete"])
            self.assertTrue(failures)

    def test_preprocessing_occurs_once_before_orders(self) -> None:
        rows = [
            {
                "example_id": f"e-{score}-{index}",
                "prompt_id": f"p-{score}-{index}",
                "prompt": "P " + "x" * 120,
                "response": "R " + "y" * 120,
                "human_score": score,
            }
            for score in range(5)
            for index in range(2)
        ]
        processed = preprocess_examples(
            rows, {"max_prompt_characters": 80, "max_response_characters": 90}
        )
        blocks, _ = build_calibration_blocks(processed, 1, 2, 42)
        versions: dict[str, set[tuple[str, str]]] = {}
        for order in blocks[0]["orders"]:
            for row in ordered_examples(blocks[0], order):
                meta = row["text_processing"]
                versions.setdefault(row["example_id"], set()).add(
                    (meta["used_prompt_sha256"], meta["used_response_sha256"])
                )
                self.assertTrue(meta["prompt_truncated"])
                self.assertTrue(meta["response_truncated"])
        self.assertTrue(all(len(values) == 1 for values in versions.values()))

    def test_v2_core_anchors_duplicates_and_adaptive_limit(self) -> None:
        valid = copy.deepcopy(INITIAL)
        valid["criteria"].extend(
            [adaptive("Zeta", "General zeta"), adaptive("Alpha", "General alpha")]
        )
        canonical = validate_rubric_v2(valid, INITIAL)
        self.assertEqual([item["name"] for item in canonical["criteria"][3:]], ["Alpha", "Zeta"])
        changed_core = copy.deepcopy(valid)
        changed_core["criteria"][0]["description"] = "Changed"
        with self.assertRaises(ValueError):
            validate_rubric_v2(changed_core, INITIAL)
        changed_anchor = copy.deepcopy(valid)
        changed_anchor["score_anchors"]["4"] = "Changed"
        with self.assertRaises(ValueError):
            validate_rubric_v2(changed_anchor, INITIAL)
        duplicate_name = copy.deepcopy(valid)
        duplicate_name["criteria"][4]["name"] = " zEtA "
        with self.assertRaises(ValueError):
            validate_rubric_v2(duplicate_name, INITIAL)
        duplicate_description = copy.deepcopy(valid)
        duplicate_description["criteria"][4]["description"] = "General zeta"
        with self.assertRaises(ValueError):
            validate_rubric_v2(duplicate_description, INITIAL)
        too_many = copy.deepcopy(valid)
        too_many["criteria"].append(adaptive("Third", "Third adaptive"))
        with self.assertRaises(ValueError):
            validate_rubric_v2(too_many, INITIAL)

    def test_canonical_hashes_and_drift_flags(self) -> None:
        value = copy.deepcopy(INITIAL)
        value["criteria"].append(adaptive("request_body", "Uses request body exactly"))
        canonical = canonicalize_rubric_v2(value, INITIAL)
        raw = json.dumps(value, indent=2)
        hashes = rubric_hashes(raw, canonical)
        self.assertNotEqual(hashes["raw_response_sha256"], hashes["canonical_rubric_sha256"])
        report = drift_audit(
            canonical,
            INITIAL,
            [{"prompt": "Return a request_body object", "response": "request_body"}],
        )
        codes = {flag["code"] for flag in report["flags"]}
        self.assertIn("field_name_or_snake_case", codes)
        self.assertTrue(report["requires_human_inspection"])


if __name__ == "__main__":
    unittest.main()

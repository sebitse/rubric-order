from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rubric_order.data import load_dataset, normalize_records
from rubric_order.sampling import select_partitions


class DataTests(unittest.TestCase):
    def test_column_mapping_and_numeric_string_label(self) -> None:
        rows = normalize_records(
            [{"question": "Q", "answer": "A", "rating": "4"}],
            {"prompt": "question", "response": "answer", "label": "rating"},
            source="test",
        )
        self.assertEqual(rows[0]["human_score"], 4)
        self.assertEqual(rows[0]["prompt"], "Q")

    def test_invalid_labels_fail_clearly(self) -> None:
        mapping = {"prompt": "p", "response": "r", "label": "y"}
        for value in (None, "bad", 2.5, -1, 5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_records([{"p": "Q", "r": "A", "y": value}], mapping, "test")

    def test_jsonl_directory_has_official_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {"prompt": "Q", "response": "A", "helpfulness": 2}
            for split in ("train", "validation"):
                (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            config = {
                "dataset": {
                    "path": str(root),
                    "format": "auto",
                    "columns": {
                        "prompt": "prompt",
                        "response": "response",
                        "label": "helpfulness",
                    },
                }
            }
            train, validation = load_dataset(config)
            self.assertEqual(len(train), 1)
            self.assertIsNotNone(validation)
            self.assertEqual(len(validation or []), 1)

    def test_prompt_grouped_split_has_no_leakage(self) -> None:
        raw = [
            {"prompt": f"p{prompt}", "response": f"r{response}", "label": prompt % 5}
            for prompt in range(12)
            for response in range(2)
        ]
        rows = normalize_records(
            raw,
            {"prompt": "prompt", "response": "response", "label": "label"},
            "test",
        )
        calibration, heldout, metadata = select_partitions(rows, None, 4, 2, 42)
        self.assertFalse(
            {row["prompt_id"] for row in calibration} & {row["prompt_id"] for row in heldout}
        )
        self.assertEqual(metadata["prompt_leakage_count"], 0)
        self.assertEqual(len({row["prompt_id"] for row in heldout}), 4)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rubric_order.cli import smoke_test


class SmokeTests(unittest.TestCase):
    def test_complete_no_inference_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "smoke"
            result = smoke_test(output)
            self.assertEqual(result["status"], "ok")
            self.assertFalse(result["ollama_contacted"])
            self.assertFalse((output / "calls.sqlite3").exists())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_calls_made"], 0)
            dry = json.loads(Path(result["dry_run_manifest"]).read_text(encoding="utf-8"))
            self.assertFalse(dry["inference_executed"])
            self.assertFalse(dry["ollama_contacted"])


if __name__ == "__main__":
    unittest.main()

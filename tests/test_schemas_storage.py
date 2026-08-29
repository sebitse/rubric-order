from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rubric_order.schemas import (
    JUDGE_JSON_SCHEMA,
    RUBRIC_JSON_SCHEMA,
    parse_judge_output,
    parse_rubric,
    validate_judge_output,
    validate_rubric,
)
from rubric_order.storage import CallContext, CallStore, cached_validated_call

RUBRIC = {
    "schema_version": "1.0",
    "title": "Test rubric",
    "criteria": [
        {
            "name": "Fulfillment",
            "description": "Addresses the request.",
            "positive_signals": ["complete"],
            "negative_signals": ["missing"],
        }
    ],
    "score_anchors": {str(score): f"Anchor {score}" for score in range(5)},
}


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, **_kwargs: Any) -> dict[str, Any]:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return {
            "response": response,
            "latency_seconds": 0.01,
            "prompt_eval_count": 5,
            "eval_count": 2,
        }


class SchemaAndStorageTests(unittest.TestCase):
    def test_rubric_and_judge_schemas(self) -> None:
        self.assertEqual(validate_rubric(RUBRIC)["title"], "Test rubric")
        self.assertEqual(parse_rubric(json.dumps(RUBRIC))["schema_version"], "1.0")
        self.assertEqual(validate_judge_output({"score": 3}), {"score": 3})
        with self.assertRaises(ValueError):
            validate_judge_output({"score": 3, "reason": "no"})
        with self.assertRaises(ValueError):
            validate_rubric({**RUBRIC, "score_anchors": {"0": "only"}})

    def test_rubric_response_schema_matches_validator_contract(self) -> None:
        self.assertFalse(RUBRIC_JSON_SCHEMA["additionalProperties"])

        criteria_schema = RUBRIC_JSON_SCHEMA["properties"]["criteria"]
        self.assertEqual(criteria_schema["minItems"], 1)
        self.assertEqual(criteria_schema["maxItems"], 5)

        criterion_schema = criteria_schema["items"]
        expected_fields = {
            "name",
            "description",
            "positive_signals",
            "negative_signals",
        }
        self.assertFalse(criterion_schema["additionalProperties"])
        self.assertEqual(set(criterion_schema["required"]), expected_fields)
        self.assertEqual(set(criterion_schema["properties"]), expected_fields)

        anchor_schema = RUBRIC_JSON_SCHEMA["properties"]["score_anchors"]
        self.assertFalse(anchor_schema["additionalProperties"])
        self.assertEqual(set(anchor_schema["properties"]), {str(score) for score in range(5)})

    def test_retry_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            client = FakeClient(["not json"])
            with CallStore(Path(temporary) / "calls.sqlite3") as store:
                with self.assertRaises(ValueError):
                    self._call(client, store, retries=2)
                self.assertEqual(store.status_counts(), {"validation_error": 3})
            self.assertEqual(client.calls, 3)

    def test_cache_resume_and_repeat_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            client = FakeClient(['{"score": 2}'])
            with CallStore(Path(temporary) / "calls.sqlite3") as store:
                first, first_cached = self._call(client, store)
                second, second_cached = self._call(client, store)
                repeated, repeat_cached = self._call(client, store, namespace="repeat:1")
                repeated_again, repeated_again_cached = self._call(
                    client, store, namespace="repeat:1"
                )
            self.assertEqual(first, second)
            self.assertEqual(second, repeated)
            self.assertFalse(first_cached)
            self.assertTrue(second_cached)
            self.assertFalse(repeat_cached)
            self.assertTrue(repeated_again_cached)
            self.assertEqual(client.calls, 2)

    def _call(
        self,
        client: FakeClient,
        store: CallStore,
        retries: int = 2,
        namespace: str = "default",
    ) -> tuple[dict[str, Any], bool]:
        return cached_validated_call(
            client=client,
            store=store,
            context=CallContext("run", "judge", "static", 0, "random", None, 0),
            model_name="mock",
            prompt="prompt",
            model_options={"temperature": 0},
            response_schema=JUDGE_JSON_SCHEMA,
            template_hash="template",
            parser=parse_judge_output,
            max_validation_retries=retries,
            cache_namespace=namespace,
        )


if __name__ == "__main__":
    unittest.main()

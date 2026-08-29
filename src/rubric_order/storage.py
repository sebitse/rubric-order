from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context import postflight_context, sha256_text


@dataclass(frozen=True)
class CallContext:
    run_id: str
    stage: str
    method: str
    block_id: int | None
    order_name: str | None
    step: int | None
    replicate: int


class CallStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> CallStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                method TEXT NOT NULL,
                block_id INTEGER,
                order_name TEXT,
                step INTEGER,
                replicate INTEGER NOT NULL,
                attempt INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                model_options TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                prompt_template_hash TEXT NOT NULL,
                raw_response TEXT,
                parsed_response TEXT,
                status TEXT NOT NULL,
                validation_error TEXT,
                latency_seconds REAL NOT NULL,
                prompt_eval_count INTEGER,
                eval_count INTEGER,
                timestamp TEXT NOT NULL
            )
            """
        )
        additions = {
            "schema_version": "TEXT NOT NULL DEFAULT '1.0'",
            "prompt_version": "TEXT NOT NULL DEFAULT '1.0'",
            "prompt_sha256": "TEXT",
            "response_schema_sha256": "TEXT",
            "raw_response_sha256": "TEXT",
            "canonical_response_sha256": "TEXT",
            "estimated_input_tokens": "INTEGER",
            "observed_headroom_tokens": "INTEGER",
            "context_saturation": "INTEGER",
            "backend_prompt_truncated": "INTEGER",
            "cache_namespace": "TEXT NOT NULL DEFAULT 'default'",
        }
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(calls)")}
        for name, declaration in additions.items():
            if name not in columns:
                self.connection.execute(f"ALTER TABLE calls ADD COLUMN {name} {declaration}")
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS calls_cache ON calls(request_hash, status, id)"
        )
        self.connection.commit()

    def cached_success(self, request_hash: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT parsed_response FROM calls
            WHERE request_hash = ? AND status = 'success'
            ORDER BY id DESC LIMIT 1
            """,
            (request_hash,),
        ).fetchone()
        if row is None or row["parsed_response"] is None:
            return None
        return json.loads(row["parsed_response"])

    def cache_status(self, request_hash: str) -> str:
        """Return a non-mutating status for a hash in the active call store."""
        return "hit" if self.cached_success(request_hash) is not None else "fresh"

    def record(
        self,
        context: CallContext,
        attempt: int,
        model_name: str,
        model_options: dict[str, Any],
        request_hash: str,
        template_hash: str,
        raw_response: str | None,
        parsed_response: dict[str, Any] | None,
        status: str,
        validation_error: str | None,
        latency_seconds: float,
        prompt_eval_count: int | None,
        eval_count: int | None,
        *,
        schema_version: str = "1.0",
        prompt_version: str = "1.0",
        prompt_sha256: str | None = None,
        response_schema_sha256: str | None = None,
        raw_response_sha256: str | None = None,
        canonical_response_sha256: str | None = None,
        estimated_input_tokens: int | None = None,
        observed_headroom_tokens: int | None = None,
        context_saturation: bool | None = None,
        backend_prompt_truncated: bool | None = None,
        cache_namespace: str = "default",
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO calls (
                run_id, stage, method, block_id, order_name, step, replicate, attempt,
                model_name, model_options, request_hash, prompt_template_hash, raw_response,
                parsed_response, status, validation_error, latency_seconds, prompt_eval_count,
                eval_count, timestamp, schema_version, prompt_version, prompt_sha256,
                response_schema_sha256, raw_response_sha256, canonical_response_sha256,
                estimated_input_tokens, observed_headroom_tokens, context_saturation,
                backend_prompt_truncated, cache_namespace
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                context.run_id,
                context.stage,
                context.method,
                context.block_id,
                context.order_name,
                context.step,
                context.replicate,
                attempt,
                model_name,
                json.dumps(model_options, sort_keys=True, separators=(",", ":")),
                request_hash,
                template_hash,
                raw_response,
                json.dumps(parsed_response, ensure_ascii=False) if parsed_response else None,
                status,
                validation_error,
                latency_seconds,
                prompt_eval_count,
                eval_count,
                datetime.now(timezone.utc).isoformat(),
                schema_version,
                prompt_version,
                prompt_sha256,
                response_schema_sha256,
                raw_response_sha256,
                canonical_response_sha256,
                estimated_input_tokens,
                observed_headroom_tokens,
                int(context_saturation) if context_saturation is not None else None,
                int(backend_prompt_truncated) if backend_prompt_truncated is not None else None,
                cache_namespace,
            ),
        )
        self.connection.commit()

    def status_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM calls GROUP BY status"
        ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def latest_metadata(self, context: CallContext) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT schema_version, prompt_version, prompt_sha256, raw_response_sha256,
                   canonical_response_sha256, estimated_input_tokens, prompt_eval_count,
                   observed_headroom_tokens, context_saturation, backend_prompt_truncated,
                   attempt, latency_seconds, cache_namespace
            FROM calls
            WHERE run_id = ? AND stage = ? AND method = ?
              AND block_id IS ? AND order_name IS ? AND step IS ? AND replicate = ?
            ORDER BY id DESC LIMIT 1
            """,
            (
                context.run_id,
                context.stage,
                context.method,
                context.block_id,
                context.order_name,
                context.step,
                context.replicate,
            ),
        ).fetchone()
        return dict(row) if row is not None else None

    def metadata_for_request(self, request_hash: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT schema_version, prompt_version, prompt_sha256, raw_response_sha256,
                   canonical_response_sha256, estimated_input_tokens, prompt_eval_count,
                   observed_headroom_tokens, context_saturation, backend_prompt_truncated,
                   attempt, latency_seconds, cache_namespace
            FROM calls WHERE request_hash = ? AND status = 'success'
            ORDER BY id DESC LIMIT 1
            """,
            (request_hash,),
        ).fetchone()
        return dict(row) if row is not None else None


def make_request_hash(
    model_name: str,
    prompt: str,
    model_options: dict[str, Any],
    response_schema: dict[str, Any],
    template_hash: str,
    cache_namespace: str = "default",
) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "options": model_options,
        "format": response_schema,
        "template_hash": template_hash,
        "cache_namespace": cache_namespace,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_only_cache_status(path: str | Path, request_hash: str) -> str:
    """Inspect a compatible call store without creating or migrating it."""
    database = Path(path)
    if not database.exists():
        return "fresh"
    try:
        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error:
        return "unavailable"
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "calls" not in tables:
            return "unavailable"
        columns = {row[1] for row in connection.execute("PRAGMA table_info(calls)")}
        if not {"request_hash", "status"} <= columns:
            return "unavailable"
        hit = connection.execute(
            """
            SELECT 1 FROM calls
            WHERE request_hash = ? AND status = 'success'
            LIMIT 1
            """,
            (request_hash,),
        ).fetchone()
        return "hit" if hit is not None else "fresh"
    except sqlite3.Error:
        return "unavailable"
    finally:
        connection.close()


def cached_validated_call(
    *,
    client: Any,
    store: CallStore,
    context: CallContext,
    model_name: str,
    prompt: str,
    model_options: dict[str, Any],
    response_schema: dict[str, Any],
    template_hash: str,
    parser: Callable[[str], dict[str, Any]],
    max_validation_retries: int,
    bypass_cache: bool = False,
    cache_namespace: str = "default",
    schema_version: str = "1.0",
    prompt_version: str = "1.0",
    context_preflight: dict[str, Any] | None = None,
    minimum_context_headroom: int | None = None,
    expected_request_hash: str | None = None,
) -> tuple[dict[str, Any], bool]:
    if max_validation_retries > 2:
        raise ValueError("schema validation may be retried at most twice")
    request_hash = make_request_hash(
        model_name,
        prompt,
        model_options,
        response_schema,
        template_hash,
        cache_namespace,
    )
    if expected_request_hash is not None and request_hash != expected_request_hash:
        raise ValueError(
            "prepared request hash differs from execution request hash: "
            f"prepared={expected_request_hash}, execution={request_hash}"
        )
    prompt_hash = sha256_text(prompt)
    schema_hash = sha256_text(json.dumps(response_schema, sort_keys=True, separators=(",", ":")))
    record_metadata = {
        "schema_version": schema_version,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_hash,
        "response_schema_sha256": schema_hash,
        "estimated_input_tokens": (
            context_preflight.get("estimated_input_tokens") if context_preflight else None
        ),
        "cache_namespace": cache_namespace,
    }
    if not bypass_cache:
        cached = store.cached_success(request_hash)
        if cached is not None:
            return cached, True
    last_error: ValueError | None = None
    for attempt in range(max_validation_retries + 1):
        try:
            result = client.generate(
                model=model_name,
                prompt=prompt,
                options=model_options,
                response_schema=response_schema,
            )
        except Exception as exc:
            store.record(
                context=context,
                attempt=attempt,
                model_name=model_name,
                model_options=model_options,
                request_hash=request_hash,
                template_hash=template_hash,
                raw_response=None,
                parsed_response=None,
                status="request_error",
                validation_error=str(exc),
                latency_seconds=0.0,
                prompt_eval_count=None,
                eval_count=None,
                **record_metadata,
            )
            raise
        raw_response = result["response"]
        try:
            parsed = parser(raw_response)
        except ValueError as exc:
            last_error = exc
            store.record(
                context=context,
                attempt=attempt,
                model_name=model_name,
                model_options=model_options,
                request_hash=request_hash,
                template_hash=template_hash,
                raw_response=raw_response,
                parsed_response=None,
                status="validation_error",
                validation_error=str(exc),
                latency_seconds=float(result.get("latency_seconds", 0.0)),
                prompt_eval_count=result.get("prompt_eval_count"),
                eval_count=result.get("eval_count"),
                raw_response_sha256=sha256_text(raw_response),
                **record_metadata,
            )
            continue
        context_result = None
        if minimum_context_headroom is not None:
            context_result = postflight_context(
                result.get("prompt_eval_count"),
                context_length=int(model_options["num_ctx"]),
                minimum_headroom_tokens=minimum_context_headroom,
            )
        canonical_hash = sha256_text(
            json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        context_failed = bool(context_result and context_result["context_saturation"])
        store.record(
            context=context,
            attempt=attempt,
            model_name=model_name,
            model_options=model_options,
            request_hash=request_hash,
            template_hash=template_hash,
            raw_response=raw_response,
            parsed_response=parsed,
            status="context_error" if context_failed else "success",
            validation_error=context_result["reason"] if context_failed else None,
            latency_seconds=float(result.get("latency_seconds", 0.0)),
            prompt_eval_count=result.get("prompt_eval_count"),
            eval_count=result.get("eval_count"),
            raw_response_sha256=sha256_text(raw_response),
            canonical_response_sha256=canonical_hash,
            observed_headroom_tokens=(
                context_result["observed_headroom_tokens"] if context_result else None
            ),
            context_saturation=(context_result["context_saturation"] if context_result else None),
            backend_prompt_truncated=(
                context_result["backend_prompt_truncated"] if context_result else None
            ),
            **record_metadata,
        )
        if context_failed:
            raise ValueError(
                "model call failed context postflight: "
                f"{context_result['reason']}; prompt_eval_count="
                f"{context_result['prompt_eval_count']}, num_ctx={model_options['num_ctx']}"
            )
        return parsed, False
    raise ValueError(
        f"model returned invalid structured output after {max_validation_retries + 1} attempts: "
        f"{last_error}"
    )

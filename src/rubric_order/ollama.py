from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: float = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        options: dict[str, Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": response_schema,
            "options": options,
        }
        if options.get("think") is False:
            payload["think"] = False
            payload["options"] = {key: value for key, value in options.items() if key != "think"}
        started = time.perf_counter()
        response = self._request("POST", "/api/generate", payload)
        response["latency_seconds"] = time.perf_counter() - started
        if not isinstance(response.get("response"), str):
            raise RuntimeError("Ollama response did not contain text in the response field")
        return response

    def health_and_models(self, required_models: list[str] | None = None) -> dict[str, Any]:
        response = self._request("GET", "/api/tags")
        models = [entry.get("name", "") for entry in response.get("models", [])]
        required = required_models or []
        missing = [model for model in required if not _model_available(model, models)]
        return {"ok": True, "models": models, "missing": missing}

    def model_inventory(self, required_models: list[str] | None = None) -> dict[str, Any]:
        """Return read-only installed-model and server provenance."""
        tags = self._request("GET", "/api/tags")
        version = self._request("GET", "/api/version")
        entries = tags.get("models", [])
        installed = [entry.get("name", "") for entry in entries]
        required = required_models or []
        return {
            "ok": True,
            "ollama_version": version.get("version"),
            "models": [
                {
                    "name": entry.get("name"),
                    "digest": entry.get("digest"),
                    "size": entry.get("size"),
                    "details": entry.get("details"),
                }
                for entry in entries
            ],
            "required": required,
            "missing": [model for model in required if not _model_available(model, installed)],
            "read_only": True,
        }

    def unload(self, model: str) -> None:
        self._request(
            "POST",
            "/api/generate",
            {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        )

    def _request(
        self, method: str, endpoint: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed at {self.base_url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("Ollama returned a non-object JSON response")
        if value.get("error"):
            raise RuntimeError(f"Ollama error: {value['error']}")
        return value


class InferenceBudgetClient:
    def __init__(self, client: Any, max_calls: int) -> None:
        self.client = client
        self.max_calls = max_calls
        self.calls = 0

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        if self.calls >= self.max_calls:
            raise RuntimeError(
                f"inference call budget exhausted at --max-calls {self.max_calls}; resume with "
                "a larger limit after inspecting validation failures"
            )
        self.calls += 1
        return self.client.generate(**kwargs)


def _model_available(required: str, installed: list[str]) -> bool:
    if required in installed:
        return True
    if ":" not in required:
        return f"{required}:latest" in installed
    return False


def updater_options(config: dict[str, Any]) -> dict[str, Any]:
    models = config["models"]
    return {
        "temperature": models["temperature"],
        "seed": models["seed"],
        "num_ctx": models["context_length"],
        "num_predict": models["updater_max_tokens"],
        "think": models.get("think", False),
    }


def judge_options(config: dict[str, Any]) -> dict[str, Any]:
    models = config["models"]
    return {
        "temperature": models["temperature"],
        "seed": models["seed"],
        "num_ctx": models["context_length"],
        "num_predict": models["judge_max_tokens"],
        "think": models.get("think", False),
    }

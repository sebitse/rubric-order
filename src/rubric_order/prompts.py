from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .config import resolve_project_path


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    text: str
    sha256: str

    def render(self, **values: Any) -> str:
        try:
            return self.text.format(**values)
        except KeyError as exc:
            raise ValueError(
                f"missing value for prompt template {self.name}: {exc.args[0]}"
            ) from exc


def load_template(config: dict[str, Any], name: str) -> PromptTemplate:
    try:
        configured_path = config["prompts"][name]
    except KeyError as exc:
        raise ValueError(f"no prompt template configured for {name}") from exc
    path = resolve_project_path(config, configured_path)
    text = path.read_text(encoding="utf-8")
    return PromptTemplate(name=name, text=text, sha256=hashlib.sha256(text.encode()).hexdigest())

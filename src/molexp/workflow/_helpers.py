"""Internal helpers for workflow spec — code hashing, ID generation, IR parsing."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from ._graph_decl import TaskRegistration
from .protocols import JSONValue


def _callable_name(f: Callable, fallback: str = "anonymous") -> str:
    return getattr(f, "__name__", None) or fallback


def _to_snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


def _stable_workflow_id(name: str, tasks: list[TaskRegistration]) -> str:
    """Deterministic workflow ID from name + task topology."""
    parts = [name]
    for t in tasks:
        dep_str = ",".join(sorted(t.depends_on))
        parts.append(f"{t.name}:{type(t.fn_or_class).__qualname__}:[{dep_str}]")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _ir_object_list(value: JSONValue | None) -> list[dict[str, JSONValue]]:
    """Narrow a JSONValue field to a list of dicts. Tolerant of None/non-list."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _require_str(obj: dict[str, JSONValue], key: str) -> str:
    """Read a required string field from an IR object. Raises ValueError on absent/wrong type."""
    value = obj.get(key)
    if not isinstance(value, str):
        raise ValueError(
            f"IR object is missing required string field {key!r} (got {type(value).__name__})"
        )
    return value

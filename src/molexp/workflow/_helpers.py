"""Internal helpers for workflow spec — code hashing, ID generation, IR parsing."""

from __future__ import annotations

import ast
import hashlib
import inspect
import re
import textwrap
from collections.abc import Callable

from ._graph_decl import TaskRegistration
from .protocols import JSONValue, TaskBody


def _callable_name(f: Callable, fallback: str = "anonymous") -> str:
    return getattr(f, "__name__", None) or fallback


def _to_snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


def _callable_code_hash(target: TaskBody) -> str | None:
    """Best-effort AST-normalized code hash. Returns None for uninspectable targets."""
    fn = getattr(target, "execute", None) or getattr(target, "run", None) or target
    if not callable(fn):
        return None
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


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

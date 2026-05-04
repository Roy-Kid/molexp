"""Shared JSON-coercion helper used by runner, routes, and storage."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from molexp.agent.types import ArtifactRef


def to_jsonable(value: Any) -> Any:
    """Coerce ``value`` into JSON-friendly Python primitives.

    Dataclass instances flatten into bare ``{field: value}`` dicts (no
    ``__type__`` tag — that lives in the storage layer's wire format).
    Unknown objects fall back to ``repr(value)`` rather than raising,
    so this is safe to call on tool payloads of arbitrary shape.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, ArtifactRef):
        return {
            "kind": value.kind,
            "title": value.title,
            "payload": to_jsonable(value.payload),
            "path": value.path,
        }
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    return repr(value)

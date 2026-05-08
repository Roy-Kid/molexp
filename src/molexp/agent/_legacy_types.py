"""Transitional types kept alive while ``tools/`` and ``sessions/`` are
refactored to consume pydantic-ai shapes directly via
``_pydanticai/harness.py``.

Slice 2 of agent-pydanticai-as-core removes these in favor of pydantic-ai
``ToolDefinition`` / ``ToolCallPart`` etc. — do not extend.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from molexp._typing import HashablePayload, JSONMapping, JSONValue


class ToolSchema(BaseModel):
    """Tool description as the model sees it."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: JSONMapping


class ModelToolCall(BaseModel):
    """A single tool invocation requested by the model."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: JSONMapping


def to_jsonable(value: HashablePayload) -> JSONValue:
    """Best-effort converter from arbitrary objects into JSON-serializable shape."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    return str(value)


__all__ = ["ModelToolCall", "ToolSchema", "to_jsonable"]

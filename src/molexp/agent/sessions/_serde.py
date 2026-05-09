"""Best-effort JSON serializer for session-store payloads.

``to_jsonable`` accepts the loose union ``HashablePayload`` (anything
the session store might persist — pydantic models, dataclasses,
datetimes, paths, lists, dicts, primitives) and returns a
``JSONValue`` that ``json.dumps`` can write without a custom encoder.

History: this helper used to live in ``agent/_legacy_types.py`` as a
bridge during the agent-pydanticai-as-core refactor. The
rectification spec (2026-05-09) moved it here as a private utility
of the sessions subsystem — its only consumer is
:class:`SessionStore`. The function is private to ``agent/sessions/``
and not re-exported.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
from pathlib import Path

from pydantic import BaseModel

from molexp._typing import HashablePayload, JSONValue


def to_jsonable(value: HashablePayload) -> JSONValue:
    """Best-effort converter from arbitrary objects into a JSON-serializable shape.

    The resulting tree is safe to pass to ``json.dumps`` without a
    custom encoder. Falls back to ``str(value)`` for anything the
    branches don't recognize.
    """
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


__all__ = ["to_jsonable"]

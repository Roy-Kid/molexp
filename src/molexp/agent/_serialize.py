"""Shared JSON-coercion helper.

One typed-union helper coerces values into JSON-friendly Python
primitives. ``BaseModel`` instances flatten via ``model_dump(mode="json")``;
unsupported types raise ``TypeError`` rather than fall back to ``repr``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypeAliasType, Union

from pydantic import BaseModel

JsonScalar = Union[None, bool, int, float, str]
JsonValue = TypeAliasType(
    "JsonValue",
    Union[JsonScalar, list["JsonValue"], dict[str, "JsonValue"]],
)
"""Recursive JSON-shaped value type. Pydantic-compatible via TypeAliasType."""

JsonInput = Union[
    BaseModel,
    Mapping,
    Sequence,
    str,
    int,
    float,
    bool,
    None,
    datetime,
    Enum,
    Path,
]


def to_jsonable(value: JsonInput) -> JsonValue:
    """Coerce ``value`` into JSON-friendly Python primitives.

    Supports BaseModel (via ``model_dump(mode="json")``), Mapping (recursed
    with stringified keys), Sequence / list / tuple (recursed), ``datetime``
    (isoformat), ``Enum`` (value), ``Path`` (str), and scalars. Unsupported
    types raise ``TypeError`` — there is no ``repr`` fallback.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"to_jsonable: unsupported type {type(value).__name__!r}; "
        f"extend the JsonInput union or convert at the call site"
    )

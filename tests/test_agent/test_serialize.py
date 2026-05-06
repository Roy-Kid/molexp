"""Tests for the unified ``to_jsonable`` helper.

Contract: a single typed-union helper coerces values into JSON-friendly
Python primitives. BaseModel instances flatten via ``model_dump(mode="json")``.
Unsupported types raise ``TypeError`` (no ``repr()`` fallback).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pytest

from molexp.agent._serialize import to_jsonable
from molexp.agent.types import Goal, Message


class _Color(Enum):
    RED = "red"
    BLUE = "blue"


def test_primitives_pass_through() -> None:
    assert to_jsonable(None) is None
    assert to_jsonable(True) is True
    assert to_jsonable(7) == 7
    assert to_jsonable(1.5) == 1.5
    assert to_jsonable("hi") == "hi"


def test_dict_recurses() -> None:
    assert to_jsonable({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}


def test_list_and_tuple_become_list() -> None:
    assert to_jsonable([1, 2, 3]) == [1, 2, 3]
    assert to_jsonable((1, 2, 3)) == [1, 2, 3]


def test_datetime_isoformats() -> None:
    dt = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    assert to_jsonable(dt) == dt.isoformat()


def test_enum_uses_value() -> None:
    assert to_jsonable(_Color.RED) == "red"


def test_path_becomes_str() -> None:
    p = Path("/tmp/x.txt")
    assert to_jsonable(p) == "/tmp/x.txt"


def test_basemodel_uses_model_dump() -> None:
    msg = Message(role="user", content="hi", metadata={"k": "v"})
    out = to_jsonable(msg)
    assert out == {"role": "user", "content": "hi", "name": None, "metadata": {"k": "v"}}


def test_nested_basemodel_in_dict() -> None:
    goal = Goal(description="d")
    out = to_jsonable({"goal": goal})
    assert isinstance(out, dict)
    assert out["goal"]["description"] == "d"


def test_unsupported_raises_type_error() -> None:
    """No ``repr()`` fallback — unsupported types fail loudly."""

    class Custom:
        pass

    with pytest.raises(TypeError):
        to_jsonable(Custom())  # type: ignore[arg-type]

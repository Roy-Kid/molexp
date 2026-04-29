"""Tests for the artifact-shape detector and convention-based capture."""

from __future__ import annotations

import pytest

from molexp.plugins.agent_pydanticai._pydantic_ai.session import _maybe_artifact
from molexp.plugins.agent_pydanticai.types import ResultArtifactEvent


@pytest.mark.unit
def test_detects_plot_payload():
    result = {
        "kind": "plot",
        "title": "energy vs temperature",
        "data": [{"x": [1, 2, 3], "y": [4, 5, 6]}],
        "layout": {"title": "Demo"},
    }
    event = _maybe_artifact(result)
    assert isinstance(event, ResultArtifactEvent)
    assert event.kind == "plot"
    assert event.title == "energy vs temperature"
    # Title is lifted into the event itself, not duplicated in payload.
    assert "title" not in event.payload
    assert "kind" not in event.payload
    assert event.payload["data"] == [{"x": [1, 2, 3], "y": [4, 5, 6]}]


@pytest.mark.unit
def test_detects_table_payload():
    result = {
        "kind": "table",
        "columns": ["temp", "energy"],
        "rows": [[300, -10], [400, -9]],
    }
    event = _maybe_artifact(result)
    assert isinstance(event, ResultArtifactEvent)
    assert event.kind == "table"
    assert event.payload["columns"] == ["temp", "energy"]


@pytest.mark.unit
def test_detects_text_payload():
    event = _maybe_artifact({"kind": "text", "title": "summary", "body": "hello"})
    assert isinstance(event, ResultArtifactEvent)
    assert event.kind == "text"
    assert event.title == "summary"
    assert event.payload["body"] == "hello"


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        None,
        "string",
        [],
        {},
        {"foo": "bar"},
        {"kind": "unknown", "data": []},
    ],
)
def test_non_artifact_inputs_return_none(value):
    assert _maybe_artifact(value) is None

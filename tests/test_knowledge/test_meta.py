"""Tests for :class:`molexp.knowledge.ConceptMeta` — the ``meta.yaml`` model.

OKF's ``meta.yaml`` is the structured channel of a Concept (the narrative
``index.md`` is separate). ``type`` is the only required field; subtype
fields (config_hash, status, params, …) must round-trip losslessly so the
base model can read any subtype's meta without truncation.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from molexp.knowledge import ConceptMeta


def test_constructs_with_only_type() -> None:
    meta = ConceptMeta(type="run")
    assert meta.type == "run"


def test_rejects_missing_type() -> None:
    with pytest.raises(ValidationError):
        ConceptMeta.model_validate({})
    with pytest.raises(ValidationError):
        ConceptMeta.model_validate({"id": "x"})


def test_from_yaml_parses_via_safe_load() -> None:
    meta = ConceptMeta.from_yaml("type: run\nid: abc123\n")
    assert meta.type == "run"
    assert meta.id == "abc123"


def test_round_trips_unknown_extra_keys_without_loss() -> None:
    run_like = {
        "type": "run",
        "config_hash": "sha256:deadbeef",
        "status": "succeeded",
    }
    meta = ConceptMeta.model_validate(run_like)
    dumped = meta.model_dump(mode="json")
    assert dumped["config_hash"] == "sha256:deadbeef"
    assert dumped["status"] == "succeeded"

    # Full yaml round-trip preserves the subtype keys too.
    reparsed = ConceptMeta.from_yaml(meta.to_yaml())
    reparsed_dump = reparsed.model_dump(mode="json")
    assert reparsed_dump["config_hash"] == "sha256:deadbeef"
    assert reparsed_dump["status"] == "succeeded"


def test_to_yaml_is_reparsable_structured_only() -> None:
    meta = ConceptMeta(type="experiment", id="exp1", tags=["a", "b"])
    text = meta.to_yaml()
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    assert loaded["type"] == "experiment"
    assert loaded["tags"] == ["a", "b"]
    # Structured channel only — no narrative blob field.
    assert "body" not in loaded
    assert "content" not in loaded


def test_is_frozen() -> None:
    meta = ConceptMeta(type="run")
    with pytest.raises(ValidationError):
        meta.type = "experiment"  # type: ignore[misc]

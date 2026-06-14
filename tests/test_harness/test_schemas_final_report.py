"""Tests for the ``FinalReport`` schema (spec ``harness-run-mode-01-substrate``, T01).

RED before implementation: ``FinalReport`` does not exist yet, so the
module-level import fails at collection. After GREEN these assert the
frozen narrative shape: seven required ``str`` fields plus the
``limitations`` / ``next_steps`` lists defaulting to ``[]``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from molexp.harness import FinalReport

_NARRATIVE_FIELDS = (
    "title",
    "objective",
    "methods_summary",
    "test_summary",
    "execution_summary",
    "results",
    "conclusions",
)


def _full_kwargs() -> dict[str, Any]:
    return {name: f"{name} text" for name in _NARRATIVE_FIELDS}


def test_final_report_narrative_fields_and_list_defaults() -> None:
    report = FinalReport(**_full_kwargs())
    for name in _NARRATIVE_FIELDS:
        assert getattr(report, name) == f"{name} text"
    assert report.limitations == []
    assert report.next_steps == []


def test_final_report_accepts_limitations_and_next_steps() -> None:
    report = FinalReport(
        **_full_kwargs(),
        limitations=["single seed only"],
        next_steps=["sweep the temperature grid"],
    )
    assert report.limitations == ["single seed only"]
    assert report.next_steps == ["sweep the temperature grid"]


@pytest.mark.parametrize("missing_field", _NARRATIVE_FIELDS)
def test_final_report_requires_each_narrative_field(missing_field: str) -> None:
    kwargs = _full_kwargs()
    kwargs.pop(missing_field)
    with pytest.raises(ValidationError):
        FinalReport(**kwargs)


def test_final_report_is_frozen() -> None:
    report = FinalReport(**_full_kwargs())
    with pytest.raises(ValidationError):
        report.title = "mutated"  # type: ignore[misc]


def test_final_report_reexported_from_harness() -> None:
    import molexp.harness as h
    from molexp.harness.schemas import FinalReport as FromSchemas
    from molexp.harness.schemas.final_report import FinalReport as Canonical

    assert h.FinalReport is Canonical
    assert FromSchemas is Canonical
    assert FinalReport is Canonical

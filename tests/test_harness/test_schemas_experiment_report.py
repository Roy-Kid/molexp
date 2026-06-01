"""Tests for ExperimentReport (Phase 2 §4.5 schema).

Locks the wire format per harness-goal.md §4.5:
- four required fields: title, objective, system_description, experimental_design
- eight optional/defaulted fields
- frozen pydantic round-trip
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def _required_kwargs() -> dict:
    return {
        "title": "Water NEMD",
        "objective": "Measure ionic mobility under field.",
        "system_description": "SPC/E water + 64 NaCl pairs in cubic box.",
        "experimental_design": "Run 3 replicas at 300K, apply E along z.",
    }


def test_experiment_report_required_fields_only_round_trip() -> None:
    from molexp.harness.schemas.experiment_report import ExperimentReport

    report = ExperimentReport(**_required_kwargs())
    dumped = report.model_dump_json()
    rehydrated = ExperimentReport.model_validate_json(dumped)
    assert rehydrated == report


def test_experiment_report_full_round_trip() -> None:
    from molexp.harness.schemas.experiment_report import ExperimentReport

    report = ExperimentReport(
        **_required_kwargs(),
        background="Prior work shows...",
        scientific_hypothesis="Mobility scales linearly with field.",
        variables=["E_field", "temperature"],
        controlled_conditions=["volume=fixed", "particles=conserved"],
        expected_outputs=["mobility tensor", "trajectory"],
        assumptions=["classical force field is adequate"],
        risks_or_uncertainties=["polarization not modeled"],
        user_questions=["which water model?"],
    )
    dumped = report.model_dump_json()
    rehydrated = ExperimentReport.model_validate_json(dumped)
    assert rehydrated == report


def test_experiment_report_optional_defaults() -> None:
    from molexp.harness.schemas.experiment_report import ExperimentReport

    report = ExperimentReport(**_required_kwargs())
    assert report.background is None
    assert report.scientific_hypothesis is None
    assert report.variables == []
    assert report.controlled_conditions == []
    assert report.expected_outputs == []
    assert report.assumptions == []
    assert report.risks_or_uncertainties == []
    assert report.user_questions == []


def test_experiment_report_missing_required_raises() -> None:
    from molexp.harness.schemas.experiment_report import ExperimentReport

    with pytest.raises(ValidationError):
        ExperimentReport(title="x", objective="y", system_description="z")  # type: ignore[call-arg]


def test_experiment_report_is_frozen() -> None:
    from molexp.harness.schemas.experiment_report import ExperimentReport

    report = ExperimentReport(**_required_kwargs())
    with pytest.raises(ValidationError):
        report.title = "mutated"  # type: ignore[misc]


def test_experiment_report_model_json_schema_is_dict() -> None:
    """spec §3 — output_schema is expected to receive this dict."""
    from molexp.harness.schemas.experiment_report import ExperimentReport

    schema = ExperimentReport.model_json_schema()
    assert isinstance(schema, dict)
    assert "title" in schema.get("properties", {})

"""Tests for ExperimentSpec — the concrete, parameter-resolved spec (step 2).

Locks the wire format: a spec lifts the report's free-text variables /
conditions into provenance-carrying ParameterValues and resolves the open
user_questions. Frozen pydantic round-trip + json-schema shape.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def _spec_kwargs() -> dict:
    from molexp.harness.schemas.experiment_spec import (
        ResolvedQuestion,
        SpecCondition,
        SpecVariable,
    )
    from molexp.harness.schemas.parameter import ParameterValue

    return {
        "id": "spec-1",
        "experiment_report_id": "rep-1",
        "title": "Water NEMD",
        "objective": "Measure ionic mobility under field.",
        "variables": [
            SpecVariable(
                name="E_field",
                value=ParameterValue(value=1.0e6, source="agent_inferred", reason="standard"),
                unit="V/cm",
                expected_type="float",
            )
        ],
        "controlled_conditions": [
            SpecCondition(
                name="temperature",
                value=ParameterValue(value=300, source="user_provided"),
                unit="K",
            )
        ],
        "resolved_questions": [
            ResolvedQuestion(
                question="which water model?",
                answer="SPC/E",
                source="literature_default",
                confidence=0.9,
            )
        ],
        "assumptions": ["classical force field is adequate"],
    }


def test_experiment_spec_full_round_trip() -> None:
    from molexp.harness.schemas.experiment_spec import ExperimentSpec

    spec = ExperimentSpec(**_spec_kwargs())
    rehydrated = ExperimentSpec.model_validate_json(spec.model_dump_json())
    assert rehydrated == spec
    assert spec.variables[0].value.value == 1.0e6
    assert spec.variables[0].value.source == "agent_inferred"
    assert spec.resolved_questions[0].answer == "SPC/E"


def test_experiment_spec_defaults_empty() -> None:
    from molexp.harness.schemas.experiment_spec import ExperimentSpec

    spec = ExperimentSpec(id="spec-1", experiment_report_id="rep-1", title="t", objective="o")
    assert spec.variables == []
    assert spec.controlled_conditions == []
    assert spec.resolved_questions == []
    assert spec.assumptions == []


def test_experiment_spec_is_frozen() -> None:
    from molexp.harness.schemas.experiment_spec import ExperimentSpec

    spec = ExperimentSpec(**_spec_kwargs())
    with pytest.raises(ValidationError):
        spec.title = "mutated"  # type: ignore[misc]


def test_experiment_spec_missing_required_raises() -> None:
    from molexp.harness.schemas.experiment_spec import ExperimentSpec

    with pytest.raises(ValidationError):
        ExperimentSpec(id="spec-1", title="t", objective="o")  # type: ignore[call-arg]


def test_experiment_spec_model_json_schema_is_dict() -> None:
    from molexp.harness.schemas.experiment_spec import ExperimentSpec

    schema = ExperimentSpec.model_json_schema()
    assert isinstance(schema, dict)
    assert "variables" in schema.get("properties", {})

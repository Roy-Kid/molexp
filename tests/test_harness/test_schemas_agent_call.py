"""Tests for AgentCallSpec / AgentCallResult (Phase 2 §10.1).

Locks the contract every AgentGateway impl honors:
- AgentCallSpec field shape (frozen, defaults)
- AgentCallResult field shape (frozen, nested ArtifactRef)
- round-trip via model_dump_json / model_validate_json
- output_schema accepts ExperimentReport.model_json_schema() unchanged
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from molexp.harness.schemas.artifact import ArtifactRef


def _ref(*, kind: str, sha: str, id_: str = "abc12345") -> ArtifactRef:
    return ArtifactRef(
        id=id_,
        kind=kind,  # type: ignore[arg-type]
        uri=f"file:///tmp/{id_}",
        sha256=sha,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="harness",
    )


def test_agent_call_spec_minimal_round_trip() -> None:
    from molexp.harness.schemas.agent_call import AgentCallSpec

    spec = AgentCallSpec(
        agent_name="experiment_report_writer",
        input_artifact_ids=["abc12345"],
        output_schema={"type": "object"},
    )
    dumped = spec.model_dump_json()
    rehydrated = AgentCallSpec.model_validate_json(dumped)
    assert rehydrated == spec
    # defaults
    assert spec.prompt_artifact_id is None
    assert spec.temperature == 0.2
    assert spec.metadata == {}


def test_agent_call_spec_full_round_trip() -> None:
    from molexp.harness.schemas.agent_call import AgentCallSpec

    spec = AgentCallSpec(
        agent_name="x",
        input_artifact_ids=["a", "b"],
        prompt_artifact_id="p",
        output_schema={"type": "object"},
        temperature=0.7,
        metadata={"model": "deepseek-flash"},
    )
    dumped = spec.model_dump_json()
    rehydrated = AgentCallSpec.model_validate_json(dumped)
    assert rehydrated == spec


def test_agent_call_spec_is_frozen() -> None:
    from molexp.harness.schemas.agent_call import AgentCallSpec

    spec = AgentCallSpec(
        agent_name="x",
        input_artifact_ids=[],
        output_schema={},
    )
    with pytest.raises(ValidationError):
        spec.agent_name = "mutated"  # type: ignore[misc]


def test_agent_call_result_round_trip() -> None:
    from molexp.harness.schemas.agent_call import AgentCallResult

    output_ref = _ref(kind="experiment_report", sha="a" * 64, id_="out00001")
    raw_ref = _ref(kind="log", sha="b" * 64, id_="raw00001")

    result = AgentCallResult(
        output_artifact=output_ref,
        raw_response_artifact=raw_ref,
        model="deepseek-flash",
        usage={"prompt_tokens": 12, "completion_tokens": 34},
    )
    dumped = result.model_dump_json()
    rehydrated = AgentCallResult.model_validate_json(dumped)
    assert rehydrated == result
    assert result.output_artifact == output_ref
    assert result.raw_response_artifact == raw_ref


def test_agent_call_result_default_usage() -> None:
    from molexp.harness.schemas.agent_call import AgentCallResult

    output_ref = _ref(kind="experiment_report", sha="a" * 64)
    raw_ref = _ref(kind="log", sha="b" * 64, id_="raw00002")
    result = AgentCallResult(
        output_artifact=output_ref,
        raw_response_artifact=raw_ref,
        model="m",
    )
    assert result.usage == {}


def test_agent_call_result_is_frozen() -> None:
    from molexp.harness.schemas.agent_call import AgentCallResult

    output_ref = _ref(kind="experiment_report", sha="a" * 64)
    raw_ref = _ref(kind="log", sha="b" * 64, id_="raw00003")
    result = AgentCallResult(
        output_artifact=output_ref,
        raw_response_artifact=raw_ref,
        model="m",
    )
    with pytest.raises(ValidationError):
        result.model = "mutated"  # type: ignore[misc]


def test_output_schema_accepts_experiment_report_model_json_schema() -> None:
    """Real-world wiring: AgentCallSpec.output_schema gets ExperimentReport.model_json_schema()."""
    from molexp.harness.schemas.agent_call import AgentCallSpec
    from molexp.harness.schemas.experiment_report import ExperimentReport

    schema = ExperimentReport.model_json_schema()
    spec = AgentCallSpec(
        agent_name="experiment_report_writer",
        input_artifact_ids=[],
        output_schema=schema,
    )
    assert spec.output_schema == schema

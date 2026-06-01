"""Tests for the Phase-2 AgentGateway Protocol + StubAgentGateway.

Locks:
- AgentGateway is runtime_checkable
- StubAgentGateway satisfies the Protocol structurally
- register + call persists both output_artifact and raw_response_artifact
- unknown agent_name raises AgentResponseNotRegisteredError
- StubAgentGateway is NOT re-exported at molexp.harness or molexp.harness.gateways
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest


@pytest.fixture()
def artifact_store(tmp_path: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    return FileArtifactStore(root=tmp_path / "artifacts")


@pytest.fixture()
def stub(artifact_store):
    from molexp.harness.gateways.stub import StubAgentGateway

    return StubAgentGateway(artifact_store=artifact_store)


def test_agent_gateway_is_runtime_checkable_protocol(stub) -> None:
    from molexp.harness.gateways.gateway import AgentGateway

    assert isinstance(stub, AgentGateway)


def test_agent_gateway_re_exported_at_top_level() -> None:
    from molexp.harness import AgentGateway as TopLevelAgentGateway
    from molexp.harness.gateways.gateway import AgentGateway

    assert TopLevelAgentGateway is AgentGateway


def test_stub_call_returns_registered_output_and_raw_artifacts(stub, artifact_store) -> None:
    from molexp.harness.schemas.agent_call import AgentCallSpec
    from molexp.harness.schemas.experiment_report import ExperimentReport

    canned = ExperimentReport(
        title="t",
        objective="o",
        system_description="s",
        experimental_design="e",
    )
    stub.register(
        agent_name="experiment_report_writer",
        output=canned.model_dump(),
        output_kind="experiment_report",
        raw_text="<verbatim LLM transcript>",
        model="stub-model",
        usage={"prompt_tokens": 5, "completion_tokens": 10},
    )

    spec = AgentCallSpec(
        agent_name="experiment_report_writer",
        input_artifact_ids=["user-plan-id"],
        output_schema=ExperimentReport.model_json_schema(),
    )
    result = asyncio.run(stub.call(spec))

    # Both refs must already exist in the artifact store.
    assert artifact_store.get_ref(result.output_artifact.id) == result.output_artifact
    assert artifact_store.get_ref(result.raw_response_artifact.id) == result.raw_response_artifact
    assert result.output_artifact.kind == "experiment_report"
    assert result.raw_response_artifact.kind == "log"
    assert result.model == "stub-model"
    assert result.usage == {"prompt_tokens": 5, "completion_tokens": 10}


def test_stub_call_wires_parent_ids_for_provenance(stub) -> None:
    """Output artifact's parent_ids must include each input_artifact_id so
    StageRunner can wire the derived_from edges automatically.
    """
    from molexp.harness.schemas.agent_call import AgentCallSpec
    from molexp.harness.schemas.experiment_report import ExperimentReport

    stub.register(
        agent_name="experiment_report_writer",
        output={
            "title": "t",
            "objective": "o",
            "system_description": "s",
            "experimental_design": "e",
        },
        output_kind="experiment_report",
    )
    spec = AgentCallSpec(
        agent_name="experiment_report_writer",
        input_artifact_ids=["abc12345", "def67890"],
        output_schema=ExperimentReport.model_json_schema(),
    )
    result = asyncio.run(stub.call(spec))
    assert set(result.output_artifact.parent_ids) == {"abc12345", "def67890"}


def test_stub_call_raises_on_unknown_agent_name(stub) -> None:
    from molexp.harness.errors import AgentResponseNotRegisteredError, HarnessError
    from molexp.harness.gateways.stub import StubAgentGateway  # noqa: F401  imported for clarity
    from molexp.harness.schemas.agent_call import AgentCallSpec

    spec = AgentCallSpec(
        agent_name="never_registered",
        input_artifact_ids=[],
        output_schema={},
    )
    with pytest.raises(AgentResponseNotRegisteredError) as exc:
        asyncio.run(stub.call(spec))
    assert isinstance(exc.value, HarnessError)


def test_stub_agent_gateway_not_re_exported_publicly() -> None:
    """Production code must NOT see StubAgentGateway via molexp.harness or .agents."""
    import molexp.harness as harness
    import molexp.harness.gateways as agents_pkg

    assert "StubAgentGateway" not in dir(harness)
    assert "StubAgentGateway" not in dir(agents_pkg)
    # Only reachable via the stub module's full dotted path.
    stub_mod = importlib.import_module("molexp.harness.gateways.stub")
    assert hasattr(stub_mod, "StubAgentGateway")

"""Tests for the AgentGateway Protocol + its test-only StubAgentGateway.

Locks:
- ``AgentGateway`` is runtime_checkable and ``StubAgentGateway`` satisfies it
  structurally.
- ``call`` persists both the output and raw-response artifacts and wires the
  output's ``parent_ids`` for provenance edges.
- an unknown ``agent_name`` raises ``AgentResponseNotRegisteredError``.
- ``StubAgentGateway`` is NOT re-exported at ``molexp.harness`` or
  ``molexp.harness.gateways`` (production code must never reach the stub).
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

from molexp.harness.gateways.gateway import AgentGateway
from molexp.harness.gateways.stub import StubAgentGateway
from molexp.harness.schemas.agent_call import AgentCallSpec
from molexp.harness.schemas.experiment_report import ExperimentReport


@pytest.fixture()
def artifact_store(tmp_path: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    return FileArtifactStore(root=tmp_path / "artifacts")


@pytest.fixture()
def stub(artifact_store) -> StubAgentGateway:
    return StubAgentGateway(artifact_store=artifact_store)


class TestAgentGatewayProtocol:
    def test_stub_structurally_satisfies_runtime_checkable_protocol(
        self, stub: StubAgentGateway
    ) -> None:
        assert isinstance(stub, AgentGateway)

    def test_stub_is_not_re_exported_at_harness_or_gateways(self) -> None:
        """Production code must NOT see StubAgentGateway via a public import."""
        import molexp.harness as harness
        import molexp.harness.gateways as gateways_pkg

        assert "StubAgentGateway" not in dir(harness)
        assert "StubAgentGateway" not in dir(gateways_pkg)
        # Only reachable via the stub module's full dotted path.
        stub_mod = importlib.import_module("molexp.harness.gateways.stub")
        assert hasattr(stub_mod, "StubAgentGateway")


class TestStubAgentGateway:
    def test_call_persists_output_and_raw_artifacts_with_metadata(
        self, stub: StubAgentGateway, artifact_store
    ) -> None:
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
        assert (
            artifact_store.get_ref(result.raw_response_artifact.id) == result.raw_response_artifact
        )
        assert result.output_artifact.kind == "experiment_report"
        assert result.raw_response_artifact.kind == "log"
        assert result.model == "stub-model"
        assert result.usage == {"prompt_tokens": 5, "completion_tokens": 10}

    def test_call_wires_output_parent_ids_from_input_artifacts(
        self, stub: StubAgentGateway
    ) -> None:
        """Output ``parent_ids`` == input ids so StageRunner wires derived_from edges."""
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

    def test_call_raises_on_unknown_agent_name(self, stub: StubAgentGateway) -> None:
        from molexp.harness.errors import AgentResponseNotRegisteredError, HarnessError

        spec = AgentCallSpec(agent_name="never_registered", input_artifact_ids=[], output_schema={})
        with pytest.raises(AgentResponseNotRegisteredError) as exc:
            asyncio.run(stub.call(spec))
        assert isinstance(exc.value, HarnessError)

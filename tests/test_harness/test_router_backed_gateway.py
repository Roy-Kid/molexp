"""Tests for ``molexp.harness.gateways.router_backed.RouterBackedAgentGateway``.

The production :class:`AgentGateway` driven by a :class:`molexp.agent.router.Router`.
Covers the ``complete_structured`` call path (raw-before-parsed persistence,
tier / MCP resolution, the optional LLM observer, system-prompt + composed-prompt
provenance) plus the plan-agent registry data those gateways are wired from
(``plan_agents.py`` — its sole test owner).

The router fakes here implement the :class:`molexp.agent.router.Router`
Protocol with the minimum surface the gateway needs; importing
``molexp.agent.router`` is legal under the harness charter (the Protocol module
is SDK-free and ``harness → agent`` is an allowed DAG edge).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from molexp.agent.router import (
    AgenticChunk,
    FinalChunk,
    ModelTier,
    RouterTextResult,
    TextDeltaChunk,
    ThinkingDeltaChunk,
    ToolCallChunk,
    ToolResultChunk,
)
from molexp.agent.types import UsageBreakdown

# ── Stub routers (Protocol-conforming, SDK-free) ──────────────────────────


class _StubRouter:
    """Minimal :class:`Router` implementation for offline gateway tests.

    Returns ``response_text`` verbatim from ``complete_text`` and parses
    the same string into the requested ``schema`` for ``complete_structured``.
    Streaming + usage methods are inert.
    """

    def __init__(self, response_text: str, model: str = "stub-router-model") -> None:
        self._response_text = response_text
        self._model = model

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text=self._response_text, raw=None)

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[BaseModel],
        node_id: str = "",
        mcp_tools: tuple[Any, ...] = (),
    ) -> BaseModel:
        return schema.model_validate_json(self._response_text)

    def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        toolsets: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
        before_tool: object | None = None,
        after_tool: object | None = None,
        should_stop: object | None = None,
    ) -> AsyncIterator[AgenticChunk]:
        async def _empty() -> AsyncIterator[AgenticChunk]:  # pragma: no cover
            if False:
                yield  # type: ignore[unreachable]

        return _empty()

    def clear_usage(self) -> None:  # pragma: no cover - inert in tests
        return None

    def snapshot_usage(self) -> UsageBreakdown:  # pragma: no cover - inert
        return UsageBreakdown()


class _RecordingRouter:
    """:class:`Router` stub that records every ``complete_structured`` call.

    Returns ``instance`` (a pre-built, schema-valid object) regardless of the
    surrounding prose in ``user`` — simulating pydantic-ai's ``output_type``
    enforcement. ``complete_text`` raises so a test can assert the gateway no
    longer uses it.
    """

    def __init__(self, instance: BaseModel, model: str = "recording-router") -> None:
        self._instance = instance
        self._model = model
        self.structured_calls: list[dict[str, Any]] = []
        self.complete_text_calls = 0
        self.stream_agentic_calls = 0

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        self.complete_text_calls += 1
        raise AssertionError("complete_text must not be called by the structured gateway path")

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[BaseModel],
        node_id: str = "",
        mcp_tools: tuple[Any, ...] = (),
    ) -> BaseModel:
        self.structured_calls.append(
            {
                "tier": tier,
                "system": system,
                "user": user,
                "schema": schema,
                "node_id": node_id,
                "mcp_tools": mcp_tools,
            }
        )
        return self._instance

    def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        toolsets: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
        before_tool: object | None = None,
        after_tool: object | None = None,
        should_stop: object | None = None,
    ) -> AsyncIterator[AgenticChunk]:
        self.stream_agentic_calls += 1

        async def _empty() -> AsyncIterator[AgenticChunk]:  # pragma: no cover
            if False:
                yield  # type: ignore[unreachable]

        return _empty()

    def clear_usage(self) -> None:  # pragma: no cover - inert in tests
        return None

    def snapshot_usage(self) -> UsageBreakdown:  # pragma: no cover - inert
        return UsageBreakdown()


class _AgenticRouter:
    """:class:`Router` stub whose ``stream_agentic`` replays a canned ReAct trace.

    Yields thinking → tool_call → tool_result → text → ``FinalChunk(final_text)``
    so the agentic gateway path has a full multi-chunk sequence to consume.
    ``complete_structured`` / ``complete_text`` raise so a test can prove the
    agentic branch never touches the structured / text routes; every
    ``stream_agentic`` invocation is recorded in ``agentic_calls``.
    """

    def __init__(self, *, final_text: str, model: str = "agentic-router") -> None:
        self._final_text = final_text
        self._model = model
        self.agentic_calls: list[dict[str, Any]] = []
        self.structured_calls = 0

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        raise AssertionError("complete_text must not be called by the agentic gateway path")

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[BaseModel],
        node_id: str = "",
        mcp_tools: tuple[Any, ...] = (),
    ) -> BaseModel:
        self.structured_calls += 1
        raise AssertionError("complete_structured must not be called by the agentic gateway path")

    def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        toolsets: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
        before_tool: object | None = None,
        after_tool: object | None = None,
        should_stop: object | None = None,
    ) -> AsyncIterator[AgenticChunk]:
        self.agentic_calls.append(
            {
                "prompt": prompt,
                "system": system,
                "tier": tier,
                "tools": tools,
                "toolsets": toolsets,
            }
        )
        final_text = self._final_text

        async def _replay() -> AsyncIterator[AgenticChunk]:
            yield ThinkingDeltaChunk(text="reason about the plan")
            yield ToolCallChunk(tool_name="molcrafts_search", args_summary="find api")
            yield ToolResultChunk(tool_name="molcrafts_search", result_summary="found", ok=True)
            yield TextDeltaChunk(text="drafting answer")
            yield FinalChunk(text=final_text)

        return _replay()

    def clear_usage(self) -> None:  # pragma: no cover - inert in tests
        return None

    def snapshot_usage(self) -> UsageBreakdown:  # pragma: no cover - inert
        return UsageBreakdown()


class _TinyReport(BaseModel):
    title: str
    score: int


class TestRouterBackedAgentGateway:
    async def test_call_persists_raw_then_returns_typed_output(self, tmp_path: Path) -> None:
        """Happy path through ``complete_structured``: the gateway resolves the
        agent's schema + output kind, persists a ``kind="log"`` raw artifact and
        a ``kind="experiment_report"`` parsed-output artifact (both resolvable
        via ``get_ref``, both carrying the input id in ``parent_ids``), and
        reports a non-empty ``model``.
        """
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        canned_payload = {"title": "ok", "score": 7}
        router = _StubRouter(response_text=json.dumps(canned_payload))
        store = FileArtifactStore(root=tmp_path / "artifacts")

        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"tiny_writer": _TinyReport},
            output_kind_by_agent={"tiny_writer": "experiment_report"},
            tier_by_agent={"tiny_writer": ModelTier.DEFAULT},
        )

        parent = store.put_text(
            kind="user_plan", text="run a tiny experiment", created_by="test", parent_ids=[]
        )

        result = await gateway.call(
            AgentCallSpec(
                agent_name="tiny_writer",
                input_artifact_ids=[parent.id],
                output_schema=_TinyReport.model_json_schema(),
            )
        )

        # Both refs are resolvable in the store (persisted before return).
        assert store.get_ref(result.output_artifact.id) == result.output_artifact
        assert store.get_ref(result.raw_response_artifact.id) == result.raw_response_artifact

        # Kinds match the registry / audit invariant.
        assert result.output_artifact.kind == "experiment_report"
        assert result.raw_response_artifact.kind == "log"

        # Parsed + raw bytes round-trip to the canned payload.
        assert json.loads(store.get(result.output_artifact.id).decode("utf-8")) == canned_payload
        assert (
            json.loads(store.get(result.raw_response_artifact.id).decode("utf-8")) == canned_payload
        )

        # Provenance edges propagated from the spec (composed-prompt id added on
        # top — see test_call_persists_composed_prompt_artifact_with_lineage).
        assert parent.id in result.output_artifact.parent_ids
        assert parent.id in result.raw_response_artifact.parent_ids

        assert result.model  # non-empty for audit citations

    async def test_call_notifies_optional_llm_observer(self, tmp_path: Path) -> None:
        """Optional ``on_llm_call`` receives prompt + raw + artifact ids (cache projection)."""
        from molexp.harness.gateways.llm_trace import LlmCallTrace
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        canned_payload = {"title": "ok", "score": 1}
        router = _StubRouter(response_text=json.dumps(canned_payload))
        store = FileArtifactStore(root=tmp_path / "artifacts")
        seen: list[LlmCallTrace] = []

        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"tiny_writer": _TinyReport},
            output_kind_by_agent={"tiny_writer": "experiment_report"},
            tier_by_agent={"tiny_writer": ModelTier.DEFAULT},
            model="test-model",
            on_llm_call=seen.append,
        )
        parent = store.put_text(
            kind="user_plan", text="observe me", created_by="test", parent_ids=[]
        )
        result = await gateway.call(
            AgentCallSpec(
                agent_name="tiny_writer",
                input_artifact_ids=[parent.id],
                output_schema=_TinyReport.model_json_schema(),
            )
        )
        assert len(seen) == 1
        trace = seen[0]
        assert trace.agent_name == "tiny_writer"
        assert trace.model == "test-model"
        assert "observe me" in trace.prompt
        assert json.loads(trace.raw) == canned_payload
        assert trace.prompt_artifact_id
        assert trace.raw_artifact_id == result.raw_response_artifact.id

    async def test_call_invokes_complete_structured_not_complete_text(self, tmp_path: Path) -> None:
        """Planning agents drive ``complete_structured`` only — a recording router
        whose ``complete_text`` raises proves the text route is untouched; the
        registered schema + ``node_id==agent_name`` are forwarded."""
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        router = _RecordingRouter(instance=_TinyReport(title="ok", score=3))
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"tiny_writer": _TinyReport},
            output_kind_by_agent={"tiny_writer": "experiment_report"},
            tier_by_agent={"tiny_writer": ModelTier.DEFAULT},
        )

        await gateway.call(
            AgentCallSpec(
                agent_name="tiny_writer",
                input_artifact_ids=[],
                output_schema=_TinyReport.model_json_schema(),
            )
        )

        assert router.complete_text_calls == 0
        assert len(router.structured_calls) == 1
        assert router.structured_calls[0]["schema"] is _TinyReport
        assert router.structured_calls[0]["node_id"] == "tiny_writer"

    async def test_tier_by_agent_is_required_no_fallback(self, tmp_path: Path) -> None:
        """Every agent must appear in ``tier_by_agent`` — the registry tier is
        forwarded, and an unmapped agent raises rather than defaulting."""
        from molexp.harness.errors import AgentResponseNotRegisteredError
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        router = _RecordingRouter(instance=_TinyReport(title="ok", score=1))
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"heavy_one": _TinyReport, "light_one": _TinyReport},
            output_kind_by_agent={
                "heavy_one": "experiment_report",
                "light_one": "experiment_report",
            },
            tier_by_agent={"heavy_one": ModelTier.HEAVY, "light_one": ModelTier.DEFAULT},
        )

        for name in ("heavy_one", "light_one"):
            await gateway.call(
                AgentCallSpec(
                    agent_name=name,
                    input_artifact_ids=[],
                    output_schema=_TinyReport.model_json_schema(),
                )
            )

        by_node = {c["node_id"]: c["tier"] for c in router.structured_calls}
        assert by_node["heavy_one"] == ModelTier.HEAVY
        assert by_node["light_one"] == ModelTier.DEFAULT

        # Unmapped agent raises (no fallback to gateway default tier).
        bare = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"orphan": _TinyReport},
            output_kind_by_agent={"orphan": "experiment_report"},
            tier_by_agent={},  # explicit empty — must not fall back
        )
        with pytest.raises(AgentResponseNotRegisteredError, match="no model tier"):
            await bare.call(
                AgentCallSpec(
                    agent_name="orphan",
                    input_artifact_ids=[],
                    output_schema=_TinyReport.model_json_schema(),
                )
            )

    async def test_spec_tier_and_use_mcp_override_the_registry(self, tmp_path: Path) -> None:
        """Per-call ``AgentCallSpec.tier`` / ``use_mcp`` override the registry."""
        from molexp.agent.router import McpToolSpec
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        router = _RecordingRouter(instance=_TinyReport(title="ok", score=1))
        store = FileArtifactStore(root=tmp_path / "artifacts")
        molmcp = McpToolSpec(name="molmcp", command="molmcp")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"coder": _TinyReport},
            output_kind_by_agent={"coder": "experiment_report"},
            tier_by_agent={"coder": ModelTier.DEFAULT},
            mcp_tools_by_agent={"coder": (molmcp,)},
        )
        # First pass: default tier, MCP suppressed.
        await gateway.call(
            AgentCallSpec(
                agent_name="coder",
                input_artifact_ids=[],
                output_schema=_TinyReport.model_json_schema(),
                tier="default",
                use_mcp=False,
            )
        )
        # Repair: heavy tier + MCP enabled.
        await gateway.call(
            AgentCallSpec(
                agent_name="coder",
                input_artifact_ids=[],
                output_schema=_TinyReport.model_json_schema(),
                tier="heavy",
                use_mcp=True,
            )
        )
        assert router.structured_calls[0]["tier"] == ModelTier.DEFAULT
        assert router.structured_calls[0]["mcp_tools"] == ()
        assert router.structured_calls[1]["tier"] == ModelTier.HEAVY
        assert router.structured_calls[1]["mcp_tools"] == (molmcp,)

    async def test_registry_mcp_tools_forwarded_by_default(self, tmp_path: Path) -> None:
        """Without a per-call override, a mapped agent's MCP tools flow to the
        router and an unmapped agent gets none."""
        from molexp.agent.router import McpToolSpec
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        router = _RecordingRouter(instance=_TinyReport(title="ok", score=1))
        store = FileArtifactStore(root=tmp_path / "artifacts")
        molmcp = McpToolSpec(name="molmcp", command="molmcp")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"coder": _TinyReport, "reader": _TinyReport},
            output_kind_by_agent={"coder": "experiment_report", "reader": "experiment_report"},
            tier_by_agent={"coder": ModelTier.DEFAULT, "reader": ModelTier.DEFAULT},
            mcp_tools_by_agent={"coder": (molmcp,)},
        )

        for name in ("coder", "reader"):
            await gateway.call(
                AgentCallSpec(
                    agent_name=name,
                    input_artifact_ids=[],
                    output_schema=_TinyReport.model_json_schema(),
                )
            )

        by_node = {c["node_id"]: c["mcp_tools"] for c in router.structured_calls}
        assert by_node["coder"] == (molmcp,)
        assert by_node["reader"] == ()

    async def test_call_raises_on_unknown_agent_name(self, tmp_path: Path) -> None:
        """Unregistered ``agent_name`` raises ``AgentResponseNotRegisteredError``
        (a :class:`HarnessError` subclass, so generic handlers catch both backends)."""
        from molexp.harness.errors import AgentResponseNotRegisteredError, HarnessError
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        router = _StubRouter(response_text='{"title": "x", "score": 1}')
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"tiny_writer": _TinyReport},
            output_kind_by_agent={"tiny_writer": "experiment_report"},
            tier_by_agent={"tiny_writer": ModelTier.DEFAULT},
        )

        with pytest.raises(AgentResponseNotRegisteredError) as exc_info:
            await gateway.call(
                AgentCallSpec(agent_name="unknown_agent", input_artifact_ids=[], output_schema={})
            )
        assert isinstance(exc_info.value, HarnessError)

    async def test_gateway_forwards_system_prompt_into_complete_structured(
        self, tmp_path: Path
    ) -> None:
        """A gateway built with ``prompts_by_agent()`` forwards the matching
        SYSTEM_PROMPT as the ``system=`` arg into ``complete_structured``."""
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.prompts import prompts_by_agent
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        prompts = prompts_by_agent()
        agent_name = "experiment_report_writer"

        router = _RecordingRouter(instance=_TinyReport(title="wired", score=1))
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={agent_name: _TinyReport},
            output_kind_by_agent={agent_name: "experiment_report"},
            tier_by_agent={agent_name: ModelTier.DEFAULT},
            system_prompt_by_agent=prompts,
        )

        await gateway.call(
            AgentCallSpec(
                agent_name=agent_name,
                input_artifact_ids=[],
                output_schema=_TinyReport.model_json_schema(),
            )
        )

        assert len(router.structured_calls) == 1
        assert router.structured_calls[0]["system"] == prompts[agent_name]

    async def test_call_persists_composed_prompt_artifact_with_lineage(
        self, tmp_path: Path
    ) -> None:
        """The composed prompt is persisted as a ``kind="prompt"`` artifact and
        threaded into the output + raw lineage (without dropping the input ids),
        so audit replay can reconstruct the exact LLM *input*. The prompt
        artifact itself derives from the input it was composed from."""
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        router = _RecordingRouter(instance=_TinyReport(title="audited", score=9))
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"tiny_writer": _TinyReport},
            output_kind_by_agent={"tiny_writer": "experiment_report"},
            tier_by_agent={"tiny_writer": ModelTier.DEFAULT},
        )

        parent = store.put_text(
            kind="user_plan", text="run a tiny experiment", created_by="test", parent_ids=[]
        )

        result = await gateway.call(
            AgentCallSpec(
                agent_name="tiny_writer",
                input_artifact_ids=[parent.id],
                output_schema=_TinyReport.model_json_schema(),
            )
        )

        prompt_refs = store.list_by_kind("prompt")
        assert len(prompt_refs) == 1
        prompt_ref = prompt_refs[0]

        # The persisted prompt is the exact composed prompt (the single input's text).
        assert store.get(prompt_ref.id).decode("utf-8") == "run a tiny experiment"

        # Output + raw lineage include BOTH the input id and the prompt id.
        assert parent.id in result.output_artifact.parent_ids
        assert prompt_ref.id in result.output_artifact.parent_ids
        assert parent.id in result.raw_response_artifact.parent_ids
        assert prompt_ref.id in result.raw_response_artifact.parent_ids

        # The prompt artifact derives from the input it was composed from.
        assert parent.id in prompt_ref.parent_ids


class TestStructuredCallModeIsDefault:
    """``AgentCallSpec.call_mode`` defaults to ``"structured"`` — every call
    that predates the agentic branch keeps driving ``complete_structured``."""

    def test_default_call_mode_is_structured(self) -> None:
        """A spec constructed without ``call_mode`` reports ``"structured"``."""
        from molexp.harness.schemas import AgentCallSpec

        spec = AgentCallSpec(
            agent_name="tiny_writer",
            input_artifact_ids=[],
            output_schema={},
        )
        assert spec.call_mode == "structured"

    async def test_structured_mode_never_calls_stream_agentic(self, tmp_path: Path) -> None:
        """An explicit ``call_mode="structured"`` drives ``complete_structured``
        once and NEVER opens the agentic stream (record-style assertion)."""
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        router = _RecordingRouter(instance=_TinyReport(title="ok", score=1))
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"tiny_writer": _TinyReport},
            output_kind_by_agent={"tiny_writer": "experiment_report"},
            tier_by_agent={"tiny_writer": ModelTier.DEFAULT},
        )

        spec = AgentCallSpec(
            agent_name="tiny_writer",
            input_artifact_ids=[],
            output_schema=_TinyReport.model_json_schema(),
            call_mode="structured",
        )
        assert spec.call_mode == "structured"

        await gateway.call(spec)

        assert len(router.structured_calls) == 1
        assert router.stream_agentic_calls == 0


class TestAgenticCallMode:
    """``call_mode="agentic"`` drives the emergent tool loop
    (:meth:`Router.stream_agentic`) instead of one structured round-trip."""

    async def test_agentic_mode_drives_stream_agentic_and_parses_final(
        self, tmp_path: Path
    ) -> None:
        """The gateway opens ``stream_agentic`` (never ``complete_structured``),
        consumes the canned tool_call/tool_result/final sequence, and parses
        ``FinalChunk.text`` into the registered schema, persisted under the
        agent's output kind."""
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        canned = {"title": "from-react", "score": 42}
        router = _AgenticRouter(final_text=json.dumps(canned))
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"agentic_writer": _TinyReport},
            output_kind_by_agent={"agentic_writer": "experiment_plan"},
            tier_by_agent={"agentic_writer": ModelTier.HEAVY},
        )

        parent = store.put_text(
            kind="user_plan", text="drive a react loop", created_by="test", parent_ids=[]
        )
        result = await gateway.call(
            AgentCallSpec(
                agent_name="agentic_writer",
                input_artifact_ids=[parent.id],
                output_schema=_TinyReport.model_json_schema(),
                call_mode="agentic",
            )
        )

        # Agentic route taken exactly once; structured route untouched.
        assert len(router.agentic_calls) == 1
        assert router.structured_calls == 0

        # FinalChunk.text parsed into the registered schema, stored as its kind.
        assert result.output_artifact.kind == "experiment_plan"
        assert json.loads(store.get(result.output_artifact.id).decode("utf-8")) == canned

    async def test_agentic_mode_persists_raw_trace_before_parsed_with_lineage(
        self, tmp_path: Path
    ) -> None:
        """The raw ``kind="log"`` artifact holds the serialized ReAct trace and is
        persisted BEFORE the parsed output; both carry the same lineage as the
        structured path (input id + composed-prompt id)."""
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        router = _AgenticRouter(final_text=json.dumps({"title": "audited", "score": 5}))
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"agentic_writer": _TinyReport},
            output_kind_by_agent={"agentic_writer": "experiment_plan"},
            tier_by_agent={"agentic_writer": ModelTier.HEAVY},
        )

        parent = store.put_text(
            kind="user_plan", text="audit the react trace", created_by="test", parent_ids=[]
        )
        result = await gateway.call(
            AgentCallSpec(
                agent_name="agentic_writer",
                input_artifact_ids=[parent.id],
                output_schema=_TinyReport.model_json_schema(),
                call_mode="agentic",
            )
        )

        # Exactly one raw log holding the serialized ReAct trace (tool call +
        # tool result both survive into the persisted record).
        raw_logs = store.list_by_kind("log")
        assert len(raw_logs) == 1
        assert raw_logs[0].id == result.raw_response_artifact.id
        raw_text = store.get(raw_logs[0].id).decode("utf-8")
        assert "molcrafts_search" in raw_text
        assert "found" in raw_text

        # Raw persisted before parsed (the invalid-json test proves it strongly).
        assert result.raw_response_artifact.kind == "log"
        assert raw_logs[0].created_at <= result.output_artifact.created_at

        # Lineage identical to the structured path: input id + composed prompt id.
        prompt_ref = store.list_by_kind("prompt")[0]
        assert parent.id in result.output_artifact.parent_ids
        assert prompt_ref.id in result.output_artifact.parent_ids
        assert parent.id in result.raw_response_artifact.parent_ids
        assert prompt_ref.id in result.raw_response_artifact.parent_ids

    async def test_agentic_mode_never_calls_complete_structured(self, tmp_path: Path) -> None:
        """A completed agentic call leaves ``complete_structured`` untouched —
        the stub raises there, so a clean run is itself the proof."""
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        router = _AgenticRouter(final_text=json.dumps({"title": "ok", "score": 1}))
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"agentic_writer": _TinyReport},
            output_kind_by_agent={"agentic_writer": "experiment_plan"},
            tier_by_agent={"agentic_writer": ModelTier.HEAVY},
        )

        await gateway.call(
            AgentCallSpec(
                agent_name="agentic_writer",
                input_artifact_ids=[],
                output_schema=_TinyReport.model_json_schema(),
                call_mode="agentic",
            )
        )

        assert router.structured_calls == 0
        assert len(router.agentic_calls) == 1

    async def test_agentic_invalid_final_json_raises_after_raw_persisted(
        self, tmp_path: Path
    ) -> None:
        """When ``FinalChunk.text`` is not valid schema JSON the call RAISES (no
        silent fallback) — but the raw ReAct trace was already persisted, and no
        parsed-output artifact is written."""
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        router = _AgenticRouter(final_text="this is not valid schema json")
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"agentic_writer": _TinyReport},
            output_kind_by_agent={"agentic_writer": "experiment_plan"},
            tier_by_agent={"agentic_writer": ModelTier.HEAVY},
        )

        parent = store.put_text(
            kind="user_plan", text="bad final chunk", created_by="test", parent_ids=[]
        )
        with pytest.raises(ValueError):
            await gateway.call(
                AgentCallSpec(
                    agent_name="agentic_writer",
                    input_artifact_ids=[parent.id],
                    output_schema=_TinyReport.model_json_schema(),
                    call_mode="agentic",
                )
            )

        # Raw ReAct trace already on disk despite the parse failure.
        raw_logs = store.list_by_kind("log")
        assert len(raw_logs) == 1
        assert "molcrafts_search" in store.get(raw_logs[0].id).decode("utf-8")

        # No parsed output artifact was persisted.
        assert store.list_by_kind("experiment_plan") == []


class TestCallModeValidation:
    """``call_mode`` is a closed ``Literal`` — unknown values are rejected."""

    def test_illegal_call_mode_rejected_by_pydantic(self) -> None:
        from pydantic import ValidationError

        from molexp.harness.schemas import AgentCallSpec

        with pytest.raises(ValidationError):
            AgentCallSpec(
                agent_name="tiny_writer",
                input_artifact_ids=[],
                output_schema={},
                call_mode="totally-bogus",  # type: ignore[arg-type]
            )


class TestPlanAgentRegistry:
    """The ``plan_agents.py`` registry the production gateway is wired from
    (this file is its sole test owner)."""

    def test_mcp_servers_cover_code_writers_only(self) -> None:
        """The codegen agents are wired to consult molmcp; readers are not."""
        from molexp.harness.gateways import plan_agent_mcp_servers, plan_agent_responses

        servers = plan_agent_mcp_servers()
        assert set(servers) <= set(plan_agent_responses())
        for coder in ("workflow_source_writer", "test_code_file_writer"):
            assert servers[coder] == ("molmcp",)
        assert "plan_reviewer" not in servers

    def test_tiers_cover_every_agent_explicitly(self) -> None:
        """Every registered agent has an explicit tier — no omissions/fallback.

        Authors (including codegen) are HEAVY; only plan_reviewer is DEFAULT.
        """
        from molexp.harness.gateways import plan_agent_responses, plan_agent_tiers

        responses = plan_agent_responses()
        tiers = plan_agent_tiers()
        assert set(tiers) == set(responses)  # full coverage, no gateway fallback
        assert tiers["plan_reviewer"] == ModelTier.DEFAULT
        authors = set(responses) - {"plan_reviewer"}
        assert all(tiers[a] == ModelTier.HEAVY for a in authors)

"""Tests for ``RouterBackedAgentGateway``.

Originally written for spec 03a (T2) against the ``complete_text`` +
``schema.model_validate_json`` path. Spec
``plan-mode-revival-02-structured-planning`` switches the gateway onto
``router.complete_structured`` (pydantic-ai's native ``output_type`` +
``output_retries``), which returns a parsed ``schema`` instance directly.
This file is extended with the structured-path, prose-resilience, and
prompt-wiring cases for that switch.

The router fakes here implement the :class:`molexp.agent.router.Router`
Protocol with the minimum surface needed by the gateway. Importing
``molexp.agent.router`` is legal under the harness charter — the Protocol
module is SDK-free and ``harness → agent`` is an allowed DAG edge.
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
    ModelTier,
    Router,
    RouterTextResult,
)
from molexp.agent.types import UsageBreakdown

# ── Stub router (Protocol-conforming, SDK-free) ───────────────────────────


class _StubRouter:
    """Minimal :class:`Router` implementation for offline gateway tests.

    Returns ``response_text`` verbatim from ``complete_text`` and parses
    the same string into the requested ``schema`` for
    ``complete_structured``. Streaming + usage methods are inert.
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
    ) -> BaseModel:
        return schema.model_validate_json(self._response_text)

    def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
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

    Returns ``instance`` (a pre-built, schema-valid object) regardless of
    the surrounding prose in ``user`` — simulating pydantic-ai's
    ``output_type`` enforcement, which yields a valid object even when the
    raw model turn is wrapped in markdown/prose. ``complete_text`` raises
    if invoked so a test can assert the gateway no longer uses it.
    """

    def __init__(self, instance: BaseModel, model: str = "recording-router") -> None:
        self._instance = instance
        self._model = model
        self.structured_calls: list[dict[str, Any]] = []
        self.complete_text_calls = 0

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
    ) -> BaseModel:
        self.structured_calls.append(
            {
                "tier": tier,
                "system": system,
                "user": user,
                "schema": schema,
                "node_id": node_id,
            }
        )
        return self._instance

    def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        async def _empty() -> AsyncIterator[AgenticChunk]:  # pragma: no cover
            if False:
                yield  # type: ignore[unreachable]

        return _empty()

    def clear_usage(self) -> None:  # pragma: no cover - inert in tests
        return None

    def snapshot_usage(self) -> UsageBreakdown:  # pragma: no cover - inert
        return UsageBreakdown()


def _assert_stub_is_router() -> None:
    """Sanity check the stubs satisfy the runtime-checkable Protocol."""
    assert isinstance(_StubRouter(response_text="{}"), Router)


class _TinyReport(BaseModel):
    title: str
    score: int


# ── Happy path: structured output persisted + returned ────────────────────


@pytest.mark.asyncio
async def test_call_persists_raw_then_returns_typed_output(tmp_path: Path) -> None:
    """ac-002 — happy path through the ``complete_structured`` route.

    The gateway must:

    * resolve ``agent_name="tiny_writer"`` to the registered schema +
      output kind;
    * persist a ``kind="log"`` raw artifact holding
      ``instance.model_dump_json()`` through the injected store;
    * persist a ``kind="experiment_report"`` parsed-output artifact;
    * wire ``parent_ids = spec.input_artifact_ids`` on both refs so
      :class:`StageRunner` can emit ``derived_from`` edges;
    * return an :class:`AgentCallResult` whose ``output_artifact`` and
      ``raw_response_artifact`` are resolvable via ``get_ref``.
    """
    from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
    from molexp.harness.schemas import AgentCallSpec
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    _assert_stub_is_router()

    canned_payload = {"title": "ok", "score": 7}
    canned_json = json.dumps(canned_payload)

    router = _StubRouter(response_text=canned_json)
    store = FileArtifactStore(root=tmp_path / "artifacts")

    gateway = RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses={"tiny_writer": _TinyReport},
        output_kind_by_agent={"tiny_writer": "experiment_report"},
    )

    parent = store.put_text(
        kind="user_plan",
        text="run a tiny experiment",
        created_by="test",
        parent_ids=[],
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

    # Parsed output bytes parse back to the canned payload.
    parsed_output = json.loads(store.get(result.output_artifact.id).decode("utf-8"))
    assert parsed_output == canned_payload

    # Raw "log" artifact holds the instance's model_dump_json (semantically).
    raw_bytes = store.get(result.raw_response_artifact.id).decode("utf-8")
    assert json.loads(raw_bytes) == canned_payload

    # Provenance edges propagated from the spec (input ids retained; the
    # composed-prompt artifact id is added on top — see the dedicated test).
    assert parent.id in result.output_artifact.parent_ids
    assert parent.id in result.raw_response_artifact.parent_ids

    # Result metadata: ``model`` field is non-empty for audit citations.
    assert result.model


# ── ac-001: gateway uses complete_structured, never complete_text ─────────


@pytest.mark.asyncio
async def test_call_invokes_complete_structured_not_complete_text(
    tmp_path: Path,
) -> None:
    """ac-001 — planning agents drive ``complete_structured`` only.

    A recording router whose ``complete_text`` raises proves the gateway
    no longer touches the text route; ``complete_structured`` is recorded
    with the registered schema.
    """
    from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
    from molexp.harness.schemas import AgentCallSpec
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    instance = _TinyReport(title="ok", score=3)
    router = _RecordingRouter(instance=instance)
    store = FileArtifactStore(root=tmp_path / "artifacts")
    gateway = RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses={"tiny_writer": _TinyReport},
        output_kind_by_agent={"tiny_writer": "experiment_report"},
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
    # node_id is forwarded as the agent_name for traceability.
    assert router.structured_calls[0]["node_id"] == "tiny_writer"


# ── ac-003: prose-wrapped output still yields a schema-valid object ───────


@pytest.mark.asyncio
async def test_prose_wrapped_output_yields_valid_artifact(tmp_path: Path) -> None:
    """ac-003 — the structured path closes the prose-crash regression.

    A router that returns a valid instance regardless of surrounding prose
    (as pydantic-ai's ``output_type`` enforcement does) lets ``call``
    return a resolvable, schema-valid output artifact instead of raising —
    where the OLD ``complete_text`` + ``model_validate_json`` path crashed
    on prose.
    """
    from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
    from molexp.harness.schemas import AgentCallResult, AgentCallSpec
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    instance = _TinyReport(title="resilient", score=42)
    router = _RecordingRouter(instance=instance)
    store = FileArtifactStore(root=tmp_path / "artifacts")
    gateway = RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses={"tiny_writer": _TinyReport},
        output_kind_by_agent={"tiny_writer": "experiment_report"},
    )

    result = await gateway.call(
        AgentCallSpec(
            agent_name="tiny_writer",
            input_artifact_ids=[],
            output_schema=_TinyReport.model_json_schema(),
        )
    )

    assert isinstance(result, AgentCallResult)
    parsed = json.loads(store.get(result.output_artifact.id).decode("utf-8"))
    assert parsed == {"title": "resilient", "score": 42}


# ── ac-004: unknown agent_name still raises ───────────────────────────────


@pytest.mark.asyncio
async def test_call_raises_on_unknown_agent_name(tmp_path: Path) -> None:
    """ac-004 — unregistered ``agent_name`` raises ``AgentResponseNotRegisteredError``.

    Parity with :class:`StubAgentGateway`; the exception is a
    :class:`HarnessError` subclass so generic handlers catch both backends.
    """
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
    )

    with pytest.raises(AgentResponseNotRegisteredError) as exc_info:
        await gateway.call(
            AgentCallSpec(
                agent_name="unknown_agent",
                input_artifact_ids=[],
                output_schema={},
            )
        )
    assert isinstance(exc_info.value, HarnessError)


# ── ac-002: raw persisted before parsed (both with parent_ids) ────────────


@pytest.mark.asyncio
async def test_raw_log_persisted_with_parent_ids_alongside_output(
    tmp_path: Path,
) -> None:
    """ac-002 — kind="log" raw + registered-kind output both carry parent_ids.

    The §10.2 invariant: the raw record (now ``instance.model_dump_json()``)
    is persisted as ``kind="log"`` and the parsed output under the
    registered kind, both with ``parent_ids == spec.input_artifact_ids``.
    """
    from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
    from molexp.harness.schemas import AgentCallSpec
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    instance = _TinyReport(title="audited", score=9)
    router = _RecordingRouter(instance=instance)
    store = FileArtifactStore(root=tmp_path / "artifacts")
    gateway = RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses={"tiny_writer": _TinyReport},
        output_kind_by_agent={"tiny_writer": "experiment_report"},
    )

    parent = store.put_text(
        kind="user_plan",
        text="plan",
        created_by="test",
        parent_ids=[],
    )

    result = await gateway.call(
        AgentCallSpec(
            agent_name="tiny_writer",
            input_artifact_ids=[parent.id],
            output_schema=_TinyReport.model_json_schema(),
        )
    )

    # Raw log artifact present, holds the model_dump_json of the instance.
    log_refs = store.list_by_kind("log")
    log_payloads = [store.get(ref.id).decode("utf-8") for ref in log_refs]
    assert json.loads(instance.model_dump_json()) in [json.loads(p) for p in log_payloads]

    # Both refs carry the spec's parent_ids (plus the composed-prompt id).
    assert result.raw_response_artifact.kind == "log"
    assert parent.id in result.raw_response_artifact.parent_ids
    assert parent.id in result.output_artifact.parent_ids


# ── ac-007: gateway forwards the per-agent SYSTEM_PROMPT into structured ──


@pytest.mark.asyncio
async def test_gateway_forwards_system_prompt_into_complete_structured(
    tmp_path: Path,
) -> None:
    """ac-007 — a gateway built with ``prompts_by_agent()`` forwards the
    matching ``SYSTEM_PROMPT`` as the ``system=`` arg into
    ``complete_structured`` for a known ``agent_name``.
    """
    from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
    from molexp.harness.prompts import prompts_by_agent
    from molexp.harness.schemas import AgentCallSpec
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    prompts = prompts_by_agent()
    agent_name = "experiment_report_writer"

    instance = _TinyReport(title="wired", score=1)
    router = _RecordingRouter(instance=instance)
    store = FileArtifactStore(root=tmp_path / "artifacts")
    gateway = RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses={agent_name: _TinyReport},
        output_kind_by_agent={agent_name: "experiment_report"},
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


# ── prompt provenance: composed prompt persisted + wired into lineage ─────


class TestRouterBackedAgentGatewayPromptProvenance:
    """The gateway persists the composed prompt as a first-class artifact."""

    @pytest.mark.asyncio
    async def test_call_persists_composed_prompt_artifact_with_lineage(
        self,
        tmp_path: Path,
    ) -> None:
        """The composed prompt is persisted as a ``kind="prompt"`` artifact and
        threaded into the output + raw lineage — without dropping the input ids.

        Audit replay can then reconstruct the exact LLM *input*, not just its
        response. The prompt artifact itself derives from the input artifacts.
        """
        from molexp.harness.gateways.router_backed import RouterBackedAgentGateway
        from molexp.harness.schemas import AgentCallSpec
        from molexp.harness.store.file_artifact_store import FileArtifactStore

        instance = _TinyReport(title="audited", score=9)
        router = _RecordingRouter(instance=instance)
        store = FileArtifactStore(root=tmp_path / "artifacts")
        gateway = RouterBackedAgentGateway(
            router=router,
            artifact_store=store,
            agent_responses={"tiny_writer": _TinyReport},
            output_kind_by_agent={"tiny_writer": "experiment_report"},
        )

        parent = store.put_text(
            kind="user_plan",
            text="run a tiny experiment",
            created_by="test",
            parent_ids=[],
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

        # The persisted prompt is the exact composed prompt (here, the single
        # input artifact's text).
        assert store.get(prompt_ref.id).decode("utf-8") == "run a tiny experiment"

        # Output + raw lineage include BOTH the input id and the prompt id.
        assert parent.id in result.output_artifact.parent_ids
        assert prompt_ref.id in result.output_artifact.parent_ids
        assert parent.id in result.raw_response_artifact.parent_ids
        assert prompt_ref.id in result.raw_response_artifact.parent_ids

        # The prompt artifact derives from the input it was composed from.
        assert parent.id in prompt_ref.parent_ids

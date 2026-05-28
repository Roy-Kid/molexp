"""Tests for ``RouterBackedAgentGateway`` (spec 03a, T2).

This file is intentionally RED before T5 implements the gateway: the
import of ``RouterBackedAgentGateway`` fails, which proves the failure
mode is "class does not exist yet" — the correct RED for write-mode TDD.

The router fake here implements the
:class:`molexp.agent.router.Router` Protocol with the minimum surface
needed by the gateway: a canned :class:`RouterTextResult` from
``complete_text`` plus a structurally-equivalent
``complete_structured`` that parses the same canned JSON. Importing
``molexp.agent.router`` is legal under spec 03a — the Protocol module
is SDK-free and harness → agent is now an allowed DAG edge.
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


def _assert_stub_is_router() -> None:
    """Sanity check the stub satisfies the runtime-checkable Protocol."""
    assert isinstance(_StubRouter(response_text="{}"), Router)


# ── T2: happy-path RED test ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_persists_raw_then_returns_typed_output(tmp_path: Path) -> None:
    """ac-004 — happy path: registered ``agent_name``, valid JSON response.

    The gateway must:

    * resolve ``agent_name="tiny_writer"`` to the registered schema +
      output kind;
    * persist a ``kind="log"`` raw artifact through the injected store;
    * persist a ``kind="experiment_report"`` parsed-output artifact;
    * wire ``parent_ids = spec.input_artifact_ids`` on both refs so
      :class:`StageRunner` can emit ``derived_from`` edges;
    * return an :class:`AgentCallResult` whose ``output_artifact`` and
      ``raw_response_artifact`` are resolvable via
      ``artifact_store.get_ref``.
    """
    # Import inside the test body so the RED failure mode is the missing
    # symbol (``ImportError``) rather than collection-time crash, matching
    # the project's test-body-imports convention for stage tests.
    from molexp.harness.agents.router_backed import RouterBackedAgentGateway
    from molexp.harness.schemas import AgentCallSpec
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    _assert_stub_is_router()

    class TinyReport(BaseModel):
        title: str
        score: int

    canned_payload = {"title": "ok", "score": 7}
    canned_json = json.dumps(canned_payload)

    router = _StubRouter(response_text=canned_json)
    store = FileArtifactStore(root=tmp_path / "artifacts")

    gateway = RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses={"tiny_writer": TinyReport},
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
            output_schema=TinyReport.model_json_schema(),
        )
    )

    # Both refs are resolvable in the store (i.e. were persisted before
    # the call returned).
    assert store.get_ref(result.output_artifact.id) == result.output_artifact
    assert store.get_ref(result.raw_response_artifact.id) == result.raw_response_artifact

    # Kinds match the registry / audit invariant.
    assert result.output_artifact.kind == "experiment_report"
    assert result.raw_response_artifact.kind == "log"

    # Parsed output bytes parse back to the canned payload (the gateway
    # may use compact or indented JSON — assert semantically, not on the
    # exact byte sequence).
    parsed_output = json.loads(store.get(result.output_artifact.id).decode("utf-8"))
    assert parsed_output == canned_payload

    # Provenance edges propagated from the spec.
    assert result.output_artifact.parent_ids == [parent.id]
    assert result.raw_response_artifact.parent_ids == [parent.id]

    # Result metadata: ``model`` field is non-empty so audit reports can
    # cite which router answered.
    assert result.model


# ── T3: error parity with StubAgentGateway ────────────────────────────────


@pytest.mark.asyncio
async def test_call_raises_on_unknown_agent_name(tmp_path: Path) -> None:
    """ac-005 — unregistered ``agent_name`` raises ``AgentResponseNotRegisteredError``.

    Parity with :class:`StubAgentGateway`'s behavior so harness stages can
    raise the same exception regardless of which backend is wired in.
    """
    from molexp.harness.agents.router_backed import RouterBackedAgentGateway
    from molexp.harness.errors import AgentResponseNotRegisteredError, HarnessError
    from molexp.harness.schemas import AgentCallSpec
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    class TinyReport(BaseModel):
        title: str

    router = _StubRouter(response_text='{"title": "x"}')
    store = FileArtifactStore(root=tmp_path / "artifacts")
    gateway = RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses={"tiny_writer": TinyReport},
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
    # AgentResponseNotRegisteredError must be a HarnessError subclass so
    # generic harness error handlers catch both stub + router-backed.
    assert isinstance(exc_info.value, HarnessError)


# ── T4: audit ordering invariant (§10.2) ──────────────────────────────────


@pytest.mark.asyncio
async def test_raw_artifact_persisted_before_typed_output_on_parse_failure(
    tmp_path: Path,
) -> None:
    """ac-006 — even when structured-output parsing fails, the raw artifact
    must already be in the store. This is the audit invariant from
    ``.claude/notes/harness-goal.md`` §10.2: audit replay can always
    recover what the LLM emitted, independent of whether downstream
    parsing succeeded.
    """
    from pydantic import ValidationError

    from molexp.harness.agents.router_backed import RouterBackedAgentGateway
    from molexp.harness.schemas import AgentCallSpec
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    class StrictSchema(BaseModel):
        required_int: int

    # Schema requires int; the router returns a string. ``model_validate_json``
    # will raise ``ValidationError`` AFTER the gateway has already persisted
    # the raw response.
    malformed = '{"required_int": "not an int"}'
    router = _StubRouter(response_text=malformed)
    store = FileArtifactStore(root=tmp_path / "artifacts")
    gateway = RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses={"strict_writer": StrictSchema},
        output_kind_by_agent={"strict_writer": "experiment_report"},
    )

    with pytest.raises(ValidationError):
        await gateway.call(
            AgentCallSpec(
                agent_name="strict_writer",
                input_artifact_ids=[],
                output_schema=StrictSchema.model_json_schema(),
            )
        )

    # Audit invariant: raw artifact is in the store.
    raw_refs = store.list_by_kind("log")
    raw_payloads = [store.get(ref.id).decode("utf-8") for ref in raw_refs]
    assert malformed in raw_payloads

    # And no parsed-output artifact was persisted — partial state would be
    # ambiguous.
    assert store.list_by_kind("experiment_report") == []

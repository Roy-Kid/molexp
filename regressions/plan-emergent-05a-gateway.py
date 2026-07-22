"""Regression: both ``RouterBackedAgentGateway`` dispatch branches, offline.

Binding runtime example for spec ``plan-emergent-05a-gateway``. A
library-external user drives the **production** agent gateway
(:class:`molexp.harness.RouterBackedAgentGateway`) end to end, wired from the
shared plan-agent registry (``plan_agent_responses`` / ``plan_agent_tiers`` /
``plan_output_kinds`` / ``plan_system_prompts``) exactly as the CLI and server
shells build it — using only documented import surfaces. A single
process-internal fake :class:`molexp.agent.router.Router` backs BOTH gateway
paths so one gateway proves the ``call_mode`` fork.

Two reference branch checks:

1. **structured** (``call_mode="structured"``) — one ``plan_report_renderer``
   call routes through ``Router.complete_structured`` (returning an
   :class:`ExperimentReport`), persisting a ``kind="plan_report"`` output
   artifact and a ``kind="log"`` raw artifact — both resolvable through the
   store, both carrying the composed-prompt artifact id in their lineage — and
   NEVER opens ``Router.stream_agentic``.
2. **agentic** (``call_mode="agentic"``) — one ``create_experiment_plan`` call
   drives ``Router.stream_agentic``, consuming a canned ReAct chunk trace whose
   terminal ``FinalChunk.text`` is valid :class:`ExperimentSpec` JSON. It
   persists a ``kind="experiment_plan"`` output and a ``kind="log"`` raw
   artifact (the serialized ReAct trace, naming the dispatched tool) — both
   resolvable, both threading the composed-prompt id — and NEVER calls
   ``Router.complete_structured``.

Deterministic + offline: no network, no subprocess, no server, no CLI, no real
LLM. Run standalone with ``python regressions/plan-emergent-05a-gateway.py``;
the final line on success is ``plan-emergent-05a-gateway: ok``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

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
from molexp.harness import FileArtifactStore, RouterBackedAgentGateway
from molexp.harness.gateways.plan_agents import (
    plan_agent_responses,
    plan_agent_tiers,
    plan_output_kinds,
    plan_system_prompts,
)
from molexp.harness.schemas import (
    AgentCallResult,
    AgentCallSpec,
    ExperimentReport,
    ExperimentSpec,
    PlanArtifactRef,
)


class _PlanRouter:
    """Process-internal fake :class:`Router` driving BOTH gateway branches.

    * ``complete_structured`` returns a pre-built :class:`ExperimentReport` (the
      structured ``plan_report_renderer`` output) and counts each call.
    * ``stream_agentic`` replays a canned ReAct trace whose terminal
      ``FinalChunk.text`` is valid :class:`ExperimentSpec` JSON (the agentic
      ``create_experiment_plan`` output) and counts each call.

    ``complete_text`` raises — neither branch uses the text route. The usage
    methods are inert, mirroring the stub in
    ``tests/test_harness/test_router_backed_gateway.py``. The opaque
    ``message_history`` / ``tools`` / hook parameters follow the SDK-free
    ``Router`` Protocol shape (``tuple[Any, ...]`` / ``object | None``).
    """

    def __init__(self, *, report: ExperimentReport, spec_json: str) -> None:
        self._report = report
        self._spec_json = spec_json
        self.structured_calls = 0
        self.agentic_calls = 0

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        del prompt, system, message_history, tier
        raise AssertionError("complete_text must not be called by either gateway branch")

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
        del tier, system, user, schema, node_id, mcp_tools
        self.structured_calls += 1
        return self._report

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
        del prompt, system, tools, toolsets, tier, message_history
        del before_tool, after_tool, should_stop
        self.agentic_calls += 1
        spec_json = self._spec_json

        async def _replay() -> AsyncIterator[AgenticChunk]:
            yield ThinkingDeltaChunk(text="reason about the LJ melting experiment")
            yield ToolCallChunk(
                tool_name="molcrafts_search", args_summary="lennard jones potential"
            )
            yield ToolResultChunk(
                tool_name="molcrafts_search", result_summary="found lj_potential", ok=True
            )
            yield TextDeltaChunk(text="drafting the experiment spec")
            yield FinalChunk(text=spec_json)

        return _replay()

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _build_gateway(store: FileArtifactStore, router: _PlanRouter) -> RouterBackedAgentGateway:
    """Wire the gateway from the shared plan-agent registry (CLI/server parity)."""
    return RouterBackedAgentGateway(
        router=router,
        artifact_store=store,
        agent_responses=plan_agent_responses(),
        output_kind_by_agent=plan_output_kinds(),
        tier_by_agent=plan_agent_tiers(),
        system_prompt_by_agent=plan_system_prompts(),
    )


def _prompt_parent(store: FileArtifactStore, ref: PlanArtifactRef) -> PlanArtifactRef:
    """Return the single ``kind="prompt"`` lineage parent of *ref*."""
    prompts = [store.get_ref(pid) for pid in ref.parent_ids if store.get_ref(pid).kind == "prompt"]
    assert len(prompts) == 1, f"expected exactly one prompt parent, got {len(prompts)}"
    return prompts[0]


async def _check_structured_branch(
    gateway: RouterBackedAgentGateway, store: FileArtifactStore, router: _PlanRouter
) -> None:
    """Branch 1: ``plan_report_renderer`` drives ``complete_structured`` only."""
    parent = store.put_text(
        kind="user_plan",
        text="Estimate the LJ melting point via stepwise heating.",
        created_by="regression",
        parent_ids=[],
    )
    agentic_before = router.agentic_calls
    result: AgentCallResult = await gateway.call(
        AgentCallSpec(
            agent_name="plan_report_renderer",
            input_artifact_ids=[parent.id],
            output_schema=ExperimentReport.model_json_schema(),
            call_mode="structured",
        )
    )
    output = result.output_artifact
    raw = result.raw_response_artifact
    prompt_ref = _prompt_parent(store, output)

    # Reference observations (printed before asserting).
    print(f"[structured] output artifact kind        = {output.kind!r}")
    print(f"[structured] raw artifact kind            = {raw.kind!r}")
    print(f"[structured] output resolvable in store   = {store.get_ref(output.id) == output}")
    print(f"[structured] raw resolvable in store      = {store.get_ref(raw.id) == raw}")
    print(f"[structured] prompt id in output lineage  = {prompt_ref.id in output.parent_ids}")
    print(f"[structured] stream_agentic call count    = {router.agentic_calls}")

    assert output.kind == "plan_report", (
        f"structured output kind must be plan_report, got {output.kind!r}"
    )
    assert raw.kind == "log", f"raw response kind must be log, got {raw.kind!r}"
    assert store.get_ref(output.id) == output, "output artifact must resolve through the store"
    assert store.get_ref(raw.id) == raw, "raw artifact must resolve through the store"
    assert prompt_ref.id in output.parent_ids, "output lineage must include the composed-prompt id"
    assert prompt_ref.id in raw.parent_ids, "raw lineage must include the composed-prompt id"
    assert parent.id in output.parent_ids, "output lineage must retain the input id"
    assert router.agentic_calls == agentic_before, "structured branch must NOT open stream_agentic"


async def _check_agentic_branch(
    gateway: RouterBackedAgentGateway,
    store: FileArtifactStore,
    router: _PlanRouter,
    spec_json: str,
) -> None:
    """Branch 2: ``create_experiment_plan`` drives ``stream_agentic`` only."""
    parent = store.put_text(
        kind="user_plan",
        text="Ground a concrete LJ melting plan on the live toolchain.",
        created_by="regression",
        parent_ids=[],
    )
    structured_before = router.structured_calls
    result: AgentCallResult = await gateway.call(
        AgentCallSpec(
            agent_name="create_experiment_plan",
            input_artifact_ids=[parent.id],
            output_schema=ExperimentSpec.model_json_schema(),
            call_mode="agentic",
        )
    )
    output = result.output_artifact
    raw = result.raw_response_artifact
    prompt_ref = _prompt_parent(store, output)
    raw_text = store.get(raw.id).decode("utf-8")
    output_json = json.loads(store.get(output.id).decode("utf-8"))

    # Reference observations (printed before asserting).
    print(f"[agentic] output artifact kind           = {output.kind!r}")
    print(f"[agentic] raw artifact kind              = {raw.kind!r}")
    print(f"[agentic] tool name in raw ReAct trace   = {'molcrafts_search' in raw_text}")
    print(f"[agentic] output resolvable in store     = {store.get_ref(output.id) == output}")
    print(f"[agentic] raw resolvable in store        = {store.get_ref(raw.id) == raw}")
    print(f"[agentic] prompt id in output lineage    = {prompt_ref.id in output.parent_ids}")
    print(f"[agentic] complete_structured call count = {router.structured_calls}")

    assert output.kind == "experiment_plan", (
        f"agentic output kind must be experiment_plan, got {output.kind!r}"
    )
    assert raw.kind == "log", f"raw response kind must be log, got {raw.kind!r}"
    assert "molcrafts_search" in raw_text, "raw ReAct trace must name the dispatched tool"
    assert store.get_ref(output.id) == output, "output artifact must resolve through the store"
    assert store.get_ref(raw.id) == raw, "raw artifact must resolve through the store"
    assert output_json == json.loads(spec_json), (
        "persisted plan must equal the parsed FinalChunk JSON"
    )
    assert prompt_ref.id in output.parent_ids, "output lineage must include the composed-prompt id"
    assert prompt_ref.id in raw.parent_ids, "raw lineage must include the composed-prompt id"
    assert parent.id in output.parent_ids, "output lineage must retain the input id"
    assert router.structured_calls == structured_before, (
        "agentic branch must NOT call complete_structured"
    )


async def main() -> int:
    """Drive both gateway branches through one wiring; print the success marker."""
    tmp = Path(tempfile.mkdtemp(prefix="plan-emergent-05a-gateway-"))
    try:
        report = ExperimentReport(
            title="LJ melting point",
            objective="Estimate the melting temperature of a Lennard-Jones solid.",
            system_description="A Lennard-Jones solid in a periodic simulation box.",
            experimental_design="Heat the solid stepwise and detect the solid->liquid transition.",
        )
        spec_json = ExperimentSpec(
            id="spec-lj-melt",
            experiment_report_id="report-lj-melt",
            title="LJ melting point",
            objective="Estimate the melting temperature of a Lennard-Jones solid.",
        ).model_dump_json()

        router = _PlanRouter(report=report, spec_json=spec_json)
        store = FileArtifactStore(root=tmp / "artifacts")
        gateway = _build_gateway(store, router)

        print("== branch 1: structured plan_report_renderer ==")
        await _check_structured_branch(gateway, store, router)
        print("== branch 2: agentic create_experiment_plan ==")
        await _check_agentic_branch(gateway, store, router, spec_json)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("plan-emergent-05a-gateway: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

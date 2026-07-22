"""Tests for the shared plan-agent registry (``molexp.harness.gateways.plan_agents``).

The CLI (``molexp plan``) and the server both build their gateway from these
five maps, so they must stay internally consistent: the four *dense* maps
(``plan_agent_responses`` / ``plan_agent_tiers`` / ``plan_output_kinds`` /
``plan_system_prompts``) declare identical key sets, and the *sparse*
``plan_agent_mcp_servers`` map is a subset of them.

This spec (``plan-emergent-05a-gateway``) adds two agents to the registry —
``create_experiment_plan`` (the agentic planner, molmcp-grounded, HEAVY,
output kind ``experiment_plan``, schema :class:`ExperimentSpec`) and
``plan_report_renderer`` (HEAVY, no molmcp, output kind ``plan_report``,
schema :class:`ExperimentReport``) — plus the two new well-known artifact
kinds their outputs are persisted under.

The gateway is exercised offline: :func:`build_plan_gateway` receives an inert
Router stub so no LLM is ever called, and a workspace-scoped ``molmcp`` entry
satisfies the codegen fail-close.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from molexp.agent.router import AgenticChunk, ModelTier, RouterTextResult
from molexp.agent.types import UsageBreakdown

# ── Inert Router stub (Protocol-conforming, never actually dispatched) ────────


class _InertRouter:
    """A :class:`Router` that ``build_plan_gateway`` can pass through without
    ever calling — the smoke test only checks how the gateway is *wired*."""

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:  # pragma: no cover - never dispatched
        raise AssertionError("router must not be called by the wiring smoke test")

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[Any],
        node_id: str = "",
        mcp_tools: tuple[Any, ...] = (),
    ) -> Any:  # pragma: no cover - never dispatched
        raise AssertionError("router must not be called by the wiring smoke test")

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
    ) -> AsyncIterator[AgenticChunk]:  # pragma: no cover - never dispatched
        async def _empty() -> AsyncIterator[AgenticChunk]:
            if False:
                yield  # type: ignore[unreachable]

        return _empty()

    def clear_usage(self) -> None:  # pragma: no cover - inert
        return None

    def snapshot_usage(self) -> UsageBreakdown:  # pragma: no cover - inert
        return UsageBreakdown()


class TestDenseMapsAgree:
    """The four dense maps declare one identical key set covering both new agents."""

    def test_four_dense_maps_have_identical_key_sets(self) -> None:
        from molexp.harness.gateways import (
            plan_agent_responses,
            plan_agent_tiers,
            plan_output_kinds,
            plan_system_prompts,
        )

        responses = set(plan_agent_responses())
        assert set(plan_agent_tiers()) == responses
        assert set(plan_output_kinds()) == responses
        assert set(plan_system_prompts()) == responses

    def test_new_agents_present_in_every_dense_map(self) -> None:
        from molexp.harness.gateways import (
            plan_agent_responses,
            plan_agent_tiers,
            plan_output_kinds,
            plan_system_prompts,
        )

        for agent in ("create_experiment_plan", "plan_report_renderer"):
            assert agent in plan_agent_responses()
            assert agent in plan_agent_tiers()
            assert agent in plan_output_kinds()
            assert agent in plan_system_prompts()


class TestNewAgentSchemasAndKinds:
    """The two new agents return the pinned schemas and persist under the pinned kinds."""

    def test_response_schemas_are_pinned(self) -> None:
        from molexp.harness.gateways import plan_agent_responses
        from molexp.harness.schemas import ExperimentReport, ExperimentSpec

        responses = plan_agent_responses()
        assert responses["create_experiment_plan"] is ExperimentSpec
        assert responses["plan_report_renderer"] is ExperimentReport

    def test_output_kinds_are_pinned(self) -> None:
        from molexp.harness.gateways import plan_output_kinds

        kinds = plan_output_kinds()
        assert kinds["create_experiment_plan"] == "experiment_plan"
        assert kinds["plan_report_renderer"] == "plan_report"

    def test_new_agents_run_heavy(self) -> None:
        from molexp.harness.gateways import plan_agent_tiers

        tiers = plan_agent_tiers()
        assert tiers["create_experiment_plan"] == ModelTier.HEAVY
        assert tiers["plan_report_renderer"] == ModelTier.HEAVY


class TestMcpServersMap:
    """``plan_agent_mcp_servers`` stays a subset of the responses map and grounds
    only the agentic planner (``create_experiment_plan``), not the renderer."""

    def test_mcp_servers_is_subset_of_responses(self) -> None:
        from molexp.harness.gateways import plan_agent_mcp_servers, plan_agent_responses

        assert set(plan_agent_mcp_servers()) <= set(plan_agent_responses())

    def test_create_experiment_plan_requires_molmcp(self) -> None:
        from molexp.harness.gateways import plan_agent_mcp_servers

        assert plan_agent_mcp_servers()["create_experiment_plan"] == ("molmcp",)

    def test_plan_report_renderer_needs_no_mcp(self) -> None:
        from molexp.harness.gateways import plan_agent_mcp_servers

        assert "plan_report_renderer" not in plan_agent_mcp_servers()


class TestWellKnownArtifactKinds:
    """The two new output kinds are enumerated as well-known artifact kinds."""

    def test_new_kinds_are_well_known(self) -> None:
        from molexp.harness.schemas import WELL_KNOWN_ARTIFACT_KINDS

        assert "experiment_plan" in WELL_KNOWN_ARTIFACT_KINDS
        assert "plan_report" in WELL_KNOWN_ARTIFACT_KINDS


class TestBuildPlanGatewaySmoke:
    """``build_plan_gateway`` constructs a gateway from the extended maps without
    tripping the ``responses`` ↔ ``output_kinds`` consistency check."""

    def test_build_plan_gateway_wires_new_agents(self, tmp_path: Path) -> None:
        from molexp.harness import RouterBackedAgentGateway
        from molexp.services.plan_runtime.gateway import (
            build_plan_gateway,
            reset_plan_gateway_factory,
        )
        from molexp.workspace import Workspace

        reset_plan_gateway_factory()  # ignore any factory another test installed

        ws = Workspace(root=tmp_path / "lab", name="lab")
        run = ws.add_project("p").add_experiment("e").add_run(id="planr1")

        # Workspace-scoped molmcp so the codegen fail-close (which now also
        # covers ``create_experiment_plan``) resolves without launching anything.
        wsroot = tmp_path / "wsroot"
        wsroot.mkdir()
        (wsroot / "mcp.json").write_text(
            json.dumps(
                {"mcpServers": {"molmcp": {"type": "stdio", "command": "molmcp", "args": []}}}
            ),
            encoding="utf-8",
        )

        gateway = build_plan_gateway(
            model="stub:model",
            run=run,
            router=_InertRouter(),
            workspace_root=wsroot,
        )

        assert isinstance(gateway, RouterBackedAgentGateway)
        # The extended maps flow through unchanged (whitebox pin on the wiring).
        assert gateway._agent_responses["create_experiment_plan"] is not None
        assert gateway._agent_responses["plan_report_renderer"] is not None
        assert gateway._output_kinds["create_experiment_plan"] == "experiment_plan"
        assert gateway._output_kinds["plan_report_renderer"] == "plan_report"

"""Tests for ``PydanticAICapabilityProbe`` (Phase 4).

The probe wraps two ``pydantic_ai.Agent`` instances. The cheap unit
tests below exercise the wiring with a monkeypatched ``Agent`` spy so
we never spin up a real LLM or MCP server. A second optional smoke
test (``test_real_molmcp_smoke``) actually starts a tiny stdio MCP
server in-process; it is gated behind ``MOLEXP_RUN_MCP_SMOKE=1`` so
plain ``pytest`` runs do not pay the cost.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytest.importorskip("pydantic_ai")

from molexp.agent._pydanticai.capability_probe import (
    PydanticAICapabilityProbe,
    build_discovery_agent,
    build_needs_agent,
)
from molexp.agent.modes.plan.capability import (
    CapabilityEvidenceBatch,
    CapabilityNeed,
    CapabilityNeedReport,
)

# ── Spy plumbing ───────────────────────────────────────────────────────────


class _AgentResult:
    def __init__(self, output: object) -> None:
        self.output = output


class _AgentSpy:
    """Captures kwargs handed to pydantic-ai's ``Agent`` constructor."""

    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs
        self._output = kwargs.get("output_type")

    async def run(self, prompt: str) -> _AgentResult:
        del prompt
        if self._output is CapabilityNeedReport:
            return _AgentResult(
                CapabilityNeedReport(discovery_required=False, rationale_summary="spy")
            )
        if self._output is CapabilityEvidenceBatch:
            return _AgentResult(CapabilityEvidenceBatch(discovery_skipped=False))
        return _AgentResult(None)

    async def __aenter__(self) -> _AgentSpy:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset_spy() -> None:
    _AgentSpy.last_kwargs = None


# ── Builders forward kwargs verbatim ───────────────────────────────────────


def test_build_needs_agent_uses_no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("molexp.agent._pydanticai.capability_probe.Agent", _AgentSpy)
    build_needs_agent("test-model")
    captured = _AgentSpy.last_kwargs
    assert captured is not None
    assert captured.get("output_type") is CapabilityNeedReport
    assert "toolsets" not in captured  # no MCP attached
    assert "tools" not in captured


def test_build_discovery_agent_attaches_mcp_via_toolsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("molexp.agent._pydanticai.capability_probe.Agent", _AgentSpy)
    build_discovery_agent(
        "test-model",
        command="molmcp",
        args=("gateway",),
        env={"FOO": "bar"},
        retries=2,
    )
    captured = _AgentSpy.last_kwargs
    assert captured is not None
    assert captured["output_type"] is CapabilityEvidenceBatch
    toolsets = captured["toolsets"]
    assert len(toolsets) == 1
    # ``MCPServerStdio`` is real (not spied) — confirming the type is
    # enough proof that the molmcp entry made it into ``toolsets`` as
    # an MCP server rather than a placeholder, without coupling the
    # test to private attribute names that vary across SDK releases.
    from pydantic_ai.mcp import MCPServerStdio

    assert isinstance(toolsets[0], MCPServerStdio)
    assert captured["retries"] == 2


# ── Probe end-to-end (monkeypatched Agent) ─────────────────────────────────


@pytest.mark.asyncio
async def test_probe_draft_needs_returns_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe's ``draft_needs`` returns the spy's structured output verbatim."""
    monkeypatch.setattr("molexp.agent._pydanticai.capability_probe.Agent", _AgentSpy)
    probe = PydanticAICapabilityProbe(
        model="test-model",
        molmcp_command="molmcp",
    )

    from molexp.agent.modes.plan.schemas import (
        PlanBrief,
        WorkflowContract,
    )

    plan_brief = PlanBrief(overview="o", chosen_method="m")
    contract = WorkflowContract(workflow_id="w", task_io=())
    report = await probe.draft_needs(plan_brief=plan_brief, contract=contract, briefs=())
    assert isinstance(report, CapabilityNeedReport)
    assert report.discovery_required is False


@pytest.mark.asyncio
async def test_probe_discover_short_circuits_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``discovery_required=False`` skips the agent entirely; no MCP startup."""
    monkeypatch.setattr("molexp.agent._pydanticai.capability_probe.Agent", _AgentSpy)
    probe = PydanticAICapabilityProbe(model="test-model", molmcp_command="molmcp")
    pure = CapabilityNeedReport(discovery_required=False)
    batch = await probe.discover(pure)
    assert batch.discovery_skipped is True
    assert batch.evidence == ()


@pytest.mark.asyncio
async def test_probe_discover_runs_agent_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``discovery_required=True`` triggers the spy's ``run`` method."""
    monkeypatch.setattr("molexp.agent._pydanticai.capability_probe.Agent", _AgentSpy)
    probe = PydanticAICapabilityProbe(model="test-model", molmcp_command="molmcp")
    needed = CapabilityNeedReport(
        discovery_required=True,
        needs=(CapabilityNeed(task_id="prepare", capability="x"),),
    )
    batch = await probe.discover(needed)
    assert isinstance(batch, CapabilityEvidenceBatch)
    assert batch.discovery_skipped is False


# ── Optional real-molmcp smoke ─────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("MOLEXP_RUN_MCP_SMOKE") != "1",
    reason="set MOLEXP_RUN_MCP_SMOKE=1 to run the real-molmcp smoke",
)
async def test_real_molmcp_smoke() -> None:
    """End-to-end smoke against a real ``molmcp`` subprocess.

    Only runs when ``MOLEXP_RUN_MCP_SMOKE=1`` is set in the
    environment. Validates that the probe can start the molmcp
    subprocess, hand it to pydantic-ai, and round-trip a discovery
    request through the LLM. Requires both ``molmcp`` on PATH and a
    pydantic-ai-compatible model configured via env vars (e.g.
    ``OPENAI_API_KEY``).
    """
    probe = PydanticAICapabilityProbe(
        model=os.environ.get("MOLEXP_TEST_MODEL", "openai:gpt-4o-mini"),
        molmcp_command="molmcp",
        molmcp_args=("gateway",),
    )
    try:
        report = CapabilityNeedReport(
            discovery_required=True,
            needs=(
                CapabilityNeed(
                    task_id="prepare",
                    capability="construct a peptide",
                    expected_kind="class",
                    query_hints=("peptide", "builder"),
                ),
            ),
        )
        batch = await probe.discover(report)
    finally:
        await probe.aclose()
    assert isinstance(batch, CapabilityEvidenceBatch)

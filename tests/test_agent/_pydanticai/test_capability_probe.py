"""Tests for ``PydanticAICapabilityProbe`` (Phase 4).

The probe wraps two ``pydantic_ai.Agent`` instances. The cheap unit
tests below exercise the wiring with a monkeypatched ``Agent`` spy so
we never spin up a real LLM or MCP server. A second optional smoke
test (``test_real_molmcp_smoke``) actually starts a tiny stdio MCP
server in-process; it is gated behind ``MOLEXP_RUN_MCP_SMOKE=1`` so
plain ``pytest`` runs do not pay the cost.
"""

from __future__ import annotations

import shutil
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
    agent, server = build_discovery_agent(
        "test-model",
        command="molmcp",
        args=(),
        env={"FOO": "bar"},
        retries=2,
    )
    assert isinstance(agent, _AgentSpy)
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
    assert server is toolsets[0]
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

    from molexp.agent.modes.plan.schemas import PlanBrief

    plan_brief = PlanBrief(overview="o", chosen_method="m")
    report = await probe.draft_needs(plan_brief=plan_brief)
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


# ── MCP-handshake contract (no LLM required) ──────────────────────────────


@pytest.mark.asyncio
@pytest.mark.skipif(
    shutil.which("molmcp") is None,
    reason="molmcp not installed on PATH; cannot validate the stdio contract",
)
async def test_molmcp_stdio_handshake_succeeds() -> None:
    """``MCPServerStdio(molmcp)`` completes the MCP initialize handshake.

    This is the test that the previous coverage was missing. The
    monkeypatched ``Agent`` spy tests above never actually launch
    ``molmcp``; the ``test_real_molmcp_smoke`` below double-gates on an
    env flag *and* an LLM API key. So a CLI/contract drift between
    ``molexp.agent.mcp.defaults`` and the ``molmcp`` argparse layer
    flew under the radar until runtime.

    Here we drive only the stdio MCP handshake (``__aenter__`` →
    ``initialize`` → ``__aexit__``) using the *exact* command + args
    seeded into ``mcp.json``. No LLM, no agent, no env-var gate — just
    a subprocess + MCP initialize.
    """
    from pydantic_ai.mcp import MCPServerStdio

    from molexp.agent.mcp.defaults import MCP_DEFAULTS

    name, spec = MCP_DEFAULTS[0]
    assert name == "molmcp"
    server = MCPServerStdio(spec["command"], list(spec["args"]))
    async with server:
        pass  # successful __aenter__ implies a clean MCP initialize round-trip

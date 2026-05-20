"""``PydanticAICapabilityProbe`` tests — the production capability probe.

The probe is the molmcp-backed
:class:`~molexp.agent.modes.plan.protocols.CapabilityProbe` implementation
PlanMode uses in production. It wraps two ``pydantic_ai.Agent``\\ s:
a no-tool needs drafter and an MCP-attached evidence gatherer. These
tests exercise the construction + the narrowed ``probe(*, intent)``
protocol shape with an offline ``TestModel`` (no live LLM, no real MCP
subprocess for the needs path).
"""

from __future__ import annotations

import pytest

from molexp.agent.modes._planning import IntentSpec, ResourceBudget, RiskLevel
from molexp.agent.modes.plan.protocols import CapabilityProbe, ProbeResult

pytest.importorskip("pydantic_ai")


def _intent() -> IntentSpec:
    return IntentSpec(
        objective="run an MD simulation of bulk water",
        non_goals=(),
        required_outputs=("trajectory",),
        constraints=(),
        assumptions=(),
        missing_information=(),
        success_criteria=(),
        allowed_side_effects=(),
        budget=ResourceBudget(max_cost_usd=None, max_wall_seconds=None, model_tier=None),
        risk_level=RiskLevel.low,
    )


def test_probe_module_exists() -> None:
    from molexp.agent._pydanticai import capability_probe

    assert hasattr(capability_probe, "PydanticAICapabilityProbe")


def test_probe_satisfies_protocol() -> None:
    from pydantic_ai.models.test import TestModel

    from molexp.agent._pydanticai.capability_probe import PydanticAICapabilityProbe

    probe = PydanticAICapabilityProbe(
        model=TestModel(),
        molmcp_command="molmcp",
    )
    assert isinstance(probe, CapabilityProbe)


@pytest.mark.asyncio
async def test_probe_returns_probe_result_offline() -> None:
    """The needs path runs offline; the MCP path degrades gracefully.

    ``TestModel`` produces a structurally valid (but empty) drafted-needs
    set; with no real molmcp subprocess the evidence batch comes back
    empty. The probe must still return a well-formed :class:`ProbeResult`.
    """
    from pydantic_ai.models.test import TestModel

    from molexp.agent._pydanticai.capability_probe import PydanticAICapabilityProbe

    async with PydanticAICapabilityProbe(
        model=TestModel(),
        molmcp_command="this-binary-does-not-exist",
    ) as probe:
        result = await probe.probe(intent=_intent())
    assert isinstance(result, ProbeResult)
    # Evidence batch and drafted needs are always present (possibly empty).
    assert result.evidence is not None
    assert isinstance(result.drafted_needs, tuple)

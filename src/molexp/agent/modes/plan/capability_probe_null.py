"""``NullCapabilityProbe`` — the fail-closed capability-probe fallback.

When no MCP source is configured, ``ExploreCapabilities`` falls back to
this probe: it returns an empty :class:`ProbeResult` (no drafted needs,
no evidence). PlanMode then synthesizes a plan against an empty
capability graph — every project-specific API binding fails the
plan-graph preflight, which is the intended fail-closed behaviour (the
plan cannot be approved without evidenced capabilities).

The production probe is
:class:`~molexp.agent._pydanticai.capability_probe.PydanticAICapabilityProbe`;
this one carries no SDK dependency so it is always importable.
"""

from __future__ import annotations

from molexp.agent.modes._planning import IntentSpec
from molexp.agent.modes.plan.protocols import ProbeResult

__all__ = ["NullCapabilityProbe"]


class NullCapabilityProbe:
    """A :class:`~molexp.agent.modes.plan.protocols.CapabilityProbe` that finds nothing.

    Used as the fail-closed fallback when no capability source is wired.
    Every :meth:`probe` call returns an empty :class:`ProbeResult`.
    """

    async def probe(self, *, intent: IntentSpec) -> ProbeResult:
        """Return an empty :class:`ProbeResult` — discovers no capability."""
        del intent  # the null probe ignores intent by design
        return ProbeResult()

"""The narrowed ``CapabilityProbe`` protocol consumed by ``ExploreCapabilities``.

PlanMode's ``ExploreCapabilities`` stage probes the workspace through a
:class:`CapabilityProbe` — a single-method abstraction over capability
discovery. The probe takes a typed
:class:`~molexp.agent.modes._planning.IntentSpec` and returns a flat
:class:`ProbeResult` (drafted needs + an evidence batch);
:func:`~molexp.agent.modes.plan.capability_projection.capability_projection`
turns the flat result into the typed ``CapabilityGraph``.

Two concrete implementations:

* :class:`~molexp.agent.modes.plan.capability_probe_null.NullCapabilityProbe`
  — the fail-closed fallback when no MCP source is configured: returns
  an empty :class:`ProbeResult`.
* :class:`~molexp.agent._pydanticai.capability_probe.PydanticAICapabilityProbe`
  — the production, molmcp-backed probe (a sanctioned ``_pydanticai/``
  pydantic-ai construction).

This module imports neither pydantic SDK — it is a pure protocol +
frozen data type.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes._planning import IntentSpec
from molexp.agent.modes.plan.capability_evidence import (
    CapabilityEvidenceBatch,
    DraftedNeed,
)

__all__ = ["CapabilityProbe", "ProbeResult"]


class ProbeResult(BaseModel):
    """The flat output of one :meth:`CapabilityProbe.probe` call.

    Carries the drafted needs *and* the evidence batch gathered for
    them; :func:`capability_projection` consumes both to build the
    typed :class:`~molexp.agent.modes._planning.CapabilityGraph`.

    Attributes:
        drafted_needs: The capabilities the probe judged the plan needs.
        evidence: The flat evidence batch gathered for those needs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    drafted_needs: tuple[DraftedNeed, ...] = ()
    evidence: CapabilityEvidenceBatch = CapabilityEvidenceBatch()


@runtime_checkable
class CapabilityProbe(Protocol):
    """A single-method abstraction over capability discovery.

    ``ExploreCapabilities`` calls :meth:`probe` with the typed
    ``IntentSpec`` and projects the returned :class:`ProbeResult` into a
    typed ``CapabilityGraph``. Concrete probes own the source-specific
    discovery policy (MCP introspection, transport selection); PlanMode
    deals only in the typed result.
    """

    async def probe(self, *, intent: IntentSpec) -> ProbeResult:
        """Discover the capabilities ``intent`` needs from the workspace.

        Args:
            intent: The typed user-intent contract to probe against.

        Returns:
            A :class:`ProbeResult` with the drafted needs and the
            evidence batch gathered for them.
        """
        ...

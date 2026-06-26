"""``CapabilitySelection`` — the LLM's pick of which grounded capabilities
*this* experiment actually needs.

``ResolveCapabilities`` (plan step 3) grounds the full molcrafts toolchain via
molmcp (~hundreds of capabilities), then asks an LLM to choose the minimal set
needed to realize the concrete :class:`~molexp.harness.schemas.ExperimentSpec`.
This is that structured pick: a short list of ``capability_id``\\ s, each with a
one-line reason. The stage renders only these into the binder-facing
``capability_catalog`` artifact, so the catalog shows *what we need* rather than
the whole index. ``id``\\ s the registry doesn't recognize are dropped by the
stage (the registry stays the source of truth for what binds).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["CapabilitySelection", "SelectedCapability"]


class SelectedCapability(BaseModel):
    """One capability the LLM chose for the experiment, with its rationale."""

    model_config = ConfigDict(frozen=True)

    id: str
    """The ``capability_id`` — must be one of the ids in the grounded registry."""
    reason: str
    """Why this experiment needs this capability (one line)."""


class CapabilitySelection(BaseModel):
    """The LLM's minimal set of capabilities chosen for the experiment."""

    model_config = ConfigDict(frozen=True)

    selected: list[SelectedCapability] = Field(default_factory=list)
    notes: str = ""

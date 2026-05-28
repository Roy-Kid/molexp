"""``HarnessRunContext`` — service container handed to every ``Stage``.

Renamed from the harness-goal's ``RunContext`` to disambiguate from
:class:`molexp.workspace.RunContext` (workspace's per-Run execution
metadata). The full dotted path stays unambiguous; the short class name
matches the package.

Phase 1 carried three stores (artifact / event / provenance). Phase 7
adds three optional service fields (``capability_registry``,
``agent_gateway``, ``approval_policy``) — all default to ``None`` so
Phase-1..6 callers stay byte-identical. Stages introduced in Phase 7+
read these services from the ctx rather than via constructor injection.

Plain Python class (not pydantic) because the fields hold live store
instances with active SQLite connections — exactly the case CLAUDE.md
calls out as forbidden under ``arbitrary_types_allowed`` in pydantic
models. Frozen via ``__slots__`` + a ``__setattr__`` guard so callers can't
mutate the context mid-run.
"""

from __future__ import annotations

from pathlib import Path

from molexp.harness.gateways.gateway import AgentGateway
from molexp.harness.registry.capability_registry import CapabilityRegistry
from molexp.harness.schemas.policy import ApprovalPolicy
from molexp.harness.store.artifact_store import ArtifactStore
from molexp.harness.store.event_log import EventLog
from molexp.harness.store.provenance_store import ProvenanceStore

__all__ = ["HarnessRunContext"]


class HarnessRunContext:
    """Read-only services container threaded through ``StageRunner``."""

    # Class-level annotations so static type checkers can see the attributes.
    run_id: str
    workspace_root: Path
    artifact_store: ArtifactStore
    event_log: EventLog
    provenance_store: ProvenanceStore
    capability_registry: CapabilityRegistry | None
    agent_gateway: AgentGateway | None
    approval_policy: ApprovalPolicy | None

    __slots__ = (
        "_frozen",
        "agent_gateway",
        "approval_policy",
        "artifact_store",
        "capability_registry",
        "event_log",
        "provenance_store",
        "run_id",
        "workspace_root",
    )

    def __init__(
        self,
        *,
        run_id: str,
        workspace_root: Path,
        artifact_store: ArtifactStore,
        event_log: EventLog,
        provenance_store: ProvenanceStore,
        capability_registry: CapabilityRegistry | None = None,
        agent_gateway: AgentGateway | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        # Bypass our own __setattr__ guard while constructing.
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "workspace_root", Path(workspace_root))
        object.__setattr__(self, "artifact_store", artifact_store)
        object.__setattr__(self, "event_log", event_log)
        object.__setattr__(self, "provenance_store", provenance_store)
        object.__setattr__(self, "capability_registry", capability_registry)
        object.__setattr__(self, "agent_gateway", agent_gateway)
        object.__setattr__(self, "approval_policy", approval_policy)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(f"HarnessRunContext is frozen; cannot assign {name!r}")
        object.__setattr__(self, name, value)

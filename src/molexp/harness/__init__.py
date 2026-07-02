"""molexp.harness — lineage-first scientific workflow harness.

Top-level package, sibling of :mod:`molexp.workspace`, :mod:`molexp.workflow`,
and :mod:`molexp.agent`. Owns state, artifact lineage, validation, approval,
execution, and audit for harness-driven runs; agents are demoted to
proposal generators behind hard boundaries defined in
``.claude/notes/harness-goal.md``.

Provenance split (one owner per concern): the harness records
**pipeline-artifact lineage only** — which stage of which run produced which
artifact, derived from which prior artifact (``ArtifactLineageStore``)
— plus the audit event timeline. **Run-level provenance** (params, merged
config + ``config_hash``, profile, workflow identity, execution history,
environment) is owned by :mod:`molexp.workspace` (``RunMetadata`` /
per-scope ``AssetManifest``); code-version and environment capture belong
there, never here.

Public surface (deliberately small): the Mode/Stage execution machinery,
the one shipped pipeline (:class:`PlanMode`), the stores its artifacts and
audit trail live in, the executor seam, the approval gate, and the agent
gateway. Everything else — stage classes, schemas, validators, policies,
curation actions — is imported via its full submodule path
(``molexp.harness.stages`` / ``.schemas`` / ``.validators`` / …); those
subpackages are stable import targets but are not re-exported here.

Dependency direction: ``molexp.harness → molexp.workspace`` plus a single
sanctioned edge ``molexp.harness → molexp.agent.router`` (the SDK-free
Protocol module — see spec ``harness-as-mode-substrate-03a``).
Imports from ``molexp.workflow``, ``molexp.plugins``, ``molexp.server``,
``molexp.cli``, or ``molexp.services`` are forbidden, and ``pydantic_ai`` /
``pydantic_graph`` must not be transitively pulled in — see
``tests/test_harness/test_import_guard.py``.
"""

from __future__ import annotations

from molexp.harness.audit import replay_metadata
from molexp.harness.core import HarnessRunContext, Stage, StageRunner
from molexp.harness.errors import StageExecutionError
from molexp.harness.executors import DryRunExecutor, Executor, LocalExecutor
from molexp.harness.gateways import AgentGateway, RouterBackedAgentGateway
from molexp.harness.mode import Mode
from molexp.harness.modes import PlanMode
from molexp.harness.registry import CapabilityRegistry
from molexp.harness.schemas import ModeResult
from molexp.harness.stages import ApprovalGate
from molexp.harness.store import (
    ArtifactStore,
    FileArtifactStore,
    SQLiteArtifactLineageStore,
    SQLiteEventLog,
)

__all__ = [
    "AgentGateway",
    "ApprovalGate",
    "ArtifactStore",
    "CapabilityRegistry",
    "DryRunExecutor",
    "Executor",
    "FileArtifactStore",
    "HarnessRunContext",
    "LocalExecutor",
    "Mode",
    "ModeResult",
    "PlanMode",
    "RouterBackedAgentGateway",
    "SQLiteArtifactLineageStore",
    "SQLiteEventLog",
    "Stage",
    "StageExecutionError",
    "StageRunner",
    "replay_metadata",
]

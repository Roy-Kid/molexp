"""molexp.harness — plugin host (atom = AgentCall) plus default orchestration plugins.

The host lives at :mod:`molexp.harness.host` (not re-exported here). Workspace,
workflow, agent, and science adapters mount as plugins; :func:`run_plan`
and :class:`ChatMode` are bundles. See ``.claude/notes/harness-plugins.md``.

Provenance split (one owner per concern): the harness records
**pipeline-artifact lineage only** — which stage of which run produced which
artifact, derived from which prior artifact (``ArtifactLineageStore``)
— plus the audit event timeline. **Run-level provenance** (params, merged
config + ``config_hash``, profile, workflow identity, execution history,
environment) is owned by :mod:`molexp.workspace` (``RunMetadata`` /
per-scope ``AssetManifest``); code-version and environment capture belong
there, never here.

Public surface (deliberately small): the Stage execution machinery, the two
shipped bundles (:class:`ChatMode` + :func:`run_plan`), the stores
their artifacts and audit trail live in, the executor seam, the approval
gate, and the agent gateway. Everything else — stage classes, schemas,
validators, policies, curation actions — is imported via its full submodule
path (``molexp.harness.stages`` / ``.schemas`` / ``.validators`` / …).

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
from molexp.harness.errors import ApprovalPendingError, StageExecutionError
from molexp.harness.executors import DryRunExecutor, Executor, LocalExecutor
from molexp.harness.gateways import AgentGateway, RouterBackedAgentGateway
from molexp.harness.modes import ChatMode, chat_loop_config, run_plan
from molexp.harness.registry import CapabilityRegistry
from molexp.harness.schemas import ModeResult
from molexp.harness.stages import ApprovalGate
from molexp.harness.store import (
    ArtifactStore,
    FileArtifactStore,
    SQLiteApprovalStore,
    SQLiteArtifactLineageStore,
    SQLiteEventLog,
)

__all__ = [
    "AgentGateway",
    "ApprovalGate",
    "ApprovalPendingError",
    "ArtifactStore",
    "CapabilityRegistry",
    "ChatMode",
    "DryRunExecutor",
    "Executor",
    "FileArtifactStore",
    "HarnessRunContext",
    "LocalExecutor",
    "ModeResult",
    "RouterBackedAgentGateway",
    "SQLiteApprovalStore",
    "SQLiteArtifactLineageStore",
    "SQLiteEventLog",
    "Stage",
    "StageExecutionError",
    "StageRunner",
    "chat_loop_config",
    "replay_metadata",
    "run_plan",
]

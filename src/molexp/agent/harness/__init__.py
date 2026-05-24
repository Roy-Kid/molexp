"""Agent harness — shared pi-inspired runtime for the four agent modes.

This package is the orchestration-infrastructure layer every
:class:`~molexp.agent.mode.AgentMode` (Plan / Author / Run / Review)
sits on. It owns strictly what pydantic-ai does *not*:

- a typed orchestration-level :data:`~molexp.agent.harness.events.AgentEvent`
  stream emitted *as a mode runs* (:mod:`events`);
- an append-only :class:`~molexp.agent.harness.session.Session`
  entry-tree backed by a :class:`SessionStorage` repository
  (:mod:`session`, :mod:`session_entry`, :mod:`session_storage`);
- LLM-driven context compaction (:mod:`compaction`);
- an :class:`~molexp.agent.harness.execution_env.ExecutionEnv`
  subprocess + scratch-dir abstraction (:mod:`execution_env`);
- the :class:`~molexp.agent.harness.harness.AgentHarness` runtime
  object with a typed :class:`HookRegistry` (:mod:`hooks`,
  :mod:`harness`).

Import-boundary invariant: this package imports neither ``pydantic_ai``
nor ``pydantic_graph``. The one LLM need — the compaction
summarization call — flows through the :class:`~molexp.agent.router.Router`
protocol, whose concrete pydantic-ai implementation lives behind the
``agent/_pydanticai/`` firewall and is injected at runtime.
"""

from __future__ import annotations

from molexp.agent.harness.compaction import (
    CompactionPlan,
    CompactionSettings,
    estimate_tokens,
    prepare_compaction,
)
from molexp.agent.harness.events import (
    AgentEvent,
    AnyAgentEvent,
    ApprovalDecidedEvent,
    ApprovalRequestedEvent,
    ArtifactWrittenEvent,
    ClarificationRequiredEvent,
    CompactionPerformedEvent,
    ErrorEvent,
    EventSink,
    ModeCompletedEvent,
    ModeStartedEvent,
    PlanEmittedEvent,
    PreflightFailedEvent,
    RepairProposedEvent,
    StageCompletedEvent,
    StageStartedEvent,
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from molexp.agent.harness.execution_env import (
    ExecResult,
    ExecutionEnv,
    ExecutionError,
    LocalExecutionEnv,
)
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.harness.hooks import (
    HookContext,
    HookPoint,
    HookRegistry,
)
from molexp.agent.harness.pipeline import execute_pipeline
from molexp.agent.harness.repair import RepairPolicy
from molexp.agent.harness.session import Session
from molexp.agent.harness.session_entry import (
    ApprovalEntry,
    ArtifactEntry,
    CompactionEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionEntry,
    StageEntry,
)
from molexp.agent.harness.session_storage import (
    InMemorySessionStorage,
    JsonlSessionStorage,
    SessionStorage,
)
from molexp.agent.harness.stage import NameOnlyStage, Stage

__all__ = [
    "AgentEvent",
    "AgentHarness",
    "AnyAgentEvent",
    "ApprovalDecidedEvent",
    "ApprovalEntry",
    "ApprovalRequestedEvent",
    "ArtifactEntry",
    "ArtifactWrittenEvent",
    "ClarificationRequiredEvent",
    "CompactionEntry",
    "CompactionPerformedEvent",
    "CompactionPlan",
    "CompactionSettings",
    "ErrorEvent",
    "EventSink",
    "ExecResult",
    "ExecutionEnv",
    "ExecutionError",
    "HookContext",
    "HookPoint",
    "HookRegistry",
    "InMemorySessionStorage",
    "JsonlSessionStorage",
    "LocalExecutionEnv",
    "MessageEntry",
    "ModeCompletedEvent",
    "ModeStartedEvent",
    "ModelChangeEntry",
    "NameOnlyStage",
    "PlanEmittedEvent",
    "PreflightFailedEvent",
    "RepairPolicy",
    "RepairProposedEvent",
    "Session",
    "SessionEntry",
    "SessionStorage",
    "Stage",
    "StageCompletedEvent",
    "StageEntry",
    "StageStartedEvent",
    "TokenDeltaEvent",
    "ToolCallCompletedEvent",
    "ToolCallStartedEvent",
    "estimate_tokens",
    "execute_pipeline",
    "prepare_compaction",
]

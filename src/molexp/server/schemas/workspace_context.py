"""Response schema for ``GET /api/workspace/context``.

A camelCase serialization wrapper over the workspace-layer ``WorkspaceContext``
read-model (``molexp.workspace.workspace_context``). It is a presentation DTO —
not a second model — built directly from the frozen assembler output via
:meth:`WorkspaceContextResponse.from_context`. Field names follow the existing
server convention (camelCase field names, e.g. ``projectCount``).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from molexp._typing import JSONValue

if TYPE_CHECKING:
    from molexp.workspace.workspace_context import (
        ArtifactRef,
        ContextFocus,
        ExperimentRef,
        HealthFlag,
        KnowledgeRef,
        ProjectRef,
        RunRef,
        WorkflowRef,
        WorkspaceContext,
        WorkspaceRef,
    )


class WorkspaceRefResponse(BaseModel):
    id: str
    name: str
    root: str
    targets: list[str] = []

    @classmethod
    def from_ref(cls, ref: WorkspaceRef) -> WorkspaceRefResponse:
        return cls(id=ref.id, name=ref.name, root=ref.root, targets=list(ref.targets))


class ProjectRefResponse(BaseModel):
    id: str
    name: str

    @classmethod
    def from_ref(cls, ref: ProjectRef) -> ProjectRefResponse:
        return cls(id=ref.id, name=ref.name)


class ExperimentRefResponse(BaseModel):
    id: str
    name: str
    projectId: str
    parameterSpace: dict[str, JSONValue] = {}

    @classmethod
    def from_ref(cls, ref: ExperimentRef) -> ExperimentRefResponse:
        return cls(
            id=ref.id,
            name=ref.name,
            projectId=ref.project_id,
            parameterSpace=dict(ref.parameter_space),
        )


class WorkflowRefResponse(BaseModel):
    experimentId: str
    name: str
    irHash: str | None = None

    @classmethod
    def from_ref(cls, ref: WorkflowRef) -> WorkflowRefResponse:
        return cls(experimentId=ref.experiment_id, name=ref.name, irHash=ref.ir_hash)


class RunRefResponse(BaseModel):
    runId: str
    experimentId: str
    projectId: str
    status: str
    configHash: str | None = None
    startedAt: datetime | None = None
    finishedAt: datetime | None = None
    currentExecutionId: str | None = None

    @classmethod
    def from_ref(cls, ref: RunRef) -> RunRefResponse:
        return cls(
            runId=ref.run_id,
            experimentId=ref.experiment_id,
            projectId=ref.project_id,
            status=ref.status,
            configHash=ref.config_hash,
            startedAt=ref.started_at,
            finishedAt=ref.finished_at,
            currentExecutionId=ref.current_execution_id,
        )


class ArtifactRefResponse(BaseModel):
    assetId: str
    scope: str
    kind: str
    path: str
    contentHash: str | None = None
    runId: str | None = None
    executionId: str | None = None
    taskId: str | None = None

    @classmethod
    def from_ref(cls, ref: ArtifactRef) -> ArtifactRefResponse:
        return cls(
            assetId=ref.asset_id,
            scope=ref.scope,
            kind=ref.kind,
            path=ref.path,
            contentHash=ref.content_hash,
            runId=ref.run_id,
            executionId=ref.execution_id,
            taskId=ref.task_id,
        )


class KnowledgeRefResponse(BaseModel):
    path: str
    type: str
    title: str
    id: str | None = None

    @classmethod
    def from_ref(cls, ref: KnowledgeRef) -> KnowledgeRefResponse:
        return cls(path=ref.path, type=ref.type, title=ref.title, id=ref.id)


class HealthFlagResponse(BaseModel):
    kind: str
    ref: str
    detail: str

    @classmethod
    def from_flag(cls, flag: HealthFlag) -> HealthFlagResponse:
        return cls(kind=flag.kind, ref=flag.ref, detail=flag.detail)


class ContextFocusResponse(BaseModel):
    projectId: str | None = None
    experimentId: str | None = None
    runId: str | None = None
    selectedObjectRefs: list[str] = []

    @classmethod
    def from_focus(cls, focus: ContextFocus) -> ContextFocusResponse:
        return cls(
            projectId=focus.project_id,
            experimentId=focus.experiment_id,
            runId=focus.run_id,
            selectedObjectRefs=list(focus.selected_object_refs),
        )


class WorkspaceContextResponse(BaseModel):
    """Camel-cased HTTP view of the canonical ``WorkspaceContext`` read-model."""

    workspace: WorkspaceRefResponse
    focus: ContextFocusResponse
    projects: list[ProjectRefResponse] = []
    experiments: list[ExperimentRefResponse] = []
    workflows: list[WorkflowRefResponse] = []
    recentRuns: list[RunRefResponse] = []
    failedRuns: list[RunRefResponse] = []
    runningRuns: list[RunRefResponse] = []
    artifacts: list[ArtifactRefResponse] = []
    knowledge: list[KnowledgeRefResponse] = []
    openQuestions: list[KnowledgeRefResponse] = []
    staleOrMissing: list[HealthFlagResponse] = []

    @classmethod
    def from_context(cls, ctx: WorkspaceContext) -> WorkspaceContextResponse:
        """Build the HTTP view from a frozen :class:`WorkspaceContext`."""
        return cls(
            workspace=WorkspaceRefResponse.from_ref(ctx.workspace),
            focus=ContextFocusResponse.from_focus(ctx.focus),
            projects=[ProjectRefResponse.from_ref(r) for r in ctx.projects],
            experiments=[ExperimentRefResponse.from_ref(r) for r in ctx.experiments],
            workflows=[WorkflowRefResponse.from_ref(r) for r in ctx.workflows],
            recentRuns=[RunRefResponse.from_ref(r) for r in ctx.recent_runs],
            failedRuns=[RunRefResponse.from_ref(r) for r in ctx.failed_runs],
            runningRuns=[RunRefResponse.from_ref(r) for r in ctx.running_runs],
            artifacts=[ArtifactRefResponse.from_ref(r) for r in ctx.artifacts],
            knowledge=[KnowledgeRefResponse.from_ref(r) for r in ctx.knowledge],
            openQuestions=[KnowledgeRefResponse.from_ref(r) for r in ctx.open_questions],
            staleOrMissing=[HealthFlagResponse.from_flag(r) for r in ctx.stale_or_missing],
        )

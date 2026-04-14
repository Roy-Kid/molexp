"""Pydantic response models for MolExp API.

All ``from_model()`` methods take typed domain objects — no ``Any``,
no ``getattr`` guessing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from molexp.workspace import (
    Asset,
    Experiment,
    Project,
    Run,
)

# ── Project ─────────────────────────────────────────────────────────────────


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    owner: str = ""
    tags: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    created: str
    experimentCount: int | None = None

    @classmethod
    def from_model(cls, project: Project, experiment_count: int | None = None) -> ProjectResponse:
        return cls(
            id=project.id,
            name=project.name,
            description=project.description,
            owner=project.owner,
            tags=project.tags,
            config=project.config,
            created=project.created_at.isoformat(),
            experimentCount=experiment_count,
        )


class ProjectListResponse(BaseModel):
    projects: list[ProjectResponse]
    total: int


# ── Experiment ──────────────────────────────────────────────────────────────


class ExperimentResponse(BaseModel):
    id: str
    projectId: str
    name: str
    description: str = ""
    workflow: str | None = None
    workflowType: str | None = None
    gitCommit: str | None = None
    parameterSpace: dict[str, Any] = Field(default_factory=dict)
    created: str
    runCount: int | None = None
    runs: list["RunSummary"] = Field(default_factory=list)

    @classmethod
    def from_model(
        cls, experiment: Experiment, runs: list[Run] | None = None
    ) -> ExperimentResponse:
        run_list = []
        if runs:
            run_list = [
                RunSummary(
                    id=r.id,
                    status=r.status,
                    created=r.metadata.created_at.isoformat(),
                    parameters=r.parameters,
                )
                for r in runs
            ]
        return cls(
            id=experiment.id,
            projectId=experiment.project.id,
            name=experiment.name,
            description=experiment.description,
            workflow=experiment.metadata.workflow_source,
            workflowType=experiment.metadata.workflow_type,
            gitCommit=experiment.metadata.git_commit,
            parameterSpace=experiment.metadata.parameter_space,
            created=experiment.created_at.isoformat(),
            runCount=len(runs) if runs else None,
            runs=run_list,
        )


# ── Run ─────────────────────────────────────────────────────────────────────


class RunSummary(BaseModel):
    id: str
    status: str
    created: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class WorkflowSnapshotResponse(BaseModel):
    source: str
    gitCommit: str | None = None
    codeHash: str | None = None
    configHash: str | None = None


class RunResponse(BaseModel):
    id: str
    projectId: str
    experimentId: str
    status: str
    created: str
    finished: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    workflow: WorkflowSnapshotResponse | None = None
    error: dict[str, str] | None = None
    executorInfo: dict[str, Any] = Field(default_factory=dict)
    profile: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    configHash: str | None = None

    @classmethod
    def from_model(cls, run: Run) -> RunResponse:
        wf_snap = None
        if run.metadata.workflow_snapshot:
            s = run.metadata.workflow_snapshot
            wf_snap = WorkflowSnapshotResponse(
                source=s.source,
                gitCommit=s.git_commit,
                codeHash=s.code_hash,
                configHash=s.config_hash,
            )
        error = None
        if run.metadata.error:
            error = {
                "type": run.metadata.error.type,
                "message": run.metadata.error.message,
            }
        return cls(
            id=run.id,
            projectId=run.experiment.project.id,
            experimentId=run.experiment.id,
            status=run.status,
            created=run.metadata.created_at.isoformat(),
            finished=run.metadata.finished_at.isoformat() if run.metadata.finished_at else None,
            parameters=run.parameters,
            workflow=wf_snap,
            error=error,
            executorInfo=run.metadata.executor_info,
            profile=run.metadata.profile,
            config=run.metadata.config,
            configHash=run.metadata.config_hash,
        )


class RunStatusResponse(BaseModel):
    id: str
    status: str
    finished: str | None = None


# ── Asset ───────────────────────────────────────────────────────────────────


class AssetResponse(BaseModel):
    id: str
    name: str
    created: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_model(cls, asset: Asset) -> AssetResponse:
        return cls(
            id=asset.asset_id,
            name=asset.name,
            created=asset.created_at.isoformat(),
            metadata=asset.metadata,
        )


# ── Workspace ───────────────────────────────────────────────────────────────


class WorkspaceInfoResponse(BaseModel):
    root: str
    projectCount: int
    assetCount: int


class FolderEntryResponse(BaseModel):
    name: str
    path: str
    type: str
    size: int | None = None


class FolderBrowseResponse(BaseModel):
    path: str
    entries: list[FolderEntryResponse]


class WorkspaceFolderResponse(BaseModel):
    id: str
    path: str
    name: str
    added_at: str


class FileContentResponse(BaseModel):
    content: str


# ── Execution ───────────────────────────────────────────────────────────────


class ExecutionPlanResponse(BaseModel):
    plan: list[str]
    nodeCount: int


class CacheStatsResponse(BaseModel):
    storeDir: str
    entryCount: int


class CacheClearResponse(BaseModel):
    removedCount: int


# ── Agent ───────────────────────────────────────────────────────────────────


class SessionEventResponse(BaseModel):
    type: str
    ts: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentSessionResponse(BaseModel):
    sessionId: str
    status: str
    goalDescription: str
    createdAt: str
    events: list[SessionEventResponse] = Field(default_factory=list)


class AgentSessionListResponse(BaseModel):
    sessions: list[AgentSessionResponse]
    total: int


# ── Plugin Registry ─────────────────────────────────────────────────────────


class UiPluginResponse(BaseModel):
    id: str
    title: str
    description: str = ""
    uiModule: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_descriptor(cls, descriptor) -> "UiPluginResponse":
        return cls(
            id=descriptor.id,
            title=descriptor.title,
            description=descriptor.description,
            uiModule=descriptor.ui_module,
            capabilities=list(descriptor.capabilities),
            metadata=descriptor.metadata,
        )


class UiPluginListResponse(BaseModel):
    plugins: list[UiPluginResponse]
    total: int


# ── Generic ─────────────────────────────────────────────────────────────────


class MessageResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    workspace_available: bool
    capabilities: dict[str, bool] = Field(default_factory=dict)


# ── Run logs / execution ─────────────────────────────────────────────────────


class RunLogsResponse(BaseModel):
    """Contents of job.out and job.err for a run."""

    stdout: str | None = None
    stderr: str | None = None


class WorkflowStepInfo(BaseModel):
    """Human-readable summary of one workflow execution step."""

    index: int
    status: str  # pending | running | success | error
    step_outputs: dict[str, Any] = Field(default_factory=dict)


class RunExecutionResponse(BaseModel):
    """Workflow execution state read from workflow.json."""

    execution_id: str | None = None
    status: str = "not_started"  # running | completed | failed | not_started
    steps: list[WorkflowStepInfo] = Field(default_factory=list)
    end: dict[str, Any] | None = None

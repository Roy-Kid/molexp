"""Pydantic response models for MolExp API.

This module defines standardized response models used across all API endpoints.
Using explicit models instead of manual dict construction improves:
- Type safety and IDE support
- Automatic API documentation (OpenAPI/Swagger)
- Consistency across endpoints
- Response validation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ============================================================================
# Base Models
# ============================================================================


class TimestampMixin(BaseModel):
    """Mixin for models with timestamps."""

    created: str = Field(..., description="ISO 8601 creation timestamp")


class EntityMixin(BaseModel):
    """Mixin for entity models with an ID."""

    id: str


# ============================================================================
# Project Responses
# ============================================================================


class ProjectResponse(EntityMixin, TimestampMixin):
    """Project response model."""

    projectId: str
    name: str
    description: str = ""
    owner: str = ""
    tags: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    experimentCount: int | None = None

    @classmethod
    def from_model(
        cls, project: Any, experiment_count: int | None = None
    ) -> "ProjectResponse":
        """Create response from Project model."""
        return cls(
            id=project.project_id,
            projectId=project.project_id,
            name=project.name,
            description=project.description,
            owner=project.owner,
            tags=project.tags,
            created=project.created_at.isoformat(),
            config=project.config,
            experimentCount=experiment_count,
        )


class ProjectListResponse(BaseModel):
    """List of projects."""

    projects: list[ProjectResponse]
    total: int


class ExperimentSummary(EntityMixin, TimestampMixin):
    """Experiment summary for nested responses."""

    name: str


# ============================================================================
# Experiment Responses
# ============================================================================


class ExperimentResponse(EntityMixin, TimestampMixin):
    """Experiment response model."""

    experimentId: str
    projectId: str
    name: str
    description: str = ""
    workflow: str = Field(..., description="Workflow source path")
    workflowType: str | None = None
    gitCommit: str | None = None
    parameterSpace: dict[str, Any] = Field(default_factory=dict)
    defaultInputs: list[dict[str, Any]] = Field(default_factory=list)
    runCount: int | None = None
    runs: list["RunSummary"] = Field(default_factory=list)

    @classmethod
    def from_model(
        cls,
        experiment: Any,
        runs: list[Any] | None = None,
    ) -> "ExperimentResponse":
        """Create response from Experiment model."""
        run_list = []
        if runs:
            run_list = [
                RunSummary(
                    id=r.run_id,
                    status=r.status.value,
                    created=r.created_at.isoformat(),
                    parameters=r.parameters,
                )
                for r in runs
            ]

        return cls(
            id=experiment.experiment_id,
            experimentId=experiment.experiment_id,
            projectId=experiment.project_id,
            name=experiment.name,
            description=experiment.description,
            workflow=experiment.workflow_template.source,
            workflowType=experiment.workflow_template.type,
            gitCommit=experiment.workflow_template.git_commit,
            created=experiment.created_at.isoformat(),
            parameterSpace=experiment.parameter_space,
            defaultInputs=[
                {"assetId": ref.asset_id, "role": ref.role}
                for ref in experiment.default_inputs
            ],
            runCount=len(runs) if runs else None,
            runs=run_list,
        )


# ============================================================================
# Run Responses
# ============================================================================


class RunSummary(EntityMixin, TimestampMixin):
    """Run summary for list responses."""

    status: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class WorkflowSnapshotResponse(BaseModel):
    """Workflow snapshot details."""

    file: str
    gitCommit: str | None = None
    serializedGraph: str | None = None


class AssetRefResponse(BaseModel):
    """Asset reference response."""

    assetId: str
    role: str
    producerRunId: str | None = None
    accessedAt: str | None = None
    producedAt: str | None = None


class AssetRefsResponse(BaseModel):
    """Collection of asset references."""

    inputs: list[AssetRefResponse] = Field(default_factory=list)
    outputs: list[AssetRefResponse] = Field(default_factory=list)


class ContextSnapshotResponse(BaseModel):
    """Execution context snapshot."""

    environment: dict[str, str] = Field(default_factory=dict)
    dependencies: dict[str, str] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)


class RunResponse(EntityMixin, TimestampMixin):
    """Full run response model."""

    runId: str
    projectId: str
    experimentId: str
    status: str
    finished: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    workflow: WorkflowSnapshotResponse | None = None
    executorInfo: dict[str, Any] = Field(default_factory=dict)
    workingDir: str | None = None
    logsDir: str | None = None
    assetRefs: AssetRefsResponse | None = None
    context: ContextSnapshotResponse | None = None

    @classmethod
    def from_model(
        cls,
        run: Any,
        asset_refs: Any | None = None,
        context: Any | None = None,
    ) -> "RunResponse":
        """Create response from Run model."""
        asset_refs_response = None
        if asset_refs:
            asset_refs_response = AssetRefsResponse(
                inputs=[
                    AssetRefResponse(
                        assetId=ref.asset_id,
                        role=ref.role,
                        producerRunId=ref.producer_run_id,
                        accessedAt=(
                            ref.accessed_at.isoformat() if ref.accessed_at else None
                        ),
                    )
                    for ref in asset_refs.inputs
                ],
                outputs=[
                    AssetRefResponse(
                        assetId=ref.asset_id,
                        role=ref.role,
                        producerRunId=ref.producer_run_id,
                        producedAt=(
                            ref.produced_at.isoformat() if ref.produced_at else None
                        ),
                    )
                    for ref in asset_refs.outputs
                ],
            )

        context_response = None
        if context:
            context_response = ContextSnapshotResponse(
                environment=context.environment,
                dependencies=context.dependencies,
                hardware=context.hardware,
            )

        return cls(
            id=run.run_id,
            runId=run.run_id,
            projectId=run.project_id,
            experimentId=run.experiment_id,
            status=run.status.value,
            created=run.created_at.isoformat(),
            finished=run.finished_at.isoformat() if run.finished_at else None,
            parameters=run.parameters,
            workflow=WorkflowSnapshotResponse(
                file=run.workflow_snapshot.workflow_file,
                gitCommit=run.workflow_snapshot.git_commit,
                serializedGraph=run.workflow_snapshot.serialized_graph,
            ),
            executorInfo=run.executor_info,
            workingDir=run.working_dir,
            logsDir=run.logs_dir,
            assetRefs=asset_refs_response,
            context=context_response,
        )


# ============================================================================
# Asset Responses
# ============================================================================


class AssetFileResponse(BaseModel):
    """Asset file details."""

    path: str
    size: int
    hash: str


class AssetResponse(EntityMixin, TimestampMixin):
    """Asset response model."""

    assetId: str
    type: str
    format: str
    size: int
    contentHash: str
    mimeType: str = ""
    producerRunId: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    files: list[AssetFileResponse] = Field(default_factory=list)

    @classmethod
    def from_model(cls, asset: Any) -> "AssetResponse":
        """Create response from Asset model."""
        return cls(
            id=asset.asset_id,
            assetId=asset.asset_id,
            type=asset.type.value,
            format=asset.format,
            size=asset.size_bytes,
            contentHash=asset.content_hash,
            mimeType=asset.mime_type,
            created=asset.created_at.isoformat(),
            producerRunId=asset.producer_run_id,
            tags=asset.tags,
            metadata=asset.metadata,
            files=[
                AssetFileResponse(path=f.path, size=f.size, hash=f.hash)
                for f in asset.files
            ],
        )


# ============================================================================
# Workspace Responses
# ============================================================================


class WorkspaceInfoResponse(BaseModel):
    """Workspace info response."""

    root: str
    projectCount: int
    assetCount: int


class DashboardStatsResponse(BaseModel):
    """Dashboard statistics response."""

    totalExperiments: int
    activeWorkflows: int
    dataUsage: str
    computeHours: str
    recentExperiments: list[dict[str, Any]]


class FolderEntryResponse(BaseModel):
    """Folder entry for browsing."""

    name: str
    path: str
    type: str  # "file" or "directory"
    size: int | None = None


class FolderBrowseResponse(BaseModel):
    """Folder browse response."""

    path: str
    entries: list[FolderEntryResponse]


class WorkspaceFolderResponse(BaseModel):
    """Workspace folder response."""

    id: str
    path: str
    name: str
    added_at: str


class FileContentResponse(BaseModel):
    """File content response."""

    content: str


# ============================================================================
# Execution Responses
# ============================================================================


class ExecutionPlanResponse(BaseModel):
    """Execution plan response."""

    plan: list[str]
    nodeCount: int


class RunStatusResponse(BaseModel):
    """Run status update response."""

    id: str
    status: str
    finished: str | None = None


# ============================================================================
# Node Responses
# ============================================================================


class NodePortResponse(BaseModel):
    """Node port definition."""

    id: str
    label: str
    type: str
    required: bool = True


class NodeResponse(BaseModel):
    """Node type response."""

    id: str
    label: str
    category: str
    description: str
    inputs: list[NodePortResponse]
    outputs: list[NodePortResponse]
    icon: str | None = None
    tags: list[str] = Field(default_factory=list)
    config_schema: dict[str, Any] = Field(default_factory=dict)


class NodeListResponse(BaseModel):
    """List of node types."""

    nodes: list[NodeResponse]


# ============================================================================
# Generic Responses
# ============================================================================


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    workspace_available: bool
    ir_available: bool


class EntityClassificationResponse(BaseModel):
    """Entity classification response."""

    indexed: bool
    kind: str
    path: str
    metadata: dict[str, Any] | None = None


class WorkspaceScanResponse(BaseModel):
    """Workspace scan response."""

    total: int
    entities: list[EntityClassificationResponse]

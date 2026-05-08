"""Pydantic response models for MolExp API.

All ``from_model()`` methods take typed domain objects — no ``Any``,
no ``getattr`` guessing.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from molexp.workspace import (
    Asset,
    Experiment,
    Project,
    Run,
)


def _read_context_results(run: Run) -> dict[str, Any]:
    """Read the ``context.results`` block from run.json on disk.

    The ``Context`` object is owned by the active ``RunContext`` only; once
    a run has finished, the only place ``results`` survives is the
    ``context`` sub-object inside ``run.json``. We read it lazily so the
    REST response can show "what did this run produce" without bringing
    runtime state into the persisted ``RunMetadata`` model.
    """
    run_json = run.run_dir / "run.json"
    if not run_json.exists():
        return {}
    try:
        with open(run_json) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    ctx = data.get("context") or {}
    results = ctx.get("results") or {}
    return dict(results) if isinstance(results, dict) else {}


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
    defaultTarget: str | None = None
    created: str
    runCount: int | None = None
    runs: list[RunSummary] = Field(default_factory=list)

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
                    finished=(
                        r.metadata.finished_at.isoformat() if r.metadata.finished_at else None
                    ),
                    parameters=r.parameters,
                    results=_read_context_results(r),
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
            defaultTarget=experiment.metadata.default_target,
            created=experiment.created_at.isoformat(),
            runCount=len(runs) if runs else None,
            runs=run_list,
        )


# ── Run ─────────────────────────────────────────────────────────────────────


class RunSummary(BaseModel):
    id: str
    status: str
    created: str
    finished: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)


class WorkflowSnapshotResponse(BaseModel):
    source: str
    gitCommit: str | None = None
    codeHash: str | None = None
    configHash: str | None = None


class ExecutionRecordResponse(BaseModel):
    """One execution attempt of a Run.

    Mirrors :class:`molexp.workspace.models.ExecutionRecord` with
    JSON-friendly field names so the UI can render a per-attempt
    timeline.
    """

    executionId: str
    startedAt: str
    finishedAt: str | None = None
    status: str
    schedulerJobId: str | None = None


class RunResponse(BaseModel):
    id: str
    projectId: str
    experimentId: str
    status: str
    created: str
    finished: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    workflow: WorkflowSnapshotResponse | None = None
    workflowSource: str | None = None
    error: dict[str, str] | None = None
    executorInfo: dict[str, Any] = Field(default_factory=dict)
    profile: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    configHash: str | None = None
    executionHistory: list[ExecutionRecordResponse] = Field(default_factory=list)
    target: str | None = None

    @classmethod
    def from_model(cls, run: Run) -> RunResponse:
        wf_snap = None
        wf_source: str | None = None
        if run.metadata.workflow_snapshot:
            s = run.metadata.workflow_snapshot
            wf_snap = WorkflowSnapshotResponse(
                source=s.source,
                gitCommit=s.git_commit,
                codeHash=s.code_hash,
                configHash=s.config_hash,
            )
            wf_source = s.source
        error = None
        if run.metadata.error:
            error = {
                "type": run.metadata.error.type,
                "message": run.metadata.error.message,
            }
        history = [
            ExecutionRecordResponse(
                executionId=rec.execution_id,
                startedAt=rec.started_at.isoformat(),
                finishedAt=rec.finished_at.isoformat() if rec.finished_at else None,
                status=rec.status,
                schedulerJobId=rec.scheduler_job_id,
            )
            for rec in run.metadata.execution_history
        ]
        return cls(
            id=run.id,
            projectId=run.experiment.project.id,
            experimentId=run.experiment.id,
            status=run.status,
            created=run.metadata.created_at.isoformat(),
            finished=run.metadata.finished_at.isoformat() if run.metadata.finished_at else None,
            parameters=run.parameters,
            results=_read_context_results(run),
            workflow=wf_snap,
            workflowSource=wf_source,
            error=error,
            executorInfo=run.metadata.executor_info,
            profile=run.metadata.profile,
            config=run.metadata.config,
            configHash=run.metadata.config_hash,
            executionHistory=history,
            target=run.metadata.target,
        )


class RunStatusResponse(BaseModel):
    id: str
    status: str
    finished: str | None = None


# ── Asset ───────────────────────────────────────────────────────────────────


class AssetResponse(BaseModel):
    """Serialized typed ``Asset``.

    ``kind`` is the discriminator (``data`` / ``artifact`` / ``log`` / …).
    ``extra`` carries subclass-specific fields so the frontend can render
    per-kind details without a separate schema per kind.
    ``content_hash`` is the sha256 (``"sha256:<hex>"``) of the payload
    when the asset is content-addressable; ``None`` for streaming kinds.
    """

    id: str
    name: str
    kind: str
    scope_kind: str
    scope_ids: list[str]
    path: str
    created_at: str
    updated_at: str
    producer: dict[str, Any] | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
    content_hash: str | None = None

    @classmethod
    def from_model(cls, asset: Asset) -> AssetResponse:
        from molexp.workspace.assets import ASSET_ADAPTER

        dumped = ASSET_ADAPTER.dump_python(asset, mode="json")
        common_fields = {
            "asset_id",
            "name",
            "scope",
            "path",
            "created_at",
            "updated_at",
            "producer",
            "tags",
            "kind",
            "content_hash",
        }
        extra = {k: v for k, v in dumped.items() if k not in common_fields}
        return cls(
            id=asset.asset_id,
            name=asset.name,
            kind=dumped["kind"],
            scope_kind=asset.scope.kind,
            scope_ids=list(asset.scope.ids),
            path=str(asset.path),
            created_at=asset.created_at.isoformat(),
            updated_at=asset.updated_at.isoformat(),
            producer=asset.producer.model_dump() if asset.producer else None,
            tags=dict(asset.tags),
            extra=extra,
            content_hash=asset.content_hash,
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


class SessionStatsResponse(BaseModel):
    inputTokens: int = 0
    outputTokens: int = 0
    cacheReadTokens: int = 0
    cacheWriteTokens: int = 0
    totalTokens: int = 0
    requests: int = 0
    toolCalls: int = 0
    events: int = 0
    startedAt: str | None = None
    completedAt: str | None = None
    durationSeconds: float | None = None


class AgentSessionResponse(BaseModel):
    sessionId: str
    status: str
    goalDescription: str
    createdAt: str
    events: list[SessionEventResponse] = Field(default_factory=list)
    stats: SessionStatsResponse = Field(default_factory=SessionStatsResponse)
    planMode: bool = False
    skillId: str | None = None


class AgentSessionListResponse(BaseModel):
    sessions: list[AgentSessionResponse]
    total: int


class AgentTaskResponse(BaseModel):
    """User-facing task wrapper around one current runtime session.

    ``taskId`` is the product identifier the UI should route on; ``sessionId``
    is the lower-level runtime handle used to continue the active execution.
    """

    taskId: str
    title: str
    goal: str
    status: str
    createdAt: str
    updatedAt: str | None = None
    sessionId: str
    events: list[SessionEventResponse] = Field(default_factory=list)
    stats: SessionStatsResponse = Field(default_factory=SessionStatsResponse)
    planMode: bool = False
    skillId: str | None = None


class AgentTaskListResponse(BaseModel):
    tasks: list[AgentTaskResponse]
    total: int


class ReviewTargetRefResponse(BaseModel):
    type: str
    id: str
    taskId: str | None = None
    sessionId: str | None = None


class ReviewItemResponse(BaseModel):
    id: str
    kind: str
    title: str
    description: str | None = None
    riskLevel: str
    status: str
    targetRef: ReviewTargetRefResponse
    createdAt: str
    resolvedAt: str | None = None
    resolutionComment: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewListResponse(BaseModel):
    reviews: list[ReviewItemResponse]
    total: int


class CommandParameterSpec(BaseModel):
    """One ``{{param}}`` slot in a slash command's goal_template."""

    name: str
    required: bool = True


class CommandSpec(BaseModel):
    """A single slash command — skill-backed or builtin."""

    slashName: str
    name: str
    description: str = ""
    parameters: list[CommandParameterSpec] = Field(default_factory=list)
    defaultPlanMode: bool = False
    isBuiltin: bool = False
    skillId: str | None = None


class CommandListResponse(BaseModel):
    commands: list[CommandSpec] = Field(default_factory=list)


class CommandParseResponse(BaseModel):
    """Mirror of :class:`molexp.agent.skills.commands.ParsedCommand`."""

    kind: Literal["skill", "builtin", "error"]
    name: str = ""
    skillId: str = ""
    parameters: dict[str, str] = Field(default_factory=dict)
    planMode: bool = False
    error: str = ""


class AgentSystemPromptResponse(BaseModel):
    """Per-session system prompt breakdown for the inspector."""

    base: str
    workspaceInstructions: str = ""
    skillInstructions: str = ""
    sessionOverride: str | None = None
    planMode: bool = False
    effective: str


# ── Plugin Registry ─────────────────────────────────────────────────────────


class UiPluginResponse(BaseModel):
    """Per-bundle entry returned by ``GET /api/plugins``.

    Carries no UI semantics — those live in each bundle's own
    ``manifest.json`` (fetched by the browser-side loader). The shape
    is deliberately minimal: a stable ``id``, plus the two URLs the
    frontend needs to fetch the manifest and dynamic-import the entry.
    """

    id: str
    manifestUrl: str
    entryUrl: str


class UiPluginListResponse(BaseModel):
    plugins: list[UiPluginResponse]
    total: int


# ── Task-type registry ──────────────────────────────────────────────────────


class TaskTypeResponse(BaseModel):
    """Single registered task type the agent / UI can compose into IR."""

    slug: str = Field(..., description="Registry slug, e.g. 'core.add'")
    description: str = Field("", description="Human-readable summary")


class TaskTypeListResponse(BaseModel):
    task_types: list[TaskTypeResponse]
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
    """Per-execution stdout/stderr for a run.

    ``execution_id`` is the attempt these logs belong to; the server
    defaults to the most recent attempt when no specific execution is
    requested.  Each value is the full content of
    ``executions/<execution_id>/{stdout,stderr}.log`` (or ``None`` if the
    file is absent — e.g. local executions skip stdout capture).
    """

    execution_id: str | None = None
    stdout: str | None = None
    stderr: str | None = None


class MetricSeriesResponse(BaseModel):
    """Summary for one metric series in a run-local metrics query."""

    key: str
    type: str
    count: int
    latestStep: int | float | None = None
    latestTimestamp: str | None = None
    latestValue: Any | None = None


class RunMetricsResponse(BaseModel):
    """Run-local metrics query response."""

    nextLine: int = 0
    records: list[dict[str, Any]] = Field(default_factory=list)
    series: list[MetricSeriesResponse] = Field(default_factory=list)
    parseErrors: int = 0


class RunFileTextResponse(BaseModel):
    """Raw UTF-8 text content of a file under a run directory."""

    path: str
    content: str
    size: int


class LammpsThermoStage(BaseModel):
    """One ``Per MPI rank ... Loop time`` block as columns + numeric rows."""

    columns: list[str] = Field(default_factory=list)
    rows: list[list[float]] = Field(default_factory=list)


class LammpsLogResponse(BaseModel):
    """Parsed LAMMPS log thermo stages, produced by ``molpy.io.LAMMPSLog``."""

    path: str
    version: str | None = None
    nStages: int = 0
    stages: list[LammpsThermoStage] = Field(default_factory=list)


class WorkflowStepInfo(BaseModel):
    """Human-readable summary of one workflow execution step."""

    index: int
    status: str  # pending | running | success | error
    outputs: dict[str, Any] = Field(default_factory=dict)


class RunExecutionResponse(BaseModel):
    """Workflow execution state read from workflow.json."""

    execution_id: str | None = None
    status: str = "not_started"  # running | completed | failed | not_started
    steps: list[WorkflowStepInfo] = Field(default_factory=list)
    end: dict[str, Any] | None = None


# ── Asset lineage (Producer.inputs DAG) ─────────────────────────────────────


class AssetLineageNode(BaseModel):
    """One node in an asset's lineage neighborhood.

    Carries just enough to render a clickable card in the UI; full
    asset detail is available via ``GET /api/assets/{id}``.
    """

    id: str
    name: str
    kind: str
    scope_kind: str


class AssetLineageResponse(BaseModel):
    """Upstream + downstream neighbours of an asset in the lineage DAG.

    ``ancestors`` is the transitive set of upstream asset_ids reached
    by walking ``producer.inputs`` in reverse; ``descendants`` is the
    transitive forward set. The starting asset is excluded from both.
    """

    asset_id: str
    ancestors: list[AssetLineageNode] = Field(default_factory=list)
    descendants: list[AssetLineageNode] = Field(default_factory=list)


# ── Catalog / file lineage ──────────────────────────────────────────────────


class CatalogProducerInfo(BaseModel):
    """Producer metadata for a catalog entry."""

    runId: str | None = None
    taskId: str | None = None
    executionId: str | None = None


class CatalogScopeInfo(BaseModel):
    """Scope chain that owns this asset (project/experiment/run ids)."""

    kind: str
    projectId: str | None = None
    experimentId: str | None = None
    runId: str | None = None


class CatalogSibling(BaseModel):
    """Other outputs from the same producer.task_id."""

    assetId: str
    name: str
    kind: str
    relPath: str


class CatalogByPathResponse(BaseModel):
    """Reverse-lookup: which run/experiment/project produced a file?"""

    matched: bool
    workspaceRelPath: str
    assetId: str | None = None
    assetKind: str | None = None
    producer: CatalogProducerInfo | None = None
    scope: CatalogScopeInfo | None = None
    siblings: list[CatalogSibling] = Field(default_factory=list)


class RunFileNode(BaseModel):
    """One node in a run's output file tree."""

    name: str
    relPath: str  # relative to run_dir
    type: str  # 'file' | 'folder'
    size: int | None = None
    modified: float | None = None
    assetId: str | None = None
    assetKind: str | None = None
    taskId: str | None = None
    children: list[RunFileNode] = Field(default_factory=list)


RunFileNode.model_rebuild()


class RunFilesResponse(BaseModel):
    """Per-run output file tree, enriched with catalog producer metadata."""

    runId: str
    runDir: str
    nodes: list[RunFileNode] = Field(default_factory=list)


# ── Experiment comparison ───────────────────────────────────────────────────


class ComparisonRunRow(BaseModel):
    """One run row in the experiment comparison matrix."""

    runId: str
    status: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    durationSec: float | None = None
    created: str
    finished: str | None = None
    error: dict[str, str] | None = None


class ExperimentComparisonResponse(BaseModel):
    """Comparison matrix: parameter columns x run rows + metric columns."""

    experimentId: str
    projectId: str
    paramKeys: list[str] = Field(default_factory=list)
    metricKeys: list[str] = Field(default_factory=list)
    runs: list[ComparisonRunRow] = Field(default_factory=list)


# ── Run actions ─────────────────────────────────────────────────────────────


class RunActionResponse(BaseModel):
    """Result of an actionable mutation on a run."""

    runId: str
    status: str
    message: str | None = None


class RunRerunResponse(BaseModel):
    """A new run cloned from an existing one."""

    sourceRunId: str
    newRunId: str
    projectId: str
    experimentId: str
    status: str


# ── Skills / MCP / Tool admin ───────────────────────────────────────────────


class SkillResponse(BaseModel):
    """A saved skill (goal template + tool scope + system addendum)."""

    id: str
    name: str
    description: str = ""
    goalTemplate: str
    slashName: str = ""
    instructions: str = ""
    defaultPlanMode: bool = False
    constraints: list[str] = Field(default_factory=list)
    successCriteria: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    allowedTools: list[str] = Field(default_factory=list)
    deniedTools: list[str] = Field(default_factory=list)
    requiresExitTool: str = ""
    builtin: bool = False
    scope: str = "workspace"
    createdAt: str = ""
    updatedAt: str = ""


class SkillListResponse(BaseModel):
    skills: list[SkillResponse] = Field(default_factory=list)


class ToolParameterResponse(BaseModel):
    name: str
    annotation: str = "Any"
    required: bool = False


class AgentToolResponse(BaseModel):
    """One tool exposed to the agent — native or MCP-discovered.

    For MCP tools, ``source`` is ``"mcp:<server-name>"`` so the UI can
    group by server. Native tools keep ``source = "native"``.
    """

    name: str
    description: str = ""
    parameters: list[ToolParameterResponse] = Field(default_factory=list)
    requiresApproval: bool = False
    source: str = "native"


class McpToolGroupResponse(BaseModel):
    """Per-server status surface for the Tools panel.

    Even when a server is offline / misconfigured / unauthorized we want
    the UI to render *something* under that server's heading — a row with
    the error keeps users oriented instead of silently dropping the group.
    """

    server: str
    scope: Literal["native", "user", "workspace"]
    ok: bool
    toolCount: int = 0
    error: str | None = None


class AgentToolListResponse(BaseModel):
    tools: list[AgentToolResponse] = Field(default_factory=list)
    mcpGroups: list[McpToolGroupResponse] = Field(default_factory=list)


class CustomToolHttpInvokerResponse(BaseModel):
    """Read-only view of a user/workspace HTTP-webhook tool's wiring.

    Header values are returned **with secret references intact**
    (``${SECRET:KEY}``); the actual secret value never leaves the
    server.
    """

    kind: Literal["http"] = "http"
    url: str
    method: Literal["GET", "POST", "PUT", "DELETE"] = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    bodyTemplate: str = ""


class CustomToolPythonInvokerResponse(BaseModel):
    """Read-only view of a Python-implementation tool reference."""

    kind: Literal["python"] = "python"
    target: str


class CustomToolResponse(BaseModel):
    """Single user/workspace/registration-tier tool record.

    Mirrors the `AgentToolResponse` shape but adds the persistence
    metadata (`scope`, `shadowed`, `valid`, `createdAt`, `updatedAt`)
    needed for tier-aware listing and inline error reporting.
    """

    id: str
    name: str
    description: str = ""
    category: Literal["workspace", "workflow", "chat", "control", "web"] = "workspace"
    mutates: bool = False
    requiresApproval: bool = False
    parametersSchema: dict[str, object] = Field(default_factory=dict)
    invoker: CustomToolHttpInvokerResponse | CustomToolPythonInvokerResponse = Field(
        discriminator="kind"
    )
    scope: Literal["native", "user", "workspace"] = "user"
    shadowed: bool = False
    valid: bool = True
    invalidReason: str = ""
    builtin: bool = False
    createdAt: str = ""
    updatedAt: str = ""


class CustomToolListResponse(BaseModel):
    tools: list[CustomToolResponse] = Field(default_factory=list)


class McpAuthSummary(BaseModel):
    """Public-safe view of a server's structured auth settings.

    Token values, refresh tokens, and client secrets are never exposed —
    only metadata the UI needs to render the connection card. ``connected``
    indicates the token store on disk has at least one persisted token
    (rough proxy for "user has completed Connect at least once").
    """

    type: Literal["oauth2"]
    scopes: list[str] = Field(default_factory=list)
    clientId: str | None = None
    connected: bool = False


class McpServerResponse(BaseModel):
    """One MCP server entry, possibly merged across scopes.

    ``shadowed`` is True when this entry exists at User scope but is
    overridden by a Workspace entry of the same name. ``unresolvedSecrets``
    lists ``${SECRET:KEY}`` references that have no value in either secret
    store — the runtime skips such entries.
    """

    name: str
    scope: Literal["native", "user", "workspace"]
    transport: str = ""
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    envKeys: list[str] = Field(default_factory=list)
    headerKeys: list[str] = Field(default_factory=list)
    secretRefs: list[str] = Field(default_factory=list)
    unresolvedSecrets: list[str] = Field(default_factory=list)
    shadowed: bool = False
    valid: bool = True
    invalidReason: str = ""
    auth: McpAuthSummary | None = None


class McpServerListResponse(BaseModel):
    """Merged view of both scopes plus the resolved file paths.

    ``workspacePath`` and ``userPath`` are the absolute paths the store
    would read/write at each scope (whether or not the file currently
    exists) — useful for UI tooltips like "Edit ~/.molexp/mcp.json".
    """

    workspacePath: str
    userPath: str
    servers: list[McpServerResponse] = Field(default_factory=list)


class McpServerTestResponse(BaseModel):
    """Outcome of probing an MCP server (subprocess spawn or HTTP handshake)."""

    ok: bool
    name: str
    scope: Literal["native", "user", "workspace"]
    transport: str
    latencyMs: int = 0
    toolCount: int = 0
    error: str | None = None


class McpOAuthStartResponse(BaseModel):
    """Result of POST /mcp/servers/{name}/oauth/start.

    The UI opens ``authorizeUrl`` in a popup; once the IdP bounces back to
    the SPA the SPA POSTs ``code``+``state`` to the callback endpoint to
    finish the flow.
    """

    name: str
    scope: Literal["native", "user", "workspace"]
    authorizeUrl: str


class McpOAuthStatusResponse(BaseModel):
    """Whether the named server currently has a usable OAuth token on disk.

    ``hasTokens`` is True after a successful Connect; False if the user has
    never connected, has disconnected, or the token file got corrupted.
    """

    name: str
    scope: Literal["native", "user", "workspace"]
    hasTokens: bool
    scopes: list[str] = Field(default_factory=list)


class McpSecretRefRow(BaseModel):
    """One row in the secrets list — key + which servers reference it."""

    key: str
    isSet: bool
    referencedBy: list[str] = Field(default_factory=list)


class McpSecretListResponse(BaseModel):
    """Secrets at the requested scope. Plaintext values are never returned."""

    scope: Literal["native", "user", "workspace"]
    path: str
    secrets: list[McpSecretRefRow] = Field(default_factory=list)


# ── Agent provider config ───────────────────────────────────────────────────


class AgentProviderResponse(BaseModel):
    """Public view of the workspace's LLM provider config — never the raw key.

    ``apiKeyPreview`` is a masked rendering ("sk-...1234"); ``apiKeySet``
    is the boolean the UI uses to gate the "ready" indicator.
    """

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    baseUrl: str = ""
    apiKeyPreview: str = ""
    apiKeySet: bool = False
    instructions: str = ""
    supportedProviders: list[str] = Field(default_factory=list)


class AgentProviderTestResponse(BaseModel):
    """Result of probing the configured provider with a minimal request.

    ``ok=True`` means we got a model response back. ``latencyMs`` is the
    wall-clock RTT for the probe; ``error`` is filled only on failure
    with a short, user-readable description (no stack trace, no key).
    """

    ok: bool = False
    provider: str = ""
    model: str = ""
    latencyMs: int = 0
    reply: str = ""
    error: str | None = None


class AgentHealthResponse(BaseModel):
    """Whether the agent runtime is ready to start a new session.

    ``ready=False`` indicates a configuration problem the user can
    resolve in Agent Settings (most commonly: no API key). ``source``
    is one of ``"stored"`` (workspace config), ``"env"`` (process env
    var), or ``"none"`` (not configured).
    """

    ready: bool = False
    provider: str = ""
    model: str = ""
    source: str = "none"
    reason: str = ""
    envVar: str = ""

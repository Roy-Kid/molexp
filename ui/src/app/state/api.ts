import { AssetsService } from "@/api/generated/services/AssetsService";
import { ExecutionService } from "@/api/generated/services/ExecutionService";
import { ExperimentsService } from "@/api/generated/services/ExperimentsService";
import { ProjectsService } from "@/api/generated/services/ProjectsService";
import { RunsService } from "@/api/generated/services/RunsService";
import { WorkflowService } from "@/api/generated/services/WorkflowService";
import { WorkspaceService } from "@/api/generated/services/WorkspaceService";
import {
  buildFlowgramDocument,
  type FlowgramDocument,
  parseTaskGraphIr,
} from "@/app/renderers/flowgram-document";
import type {
  AgentSessionSummary,
  ApiAgentSession,
  ApiAssetResponse,
  ApiCacheClear,
  ApiCacheStats,
  ApiExperimentResponse,
  ApiProjectResponse,
  ApiRunResponse,
  AssetSummary,
  ConsoleEntry,
  ExperimentCreateRequest,
  ExperimentSummary,
  ProjectCreateRequest,
  ProjectSummary,
  RunCreateRequest,
  RunSummary,
  WorkflowSummary,
  WorkspaceSnapshot,
  WorkspaceTreeNode,
} from "@/app/types";
import type { TaskGraphJson } from "@/types/task_graph_ir";

// Local types not yet in OpenAPI. The lineage fields (`assetId`,
// `assetKind`, `producerRunId`, `producerTaskId`) are populated when
// the workspace files endpoint is called with `?include=catalog`.
interface WorkspaceFileNode {
  id?: string;
  name: string;
  path: string;
  type?: string;
  children?: WorkspaceFileNode[];
  size?: number | null;
  modified?: string | number;
  assetId?: string | null;
  assetKind?: string | null;
  producerRunId?: string | null;
  producerTaskId?: string | null;
  hasPreviewSidecar?: boolean | null;
}

interface WorkspaceFilesResponse {
  path?: string;
  children?: WorkspaceFileNode[];
}

export interface MetricRecord {
  t: string;
  k: string;
  s?: number;
  w?: string;
  v?: unknown;
  tags?: Record<string, unknown>;
}

export interface MetricSeriesSummary {
  key: string;
  type: string;
  count: number;
  latestStep?: number | null;
  latestTimestamp?: string | null;
  latestValue?: unknown;
}

export interface RunMetricsResponse {
  nextLine: number;
  records: MetricRecord[];
  series: MetricSeriesSummary[];
  parseErrors: number;
}

export interface RunMetricsQuery {
  type?: string;
  key?: string;
  sinceLine?: number;
  limit?: number;
}

export interface TensorboardScalarPoint {
  step: number;
  wallTime: number;
  value: number;
}

export interface TensorboardScalarSeries {
  tag: string;
  logdir: string;
  points: TensorboardScalarPoint[];
}

export interface TensorboardScalarsResponse {
  runId: string;
  runDir: string;
  logdirs: string[];
  series: TensorboardScalarSeries[];
}

/**
 * Error thrown by ``getRunTensorboardScalars`` — preserves the HTTP
 * status so the UI can distinguish 503 (extra not installed) from
 * generic failures. Subclasses ``Error`` so ``err instanceof Error``,
 * Sentry stacks, and error boundaries behave correctly.
 */
export class TensorboardScalarsError extends Error {
  public readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "TensorboardScalarsError";
    this.status = status;
  }
}

export type { LammpsLogResponse } from "@/api/generated/models/LammpsLogResponse";
export type { LammpsThermoStage } from "@/api/generated/models/LammpsThermoStage";
export type { RunFileTextResponse } from "@/api/generated/models/RunFileTextResponse";

export const workspaceApi = {
  getProjects: async (): Promise<ApiProjectResponse[]> => {
    return ProjectsService.listProjectsApiProjectsGet();
  },
  createProject: async (data: ProjectCreateRequest): Promise<ApiProjectResponse> => {
    return ProjectsService.createProjectApiProjectsPost(data);
  },
  deleteProject: async (projectId: string): Promise<void> => {
    await ProjectsService.deleteProjectApiProjectsIdDelete(projectId);
  },
  getExperiments: async (projectId: string): Promise<ApiExperimentResponse[]> => {
    return ExperimentsService.listExperimentsApiProjectsProjectIdExperimentsGet(projectId);
  },
  createExperiment: async (
    projectId: string,
    data: ExperimentCreateRequest,
  ): Promise<ApiExperimentResponse> => {
    return ExperimentsService.createExperimentApiProjectsProjectIdExperimentsPost(projectId, data);
  },
  deleteExperiment: async (projectId: string, experimentId: string): Promise<void> => {
    await ExperimentsService.deleteExperimentApiProjectsProjectIdExperimentsExperimentIdDelete(
      projectId,
      experimentId,
    );
  },
  getRuns: async (projectId: string, experimentId: string): Promise<ApiRunResponse[]> => {
    return RunsService.listRunsApiProjectsProjectIdExperimentsExperimentIdRunsGet(
      projectId,
      experimentId,
    );
  },
  getRun: async (
    projectId: string,
    experimentId: string,
    runId: string,
  ): Promise<ApiRunResponse> => {
    return RunsService.getRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdGet(
      projectId,
      experimentId,
      runId,
    );
  },
  getRunLogs: async (projectId: string, experimentId: string, runId: string) => {
    return RunsService.getRunLogsApiProjectsProjectIdExperimentsExperimentIdRunsRunIdLogsGet(
      projectId,
      experimentId,
      runId,
    );
  },
  getRunExecutionLogs: async (
    projectId: string,
    experimentId: string,
    runId: string,
    executionId: string,
  ) => {
    return RunsService.getRunExecutionLogsApiProjectsProjectIdExperimentsExperimentIdRunsRunIdExecutionsExecutionIdLogsGet(
      projectId,
      experimentId,
      runId,
      executionId,
    );
  },
  getRunExecution: async (projectId: string, experimentId: string, runId: string) => {
    return RunsService.getRunExecutionApiProjectsProjectIdExperimentsExperimentIdRunsRunIdExecutionGet(
      projectId,
      experimentId,
      runId,
    );
  },
  getRunLammpsLog: async (projectId: string, experimentId: string, runId: string, path: string) => {
    return RunsService.getRunLammpsLogApiProjectsProjectIdExperimentsExperimentIdRunsRunIdLammpsLogGet(
      projectId,
      experimentId,
      runId,
      path,
    );
  },
  getRunFileText: async (projectId: string, experimentId: string, runId: string, path: string) => {
    return RunsService.getRunFileTextApiProjectsProjectIdExperimentsExperimentIdRunsRunIdFileTextGet(
      projectId,
      experimentId,
      runId,
      path,
    );
  },
  getRunTensorboardScalars: async (
    projectId: string,
    experimentId: string,
    runId: string,
    opts: { tag?: string[]; logdir?: string } = {},
  ): Promise<TensorboardScalarsResponse> => {
    const params = new URLSearchParams();
    if (opts.logdir) params.set("logdir", opts.logdir);
    for (const t of opts.tag ?? []) params.append("tag", t);
    const suffix = params.toString() ? `?${params.toString()}` : "";
    const response = await fetch(
      `/api/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(
        experimentId,
      )}/runs/${encodeURIComponent(runId)}/tensorboard/scalars${suffix}`,
    );
    if (!response.ok) {
      const contentType = response.headers.get("Content-Type") ?? "";
      const text = await response.text();
      let message = `Failed to fetch tensorboard scalars: ${response.statusText}`;
      if (contentType.includes("application/json")) {
        try {
          const body = JSON.parse(text);
          if (typeof body?.detail === "string") message = body.detail;
        } catch {
          // ignore — fall through to generic statusText
        }
      } else if (text && text.length < 500) {
        // Non-JSON bodies (proxy HTML, uvicorn text errors) are short
        // enough to surface verbatim; long bodies are noise.
        message = `${message}: ${text.trim()}`;
      }
      throw new TensorboardScalarsError(response.status, message);
    }
    return response.json();
  },
  getRunMetrics: async (
    projectId: string,
    experimentId: string,
    runId: string,
    query: RunMetricsQuery = {},
  ): Promise<RunMetricsResponse> => {
    const params = new URLSearchParams();
    if (query.type) params.set("type", query.type);
    if (query.key) params.set("key", query.key);
    if (query.sinceLine !== undefined) params.set("since_line", String(query.sinceLine));
    if (query.limit !== undefined) params.set("limit", String(query.limit));

    const suffix = params.toString() ? `?${params.toString()}` : "";
    const response = await fetch(
      `/api/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(
        experimentId,
      )}/runs/${encodeURIComponent(runId)}/metrics${suffix}`,
    );
    if (!response.ok) {
      throw new Error(`Failed to fetch run metrics: ${response.statusText}`);
    }
    return response.json();
  },
  createRun: async (
    projectId: string,
    experimentId: string,
    data: RunCreateRequest,
  ): Promise<ApiRunResponse> => {
    return RunsService.createRunApiProjectsProjectIdExperimentsExperimentIdRunsPost(
      projectId,
      experimentId,
      data,
    );
  },
  updateRunStatus: async (
    projectId: string,
    experimentId: string,
    runId: string,
    status: string,
  ): Promise<void> => {
    await RunsService.updateRunStatusApiProjectsProjectIdExperimentsExperimentIdRunsRunIdStatusPatch(
      projectId,
      experimentId,
      runId,
      { status },
    );
  },
  getAssets: async (): Promise<ApiAssetResponse[]> => {
    return AssetsService.listAssetsApiAssetsGet();
  },
  getProjectAssets: async (projectId: string): Promise<ApiAssetResponse[]> => {
    // Manually fetch until client is regenerated
    const response = await fetch(`/api/projects/${projectId}/assets`);
    if (!response.ok) {
      throw new Error(`Failed to fetch project assets: ${response.statusText}`);
    }
    return response.json();
  },
  getRunAssets: async (runId: string): Promise<ApiAssetResponse[]> => {
    return AssetsService.listAssetsApiAssetsGet(undefined, undefined, runId);
  },
  getAssetLineage: async (assetId: string) => {
    return AssetsService.getAssetLineageApiAssetsAssetIdLineageGet(assetId);
  },
  openWorkspace: async (path: string, createIfMissing = false): Promise<void> => {
    await WorkspaceService.openWorkspaceApiWorkspaceOpenPost({
      path,
      create_if_missing: createIfMissing,
    });
  },
  createDirectory: async (path: string): Promise<void> => {
    await WorkspaceService.createDirectoryApiWorkspaceDirectoriesPost({
      folder_id: "workspace",
      path,
    });
  },
  writeFile: async (path: string, content = ""): Promise<void> => {
    await WorkspaceService.writeFileApiWorkspaceFilesPut({ folder_id: "workspace", path, content });
  },
  getWorkspaceFileText: async (path: string): Promise<string> => {
    const response = await WorkspaceService.readWorkspaceFileApiWorkspaceFileGet(path);
    return response.content;
  },
  getCacheStats: async (): Promise<ApiCacheStats> => {
    return ExecutionService.getCacheStatsApiCacheStatsGet();
  },
  clearCache: async (): Promise<ApiCacheClear> => {
    return ExecutionService.clearCacheApiCacheDelete();
  },
  getWorkspaceFileBlob: async (path: string): Promise<Blob> => {
    // The generated client currently returns 'any' (JSON) for blob endpoint if not configured for binary.
    // For now we might need to fallback to manual fetch for Blob if strictly required,
    // or assume the generated method returns a Blob if we tweak it.
    // However, looking at WorkspaceService.ts, readWorkspaceFileBlobApiWorkspaceFileBlobGet returns CancelablePromise<any>.
    // It calls __request which typically returns JSON.
    // We will stick to raw fetch for this one specific binary endpoint to ensure Blob return.
    const response = await fetch(`/api/workspace/file/blob?path=${encodeURIComponent(path)}`);
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status} ${response.statusText}`);
    }
    return response.blob();
  },

  /**
   * Fetch the workspace file tree, optionally enriched with catalog
   * lineage metadata (`assetId`, `assetKind`, `producerRunId`,
   * `producerTaskId`) for nodes that match a registered asset.
   */
  getWorkspaceTree: async (
    options: { path?: string; maxDepth?: number; includeCatalog?: boolean } = {},
  ): Promise<WorkspaceTreeNodeRaw> => {
    const params = new URLSearchParams();
    params.set("path", options.path ?? "");
    params.set("max_depth", String(options.maxDepth ?? 8));
    if (options.includeCatalog) {
      params.set("include", "catalog");
    }
    const response = await fetch(`/api/workspace/files?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Reverse-lookup: which run/experiment/project produced this file.
   */
  getCatalogByPath: async (path: string): Promise<CatalogByPathResponse> => {
    const response = await fetch(`/api/catalog/by-path?path=${encodeURIComponent(path)}`);
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
  },

  /** Fetch the per-run output file tree, enriched with catalog data. */
  getRunFiles: async (
    projectId: string,
    experimentId: string,
    runId: string,
  ): Promise<RunFilesResponse> => {
    return RunsService.getRunFilesApiProjectsProjectIdExperimentsExperimentIdRunsRunIdFilesGet(
      projectId,
      experimentId,
      runId,
    ) as unknown as Promise<RunFilesResponse>;
  },

  /** Fetch the experiment comparison sweep matrix. */
  getExperimentComparison: async (
    projectId: string,
    experimentId: string,
  ): Promise<ExperimentComparisonResponse> => {
    return ExperimentsService.getExperimentComparisonApiProjectsProjectIdExperimentsExperimentIdComparisonGet(
      projectId,
      experimentId,
    ) as unknown as Promise<ExperimentComparisonResponse>;
  },

  /** Best-effort kill: marks the run as cancelled. */
  killRun: async (
    projectId: string,
    experimentId: string,
    runId: string,
  ): Promise<RunActionResponse> => {
    return RunsService.killRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdKillPost(
      projectId,
      experimentId,
      runId,
    ) as unknown as Promise<RunActionResponse>;
  },

  /** Clone an existing run's parameters into a fresh run. */
  rerunRun: async (
    projectId: string,
    experimentId: string,
    runId: string,
  ): Promise<RunRerunResponse> => {
    return RunsService.rerunRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdRerunPost(
      projectId,
      experimentId,
      runId,
    ) as unknown as Promise<RunRerunResponse>;
  },

  /** Stream URL for a run export zip — used directly via <a href>. */
  runExportUrl: (projectId: string, experimentId: string, runId: string): string => {
    return `/api/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(experimentId)}/runs/${encodeURIComponent(runId)}/export`;
  },
};

// ── Local types for endpoints not strictly modelled in the generated client ──

export interface WorkspaceTreeNodeRaw {
  id?: string;
  name: string;
  path: string;
  type?: string;
  size?: number | null;
  modified?: number | string;
  children?: WorkspaceTreeNodeRaw[];
  assetId?: string | null;
  assetKind?: string | null;
  producerRunId?: string | null;
  producerTaskId?: string | null;
}

export interface CatalogByPathResponse {
  matched: boolean;
  workspaceRelPath: string;
  assetId: string | null;
  assetKind: string | null;
  producer: { runId: string | null; taskId: string | null; executionId: string | null } | null;
  scope: {
    kind: "workspace" | "project" | "experiment" | "run";
    projectId: string | null;
    experimentId: string | null;
    runId: string | null;
  } | null;
  siblings: Array<{ assetId: string; name: string; kind: string; relPath: string }>;
}

export interface RunFileNodeRaw {
  name: string;
  relPath: string;
  type: "file" | "folder";
  size: number | null;
  modified: number | null;
  assetId: string | null;
  assetKind: string | null;
  taskId: string | null;
  children: RunFileNodeRaw[];
}

export interface RunFilesResponse {
  runId: string;
  runDir: string;
  nodes: RunFileNodeRaw[];
}

export interface ComparisonRunRowRaw {
  runId: string;
  status: string;
  parameters: Record<string, unknown>;
  metrics: Record<string, unknown>;
  durationSec: number | null;
  created: string;
  finished: string | null;
  error: { type: string; message: string } | null;
}

export interface ExperimentComparisonResponse {
  experimentId: string;
  projectId: string;
  paramKeys: string[];
  metricKeys: string[];
  runs: ComparisonRunRowRaw[];
}

export interface RunActionResponse {
  runId: string;
  status: string;
  message: string | null;
}

export interface RunRerunResponse {
  sourceRunId: string;
  newRunId: string;
  projectId: string;
  experimentId: string;
  status: string;
}

export const buildEmptySnapshot = (): WorkspaceSnapshot => {
  return {
    projects: [],
    experiments: [],
    runs: [],
    assets: [],
    workflows: [],
    agentSessions: [],
    workspaceRoot: null,
    consoleEntries: [],
  };
};

export const mapProjects = (projects: ApiProjectResponse[]): ProjectSummary[] => {
  return projects.map((project) => ({
    id: project.id,
    name: project.name,
    status: "active",
    summary: project.description || "No description",
    updatedAt: project.created,
  }));
};

export const mapExperiments = (
  projectId: string,
  experiments: ApiExperimentResponse[],
): ExperimentSummary[] => {
  return experiments.map((experiment) => ({
    id: experiment.id,
    name: experiment.name,
    status: "active",
    summary: experiment.description || experiment.workflow || "No workflow",
    workflowFile: experiment.workflow ?? "",
    updatedAt: experiment.created,
    projectId,
    parameterSpace: (experiment.parameterSpace ?? {}) as Record<string, unknown>,
    workflowSource: experiment.workflow ?? null,
  }));
};

export const mapRuns = (
  projectId: string,
  experimentId: string,
  runs: ApiRunResponse[],
): RunSummary[] => {
  const mapStatus = (status: string): RunSummary["status"] => {
    if (status === "running") {
      return "running";
    }
    if (status === "succeeded") {
      return "succeeded";
    }
    if (status === "failed") {
      return "failed";
    }
    if (status === "cancelled") {
      return "cancelled";
    }
    return "pending";
  };

  return runs.map((run) => ({
    executorInfo: Object.fromEntries(
      Object.entries(run.executorInfo ?? {}).map(([key, value]) => [key, String(value)]),
    ),
    id: run.id,
    name: run.id,
    status: mapStatus(run.status),
    summary: `Status: ${run.status}`,
    updatedAt: run.finished ?? run.created,
    projectId,
    experimentId,
    profile: run.profile ?? null,
    configHash: run.configHash ?? null,
    parameters: (run.parameters ?? {}) as Record<string, unknown>,
    results: (run.results ?? {}) as Record<string, unknown>,
    workflowSource: run.workflowSource ?? run.workflow?.source ?? null,
    workflowSnapshot: run.workflow ?? null,
    startedAt: run.created ?? null,
    finishedAt: run.finished ?? null,
    executionHistory: (run.executionHistory ?? []).map((rec) => ({
      executionId: rec.executionId,
      startedAt: rec.startedAt,
      finishedAt: rec.finishedAt ?? null,
      status: rec.status,
      schedulerJobId: rec.schedulerJobId ?? null,
    })),
    errorMessage: run.error?.message ?? null,
  }));
};

const assetSize = (asset: ApiAssetResponse): number | null => {
  const extraSize = (asset.extra as Record<string, unknown> | undefined)?.size;
  return typeof extraSize === "number" ? extraSize : null;
};

const assetSummary = (asset: ApiAssetResponse): string => {
  const scope = asset.scope_kind ? `${asset.scope_kind} scope` : "unscoped";
  return `${asset.kind} · ${scope}`;
};

export const mapAssets = (assets: ApiAssetResponse[], projectId?: string): AssetSummary[] => {
  return assets.map((asset) => ({
    id: asset.id,
    name: asset.name,
    kind: asset.kind,
    status: "active",
    summary: assetSummary(asset),
    updatedAt: asset.updated_at,
    sizeBytes: assetSize(asset),
    projectId,
  }));
};

/**
 * Build a flowgram free-layout document from an experiment's `workflow_source`
 * when it is a serialized IR (`{task_configs, links}` — see `Workflow.to_dict()`
 * / `schema/workflow.json`). Returns `undefined` when the source is absent or is
 * a Python script / path rather than a serialized IR, so callers fall back to
 * the raw string.
 */
export const buildWorkflowDocument = (
  source: string | null | undefined,
): FlowgramDocument | undefined => {
  const ir = parseTaskGraphIr(source);
  if (!ir) return undefined;
  return buildFlowgramDocument(ir);
};

export const mapWorkflows = (
  experiments: ExperimentSummary[],
  rawExperiments: ApiExperimentResponse[],
): WorkflowSummary[] => {
  const experimentById = new Map(rawExperiments.map((experiment) => [experiment.id, experiment]));
  return experiments.map((experiment) => {
    const raw = experimentById.get(experiment.id);
    const source = raw?.workflow ?? null;
    const graph: TaskGraphJson | undefined = parseTaskGraphIr(source) ?? undefined;
    return {
      id: `workflow:${experiment.id}`,
      name: `${experiment.name} workflow`,
      status: "active",
      summary: graph
        ? `${graph.task_configs.length} tasks · ${graph.links.length} dependencies`
        : (source ?? "workflow"),
      updatedAt: experiment.updatedAt,
      projectId: experiment.projectId,
      experimentId: experiment.id,
      graph,
    };
  });
};

export const emptyConsoleEntries = (): ConsoleEntry[] => [];

const mapWorkspaceNode = (node: WorkspaceFileNode): WorkspaceTreeNode => {
  const isFile = node.type === "file";
  const updatedAt =
    typeof node.modified === "number"
      ? new Date(node.modified * 1000).toISOString()
      : (node.modified ?? "");
  return {
    id: node.id ?? node.path,
    name: node.name,
    path: node.path,
    kind: isFile ? "file" : "directory",
    children: (node.children ?? []).map(mapWorkspaceNode),
    sizeBytes: node.size ?? 0,
    updatedAt,
    assetId: node.assetId ?? undefined,
    hasPreviewSidecar: node.hasPreviewSidecar ?? undefined,
  };
};

export const mapWorkspaceTree = (
  rootPath: string,
  response: WorkspaceFilesResponse,
): WorkspaceTreeNode => {
  return {
    id: "workspace-root",
    name: response.path ?? rootPath,
    path: response.path ?? rootPath,
    kind: "directory",
    children: (response.children ?? []).map(mapWorkspaceNode),
    sizeBytes: 0,
    updatedAt: "",
  };
};

export const mapAgentSessions = (sessions: ApiAgentSession[]): AgentSessionSummary[] => {
  return sessions.map((s) => ({
    id: s.taskId ?? s.sessionId,
    sessionId: s.sessionId,
    goal: s.goal,
    status: s.status as AgentSessionSummary["status"],
    createdAt: s.createdAt,
    eventCount: s.events?.length ?? 0,
  }));
};

export interface ApiAgentHealth {
  ready: boolean;
  provider: string;
  model: string;
  source: "stored" | "env" | "none";
  reason: string;
  envVar: string;
}

/**
 * Thrown by createSession when the backend rejects with code
 * "agent_not_configured" (HTTP 400). Carries the structured fields so
 * the UI can route the user to the Provider settings tab.
 */
export class AgentNotConfiguredError extends Error {
  readonly code = "agent_not_configured";
  readonly provider: string;
  readonly model: string;
  readonly envVar: string;

  constructor(message: string, provider: string, model: string, envVar: string) {
    super(message);
    this.name = "AgentNotConfiguredError";
    this.provider = provider;
    this.model = model;
    this.envVar = envVar;
  }
}

/**
 * Optional overrides accepted by ``POST /api/agent/sessions``. Keep aligned
 * with :class:`molexp.server.schemas.requests.GoalCreateRequest`.
 */
export interface SessionLaunchOptions {
  planMode?: boolean;
  instructionsOverride?: string;
  skillId?: string;
}

interface ApiAgentTask {
  taskId: string;
  title: string;
  goal: string;
  status: string;
  createdAt: string;
  updatedAt?: string | null;
  sessionId?: string;
  events?: ApiAgentSession["events"];
  stats?: ApiAgentSession["stats"];
  planMode?: boolean;
  skillId?: string | null;
}

const normalizeAgentTask = (task: ApiAgentTask): ApiAgentSession => ({
  taskId: task.taskId,
  title: task.title,
  sessionId: task.sessionId ?? task.taskId,
  status: task.status,
  goal: task.goal,
  createdAt: task.createdAt,
  updatedAt: task.updatedAt,
  events: task.events ?? [],
  stats: task.stats,
  planMode: task.planMode ?? false,
  skillId: task.skillId ?? null,
});

export const agentApi = {
  listSessions: async (): Promise<ApiAgentSession[]> => {
    const response = await fetch("/api/agent-tasks");
    if (!response.ok) throw new Error(`Failed to fetch agent tasks: ${response.statusText}`);
    const data = await response.json();
    return (data.tasks ?? []).map(normalizeAgentTask);
  },

  getHealth: async (): Promise<ApiAgentHealth> => {
    const response = await fetch("/api/agent/health");
    if (!response.ok) throw new Error(`Failed to fetch agent health: ${response.statusText}`);
    return response.json();
  },

  createSession: async (
    description: string,
    successCriteria: string[] = [],
    options: SessionLaunchOptions = {},
  ): Promise<ApiAgentSession> => {
    const body: Record<string, unknown> = {
      description,
      success_criteria: successCriteria,
    };
    if (options.planMode !== undefined) body.plan_mode = options.planMode;
    if (options.instructionsOverride !== undefined)
      body.instructions_override = options.instructionsOverride;
    if (options.skillId !== undefined) body.skill_id = options.skillId;
    const response = await fetch("/api/agent-tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (response.status === 400) {
      // FastAPI nests structured errors under {"detail": {...}}
      const body = await response.json().catch(() => null);
      const detail = body?.detail;
      if (detail && typeof detail === "object" && detail.code === "agent_not_configured") {
        throw new AgentNotConfiguredError(
          String(detail.message ?? "Agent provider is not configured."),
          String(detail.provider ?? ""),
          String(detail.model ?? ""),
          String(detail.envVar ?? ""),
        );
      }
    }
    if (!response.ok) throw new Error(`Failed to create agent task: ${response.statusText}`);
    return normalizeAgentTask(await response.json());
  },

  getSession: async (sessionId: string): Promise<ApiAgentSession> => {
    const response = await fetch(`/api/agent-tasks/${sessionId}`);
    if (!response.ok) throw new Error(`Failed to fetch agent task: ${response.statusText}`);
    return normalizeAgentTask(await response.json());
  },

  streamEvents: (sessionId: string): EventSource => {
    return new EventSource(`/api/agent-tasks/${sessionId}/events`);
  },

  postMessage: async (
    sessionId: string,
    content: string,
    requestId: string | null = null,
  ): Promise<void> => {
    const response = await fetch(`/api/agent-tasks/${sessionId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, request_id: requestId }),
    });
    if (!response.ok) throw new Error(`Failed to post message: ${response.statusText}`);
  },
};

// ── Agent admin (MCP / tools / skills) ────────────────────────────────────

export type ApiMcpScope = "user" | "workspace";
export type ApiMcpTransport = "stdio" | "http" | "sse";

export interface ApiMcpAuthSummary {
  type: "oauth2";
  scopes: string[];
  clientId: string | null;
  connected: boolean;
}

export interface ApiMcpServer {
  name: string;
  scope: ApiMcpScope;
  transport: string;
  command: string | null;
  args: string[];
  url: string | null;
  envKeys: string[];
  headerKeys: string[];
  secretRefs: string[];
  unresolvedSecrets: string[];
  shadowed: boolean;
  valid: boolean;
  invalidReason: string;
  auth: ApiMcpAuthSummary | null;
}

export interface ApiMcpOAuthStatus {
  name: string;
  scope: ApiMcpScope;
  hasTokens: boolean;
  scopes: string[];
}

export interface ApiMcpOAuthStart {
  name: string;
  scope: ApiMcpScope;
  authorizeUrl: string;
}

export interface ApiMcpServerList {
  workspacePath: string;
  userPath: string;
  servers: ApiMcpServer[];
}

export interface ApiMcpServerTestResult {
  ok: boolean;
  name: string;
  scope: ApiMcpScope;
  transport: string;
  latencyMs: number;
  toolCount: number;
  error: string | null;
}

export interface ApiMcpSecretRow {
  key: string;
  isSet: boolean;
  referencedBy: string[];
}

export interface ApiMcpSecretList {
  scope: ApiMcpScope;
  path: string;
  secrets: ApiMcpSecretRow[];
}

export interface McpOAuth2AuthInput {
  type: "oauth2";
  scopes: string[];
  clientId: string | null;
}

export type McpServerSpecInput =
  | { type: "stdio"; command: string; args: string[]; env: Record<string, string> }
  | {
      type: "http" | "sse";
      url: string;
      headers: Record<string, string>;
      auth?: McpOAuth2AuthInput | null;
    };

export interface McpServerUpsertInput {
  name: string;
  scope: ApiMcpScope;
  spec: McpServerSpecInput;
}

export interface ApiToolParameter {
  name: string;
  annotation: string;
  required: boolean;
}

export interface ApiAgentTool {
  name: string;
  description: string;
  parameters: ApiToolParameter[];
  requiresApproval: boolean;
  source: string;
}

export interface ApiMcpToolGroup {
  server: string;
  scope: ApiMcpScope;
  ok: boolean;
  toolCount: number;
  error: string | null;
}

export interface ApiAgentToolList {
  tools: ApiAgentTool[];
  mcpGroups: ApiMcpToolGroup[];
}

// Tool ``source`` is either ``"native"`` for built-in tools or
// ``"mcp:<server-name>"`` for tools discovered through an MCP server.
// Helpers below keep the prefix in one place.
export const NATIVE_SOURCE = "native";

export const mcpSource = (server: string): string => `mcp:${server}`;

export const isMcpSource = (source: string): boolean => source.startsWith("mcp:");

export interface ApiSkill {
  id: string;
  name: string;
  description: string;
  goalTemplate: string;
  slashName: string;
  instructions: string;
  defaultPlanMode: boolean;
  constraints: string[];
  successCriteria: string[];
  tags: string[];
  createdAt: string;
  updatedAt: string;
}

export interface SkillUpsertInput {
  name: string;
  goalTemplate: string;
  description?: string;
  slashName?: string;
  instructions?: string;
  defaultPlanMode?: boolean;
  constraints?: string[];
  successCriteria?: string[];
  tags?: string[];
}

const _toSkillBody = (input: SkillUpsertInput) => ({
  name: input.name,
  goal_template: input.goalTemplate,
  description: input.description ?? "",
  slash_name: input.slashName ?? "",
  instructions: input.instructions ?? "",
  default_plan_mode: input.defaultPlanMode ?? false,
  constraints: input.constraints ?? [],
  success_criteria: input.successCriteria ?? [],
  tags: input.tags ?? [],
});

// ── Slash commands + system prompt ────────────────────────────────────────

export interface ApiCommandParameter {
  name: string;
  required: boolean;
}

export interface ApiCommand {
  slashName: string;
  name: string;
  description: string;
  parameters: ApiCommandParameter[];
  defaultPlanMode: boolean;
  isBuiltin: boolean;
  skillId: string | null;
}

export interface ApiCommandParse {
  kind: "skill" | "builtin" | "error";
  name: string;
  skillId: string;
  parameters: Record<string, string>;
  planMode: boolean;
  error: string;
}

export interface ApiAgentSystemPrompt {
  base: string;
  workspaceInstructions: string;
  skillInstructions: string;
  sessionOverride: string | null;
  planMode: boolean;
  effective: string;
}

/** RESERVED_SLASH_NAMES mirrors the backend whitelist for client-side validation. */
export const RESERVED_SLASH_NAMES = ["plan", "clear", "model", "help"] as const;
export const SLASH_NAME_PATTERN = /^[a-z0-9][a-z0-9-]{0,31}$/;

// Provider config — read/write the workspace's LLM provider settings.
export type ApiProviderName = "anthropic" | "openai" | "google" | "deepseek" | "openai-compatible";

export interface ApiAgentProvider {
  provider: ApiProviderName;
  model: string;
  baseUrl: string;
  apiKeyPreview: string;
  apiKeySet: boolean;
  instructions: string;
  supportedProviders: ApiProviderName[];
}

export interface ProviderUpdateInput {
  provider?: ApiProviderName;
  model?: string;
  apiKey?: string;
  baseUrl?: string;
  instructions?: string;
}

export interface ApiAgentProviderTestResult {
  ok: boolean;
  provider: string;
  model: string;
  latencyMs: number;
  reply: string;
  error: string | null;
}

const _toProviderBody = (input: ProviderUpdateInput): Record<string, unknown> => {
  const body: Record<string, unknown> = {};
  if (input.provider !== undefined) body.provider = input.provider;
  if (input.model !== undefined) body.model = input.model;
  if (input.apiKey !== undefined) body.api_key = input.apiKey;
  if (input.baseUrl !== undefined) body.base_url = input.baseUrl;
  if (input.instructions !== undefined) body.instructions = input.instructions;
  return body;
};

export const agentAdminApi = {
  getProvider: async (): Promise<ApiAgentProvider> => {
    const response = await fetch("/api/agent/provider");
    if (!response.ok) throw new Error(`Failed to fetch provider: ${response.statusText}`);
    return response.json();
  },

  updateProvider: async (input: ProviderUpdateInput): Promise<ApiAgentProvider> => {
    const response = await fetch("/api/agent/provider", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(_toProviderBody(input)),
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Failed to update provider: ${response.statusText} ${detail}`);
    }
    return response.json();
  },

  testProvider: async (input: ProviderUpdateInput): Promise<ApiAgentProviderTestResult> => {
    const response = await fetch("/api/agent/provider/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(_toProviderBody(input)),
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Failed to test provider: ${response.statusText} ${detail}`);
    }
    return response.json();
  },

  listMcpServers: async (): Promise<ApiMcpServerList> => {
    const response = await fetch("/api/agent/mcp/servers");
    if (!response.ok) throw new Error(`Failed to fetch MCP servers: ${response.statusText}`);
    return response.json();
  },

  createMcpServer: async (input: McpServerUpsertInput): Promise<ApiMcpServer> => {
    const response = await fetch("/api/agent/mcp/servers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Failed to create MCP server: ${response.statusText} ${detail}`);
    }
    return response.json();
  },

  replaceMcpServer: async (name: string, input: McpServerUpsertInput): Promise<ApiMcpServer> => {
    const response = await fetch(`/api/agent/mcp/servers/${encodeURIComponent(name)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Failed to update MCP server: ${response.statusText} ${detail}`);
    }
    return response.json();
  },

  deleteMcpServer: async (name: string, scope: ApiMcpScope): Promise<void> => {
    const response = await fetch(
      `/api/agent/mcp/servers/${encodeURIComponent(name)}?scope=${scope}`,
      { method: "DELETE" },
    );
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Failed to delete MCP server: ${response.statusText} ${detail}`);
    }
  },

  testMcpServer: async (name: string, scope: ApiMcpScope): Promise<ApiMcpServerTestResult> => {
    const response = await fetch(
      `/api/agent/mcp/servers/${encodeURIComponent(name)}/test?scope=${scope}`,
      { method: "POST" },
    );
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Failed to test MCP server: ${response.statusText} ${detail}`);
    }
    return response.json();
  },

  startMcpOauth: async (name: string, scope: ApiMcpScope): Promise<ApiMcpOAuthStart> => {
    const response = await fetch(
      `/api/agent/mcp/servers/${encodeURIComponent(name)}/oauth/start?scope=${scope}`,
      { method: "POST" },
    );
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Failed to start OAuth: ${response.statusText} ${detail}`);
    }
    return response.json();
  },

  callbackMcpOauth: async (
    name: string,
    scope: ApiMcpScope,
    code: string,
    state: string | null,
  ): Promise<ApiMcpOAuthStatus> => {
    const response = await fetch(
      `/api/agent/mcp/servers/${encodeURIComponent(name)}/oauth/callback?scope=${scope}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code, state }),
      },
    );
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`OAuth callback failed: ${response.statusText} ${detail}`);
    }
    return response.json();
  },

  getMcpOauthStatus: async (name: string, scope: ApiMcpScope): Promise<ApiMcpOAuthStatus> => {
    const response = await fetch(
      `/api/agent/mcp/servers/${encodeURIComponent(name)}/oauth?scope=${scope}`,
    );
    if (!response.ok) {
      throw new Error(`Failed to get OAuth status: ${response.statusText}`);
    }
    return response.json();
  },

  disconnectMcpOauth: async (name: string, scope: ApiMcpScope): Promise<void> => {
    const response = await fetch(
      `/api/agent/mcp/servers/${encodeURIComponent(name)}/oauth?scope=${scope}`,
      { method: "DELETE" },
    );
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Failed to disconnect OAuth: ${response.statusText} ${detail}`);
    }
  },

  listMcpSecrets: async (scope: ApiMcpScope): Promise<ApiMcpSecretList> => {
    const response = await fetch(`/api/agent/mcp/secrets?scope=${scope}`);
    if (!response.ok) {
      throw new Error(`Failed to list MCP secrets: ${response.statusText}`);
    }
    return response.json();
  },

  setMcpSecret: async (key: string, value: string, scope: ApiMcpScope): Promise<void> => {
    const response = await fetch(`/api/agent/mcp/secrets/${encodeURIComponent(key)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value, scope }),
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Failed to set MCP secret: ${response.statusText} ${detail}`);
    }
  },

  listTools: async (): Promise<ApiAgentTool[]> => {
    const response = await fetch("/api/agent/tools");
    if (!response.ok) throw new Error(`Failed to fetch tools: ${response.statusText}`);
    const data = await response.json();
    return data.tools ?? [];
  },

  listToolsAndGroups: async (): Promise<ApiAgentToolList> => {
    const response = await fetch("/api/agent/tools");
    if (!response.ok) throw new Error(`Failed to fetch tools: ${response.statusText}`);
    const data = await response.json();
    return { tools: data.tools ?? [], mcpGroups: data.mcpGroups ?? [] };
  },

  listSkills: async (): Promise<ApiSkill[]> => {
    const response = await fetch("/api/agent/skills");
    if (!response.ok) throw new Error(`Failed to fetch skills: ${response.statusText}`);
    const data = await response.json();
    return data.skills ?? [];
  },

  createSkill: async (input: SkillUpsertInput): Promise<ApiSkill> => {
    const response = await fetch("/api/agent/skills", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(_toSkillBody(input)),
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Failed to create skill: ${response.statusText} ${detail}`);
    }
    return response.json();
  },

  updateSkill: async (skillId: string, input: Partial<SkillUpsertInput>): Promise<ApiSkill> => {
    const body: Record<string, unknown> = {};
    if (input.name !== undefined) body.name = input.name;
    if (input.goalTemplate !== undefined) body.goal_template = input.goalTemplate;
    if (input.description !== undefined) body.description = input.description;
    if (input.slashName !== undefined) body.slash_name = input.slashName;
    if (input.instructions !== undefined) body.instructions = input.instructions;
    if (input.defaultPlanMode !== undefined) body.default_plan_mode = input.defaultPlanMode;
    if (input.constraints !== undefined) body.constraints = input.constraints;
    if (input.successCriteria !== undefined) body.success_criteria = input.successCriteria;
    if (input.tags !== undefined) body.tags = input.tags;
    const response = await fetch(`/api/agent/skills/${skillId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Failed to update skill: ${response.statusText} ${detail}`);
    }
    return response.json();
  },

  deleteSkill: async (skillId: string): Promise<void> => {
    const response = await fetch(`/api/agent/skills/${skillId}`, { method: "DELETE" });
    if (!response.ok) throw new Error(`Failed to delete skill: ${response.statusText}`);
  },

  launchSkill: async (
    skillId: string,
    parameters: Record<string, unknown> = {},
    options: { planMode?: boolean } = {},
  ): Promise<ApiAgentSession> => {
    const body: Record<string, unknown> = { parameters };
    if (options.planMode !== undefined) body.plan_mode = options.planMode;
    const response = await fetch(`/api/agent/skills/${skillId}/launch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(`Failed to launch skill: ${response.statusText}`);
    return response.json();
  },
};

// ── Slash commands ────────────────────────────────────────────────────────

export const commandsApi = {
  list: async (): Promise<ApiCommand[]> => {
    const response = await fetch("/api/agent/commands");
    if (!response.ok) throw new Error(`Failed to fetch commands: ${response.statusText}`);
    const data = await response.json();
    return data.commands ?? [];
  },

  parse: async (raw: string): Promise<ApiCommandParse> => {
    const response = await fetch("/api/agent/commands/parse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw }),
    });
    if (!response.ok) throw new Error(`Failed to parse command: ${response.statusText}`);
    return response.json();
  },
};

// ── Per-session prompt inspection ─────────────────────────────────────────

export const planApi = {
  getSystemPrompt: async (sessionId: string): Promise<ApiAgentSystemPrompt> => {
    const response = await fetch(`/api/agent/sessions/${sessionId}/system-prompt`);
    if (!response.ok) throw new Error(`Failed to fetch system prompt: ${response.statusText}`);
    return response.json();
  },
};

// ── Workflow document write-back (flowgram canvas) ─────────────────────────

export const workflowApi = {
  /**
   * Persist an edited workflow document through the generated WorkflowService
   * (never a hand-rolled fetch). `document` is the backend wire IR
   * ({task_configs, links, ...}); returns the server-normalized wire IR.
   */
  save: async (
    projectId: string,
    experimentId: string,
    document: Record<string, unknown>,
  ): Promise<Record<string, unknown>> => {
    const response =
      await WorkflowService.putWorkflowDocumentApiProjectsProjectIdExperimentsExperimentIdWorkflowPut(
        projectId,
        experimentId,
        { document },
      );
    return response.document as Record<string, unknown>;
  },
};

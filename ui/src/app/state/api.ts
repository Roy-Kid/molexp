import { AssetsService } from "@/api/generated/services/AssetsService";
import { ExecutionService } from "@/api/generated/services/ExecutionService";
import { ExperimentsService } from "@/api/generated/services/ExperimentsService";
import { ProjectsService } from "@/api/generated/services/ProjectsService";
import { RunsService } from "@/api/generated/services/RunsService";
import { WorkspaceService } from "@/api/generated/services/WorkspaceService";
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

export const mapWorkflows = (
  experiments: ExperimentSummary[],
  rawExperiments: ApiExperimentResponse[],
): WorkflowSummary[] => {
  const experimentById = new Map(rawExperiments.map((experiment) => [experiment.id, experiment]));
  return experiments.map((experiment) => {
    const raw = experimentById.get(experiment.id);
    const workflowPath = raw?.workflow ?? "workflow";
    return {
      id: `workflow:${experiment.id}`,
      name: `${experiment.name} workflow`,
      status: "active",
      summary: workflowPath,
      updatedAt: experiment.updatedAt,
      projectId: experiment.projectId,
      experimentId: experiment.id,
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
    id: s.sessionId,
    goalDescription: s.goalDescription,
    status: s.status as AgentSessionSummary["status"],
    createdAt: s.createdAt,
    eventCount: s.events?.length ?? 0,
  }));
};

export const agentApi = {
  listSessions: async (): Promise<ApiAgentSession[]> => {
    const response = await fetch("/api/agent/sessions");
    if (!response.ok) throw new Error(`Failed to fetch sessions: ${response.statusText}`);
    const data = await response.json();
    return data.sessions ?? [];
  },

  createSession: async (
    description: string,
    successCriteria: string[] = [],
  ): Promise<ApiAgentSession> => {
    const response = await fetch("/api/agent/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ description, success_criteria: successCriteria }),
    });
    if (!response.ok) throw new Error(`Failed to create session: ${response.statusText}`);
    return response.json();
  },

  getSession: async (sessionId: string): Promise<ApiAgentSession> => {
    const response = await fetch(`/api/agent/sessions/${sessionId}`);
    if (!response.ok) throw new Error(`Failed to fetch session: ${response.statusText}`);
    return response.json();
  },

  streamEvents: (sessionId: string): EventSource => {
    return new EventSource(`/api/agent/sessions/${sessionId}/events`);
  },

  respondApproval: async (
    sessionId: string,
    requestId: string,
    approved: boolean,
  ): Promise<void> => {
    const response = await fetch(`/api/agent/sessions/${sessionId}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: requestId, approved }),
    });
    if (!response.ok) throw new Error(`Failed to respond approval: ${response.statusText}`);
  },
};

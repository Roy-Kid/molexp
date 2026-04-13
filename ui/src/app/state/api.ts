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

// Local types not yet in OpenAPI
interface WorkspaceFileNode {
  id?: string;
  name: string;
  path: string;
  type?: string;
  children?: WorkspaceFileNode[];
  size?: number;
  modified?: string | number;
}

interface WorkspaceFilesResponse {
  path?: string;
  children?: WorkspaceFileNode[];
}

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
  getWorkspaceTree: async (path: string): Promise<WorkspaceFilesResponse> => {
    const response = await WorkspaceService.listWorkspaceFilesApiWorkspaceFilesGet(path, 8);
    return response as unknown as WorkspaceFilesResponse;
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
};

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
    summary: experiment.description || experiment.workflow,
    workflowFile: experiment.workflow,
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
    name: "runId" in run && typeof run.runId === "string" ? run.runId : run.id,
    status: mapStatus(run.status),
    summary: `Status: ${run.status}`,
    updatedAt: run.finished ?? run.created,
    projectId,
    experimentId,
  }));
};

export const mapAssets = (assets: ApiAssetResponse[], projectId?: string): AssetSummary[] => {
  return assets.map((asset) => ({
    id: asset.id,
    name: asset.assetId,
    status: "active",
    summary: `${asset.type} • ${asset.format}`,
    updatedAt: asset.created,
    sizeBytes: asset.size,
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
    const workflowPath = raw ? raw.workflow : "workflow";
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
    eventCount: s.events.length,
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

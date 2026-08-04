import type { BacklinksResponse } from "@/api/generated/models/BacklinksResponse";
import { EmbedRequest } from "@/api/generated/models/EmbedRequest";
import type { EmbedResponse } from "@/api/generated/models/EmbedResponse";
import type { EntityCard } from "@/api/generated/models/EntityCard";
import type { KnowledgeListResponse } from "@/api/generated/models/KnowledgeListResponse";
import type { KnowledgeSearchResponse } from "@/api/generated/models/KnowledgeSearchResponse";
import type { NoteDetailResponse } from "@/api/generated/models/NoteDetailResponse";
import type { NoteSummary } from "@/api/generated/models/NoteSummary";
import type { PlanDetailResponse } from "@/api/generated/models/PlanDetailResponse";
import type { PlanListResponse } from "@/api/generated/models/PlanListResponse";
import type { PlanTaskCreateRequest } from "@/api/generated/models/PlanTaskCreateRequest";
import type { PlanTaskResponse } from "@/api/generated/models/PlanTaskResponse";
import type { WorkspacePlanListResponse } from "@/api/generated/models/WorkspacePlanListResponse";
import { AssetsService } from "@/api/generated/services/AssetsService";
import { ExecutionService } from "@/api/generated/services/ExecutionService";
import { ExperimentsService } from "@/api/generated/services/ExperimentsService";
import { KnowledgeService } from "@/api/generated/services/KnowledgeService";
import { PlansService } from "@/api/generated/services/PlansService";
import { PlanTasksService } from "@/api/generated/services/PlanTasksService";
import { ProjectsService } from "@/api/generated/services/ProjectsService";
import { RunsService } from "@/api/generated/services/RunsService";
import { WorkflowService } from "@/api/generated/services/WorkflowService";
import { WorkspaceService } from "@/api/generated/services/WorkspaceService";
import { AgentUnavailableError, probeOnce, resetAgentProbes } from "@/app/state/agentProbe";
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
  ExperimentCreateRequest,
  ExperimentSummary,
  ProjectCreateRequest,
  ProjectSummary,
  RunCreateRequest,
  RunSummary,
  ServedWorkspaceSummary,
  WorkflowSummary,
  WorkspaceSnapshot,
  WorkspaceTreeNode,
} from "@/app/types";
import {
  buildFlowgramDocument,
  type FlowgramDocument,
  parseTaskGraphIr,
} from "@/components/workflow/flowgram-document";
import type { TaskGraphJson } from "@/components/workflow/task-graph-ir";

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

export type { EntityCard } from "@/api/generated/models/EntityCard";
export type { LammpsLogResponse } from "@/api/generated/models/LammpsLogResponse";
export type { LammpsThermoStage } from "@/api/generated/models/LammpsThermoStage";
export type { RunFileTextResponse } from "@/api/generated/models/RunFileTextResponse";

/** The entity kinds a knowledge document can embed (mirrors ``EmbedRequest.target_kind``). */
export type EmbedTargetKind = "run" | "experiment" | "asset" | "reference";

/** The typed provenance-edge role an embed writes (mirrors ``EmbedRequest.role``). */
export type EmbedRole = "derived_from" | "cites" | "supersedes" | "records" | "references";

const EMBED_TARGET_KIND: Record<EmbedTargetKind, EmbedRequest.target_kind> = {
  run: EmbedRequest.target_kind.RUN,
  experiment: EmbedRequest.target_kind.EXPERIMENT,
  asset: EmbedRequest.target_kind.ASSET,
  reference: EmbedRequest.target_kind.REFERENCE,
};

export const workspaceApi = {
  /** Active workspace root (+ optional remote readiness flags from newer servers). */
  getWorkspaceInfo: async (): Promise<{
    root: string;
    projectCount: number;
    assetCount: number;
    connected?: boolean | null;
    indexed?: boolean | null;
    ready?: boolean | null;
  }> => {
    // Generated type may lag openapi dump; cast keeps extra fields when present.
    return WorkspaceService.getWorkspaceInfoApiWorkspaceInfoGet() as Promise<{
      root: string;
      projectCount: number;
      assetCount: number;
      connected?: boolean | null;
      indexed?: boolean | null;
      ready?: boolean | null;
    }>;
  },
  getProjects: async (): Promise<ApiProjectResponse[]> => {
    return ProjectsService.listProjectsApiProjectsGet();
  },
  // The served-workspace set (GET /api/workspaces) is outside the generated
  // client; a plain fetch keeps it decoupled from the per-workspace routes.
  getServedWorkspaces: async (): Promise<ServedWorkspaceSummary[]> => {
    const response = await fetch("/api/workspaces");
    if (!response.ok) return [];
    const rows = (await response.json()) as Array<{
      key: string;
      label: string;
      isRemote: boolean;
      path: string | null;
      active?: boolean;
      unreachable?: boolean;
    }>;
    return rows.map((row) => ({
      key: row.key,
      label: row.label,
      isRemote: row.isRemote,
      path: row.path ?? null,
      active: row.active ?? false,
      unreachable: row.unreachable ?? false,
    }));
  },
  // Switch the active workspace (used when a user opens a non-active workspace
  // in the multi-workspace nav). Local switches by path; remote by target name
  // (which equals the served key).
  activateServedWorkspace: async (workspace: ServedWorkspaceSummary): Promise<void> => {
    const body = workspace.isRemote
      ? { kind: "remote", name: workspace.key }
      : { kind: "local", path: workspace.path };
    const response = await fetch("/api/workspace/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(`Failed to activate workspace ${workspace.key}: ${response.status}`);
    }
  },
  // Projects of one named workspace via the aggregate route
  // (GET /api/workspaces/{ws}/projects). Used when several workspaces are
  // served so each group lists its own projects without a collision.
  getProjectsForWorkspace: async (workspaceKey: string): Promise<ApiProjectResponse[]> => {
    const response = await fetch(`/api/workspaces/${encodeURIComponent(workspaceKey)}/projects`);
    if (!response.ok) {
      throw new Error(`Failed to fetch projects for workspace ${workspaceKey}: ${response.status}`);
    }
    return (await response.json()) as ApiProjectResponse[];
  },
  createProject: async (data: ProjectCreateRequest): Promise<ApiProjectResponse> => {
    return ProjectsService.createProjectApiProjectsPost(data);
  },
  deleteProject: async (projectId: string): Promise<void> => {
    await ProjectsService.deleteProjectApiProjectsProjectIdDelete(projectId);
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
  getRunExecution: async (
    projectId: string,
    experimentId: string,
    runId: string,
    executionId?: string | null,
  ) => {
    return RunsService.getRunExecutionApiProjectsProjectIdExperimentsExperimentIdRunsRunIdExecutionGet(
      projectId,
      experimentId,
      runId,
      executionId,
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
  createPlanTask: async (
    projectId: string,
    experimentId: string,
    data: PlanTaskCreateRequest,
  ): Promise<PlanTaskResponse> => {
    return PlanTasksService.createPlanTaskApiProjectsProjectIdExperimentsExperimentIdPlanTasksPost(
      projectId,
      experimentId,
      data,
    );
  },
  getPlanTask: async (
    projectId: string,
    experimentId: string,
    taskId: string,
  ): Promise<PlanTaskResponse> => {
    return PlanTasksService.getPlanTaskApiProjectsProjectIdExperimentsExperimentIdPlanTasksTaskIdGet(
      projectId,
      experimentId,
      taskId,
    );
  },
  // Generated (durable) plans: the persisted PlanMode result for an experiment.
  listPlans: async (projectId: string, experimentId: string): Promise<PlanListResponse> => {
    return PlansService.listPlansApiProjectsProjectIdExperimentsExperimentIdPlansGet(
      projectId,
      experimentId,
    );
  },
  // Every generated plan in the active workspace (the Agents hub's unified list).
  listAllPlans: async (): Promise<WorkspacePlanListResponse> => {
    return PlansService.listAllPlansApiPlansGet();
  },
  getPlan: async (
    projectId: string,
    experimentId: string,
    runId: string,
  ): Promise<PlanDetailResponse> => {
    return PlansService.getPlanApiProjectsProjectIdExperimentsExperimentIdPlansRunIdGet(
      projectId,
      experimentId,
      runId,
    );
  },
  // OKF knowledge concepts (Notes + References) for the active workspace.
  // Optional `tag` / `status` AND-narrow the note list (06 added the query
  // support); existing callers pass nothing and get the full list.
  listKnowledge: async (
    options: { tag?: string | null; status?: string | null } = {},
  ): Promise<KnowledgeListResponse> => {
    return KnowledgeService.listKnowledgeApiKnowledgeGet(
      options.tag ?? undefined,
      options.status ?? undefined,
    );
  },
  getNote: async (path: string): Promise<NoteDetailResponse> => {
    return KnowledgeService.getNoteApiKnowledgeNoteGet(path);
  },
  // Body-aware knowledge search — pure exposure of the ONE Bundle.search verb
  // (vision-loop-08); all matching semantics live server-side in the workspace.
  searchKnowledge: async (
    q: string,
    options: { type?: string | null; tag?: string | null } = {},
  ): Promise<KnowledgeSearchResponse> => {
    return KnowledgeService.searchKnowledgeApiKnowledgeSearchGet(
      q,
      options.type ?? undefined,
      options.tag ?? undefined,
    );
  },
  /**
   * The resolved summary cards for a note's embedded entities (06's card
   * resolver, ridden through ``getNote``). A thin read wrapper so the entity-card
   * UI never re-derives the request shape at the call site.
   */
  getNoteCards: async (path: string): Promise<EntityCard[]> => {
    const detail = await KnowledgeService.getNoteApiKnowledgeNoteGet(path);
    return detail.cards ?? [];
  },
  /**
   * Embed a live workspace entity into a note as one typed provenance edge
   * (06's ``POST /knowledge/doc/embed``). Maps the friendly kind onto the
   * generated ``EmbedRequest.target_kind`` enum so callers pass a plain string.
   */
  embedEntity: async (
    path: string,
    request: {
      targetKind: EmbedTargetKind;
      target: string;
      role?: EmbedRole | null;
      text?: string | null;
    },
  ): Promise<EmbedResponse> => {
    return KnowledgeService.embedDocApiKnowledgeDocEmbedPost(path, {
      target_kind: EMBED_TARGET_KIND[request.targetKind],
      target: request.target,
      role: request.role ?? null,
      text: request.text ?? null,
    });
  },
  /**
   * Rewrite a note's body (its `index.md`) through the generated
   * KnowledgeService (never a hand-rolled fetch — mirrors `workflowApi.save`).
   * Returns the server-normalized NoteDetailResponse so the caller can realign
   * its in-memory body with what was persisted.
   */
  updateNoteDoc: async (path: string, body: string): Promise<NoteDetailResponse> => {
    return KnowledgeService.editDocApiKnowledgeDocPut(path, { body });
  },
  /**
   * Update a note's tags/status (PATCH /knowledge/doc/meta) through the
   * generated KnowledgeService — never a hand-rolled fetch. Each field is
   * optional; omit one to leave it untouched (the server preserves the sibling
   * via `Note.set_tags` / `Note.set_status`). Returns the server-normalized
   * NoteSummary so the caller can realign its in-memory tags/status.
   */
  updateNoteMeta: async (
    path: string,
    patch: { tags?: string[]; status?: string },
  ): Promise<NoteSummary> => {
    return KnowledgeService.updateDocMetaApiKnowledgeDocMetaPatch(path, patch);
  },
  /**
   * Create a Note document via the generated KnowledgeService. `parentPath`
   * nests the new doc beneath an existing Note (its bundle-relative path);
   * omit it to create a root-bundle knowledge-base doc.
   */
  createKnowledgeDoc: async (
    name: string,
    options: { parentPath?: string | null; body?: string } = {},
  ): Promise<NoteSummary> => {
    return KnowledgeService.createDocApiKnowledgeDocPost({
      name,
      parentPath: options.parentPath ?? null,
      body: options.body ?? "",
    });
  },
  /** Rename a Note (PATCH /knowledge/doc with a new `name`). */
  renameKnowledgeDoc: async (path: string, name: string): Promise<NoteSummary> => {
    return KnowledgeService.moveDocApiKnowledgeDocPatch(path, { name });
  },
  /** Reparent a Note under `parentPath` (PATCH /knowledge/doc). */
  moveKnowledgeDoc: async (path: string, parentPath: string): Promise<NoteSummary> => {
    return KnowledgeService.moveDocApiKnowledgeDocPatch(path, { parentPath });
  },
  /** Delete a Note (its directory subtree) via the generated KnowledgeService. */
  deleteKnowledgeDoc: async (path: string): Promise<void> => {
    await KnowledgeService.deleteDocApiKnowledgeDocDelete(path);
  },
  /** Every Concept linking at `path` (GET /knowledge/backlinks). */
  getKnowledgeBacklinks: async (path: string): Promise<BacklinksResponse> => {
    return KnowledgeService.getBacklinksApiKnowledgeBacklinksGet(path);
  },
  /**
   * Plain URL for a browser download of a Note's portable Markdown
   * (GET /knowledge/doc/export). Used directly via `<a href>` — never fetched —
   * so the `Content-Disposition` attachment header drives the download.
   */
  knowledgeDocExportUrl: (path: string): string => {
    return `/api/knowledge/doc/export?path=${encodeURIComponent(path)}`;
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
  /**
   * List workspace directory via WorkspaceFs (Path + Fs model).
   * Prefer this over raw fetch so local/remote stay transparent.
   */
  getWorkspaceTree: async (
    options: { path?: string; maxDepth?: number; includeCatalog?: boolean } = {},
  ): Promise<WorkspaceFilesResponse> => {
    const { getWorkspaceFs } = await import("@/lib/workspace-fs");
    const fs = getWorkspaceFs();
    const dirents = await fs.listdir(options.path ?? "", {
      maxDepth: options.maxDepth ?? 2,
      includeCatalog: options.includeCatalog,
    });
    // Adapt domain dirents back to the legacy raw wire shape for mapWorkspaceTree.
    const toRaw = (d: (typeof dirents)[number]): WorkspaceFileNode => ({
      name: d.name,
      path: d.path,
      type: d.kind === "file" ? "file" : "folder",
      size: d.sizeBytes,
      modified: d.mtime ?? undefined,
      children: d.children.map(toRaw),
      assetId: d.assetId,
      hasPreviewSidecar: d.hasPreviewSidecar,
    });
    return {
      path: options.path || fs.root || "/",
      children: dirents.map(toRaw),
    };
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

  /** Canonical cancel verb: POST …/cancel (same as CLI `molexp runs cancel`). */
  killRun: async (
    projectId: string,
    experimentId: string,
    runId: string,
  ): Promise<RunActionResponse> => {
    return RunsService.cancelRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdCancelPost(
      projectId,
      experimentId,
      runId,
    ) as unknown as Promise<RunActionResponse>;
  },

  /** Resume a run in place: reopen its last non-succeeded execution, seeding completed nodes. */
  resumeRun: async (
    projectId: string,
    experimentId: string,
    runId: string,
  ): Promise<RunContinueResponse> => {
    return RunsService.resumeRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdResumePost(
      projectId,
      experimentId,
      runId,
    ) as unknown as Promise<RunContinueResponse>;
  },

  /** Rerun a run from scratch in a new execution on the same run (no clone). */
  rerunRun: async (
    projectId: string,
    experimentId: string,
    runId: string,
    fresh: boolean = false,
  ): Promise<RunContinueResponse> => {
    return RunsService.rerunRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdRerunPost(
      projectId,
      experimentId,
      runId,
      fresh,
    ) as unknown as Promise<RunContinueResponse>;
  },

  /**
   * Start a pending run by dispatching it to a compute target (the `run` verb).
   * Target-less runs 422 — those execute via `molexp run` on the host.
   */
  startRun: async (
    projectId: string,
    experimentId: string,
    runId: string,
    target: string,
    parameters?: Record<string, unknown>,
  ): Promise<RunContinueResponse> => {
    return RunsService.startRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdRunPost(
      projectId,
      experimentId,
      runId,
      { target, parameters: parameters ?? null },
    ) as unknown as Promise<RunContinueResponse>;
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

export interface RunContinueResponse {
  runId: string;
  executionId: string;
  projectId: string;
  experimentId: string;
  status: string;
}

export const buildEmptySnapshot = (): WorkspaceSnapshot => {
  return {
    workspaces: [],
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

export const mapProjects = (
  projects: ApiProjectResponse[],
  workspaceKey?: string,
): ProjectSummary[] => {
  return projects.map((project) => ({
    id: project.id,
    name: project.name,
    status: "active",
    summary: project.description || "No description",
    updatedAt: project.created,
    experimentCount: project.experimentCount ?? null,
    ...(workspaceKey ? { workspaceKey } : {}),
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
    summary: experiment.description || "",
    workflowFile: experiment.workflow ?? "",
    updatedAt: experiment.created,
    projectId,
    parameterSpace: (experiment.parameterSpace ?? {}) as Record<string, unknown>,
    workflowSource: experiment.workflow ?? null,
    planRunId: experiment.planRunId ?? null,
    runCount: experiment.runCount ?? null,
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
  return assets.map((asset) => {
    // ``scope_ids`` is the parent chain ending at the leaf scope: a run-scoped
    // asset is ``[projectId, experimentId, runId]``, an experiment-scoped one
    // ``[projectId, experimentId]``, etc. This drives the Assets nav grouping.
    const ids = asset.scope_ids ?? [];
    return {
      id: asset.id,
      name: asset.name,
      kind: asset.kind,
      status: "active",
      summary: assetSummary(asset),
      updatedAt: asset.updated_at,
      sizeBytes: assetSize(asset),
      scopeKind: asset.scope_kind,
      projectId: ids[0] ?? projectId,
      experimentId: ids[1],
      runId: ids[2],
    };
  });
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
    title: s.title ?? "",
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
  /** Canonical agent for the first turn — only ``mode``, never plan_mode. */
  mode?: "chat" | "plan";
  instructionsOverride?: string;
  skillId?: string;
  /** Mount scope (vision-loop-11): the entity whose state seeds the session. */
  projectId?: string;
  experimentId?: string;
  runId?: string;
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
  activeMode?: "chat" | "plan";
  activeTurnId?: string | null;
  activePlanTaskId?: string | null;
  skillId?: string | null;
  projectId?: string | null;
  experimentId?: string | null;
  runId?: string | null;
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
  activeMode: (task.activeMode ??
    (task.planMode ? "plan" : "chat")) as ApiAgentSession["activeMode"],
  activeTurnId: task.activeTurnId ?? null,
  activePlanTaskId: task.activePlanTaskId ?? null,
  skillId: task.skillId ?? null,
  projectId: task.projectId ?? null,
  experimentId: task.experimentId ?? null,
  runId: task.runId ?? null,
});

export const agentApi = {
  listSessions: async (): Promise<ApiAgentSession[]> => {
    const response = await fetch("/api/agent-tasks");
    if (!response.ok) throw new Error(`Failed to fetch agent tasks: ${response.statusText}`);
    const data = await response.json();
    return (data.tasks ?? []).map(normalizeAgentTask);
  },

  // Probe endpoint: routed through probeOnce so an unconfigured agent stack
  // (503) is detected once and never re-requested (see agentProbe.ts).
  getHealth: (): Promise<ApiAgentHealth> =>
    probeOnce("agent-health", async () => {
      const response = await fetch("/api/agent/health");
      if (response.status === 503) throw new AgentUnavailableError("/api/agent/health");
      if (!response.ok) throw new Error(`Failed to fetch agent health: ${response.statusText}`);
      return response.json();
    }),

  createSession: async (
    description: string,
    successCriteria: string[] = [],
    options: SessionLaunchOptions = {},
  ): Promise<ApiAgentSession> => {
    const body: Record<string, unknown> = {
      description,
      success_criteria: successCriteria,
    };
    // Canonical field only — never dual-write plan_mode.
    if (options.mode !== undefined) body.mode = options.mode;
    if (options.instructionsOverride !== undefined)
      body.instructions_override = options.instructionsOverride;
    if (options.skillId !== undefined) body.skill_id = options.skillId;
    if (options.projectId !== undefined) body.projectId = options.projectId;
    if (options.experimentId !== undefined) body.experimentId = options.experimentId;
    if (options.runId !== undefined) body.runId = options.runId;
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
    mode: "chat" | "plan" = "chat",
  ): Promise<void> => {
    const response = await fetch(`/api/agent-tasks/${sessionId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, request_id: requestId, mode }),
    });
    if (!response.ok) {
      // Surface FastAPI `detail` (e.g. missing model) — statusText alone is useless.
      let detail = "";
      try {
        const body = (await response.json()) as { detail?: unknown };
        if (typeof body.detail === "string") detail = body.detail;
        else if (Array.isArray(body.detail))
          detail = body.detail
            .map((d) => (typeof d === "object" && d && "msg" in d ? String(d.msg) : String(d)))
            .join("; ");
        else if (body.detail != null) detail = JSON.stringify(body.detail);
      } catch {
        /* ignore non-JSON error bodies */
      }
      throw new Error(
        detail || `Failed to post message: ${response.status} ${response.statusText}`,
      );
    }
  },

  /** Stop the in-flight turn for a task (idempotent when already idle). */
  cancelSession: async (sessionId: string): Promise<void> => {
    const response = await fetch(`/api/agent-tasks/${encodeURIComponent(sessionId)}/cancel`, {
      method: "POST",
    });
    if (!response.ok) throw new Error(`Failed to cancel agent task: ${response.statusText}`);
  },

  /** Drop a task (cancels live turn + removes on-disk metadata). */
  deleteSession: async (sessionId: string): Promise<void> => {
    const response = await fetch(`/api/agent-tasks/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    });
    if (!response.ok) throw new Error(`Failed to delete agent task: ${response.statusText}`);
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
  /** Non-secret env literals (e.g. MOLMCP_SOURCES) for the editor. */
  env?: Record<string, string>;
  headerKeys: string[];
  secretRefs: string[];
  unresolvedSecrets: string[];
  shadowed: boolean;
  valid: boolean;
  invalidReason: string;
  auth: ApiMcpAuthSummary | null;
  /** Parsed MOLMCP_SOURCES when this is a molmcp server. */
  knowledgeSources?: string[];
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

// Every tool belongs to an MCP server. Keep its wire-format source prefix
// centralized so the MCP list can attach tools to their owning server.
export const mcpSource = (server: string): string => `mcp:${server}`;

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

export type ApiModelTier = "cheap" | "default" | "heavy";
export type ApiTierModels = Record<ApiModelTier, string>;

export interface ApiAgentProvider {
  provider: ApiProviderName;
  model: string;
  baseUrl: string;
  apiKeyPreview: string;
  apiKeySet: boolean;
  instructions: string;
  supportedProviders: ApiProviderName[];
  /** Global cheap/default/heavy table — full ``provider:model`` ids; may cross providers. */
  models: ApiTierModels;
  configurations: ApiProviderConfiguration[];
}

export interface ApiProviderConfiguration {
  provider: ApiProviderName;
  /** Legacy per-provider tier map; prefer top-level ``models``. */
  models: ApiTierModels;
  baseUrl: string;
  apiKeyPreview: string;
  apiKeySet: boolean;
}

export interface ProviderUpdateInput {
  provider?: ApiProviderName;
  model?: string;
  models?: ApiTierModels;
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
  if (input.models !== undefined) body.models = input.models;
  if (input.apiKey !== undefined) body.api_key = input.apiKey;
  if (input.baseUrl !== undefined) body.base_url = input.baseUrl;
  if (input.instructions !== undefined) body.instructions = input.instructions;
  return body;
};

export type ApiKnowledgeSources = {
  sources: string[];
  knownPackages: string[];
  unrestricted: boolean;
  serverName: string;
  scope: string;
  configured: boolean;
};

export const agentAdminApi = {
  // Probe endpoint: routed through probeOnce so an unconfigured agent stack
  // (503) is detected once and never re-requested (see agentProbe.ts).
  getProvider: (): Promise<ApiAgentProvider> =>
    probeOnce("agent-provider", async () => {
      const response = await fetch("/api/agent/provider");
      if (response.status === 503) throw new AgentUnavailableError("/api/agent/provider");
      if (!response.ok) throw new Error(`Failed to fetch provider: ${response.statusText}`);
      return response.json();
    }),

  getKnowledgeSources: (): Promise<ApiKnowledgeSources> =>
    probeOnce("agent-knowledge-sources", async () => {
      const response = await fetch("/api/agent/knowledge-sources");
      if (response.status === 503) {
        throw new AgentUnavailableError("/api/agent/knowledge-sources");
      }
      if (!response.ok) {
        throw new Error(`Failed to fetch knowledge sources: ${response.statusText}`);
      }
      return response.json();
    }),

  updateKnowledgeSources: async (sources: string[]): Promise<ApiKnowledgeSources> => {
    const response = await fetch("/api/agent/knowledge-sources", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sources }),
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Failed to update knowledge sources: ${response.statusText} ${detail}`);
    }
    resetAgentProbes();
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
    // A saved provider can turn an unconfigured stack into a live one —
    // drop any cached "unavailable" probe outcomes so the UI re-probes.
    resetAgentProbes();
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

  listMcpServers: (): Promise<ApiMcpServerList> =>
    probeOnce("agent-mcp-servers", async () => {
      const response = await fetch("/api/agent/mcp/servers");
      if (response.status === 503) throw new AgentUnavailableError("/api/agent/mcp/servers");
      if (!response.ok) throw new Error(`Failed to fetch MCP servers: ${response.statusText}`);
      return response.json();
    }),

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

  listTools: (): Promise<ApiAgentTool[]> =>
    probeOnce("agent-tools-list", async () => {
      const response = await fetch("/api/agent/tools");
      if (response.status === 503) throw new AgentUnavailableError("/api/agent/tools");
      if (!response.ok) throw new Error(`Failed to fetch tools: ${response.statusText}`);
      const data = await response.json();
      return data.tools ?? [];
    }).catch((error: unknown) => {
      if (error instanceof AgentUnavailableError) return [];
      throw error;
    }),

  listToolsAndGroups: (): Promise<ApiAgentToolList> =>
    probeOnce("agent-tools-groups", async () => {
      const response = await fetch("/api/agent/tools");
      if (response.status === 503) throw new AgentUnavailableError("/api/agent/tools");
      if (!response.ok) throw new Error(`Failed to fetch tools: ${response.statusText}`);
      const data = await response.json();
      return { tools: data.tools ?? [], mcpGroups: data.mcpGroups ?? [] };
    }).catch((error: unknown) => {
      if (error instanceof AgentUnavailableError) return { tools: [], mcpGroups: [] };
      throw error;
    }),

  listSkills: (): Promise<ApiSkill[]> =>
    probeOnce("agent-skills", async () => {
      const response = await fetch("/api/agent/skills");
      if (response.status === 503) throw new AgentUnavailableError("/api/agent/skills");
      if (!response.ok) throw new Error(`Failed to fetch skills: ${response.statusText}`);
      const data = await response.json();
      return data.skills ?? [];
    }).catch((error: unknown) => {
      if (error instanceof AgentUnavailableError) return [];
      throw error;
    }),

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
    options: { mode?: "chat" | "plan" } = {},
  ): Promise<ApiAgentSession> => {
    const body: Record<string, unknown> = { parameters };
    if (options.mode !== undefined) body.mode = options.mode;
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
  // Probe endpoint: routed through probeOnce so an unconfigured agent stack
  // (503) is detected once and never re-requested (see agentProbe.ts).
  list: (): Promise<ApiCommand[]> =>
    probeOnce("agent-commands", async () => {
      const response = await fetch("/api/agent/commands");
      if (response.status === 503) throw new AgentUnavailableError("/api/agent/commands");
      if (!response.ok) throw new Error(`Failed to fetch commands: ${response.statusText}`);
      const data = await response.json();
      return data.commands ?? [];
    }),

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
  /**
   * System-prompt breakdown for the task inspector.
   *
   * Live surface is `/api/agent-tasks/{id}/system-prompt` (accepts task id or
   * runtime session id). The legacy `/api/agent/sessions/.../system-prompt`
   * path is retired and always 503s.
   */
  getSystemPrompt: async (taskOrSessionId: string): Promise<ApiAgentSystemPrompt> => {
    const response = await fetch(`/api/agent-tasks/${taskOrSessionId}/system-prompt`);
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      const suffix = detail ? ` — ${detail.slice(0, 200)}` : "";
      throw new Error(`Failed to fetch system prompt: ${response.statusText}${suffix}`);
    }
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

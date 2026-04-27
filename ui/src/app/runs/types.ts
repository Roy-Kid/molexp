/**
 * TypeScript mirrors of the backend `WorkspaceRunsResponse` shape.
 *
 * Kept hand-written rather than pulled from the generated OpenAPI client so
 * the runs feature module compiles before `npm run generate:api` is re-run.
 */

import type { ExecutionRowData } from "@/plugins/types";

export interface WorkspaceExecutionRow extends ExecutionRowData {
  backendMetadata: Record<string, string>;
}

export interface WorkspaceRunRow {
  id: string;
  name: string;
  projectId: string;
  projectName: string;
  experimentId: string;
  experimentName: string;
  status: string;
  backend: string | null;
  cluster: string | null;
  scheduler: string | null;
  profile: string | null;
  parameters: Record<string, unknown>;
  createdAt: string;
  finishedAt: string | null;
  executionCount: number;
  latestSchedulerJobId: string | null;
  executions: WorkspaceExecutionRow[];
}

export interface WorkspaceRunsStats {
  total: number;
  running: number;
  pending: number;
  failed: number;
  succeeded: number;
}

export interface WorkspaceRunsResponse {
  runs: WorkspaceRunRow[];
  stats: WorkspaceRunsStats;
  total: number;
  truncated: boolean;
}

export type RunsQuickView = "active" | "failed24h" | "longRunning";

export interface WorkspaceRunsFilters {
  projectId?: string[];
  experimentId?: string[];
  backend?: string[];
  cluster?: string[];
  status?: string[];
  quickView?: RunsQuickView[];
  limit?: number;
}

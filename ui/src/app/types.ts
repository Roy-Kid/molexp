export type LeftPanelView =
  | "workspace"
  | "projects"
  | "runs"
  | "asset"
  | "workflow"
  | "agent"
  | "settings";

export type SemanticObjectType =
  | "project"
  | "experiment"
  | "run"
  | "asset"
  | "workflow"
  | "workspace-file"
  | "agent";

export type BaseObjectType = "project" | "experiment" | "run" | "asset";

export type PanelKind = "editor" | "viewer" | "inspector";

export type FileKind = "yaml" | "json" | "python" | "markdown" | "text" | "image" | "unknown";

export type ContentType = "workflow-graph" | "metadata" | "log" | "text" | "metrics" | "image";

export type JsonValue = string | number | boolean | null | JsonObject | JsonValue[];

export interface JsonObject {
  [key: string]: JsonValue;
}

export type SemanticStatus =
  | "active"
  | "archived"
  | "draft"
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "skipped"
  | "waiting_for_review"
  | "approved"
  | "rejected"
  | "expired";

import type { ExperimentCreateRequest } from "../api/generated/models/ExperimentCreateRequest";
import type { ProjectCreateRequest } from "../api/generated/models/ProjectCreateRequest";
import type { RunCreateRequest } from "../api/generated/models/RunCreateRequest";

export type { ExperimentCreateRequest, ProjectCreateRequest, RunCreateRequest };

import type { AgentTaskResponse } from "../api/generated/models/AgentTaskResponse";
import type { AssetResponse } from "../api/generated/models/AssetResponse";
import type { CacheClearResponse } from "../api/generated/models/CacheClearResponse";
import type { CacheStatsResponse } from "../api/generated/models/CacheStatsResponse";
import type { ExperimentResponse } from "../api/generated/models/ExperimentResponse";
import type { ProjectResponse } from "../api/generated/models/ProjectResponse";
import type { RunResponse } from "../api/generated/models/RunResponse";
import type { RunSummary as ApiRunSummaryModel } from "../api/generated/models/RunSummary";
import type { SessionEventResponse } from "../api/generated/models/SessionEventResponse";
import type { WorkflowSnapshotResponse } from "../api/generated/models/WorkflowSnapshotResponse";

// Re-export as Api*Response for compatibility
export type ApiProjectResponse = ProjectResponse;
export type ApiExperimentResponse = ExperimentResponse;
export type ApiRunResponse = RunResponse;
export type ApiAssetResponse = AssetResponse;
export type ApiWorkflowSnapshot = WorkflowSnapshotResponse;
export type ApiRunSummary = ApiRunSummaryModel;
export type ApiCacheStats = CacheStatsResponse;
export type ApiCacheClear = CacheClearResponse;
// AgentTaskResponse is the user-facing task envelope around one runtime
// session. It already carries taskId/title/updatedAt/sessionId so legacy
// consumers that expected ``ApiAgentSession`` see the same shape.
export type ApiAgentSession = AgentTaskResponse;
export type ApiSessionEvent = SessionEventResponse;

/**
 * Known asset kinds emitted by the unified catalog. The list is open — the
 * backend may add new kinds — but these are the ones the UI renders with
 * dedicated logic.
 */
export type AssetKind =
  | "data"
  | "artifact"
  | "log"
  | "checkpoint"
  | "error_trace"
  | "execution_state"
  | "output";

export interface ProjectSummary {
  id: string;
  name: string;
  status: SemanticStatus;
  summary: string;
  updatedAt: string;
}

export interface ExperimentSummary {
  id: string;
  name: string;
  status: SemanticStatus;
  summary: string;
  workflowFile: string;
  updatedAt: string;
  projectId: string;
  parameterSpace: Record<string, unknown>;
  workflowSource: string | null;
}

export interface ExecutionRecordSummary {
  executionId: string;
  startedAt: string;
  finishedAt: string | null;
  status: string;
  schedulerJobId: string | null;
}

export interface RunSummary {
  id: string;
  name: string;
  status: SemanticStatus;
  summary: string;
  updatedAt: string;
  projectId: string;
  experimentId: string;
  executorInfo: Record<string, string>;
  profile: string | null;
  configHash: string | null;
  parameters: Record<string, unknown>;
  results: Record<string, unknown>;
  workflowSource: string | null;
  workflowSnapshot: WorkflowSnapshotResponse | null;
  startedAt: string | null;
  finishedAt: string | null;
  executionHistory: ExecutionRecordSummary[];
  errorMessage: string | null;
}

export interface AssetSummary {
  id: string;
  name: string;
  kind: AssetKind | string;
  status: SemanticStatus;
  summary: string;
  updatedAt: string;
  sizeBytes: number | null;
  projectId?: string;
}

export interface WorkflowSummary {
  id: string;
  name: string;
  status: SemanticStatus;
  summary: string;
  updatedAt: string;
  projectId: string;
  experimentId: string;
  graph?: WorkflowGraph;
}

export interface AgentSessionSummary {
  id: string;
  sessionId: string;
  goal: string;
  status: SemanticStatus;
  createdAt: string;
  eventCount: number;
}

export interface WorkflowNodeMetadata {
  nodeId: string;
  label: string;
  nodeType: "task" | "input" | "output";
  status: SemanticStatus;
  description: string;
  position: WorkflowNodePosition;
}

export interface WorkflowGraph {
  nodes: WorkflowNodeMetadata[];
  edges: WorkflowGraphEdge[];
}

export interface WorkflowNodePosition {
  x: number;
  y: number;
}

export interface WorkflowGraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
}

export interface WorkspaceTreeNode {
  id: string;
  name: string;
  path: string;
  kind: "file" | "directory";
  children: WorkspaceTreeNode[];
  sizeBytes: number;
  updatedAt: string;
}

export interface ConsoleEntry {
  id: string;
  level: "info" | "warning" | "error";
  message: string;
  timestamp: string;
}

export interface WorkspaceSnapshot {
  projects: ProjectSummary[];
  experiments: ExperimentSummary[];
  runs: RunSummary[];
  assets: AssetSummary[];
  workflows: WorkflowSummary[];
  agentSessions: AgentSessionSummary[];
  workspaceRoot: WorkspaceTreeNode | null;
  consoleEntries: ConsoleEntry[];
}

export type ObjectView = "overview" | "logs" | "metrics" | "scheduler" | "snapshot";

export interface ObjectSelection {
  objectType: BaseObjectType;
  objectId: string;
  objectView?: ObjectView;
}

export interface WorkflowSelection {
  objectType: "workflow";
  workflowId: string;
  objectId: string;
}

export interface WorkspaceFileSelection {
  objectType: "workspace-file";
  filePath: string;
  fileKind: FileKind;
  objectId: string;
}

export interface AgentSelection {
  objectType: "agent";
  objectId: string; // task_id, or "new" for the goal-input state
}

export type Selection =
  | ObjectSelection
  | WorkflowSelection
  | WorkspaceFileSelection
  | AgentSelection;

export type InspectorTarget =
  | {
      kind: "object";
      objectType: SemanticObjectType;
      objectId: string;
    }
  | {
      kind: "workflow-node";
      workflowId: string;
      nodeId: string;
    };

export interface RendererKey {
  objectType: SemanticObjectType;
  fileKind: FileKind;
  contentType: ContentType;
  panelKind: PanelKind;
}

export interface RendererProps {
  selection: Selection;
  snapshot: WorkspaceSnapshot;
  inspectorTarget: InspectorTarget;

  onInspectorTargetChange: (target: InspectorTarget) => void;
  onRefresh: () => void;
}

export interface BreadcrumbItem {
  label: string;
  to?: string;
}

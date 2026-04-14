export type LeftPanelView =
  | "workspace"
  | "projects"
  | "asset"
  | "workflow"
  | "agent";

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
  | "skipped";

import type { ExperimentCreateRequest } from "../api/generated/models/ExperimentCreateRequest";
import type { ProjectCreateRequest } from "../api/generated/models/ProjectCreateRequest";
import type { RunCreateRequest } from "../api/generated/models/RunCreateRequest";

export type { ExperimentCreateRequest, ProjectCreateRequest, RunCreateRequest };

import type { AgentSessionResponse } from "../api/generated/models/AgentSessionResponse";
import type { AssetFileResponse } from "../api/generated/models/AssetFileResponse";
import type { AssetRefResponse } from "../api/generated/models/AssetRefResponse";
import type { AssetRefsResponse } from "../api/generated/models/AssetRefsResponse";
import type { AssetResponse } from "../api/generated/models/AssetResponse";
import type { CacheClearResponse } from "../api/generated/models/CacheClearResponse";
import type { CacheStatsResponse } from "../api/generated/models/CacheStatsResponse";
import type { ContextSnapshotResponse } from "../api/generated/models/ContextSnapshotResponse";
import type { ExperimentResponse } from "../api/generated/models/ExperimentResponse";
import type { ProjectResponse } from "../api/generated/models/ProjectResponse";
import type { RunResponse } from "../api/generated/models/RunResponse";
import type { RunSummary as ApiRunSummaryModel } from "../api/generated/models/RunSummary";
import type { SessionEventResponse } from "../api/generated/models/SessionEventResponse";
import type { TaskSnapshotResponse } from "../api/generated/models/TaskSnapshotResponse";
import type { WorkflowSnapshotResponse } from "../api/generated/models/WorkflowSnapshotResponse";

// Re-export as Api*Response for compatibility
export type ApiProjectResponse = ProjectResponse;
export type ApiExperimentResponse = ExperimentResponse;
export type ApiRunResponse = RunResponse;
export type ApiAssetResponse = AssetResponse;
export type ApiAssetFile = AssetFileResponse;
export type ApiWorkflowSnapshot = WorkflowSnapshotResponse;
export type ApiContextSnapshot = ContextSnapshotResponse;
export type ApiAssetRef = AssetRefResponse;
export type ApiAssetRefs = AssetRefsResponse;
export type ApiRunSummary = ApiRunSummaryModel;
export type ApiTaskSnapshot = TaskSnapshotResponse;
export type ApiCacheStats = CacheStatsResponse;
export type ApiCacheClear = CacheClearResponse;
export type ApiAgentSession = AgentSessionResponse;
export type ApiSessionEvent = SessionEventResponse;

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
}

export interface AssetSummary {
  id: string;
  name: string;
  status: SemanticStatus;
  summary: string;
  updatedAt: string;
  sizeBytes: number;
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
  goalDescription: string;
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

export interface ObjectSelection {
  objectType: BaseObjectType;
  objectId: string;
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
  objectId: string; // session_id, or "new" for the goal-input state
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

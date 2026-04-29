import type React from "react";
import type {
  ContentType,
  FileKind,
  PanelKind,
  RendererKey,
  RendererProps,
  Selection,
  SemanticObjectType,
  WorkspaceSnapshot,
} from "@/app/types";

export type PanelSlot = "center" | "right";

export interface RendererEntry {
  key: RendererKey;
  title: string;
  panelSlot: PanelSlot;
  Component: React.ComponentType<RendererProps>;
}

export interface RenderTarget {
  panelKind: PanelKind;
  contentType: ContentType;
  fileKind: FileKind;
}

export interface RendererResolutionContext {
  key: RendererKey;
  selection: Selection;
  snapshot: WorkspaceSnapshot;
  target: RenderTarget;
}

export interface RendererContribution extends RendererEntry {
  id: string;
  priority?: number;
  matches?: (context: RendererResolutionContext) => boolean;
}

export interface FilePreviewContentProps {
  content: string;
  name: string;
  path: string;
  folderId: string;
}

export interface FilePreviewPlugin {
  id: string;
  name: string;
  extensions: string[];
  priority?: number;
  canHandle?: (props: { name: string; path: string }) => boolean;
  Component: React.ComponentType<FilePreviewContentProps>;
}

export interface EntityTabContribution {
  id: string;
  objectType: SemanticObjectType;
  value: string;
  label: string;
  priority?: number;
  Component: React.ComponentType<RendererProps>;
}

export interface FileMatchContext {
  name: string;
  relPath: string;
  size?: number | null;
  type: string;
}

export interface FileTypeMatcher {
  patterns?: string[];
  matches?: (file: FileMatchContext) => boolean;
}

export interface DiscoveredFile extends FileMatchContext {
  matchedBy: string;
}

export interface FileTypeContribution {
  id: string;
  objectType: SemanticObjectType;
  value: string;
  label: string;
  priority?: number;
  matcher: FileTypeMatcher;
  Component: React.ComponentType<RendererProps & { discoveredFiles: DiscoveredFile[] }>;
}

/**
 * Per-execution data passed to plugin renderers.
 *
 * Mirrors `WorkspaceExecutionRow` returned by `/api/workspace/runs`. The
 * `backend` discriminator decides which plugin contribution(s) light up
 * for this row; `metadata` carries scheduler-specific fields (cluster,
 * scheduler_job_id, queue, …) for table cells and detail panels.
 */
export interface ExecutionRowData {
  executionId: string;
  runId: string;
  status: string;
  startedAt: string;
  finishedAt: string | null;
  durationSeconds: number | null;
  schedulerJobId: string | null;
  backend: string | null;
  metadata: Record<string, string>;
}

export interface ExecutionColumnRenderProps {
  execution: ExecutionRowData;
}

export interface ExecutionColumnContribution {
  id: string;
  /** Backend discriminator (e.g. "molq"). Undefined = always render. */
  backend?: string;
  /** Stable column key used for header + ordering. */
  columnId: string;
  header: string;
  priority?: number;
  /** Optional fixed pixel width hint for the column header cell. */
  width?: number;
  align?: "left" | "right" | "center";
  Cell: React.ComponentType<ExecutionColumnRenderProps>;
}

export interface ExecutionDetailRenderProps {
  execution: ExecutionRowData;
  /** Parent run id for plugins that need run-level context. */
  runId: string;
}

export interface ExecutionDetailContribution {
  id: string;
  backend?: string;
  title: string;
  priority?: number;
  Component: React.ComponentType<ExecutionDetailRenderProps>;
}

export interface PluginManifest {
  id: string;
  title: string;
  description: string;
  uiModule?: string | null;
  capabilities: string[];
  metadata: Record<string, unknown>;
}

export interface UiPluginModule {
  id: string;
  register: () => void | Promise<void>;
}

export const buildRendererRegistryKey = (key: RendererKey): string => {
  return `${key.objectType}::${key.fileKind}::${key.contentType}::${key.panelKind}`;
};

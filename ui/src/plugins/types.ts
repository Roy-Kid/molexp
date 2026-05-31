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
  /**
   * Server-computed signal: the file has a same-stem `.py` preview sidecar
   * (existence-only — no user code was executed to determine it). Lets a
   * contribution light up for datasets that match no extension pattern.
   */
  hasPreviewSidecar?: boolean;
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

/**
 * One entry returned by ``GET /api/plugins`` — a discovered third-party
 * UI bundle's distribution metadata. Carries no UI semantics: `id` is
 * the entry-point name on the Python side; `manifestUrl` and `entryUrl`
 * point into the bundle's mounted directory. Real UI semantics live in
 * the bundle's ``manifest.json`` (see {@link UiBundleManifest}), fetched
 * by the loader once it sees this descriptor.
 */
export interface PluginManifest {
  id: string;
  manifestUrl: string;
  entryUrl: string;
}

/**
 * Schema of ``manifest.json`` shipped at the root of each third-party
 * UI bundle. The browser-side loader fetches it, validates the shape,
 * and checks ``api_version`` against the
 * ``UI_PLUGIN_API_VERSION`` constant frozen into this build.
 */
export interface UiBundleManifest {
  id: string;
  name: string;
  version: string;
  api_version: "1";
  /** Optional override for the entry filename. Defaults to `index.js`. */
  entry?: string;
  capabilities?: string[];
}

export interface UiPluginModule {
  id: string;
  register: () => void | Promise<void>;
}

export const buildRendererRegistryKey = (key: RendererKey): string => {
  return `${key.objectType}::${key.fileKind}::${key.contentType}::${key.panelKind}`;
};

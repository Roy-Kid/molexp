/**
 * Workspace filesystem — the frontend view of ``Workspace.fs``, the disk.
 *
 * Backend model (keep these layers distinct):
 * - ``Workspace.fs`` is the disk (``FileSystem``: local / remote / cached).
 * - A Folder is a location on that disk, not an owner of one.
 * - ``folder.files`` is the Python byte-exit (``FileStore``) at that location.
 *
 * This module is the HTTP twin of the *disk*, not of ``folder.files``.
 * Path identity is {@link WorkspacePath} (pure POSIX). I/O goes through
 * ``/api/workspace/*``, which the server routes through ``workspace.fs``.
 *
 * Design rules:
 * - Never use browser File APIs for workspace content.
 * - Prefer shallow ``listdir`` (maxDepth 1) and expand on demand.
 * - Missing remote index / empty tree is a normal state, not an error.
 */

import type { WorkspaceTreeNode } from "@/app/types";
import { basename, toApiPath, type WorkspacePath } from "@/lib/workspace-path";

// ── Wire shapes (server ``GET /api/workspace/files``) ──────────────────────

/** One node as returned by the files listing endpoint. */
export interface WorkspaceDirentRaw {
  id?: string;
  name: string;
  path: string;
  type?: string;
  children?: WorkspaceDirentRaw[];
  size?: number | null;
  modified?: string | number;
  assetId?: string | null;
  assetKind?: string | null;
  producerRunId?: string | null;
  producerTaskId?: string | null;
  hasPreviewSidecar?: boolean | null;
}

export interface WorkspaceListResultRaw {
  path?: string;
  children?: WorkspaceDirentRaw[];
}

// ── Domain shapes ──────────────────────────────────────────────────────────

export type DirentKind = "file" | "directory";

/** One entry in a directory listing (mapped from the server node). */
export interface WorkspaceDirent {
  name: string;
  path: WorkspacePath;
  kind: DirentKind;
  sizeBytes: number | null;
  mtime: number | null;
  /** Nested children when the listing used maxDepth > 1. */
  children: WorkspaceDirent[];
  /** True when this directory's children were fetched in this response. */
  childrenLoaded: boolean;
  assetId?: string;
  hasPreviewSidecar?: boolean;
}

export interface ListdirOptions {
  /** Recursion depth (0 = only the node itself; 1 = immediate children). Default 1. */
  maxDepth?: number;
  /** Request catalog lineage chips on file nodes. */
  includeCatalog?: boolean;
}

export interface WorkspaceFs {
  /**
   * Optional workspace root the FS was opened against (absolute remote path
   * or local absolute path). Used to relativise API queries.
   */
  readonly root: WorkspacePath | null;

  /** List *path* (workspace-relative or absolute under root). */
  listdir(path: WorkspacePath, options?: ListdirOptions): Promise<WorkspaceDirent[]>;

  /** Read a UTF-8 text file. */
  readText(path: WorkspacePath): Promise<string>;

  /** Read a binary file as Blob. */
  readBlob(path: WorkspacePath): Promise<Blob>;
}

// ── Mapping helpers ────────────────────────────────────────────────────────

const mtimeFromRaw = (modified: string | number | undefined): number | null => {
  if (typeof modified === "number") return modified;
  if (typeof modified === "string" && modified) {
    const ms = Date.parse(modified);
    return Number.isFinite(ms) ? ms / 1000 : null;
  }
  return null;
};

/**
 * Map a raw server node. *depthBudget* is how many more levels of children
 * the response is known to contain for this node (0 ⇒ expand must re-fetch).
 */
export const mapDirent = (raw: WorkspaceDirentRaw, depthBudget: number): WorkspaceDirent => {
  const kind: DirentKind = raw.type === "file" ? "file" : "directory";
  const kids = raw.children ?? [];
  return {
    name: raw.name,
    path: raw.path,
    kind,
    sizeBytes: raw.size ?? null,
    mtime: mtimeFromRaw(raw.modified),
    children: kids.map((c) => mapDirent(c, Math.max(0, depthBudget - 1))),
    // Files have no children; dirs are "loaded" only while the response still
    // carries nested listings (budget > 0). Budget 0 means server stopped
    // recursing — expand will listdir again.
    childrenLoaded: kind === "file" || depthBudget > 0,
    assetId: raw.assetId ?? undefined,
    hasPreviewSidecar: raw.hasPreviewSidecar ?? undefined,
  };
};

/** Convert a dirent forest into the snapshot {@link WorkspaceTreeNode} shape. */
export const direntsToTreeNodes = (dirents: WorkspaceDirent[]): WorkspaceTreeNode[] =>
  dirents.map(direntToTreeNode);

export const direntToTreeNode = (d: WorkspaceDirent): WorkspaceTreeNode => ({
  id: d.path,
  name: d.name,
  path: d.path,
  kind: d.kind,
  children: d.children.map(direntToTreeNode),
  sizeBytes: d.sizeBytes ?? 0,
  updatedAt: d.mtime != null ? new Date(d.mtime * 1000).toISOString() : "",
  assetId: d.assetId,
  hasPreviewSidecar: d.hasPreviewSidecar,
  childrenLoaded: d.childrenLoaded,
});

/**
 * Build the snapshot root node from a root listing.
 * *rootPath* is the workspace root label (absolute remote path or local path).
 */
export const treeRootFromListing = (
  rootPath: WorkspacePath,
  children: WorkspaceDirent[],
): WorkspaceTreeNode => ({
  id: "workspace-root",
  name: basename(rootPath) || rootPath || "workspace",
  path: rootPath || "/",
  kind: "directory",
  children: direntsToTreeNodes(children),
  sizeBytes: 0,
  updatedAt: "",
  childrenLoaded: true,
});

/** Merge *children* into the directory at *dirPath* inside *root* (immutable). */
export const mergeTreeChildren = (
  root: WorkspaceTreeNode,
  dirPath: WorkspacePath,
  children: WorkspaceTreeNode[],
): WorkspaceTreeNode => {
  if (root.path === dirPath || (root.id === "workspace-root" && dirPath === root.path)) {
    return { ...root, children, childrenLoaded: true };
  }
  return {
    ...root,
    children: root.children.map((child) => {
      if (child.path === dirPath) {
        return { ...child, children, childrenLoaded: true };
      }
      if (child.kind === "directory" && child.children.length > 0) {
        return mergeTreeChildren(child, dirPath, children);
      }
      // Descend if dirPath is under this child
      if (child.kind === "directory" && dirPath.startsWith(`${child.path}/`)) {
        return mergeTreeChildren(child, dirPath, children);
      }
      return child;
    }),
  };
};

// ── HTTP implementation ────────────────────────────────────────────────────

export interface HttpWorkspaceFsOptions {
  /** Workspace root absolute path (from ``GET /api/workspace/info``). */
  root?: WorkspacePath | null;
  /** Override fetch (tests). */
  fetchImpl?: typeof fetch;
  /** API prefix; default ``/api``. */
  apiBase?: string;
}

/**
 * WorkspaceFs over the molexp server HTTP surface.
 *
 * ``listdir`` → ``GET /api/workspace/files``
 * ``readText`` → ``GET /api/workspace/file``
 * ``readBlob`` → ``GET /api/workspace/file/blob``
 */
export class HttpWorkspaceFs implements WorkspaceFs {
  readonly root: WorkspacePath | null;
  private readonly fetchImpl: typeof fetch;
  private readonly apiBase: string;

  constructor(options: HttpWorkspaceFsOptions = {}) {
    this.root = options.root ?? null;
    this.fetchImpl = options.fetchImpl ?? fetch.bind(globalThis);
    this.apiBase = options.apiBase ?? "/api";
  }

  /** Return a copy bound to a known workspace root (for relative API paths). */
  withRoot(root: WorkspacePath | null): HttpWorkspaceFs {
    return new HttpWorkspaceFs({
      root,
      fetchImpl: this.fetchImpl,
      apiBase: this.apiBase,
    });
  }

  async listdir(path: WorkspacePath, options: ListdirOptions = {}): Promise<WorkspaceDirent[]> {
    const maxDepth = options.maxDepth ?? 1;
    const params = new URLSearchParams();
    params.set("path", toApiPath(path, this.root ?? undefined));
    params.set("max_depth", String(maxDepth));
    if (options.includeCatalog) {
      params.set("include", "catalog");
    }
    const response = await this.fetchImpl(`${this.apiBase}/workspace/files?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`listdir failed: ${response.status} ${response.statusText}`);
    }
    const body = (await response.json()) as WorkspaceListResultRaw;
    const rawChildren = body.children ?? [];
    // Children sit at depth 1 of this request → remaining budget is maxDepth-1.
    return rawChildren.map((c) => mapDirent(c, Math.max(0, maxDepth - 1)));
  }

  async readText(path: WorkspacePath): Promise<string> {
    const q = new URLSearchParams({ path: toApiPath(path, this.root ?? undefined) });
    const response = await this.fetchImpl(`${this.apiBase}/workspace/file?${q.toString()}`);
    if (!response.ok) {
      throw new Error(`readText failed: ${response.status} ${response.statusText}`);
    }
    const body = (await response.json()) as { content?: string };
    return body.content ?? "";
  }

  async readBlob(path: WorkspacePath): Promise<Blob> {
    const q = new URLSearchParams({ path: toApiPath(path, this.root ?? undefined) });
    const response = await this.fetchImpl(`${this.apiBase}/workspace/file/blob?${q.toString()}`);
    if (!response.ok) {
      throw new Error(`readBlob failed: ${response.status} ${response.statusText}`);
    }
    return response.blob();
  }
}

/** Process-wide default FS (root filled in once workspace/info is known). */
let defaultFs: HttpWorkspaceFs = new HttpWorkspaceFs();

export const getWorkspaceFs = (): WorkspaceFs => defaultFs;

/** Rebind the default FS to a workspace root (call after open/info). */
export const setWorkspaceFsRoot = (root: WorkspacePath | null): void => {
  defaultFs = defaultFs.withRoot(root);
};

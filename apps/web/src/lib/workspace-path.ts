/**
 * Pure POSIX path arithmetic for workspace paths — the frontend twin of
 * ``molexp.path.Path`` (PurePosixPath). No I/O, no network, no host binding.
 *
 * Pair with {@link WorkspaceFs} for ops. Never call browser/Node fs APIs here.
 *
 * Convention:
 * - Absolute paths start with ``/`` (remote roots, e.g. ``/home/me/ws``).
 * - Workspace-relative paths have no leading slash (``projects/water``).
 * - ``""`` / ``"."`` / ``"/"`` are the three legal "root-ish" forms; helpers
 *   normalise them for joining and comparison.
 */

export type WorkspacePath = string;

/** True when *path* is absolute POSIX (starts with ``/``). */
export const isAbsolute = (path: WorkspacePath): boolean => path.startsWith("/");

/** Basename of a POSIX path (``a/b/c`` → ``c``; ``/`` → ``""``). */
export const basename = (path: WorkspacePath): string => {
  const normalized = path.replace(/\/+$/, "");
  if (!normalized || normalized === ".") return "";
  const i = normalized.lastIndexOf("/");
  return i < 0 ? normalized : normalized.slice(i + 1);
};

/** Parent directory (``a/b/c`` → ``a/b``; ``/a`` → ``/``; relative top → ``""``). */
export const dirname = (path: WorkspacePath): string => {
  const normalized = path.replace(/\/+$/, "") || (isAbsolute(path) ? "/" : "");
  if (!normalized || normalized === ".") return isAbsolute(path) ? "/" : "";
  const i = normalized.lastIndexOf("/");
  if (i < 0) return "";
  if (i === 0) return "/";
  return normalized.slice(0, i);
};

/**
 * Join path segments with POSIX rules. Absolute segments reset the base
 * (same as ``pathlib.PurePosixPath.joinpath``).
 */
export const join = (...parts: WorkspacePath[]): WorkspacePath => {
  if (parts.length === 0) return "";
  let out = "";
  for (const raw of parts) {
    if (raw === undefined || raw === null) continue;
    const part = String(raw);
    if (!part || part === ".") continue;
    if (isAbsolute(part)) {
      out = part;
      continue;
    }
    if (!out || out === ".") {
      out = part;
      continue;
    }
    out = `${out.replace(/\/+$/, "")}/${part.replace(/^\/+/, "")}`;
  }
  return out;
};

/** Collapse ``.`` / ``..`` segments (does not resolve symlinks — pure string). */
export const normalize = (path: WorkspacePath): WorkspacePath => {
  if (!path || path === ".") return path === "." ? "" : path;
  const abs = isAbsolute(path);
  const parts = path.split("/").filter((p) => p && p !== ".");
  const stack: string[] = [];
  for (const p of parts) {
    if (p === "..") {
      if (stack.length > 0 && stack[stack.length - 1] !== "..") {
        stack.pop();
      } else if (!abs) {
        stack.push("..");
      }
    } else {
      stack.push(p);
    }
  }
  if (abs) return `/${stack.join("/")}`;
  return stack.join("/");
};

/**
 * Path of *path* relative to *root*. Returns *path* unchanged when it is not
 * under *root* (no ``../`` climbing — containment is fail-closed).
 */
export const relativeTo = (path: WorkspacePath, root: WorkspacePath): WorkspacePath => {
  const p = normalize(path);
  const r = normalize(root);
  if (!r || r === "." || r === "/") {
    if (r === "/" && isAbsolute(p)) return p.replace(/^\//, "");
    return p;
  }
  if (p === r) return "";
  const prefix = r.endsWith("/") ? r : `${r}/`;
  if (p.startsWith(prefix)) return p.slice(prefix.length);
  return p;
};

/** True when *path* is *root* or a descendant of *root*. */
export const isUnder = (path: WorkspacePath, root: WorkspacePath): boolean => {
  const p = normalize(path);
  const r = normalize(root);
  if (!r || r === ".") return true;
  if (p === r) return true;
  const prefix = r === "/" ? "/" : `${r.replace(/\/+$/, "")}/`;
  return p.startsWith(prefix);
};

/**
 * Relativise an absolute (or mixed) path for API query params.
 *
 * The workspace files API accepts workspace-relative paths, ``""`` for root,
 * and absolute paths that already live under the remote root. Prefer relative
 * when *workspaceRoot* is known.
 */
export const toApiPath = (path: WorkspacePath, workspaceRoot?: WorkspacePath): string => {
  if (!path || path === "." || path === "/") return "";
  if (workspaceRoot && isUnder(path, workspaceRoot)) {
    return relativeTo(path, workspaceRoot);
  }
  // Already relative
  if (!isAbsolute(path)) return path.replace(/^\.\//, "");
  return path;
};

// ── Clipboard / display paths ──────────────────────────────────────────────

/** Active served workspace slice used when qualifying remote paths. */
export interface PathDisplayWorkspace {
  /** Serve label: local abs path, or ``Host:/abs`` / ``user@host:/abs``. */
  label: string;
  isRemote: boolean;
  /** Absolute local root when local; null for remote. */
  path: string | null;
}

export interface PathDisplayContext {
  /** Workspace root absolute path (``GET /api/workspace/info`` / fs.root). */
  root?: string | null;
  /** Active served workspace (from ``GET /api/workspaces``). */
  workspace?: PathDisplayWorkspace | null;
}

/**
 * Parse ``host:/absolute`` (or ``user@host:/absolute``) from a serve label.
 * Returns null when the label is not host-qualified.
 */
export const parseHostQualifiedLabel = (label: string): { host: string; root: string } | null => {
  // SCP / serve form: everything before the first ``:/``-style host:abs split
  // — absolute path after the colon (``Arrhenius:/home/...``, ``u@h:/data``).
  const m = label.match(/^([^:]+):(\/.*)$/);
  if (!m) return null;
  return { host: m[1], root: m[2].replace(/\/+$/, "") || "/" };
};

/**
 * Compact label for chrome (ContextBar, multi-ws tree headers).
 *
 * Full serve labels like ``Arrhenius:/home/jicli594/work/mace-nve`` are the
 * identity for copy/API, but repeating them in every chrome surface is noise.
 * Display ``Host · basename`` (remote) or the directory basename (local);
 * keep the full string in ``title`` / tooltip / copy-path.
 */
export const shortWorkspaceLabel = (label: string): string => {
  const trimmed = label.trim();
  if (!trimmed) return trimmed;
  const hostQualified = parseHostQualifiedLabel(trimmed);
  if (hostQualified) {
    const leaf = basename(hostQualified.root) || hostQualified.root;
    return `${hostQualified.host} · ${leaf}`;
  }
  // Local absolute or bare path — show the leaf only when it has a parent.
  const leaf = basename(trimmed.replace(/\/+$/, ""));
  return leaf || trimmed;
};

/**
 * Absolute path for clipboard copy.
 *
 * - **Local** → POSIX absolute (``/Users/…/projects/…``).
 * - **Remote** → host-qualified (``Arrhenius:/home/jicli594/work/mace-nve/…``),
 *   same form as ``molexp serve`` workspace labels.
 *
 * *path* may be workspace-relative or already absolute under the workspace root.
 */
export const formatQualifiedPath = (path: WorkspacePath, ctx: PathDisplayContext = {}): string => {
  const workspace = ctx.workspace ?? null;
  const knownRoot = (workspace?.path || ctx.root || "").replace(/\/+$/, "");

  // Normalize to workspace-relative when we can.
  let rel = path;
  if (path === "." || path === "/" || path === "") {
    rel = "";
  } else if (knownRoot && isAbsolute(path) && isUnder(path, knownRoot)) {
    rel = relativeTo(path, knownRoot);
  } else if (isAbsolute(path) && workspace?.isRemote) {
    // Absolute remote path we could not relativise — still host-qualify.
    const parsed = workspace.label ? parseHostQualifiedLabel(workspace.label) : null;
    if (parsed) return `${parsed.host}:${path}`;
    return path;
  } else if (isAbsolute(path) && !workspace?.isRemote) {
    return path;
  } else {
    rel = path.replace(/^\.\//, "").replace(/^\/+/, "");
  }
  rel = rel.replace(/\/+$/, "");

  if (workspace?.isRemote) {
    const parsed = parseHostQualifiedLabel(workspace.label);
    if (parsed) {
      const full = rel ? join(parsed.root, rel) : parsed.root;
      return `${parsed.host}:${full}`;
    }
    // Label not host-qualified — fall through to bare absolute.
    const full = rel && knownRoot ? join(knownRoot, rel) : knownRoot || rel;
    return full;
  }

  if (!knownRoot) return rel || ".";
  return rel ? join(knownRoot, rel) : knownRoot;
};

/** On-disk relative path of a run dir (layout law: ``runs/run-<id>``). */
export const runWorkspaceRelativePath = (run: {
  projectId: string;
  experimentId: string;
  id: string;
}): string => `projects/${run.projectId}/experiments/${run.experimentId}/runs/run-${run.id}`;

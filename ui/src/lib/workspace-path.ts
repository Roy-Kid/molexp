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

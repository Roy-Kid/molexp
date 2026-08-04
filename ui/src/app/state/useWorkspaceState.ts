import { useCallback, useEffect, useRef, useState } from "react";
import {
  agentApi,
  buildEmptySnapshot,
  mapAgentSessions,
  mapAssets,
  mapExperiments,
  mapProjects,
  mapRuns,
  mapWorkflows,
  workspaceApi,
} from "@/app/state/api";
import { pulseSync } from "@/app/state/syncPulse";
import type {
  LeftPanelView,
  ProjectSummary,
  WorkspaceSnapshot,
  WorkspaceTreeNode,
} from "@/app/types";
import {
  direntsToTreeNodes,
  getWorkspaceFs,
  mergeTreeChildren,
  setWorkspaceFsRoot,
  treeRootFromListing,
} from "@/lib/workspace-fs";
import type { WorkspacePath } from "@/lib/workspace-path";

export type WorkspaceStatus = "idle" | "loading" | "ready" | "error";

export interface WorkspaceState {
  snapshot: WorkspaceSnapshot;
  status: WorkspaceStatus;
  error: Error | null;
  refresh: () => void;
  /** Lazy-expand a workspace file-tree directory via WorkspaceFs.listdir. */
  expandDirectory: (dirPath: WorkspacePath) => Promise<void>;
  /** Lazy-load experiments under a project (nav expand / open). */
  expandProject: (projectId: string) => Promise<void>;
  /** Lazy-load runs under an experiment (nav expand / open). */
  expandExperiment: (projectId: string, experimentId: string) => Promise<void>;
  /** True when this project's experiments have been fetched. */
  isProjectExpanded: (projectId: string) => boolean;
  /** True when this experiment's runs have been fetched. */
  isExperimentExpanded: (projectId: string, experimentId: string) => boolean;
}

// Slice = an independently fetchable chunk of the snapshot.
// Entity hierarchy (experiments / runs) is **not** a slice — it loads on expand.
type SnapshotSlice = "workspaces" | "workspaceTree" | "projectsList" | "assets" | "agentSessions";

// Bootstrap: shallow only. No fan-out over experiments×runs (that was 20+ HTTP
// calls on every poll and freezes remote workspaces).
const BOOTSTRAP_SLICES: readonly SnapshotSlice[] = [
  "workspaces",
  "projectsList",
  "agentSessions",
  "workspaceTree",
];

// Manual full refresh still avoids the old experimentsTree dump — expand
// caches stay warm; user re-opens folders if they want a re-fetch.
const REFRESH_SLICES: readonly SnapshotSlice[] = BOOTSTRAP_SLICES;

// Polling: only cheap / view-local slices. Never re-walk the whole entity tree.
// projects view: no interval — list is static until user expands or hits refresh.
// workspace: optional soft tree refresh is still heavy on remote → off.
// assets: load once when entering the view (see effect), not every 3s.
const VIEW_POLL_SLICES: Record<LeftPanelView, readonly SnapshotSlice[]> = {
  workspace: [],
  projects: [],
  workflow: [],
  asset: [],
  runs: [],
  agent: [],
  knowledge: [],
  settings: [],
};

const WORKSPACE_TREE_BOOTSTRAP_DEPTH = 2;

const expKey = (projectId: string, experimentId: string): string => `${projectId}/${experimentId}`;

const fetchWorkspaceTree = async (): Promise<WorkspaceSnapshot["workspaceRoot"]> => {
  try {
    try {
      const info = await workspaceApi.getWorkspaceInfo();
      if (info.root) {
        setWorkspaceFsRoot(info.root);
      }
    } catch {
      // optional
    }
    const fs = getWorkspaceFs();
    const children = await fs.listdir("", {
      maxDepth: WORKSPACE_TREE_BOOTSTRAP_DEPTH,
      includeCatalog: true,
    });
    return treeRootFromListing(fs.root ?? "/", children);
  } catch (err) {
    console.warn("Workspace tree unavailable:", err);
    return null;
  }
};

const findTreeNode = (root: WorkspaceTreeNode, path: string): WorkspaceTreeNode | null => {
  if (root.path === path) return root;
  for (const child of root.children) {
    if (child.path === path) return child;
    if (child.kind === "directory" && path.startsWith(`${child.path}/`)) {
      const hit = findTreeNode(child, path);
      if (hit) return hit;
    }
  }
  return null;
};

const fetchWorkspaces = async (): Promise<WorkspaceSnapshot["workspaces"]> => {
  try {
    return await workspaceApi.getServedWorkspaces();
  } catch (err) {
    console.warn("Served workspaces unavailable:", err);
    return [];
  }
};

const fetchProjectsList = async (
  workspaces: WorkspaceSnapshot["workspaces"],
): Promise<ProjectSummary[]> => {
  if (workspaces.length <= 1) {
    return mapProjects(await workspaceApi.getProjects());
  }
  const perWorkspace = await Promise.all(
    workspaces.map(async (ws) => {
      if (ws.unreachable) return [];
      try {
        return mapProjects(await workspaceApi.getProjectsForWorkspace(ws.key), ws.key);
      } catch (err) {
        console.warn(`Projects unavailable for workspace ${ws.key}:`, err);
        return [];
      }
    }),
  );
  return perWorkspace.flat();
};

const activeWorkspaceProjects = (snapshot: WorkspaceSnapshot): ProjectSummary[] => {
  if (snapshot.workspaces.length <= 1) return snapshot.projects;
  const activeKey = snapshot.workspaces.find((ws) => ws.active)?.key;
  return snapshot.projects.filter((project) => project.workspaceKey === activeKey);
};

const fetchAllAssets = async (projects: ProjectSummary[]): Promise<WorkspaceSnapshot["assets"]> => {
  const projectAssets = await Promise.all(
    projects.map(async (project) => {
      try {
        return mapAssets(await workspaceApi.getProjectAssets(project.id), project.id);
      } catch (err) {
        console.warn(`Failed to fetch assets for project ${project.id}:`, err);
        return [];
      }
    }),
  );
  try {
    const allAssets = [...mapAssets(await workspaceApi.getAssets()), ...projectAssets.flat()];
    return Array.from(new Map(allAssets.map((item) => [item.id, item])).values());
  } catch (err) {
    console.warn("Workspace assets unavailable:", err);
    return projectAssets.flat();
  }
};

const fetchAgentSessionsList = async (): Promise<WorkspaceSnapshot["agentSessions"]> => {
  try {
    return mapAgentSessions(await agentApi.listSessions());
  } catch (err) {
    console.warn("Agent sessions unavailable:", err);
    return [];
  }
};

const applySlicePatch = async (
  current: WorkspaceSnapshot,
  slice: SnapshotSlice,
): Promise<Partial<WorkspaceSnapshot>> => {
  switch (slice) {
    case "workspaces":
      return { workspaces: await fetchWorkspaces() };
    case "workspaceTree":
      return { workspaceRoot: await fetchWorkspaceTree() };
    case "projectsList":
      return { projects: await fetchProjectsList(current.workspaces) };
    case "assets":
      return { assets: await fetchAllAssets(activeWorkspaceProjects(current)) };
    case "agentSessions":
      return { agentSessions: await fetchAgentSessionsList() };
  }
};

const fetchSlices = async (
  current: WorkspaceSnapshot,
  slices: readonly SnapshotSlice[],
  onProgress?: (next: WorkspaceSnapshot) => void,
): Promise<WorkspaceSnapshot> => {
  let next = current;
  for (const slice of slices) {
    try {
      const patch = await applySlicePatch(next, slice);
      next = { ...next, ...patch };
      onProgress?.(next);
    } catch (err) {
      console.warn(`Snapshot slice "${slice}" failed:`, err);
    }
  }
  return next;
};

export const useWorkspaceState = (activeView?: LeftPanelView): WorkspaceState => {
  const [snapshot, setSnapshot] = useState<WorkspaceSnapshot>(buildEmptySnapshot());
  const [status, setStatus] = useState<WorkspaceStatus>("idle");
  const [error, setError] = useState<Error | null>(null);
  const inflightRef = useRef(false);
  const snapshotRef = useRef(snapshot);
  snapshotRef.current = snapshot;

  // On-demand load tracking (entity tree). Cleared only on full refresh.
  const projectsLoadedRef = useRef(new Set<string>());
  const experimentsLoadedRef = useRef(new Set<string>());
  const assetsLoadedForViewRef = useRef(false);
  // Force re-render when expand sets flip without snapshot change shape.
  const [, bump] = useState(0);

  const runFetch = useCallback((slices: readonly SnapshotSlice[], silent: boolean): void => {
    if (slices.length === 0) return;
    if (inflightRef.current) return;
    inflightRef.current = true;
    if (!silent) setStatus("loading");

    fetchSlices(snapshotRef.current, slices, (partial) => {
      snapshotRef.current = partial;
      setSnapshot(partial);
      if (!silent) setStatus("ready");
    })
      .then((nextSnapshot: WorkspaceSnapshot) => {
        snapshotRef.current = nextSnapshot;
        setSnapshot(nextSnapshot);
        setStatus("ready");
        setError(null);
      })
      .catch((err: Error) => {
        setError(err);
        setStatus((prev) => (prev === "ready" ? "ready" : "error"));
      })
      .finally(() => {
        inflightRef.current = false;
        pulseSync();
      });
  }, []);

  const refresh = useCallback((): void => {
    projectsLoadedRef.current.clear();
    experimentsLoadedRef.current.clear();
    assetsLoadedForViewRef.current = false;
    // Drop cached experiments/runs so counts fall back to server-side totals.
    setSnapshot((prev) => {
      const next = {
        ...prev,
        experiments: [],
        runs: [],
        workflows: [],
        assets: [],
      };
      snapshotRef.current = next;
      return next;
    });
    runFetch(REFRESH_SLICES, false);
  }, [runFetch]);

  const expandDirectory = useCallback(async (dirPath: WorkspacePath): Promise<void> => {
    const root = snapshotRef.current.workspaceRoot;
    if (!root) return;
    const node = findTreeNode(root, dirPath);
    if (node?.kind !== "directory") return;
    if (node.childrenLoaded) return;

    try {
      const fs = getWorkspaceFs();
      const children = await fs.listdir(dirPath, { maxDepth: 1, includeCatalog: true });
      const childNodes = direntsToTreeNodes(children);
      const nextRoot = mergeTreeChildren(root, dirPath, childNodes);
      const next = { ...snapshotRef.current, workspaceRoot: nextRoot };
      snapshotRef.current = next;
      setSnapshot(next);
    } catch (err) {
      console.warn(`expandDirectory(${dirPath}) failed:`, err);
    }
  }, []);

  const expandProject = useCallback(async (projectId: string): Promise<void> => {
    if (projectsLoadedRef.current.has(projectId)) return;
    projectsLoadedRef.current.add(projectId);
    try {
      const raw = await workspaceApi.getExperiments(projectId);
      const mapped = mapExperiments(projectId, raw);
      // Workflows for just these experiments (IR if present on the wire).
      const workflows = mapWorkflows(mapped, raw);
      setSnapshot((prev) => {
        const otherExps = prev.experiments.filter((e) => e.projectId !== projectId);
        const otherWfs = prev.workflows.filter((w) => w.projectId !== projectId);
        const next: WorkspaceSnapshot = {
          ...prev,
          experiments: [...otherExps, ...mapped],
          workflows: [...otherWfs, ...workflows],
        };
        snapshotRef.current = next;
        return next;
      });
      bump((n) => n + 1);
    } catch (err) {
      projectsLoadedRef.current.delete(projectId);
      console.warn(`expandProject(${projectId}) failed:`, err);
    }
  }, []);

  const expandExperiment = useCallback(
    async (projectId: string, experimentId: string): Promise<void> => {
      const key = expKey(projectId, experimentId);
      if (experimentsLoadedRef.current.has(key)) return;
      experimentsLoadedRef.current.add(key);
      try {
        const raw = await workspaceApi.getRuns(projectId, experimentId);
        const mapped = mapRuns(projectId, experimentId, raw);
        setSnapshot((prev) => {
          const other = prev.runs.filter(
            (r) => !(r.projectId === projectId && r.experimentId === experimentId),
          );
          const next: WorkspaceSnapshot = {
            ...prev,
            runs: [...other, ...mapped],
          };
          snapshotRef.current = next;
          return next;
        });
        bump((n) => n + 1);
      } catch (err) {
        experimentsLoadedRef.current.delete(key);
        console.warn(`expandExperiment(${key}) failed:`, err);
      }
    },
    [],
  );

  const isProjectExpanded = useCallback(
    (projectId: string): boolean => projectsLoadedRef.current.has(projectId),
    // `bump` re-renders consumers; the callback reads the live ref.
    [],
  );

  const isExperimentExpanded = useCallback(
    (projectId: string, experimentId: string): boolean =>
      experimentsLoadedRef.current.has(expKey(projectId, experimentId)),
    [],
  );

  // Bootstrap once — shallow only.
  useEffect(() => {
    runFetch(BOOTSTRAP_SLICES, false);
  }, [runFetch]);

  // Assets: load once when entering the asset view (not on every poll).
  useEffect(() => {
    if (activeView !== "asset") return;
    if (assetsLoadedForViewRef.current) return;
    assetsLoadedForViewRef.current = true;
    runFetch(["assets"], true);
  }, [activeView, runFetch]);

  // Optional view-scoped polling (currently all empty — on-demand only).
  useEffect(() => {
    if (activeView === undefined) return;
    const slices = VIEW_POLL_SLICES[activeView];
    if (slices.length === 0) return;
    // Reserved for future light polls; intentionally no default interval.
  }, [activeView]);

  return {
    snapshot,
    status,
    error,
    refresh,
    expandDirectory,
    expandProject,
    expandExperiment,
    isProjectExpanded,
    isExperimentExpanded,
  };
};

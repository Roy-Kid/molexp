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
  /**
   * Bumps on every manual refresh after entity caches are cleared. TreeView
   * re-requests lazy loads for folders that stayed open (empty childCount alone
   * does not change when the row was already "Loading…").
   */
  dataEpoch: number;
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

// Manual full refresh re-fetches bootstrap slices, then force-reloads any
// folders that were already expanded (stale-while-revalidate — UI keeps
// showing the previous children until the new payload lands).
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
  activity: [],
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
  // Always stamp workspaceKey so the multi-workspace nav filter
  // (`project.workspaceKey === ws.key`) never drops a single-ws project.
  if (workspaces.length === 0) {
    return mapProjects(await workspaceApi.getProjects());
  }
  if (workspaces.length === 1) {
    const ws = workspaces[0];
    if (ws.unreachable) return [];
    try {
      // Prefer flat /api/projects (active workspace) — same data, one RTT.
      return mapProjects(await workspaceApi.getProjects(), ws.key);
    } catch (err) {
      console.warn(`Projects unavailable for workspace ${ws.key}:`, err);
      return [];
    }
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

  // On-demand load tracking (entity tree). Cleared on full refresh, then
  // re-populated by re-expanding whatever was open.
  const projectsLoadedRef = useRef(new Set<string>());
  const experimentsLoadedRef = useRef(new Set<string>());
  // In-flight guards so a stuck "Loading…" re-trigger (TreeView still open
  // after refresh) does not fan out duplicate remote requests.
  const projectsLoadingRef = useRef(new Set<string>());
  const experimentsLoadingRef = useRef(new Set<string>());
  const assetsLoadedForViewRef = useRef(false);
  // Force re-render when expand sets flip without snapshot change shape.
  const [, bump] = useState(0);
  const [dataEpoch, setDataEpoch] = useState(0);

  const runFetch = useCallback(
    (slices: readonly SnapshotSlice[], silent: boolean): Promise<void> => {
      if (slices.length === 0) return Promise.resolve();
      if (inflightRef.current) return Promise.resolve();
      inflightRef.current = true;
      // Stay "loading" until every slice finishes — early "ready" after the first
      // slice killed the status-strip busy state (progress bar + heartbeat) while
      // workspaceTree / projectsList were still in flight.
      if (!silent) setStatus("loading");

      return fetchSlices(snapshotRef.current, slices, (partial) => {
        snapshotRef.current = partial;
        setSnapshot(partial);
        // Partial paint only — do not flip status here.
        if (!silent) pulseSync();
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
    },
    [],
  );

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

  const expandProject = useCallback(
    async (projectId: string, options?: { force?: boolean }): Promise<void> => {
      // force=true: re-fetch while keeping current children on screen (SWR).
      if (!options?.force && projectsLoadedRef.current.has(projectId)) return;
      if (projectsLoadingRef.current.has(projectId)) return;
      projectsLoadingRef.current.add(projectId);
      try {
        const raw = await workspaceApi.getExperiments(projectId);
        const mapped = mapExperiments(projectId, raw);
        // Workflows for just these experiments (IR if present on the wire).
        const workflows = mapWorkflows(mapped, raw);
        projectsLoadedRef.current.add(projectId);
        setSnapshot((prev) => {
          const otherExps = prev.experiments.filter((e) => e.projectId !== projectId);
          const otherWfs = prev.workflows.filter((w) => w.projectId !== projectId);
          const next: WorkspaceSnapshot = {
            ...prev,
            experiments: [...otherExps, ...mapped],
            workflows: [...otherWfs, ...workflows],
            // Keep project chip in sync once we know the true exp count.
            projects: prev.projects.map((p) =>
              p.id === projectId ? { ...p, experimentCount: mapped.length } : p,
            ),
          };
          snapshotRef.current = next;
          return next;
        });
        bump((n) => n + 1);
      } catch (err) {
        console.warn(`expandProject(${projectId}) failed:`, err);
        // Fail closed as "loaded" so the row leaves "Loading…" rather than
        // spinning forever; a later refresh re-attempts via force.
        projectsLoadedRef.current.add(projectId);
        bump((n) => n + 1);
      } finally {
        projectsLoadingRef.current.delete(projectId);
      }
    },
    [],
  );

  const expandExperiment = useCallback(
    async (
      projectId: string,
      experimentId: string,
      options?: { force?: boolean },
    ): Promise<void> => {
      const key = expKey(projectId, experimentId);
      if (!options?.force && experimentsLoadedRef.current.has(key)) return;
      if (experimentsLoadingRef.current.has(key)) return;
      experimentsLoadingRef.current.add(key);
      try {
        const raw = await workspaceApi.getRuns(projectId, experimentId);
        const mapped = mapRuns(projectId, experimentId, raw);
        // Mark loaded only after success — so emptyChildLabel stays "Loading…"
        // rather than "No runs" while the remote fetch is in flight (first open).
        experimentsLoadedRef.current.add(key);
        setSnapshot((prev) => {
          const other = prev.runs.filter(
            (r) => !(r.projectId === projectId && r.experimentId === experimentId),
          );
          // Stamp runCount so the right-side chip flips from "…" to "N run".
          const experiments = prev.experiments.map((e) =>
            e.projectId === projectId && e.id === experimentId
              ? { ...e, runCount: mapped.length }
              : e,
          );
          const next: WorkspaceSnapshot = {
            ...prev,
            experiments,
            runs: [...other, ...mapped],
          };
          snapshotRef.current = next;
          return next;
        });
        bump((n) => n + 1);
      } catch (err) {
        console.warn(`expandExperiment(${key}) failed:`, err);
        experimentsLoadedRef.current.add(key);
        bump((n) => n + 1);
      } finally {
        experimentsLoadingRef.current.delete(key);
      }
    },
    [],
  );

  const refresh = useCallback((): void => {
    // Stale-while-revalidate for the entity tree:
    // keep experiments / runs / workflows on screen, re-fetch open folders in
    // the background, then swap rows in place. Wiping first caused a visible
    // "Loading…" flash even when data came back quickly.
    const reopenProjects = [...projectsLoadedRef.current];
    const reopenExperiments = [...experimentsLoadedRef.current];
    const hadAssets = assetsLoadedForViewRef.current;
    assetsLoadedForViewRef.current = false;
    // Clear in-flight so a stuck load does not block the forced re-fetch.
    projectsLoadingRef.current.clear();
    experimentsLoadingRef.current.clear();
    // Help any open row still on empty/"Loading…" (never successfully loaded).
    setDataEpoch((n) => n + 1);

    void runFetch(REFRESH_SLICES, false).then(() => {
      const reloads: Promise<void>[] = [
        ...reopenProjects.map((projectId) => expandProject(projectId, { force: true })),
        ...reopenExperiments.map((key) => {
          const slash = key.indexOf("/");
          if (slash <= 0) return Promise.resolve();
          const projectId = key.slice(0, slash);
          const experimentId = key.slice(slash + 1);
          if (!projectId || !experimentId) return Promise.resolve();
          return expandExperiment(projectId, experimentId, { force: true });
        }),
      ];
      if (hadAssets || activeView === "asset") {
        assetsLoadedForViewRef.current = true;
        reloads.push(runFetch(["assets"], true));
      }
      void Promise.all(reloads);
    });
  }, [runFetch, expandProject, expandExperiment, activeView]);

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
    dataEpoch,
    expandDirectory,
    expandProject,
    expandExperiment,
    isProjectExpanded,
    isExperimentExpanded,
  };
};

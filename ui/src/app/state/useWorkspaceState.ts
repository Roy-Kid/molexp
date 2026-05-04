import { useCallback, useEffect, useRef, useState } from "react";
import {
  agentApi,
  buildEmptySnapshot,
  mapAgentSessions,
  mapAssets,
  mapExperiments,
  mapProjects,
  mapReviews,
  mapRuns,
  mapWorkflows,
  mapWorkspaceTree,
  reviewsApi,
  workspaceApi,
} from "@/app/state/api";
import type {
  ExperimentSummary,
  LeftPanelView,
  ProjectSummary,
  RunSummary,
  WorkflowSummary,
  WorkspaceSnapshot,
} from "@/app/types";

export type WorkspaceStatus = "idle" | "loading" | "ready" | "error";

export interface WorkspaceState {
  snapshot: WorkspaceSnapshot;
  status: WorkspaceStatus;
  error: Error | null;
  refresh: () => void;
}

const SNAPSHOT_POLL_INTERVAL_MS = 3000;

// Slice = an independently fetchable chunk of the snapshot. Polling refreshes
// only the slices the active view actually reads, so switching to a quiet
// view (workflow/agent) stops the unrelated fan-out fetches.
type SnapshotSlice =
  | "workspaceTree"
  | "projectsList"
  | "experimentsTree" // experiments + runs + workflows (workflows derive from experiments)
  | "assets"
  | "agentSessions"
  | "reviews";

const ALL_SLICES: readonly SnapshotSlice[] = [
  "workspaceTree",
  "projectsList",
  "experimentsTree",
  "assets",
  "agentSessions",
  "reviews",
];

// Per-view polling profile. Empty array = no polling for that view.
//   workflow: definitions are static (parsed from experiment files); manual
//             refresh covers authoring edits.
//   runs:    useWorkspaceRuns owns its own poller.
//   agent:   AgentViewer drives session state via SSE + targeted ticks.
// Slices listed before others run first; downstream slices see the freshly
// fetched value (e.g. assets reads the just-refreshed projects list).
const VIEW_POLL_SLICES: Record<LeftPanelView, readonly SnapshotSlice[]> = {
  workspace: ["workspaceTree"],
  projects: ["projectsList", "experimentsTree"],
  workflow: [],
  asset: ["projectsList", "assets"],
  runs: [],
  agent: [],
  review: ["reviews"],
  settings: [],
};

const fetchWorkspaceTree = async (): Promise<WorkspaceSnapshot["workspaceRoot"]> => {
  try {
    const tree = await workspaceApi.getWorkspaceTree({
      path: "/",
      maxDepth: 8,
      includeCatalog: true,
    });
    return mapWorkspaceTree("/", tree);
  } catch (err) {
    console.warn("Workspace tree unavailable:", err);
    return null;
  }
};

const fetchProjectsList = async (): Promise<ProjectSummary[]> => {
  return mapProjects(await workspaceApi.getProjects());
};

interface ExperimentsTreeData {
  experiments: ExperimentSummary[];
  runs: RunSummary[];
  workflows: WorkflowSummary[];
}

const fetchExperimentsTree = async (projects: ProjectSummary[]): Promise<ExperimentsTreeData> => {
  const experimentsByProject = await Promise.all(
    projects.map(async (project) => ({
      projectId: project.id,
      experiments: await workspaceApi.getExperiments(project.id),
    })),
  );

  const experimentSummaries = experimentsByProject.flatMap((item) =>
    mapExperiments(item.projectId, item.experiments),
  );

  const runsByExperiment = await Promise.all(
    experimentsByProject.flatMap((item) =>
      item.experiments.map(async (experiment) => ({
        projectId: item.projectId,
        experimentId: experiment.id,
        runs: await workspaceApi.getRuns(item.projectId, experiment.id),
      })),
    ),
  );

  const runSummaries = runsByExperiment.flatMap((item) =>
    mapRuns(item.projectId, item.experimentId, item.runs),
  );

  const rawExperiments = experimentsByProject.flatMap((item) => item.experiments);
  const workflowSummaries = mapWorkflows(experimentSummaries, rawExperiments);

  return {
    experiments: experimentSummaries,
    runs: runSummaries,
    workflows: workflowSummaries,
  };
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
  const allAssets = [...mapAssets(await workspaceApi.getAssets()), ...projectAssets.flat()];
  return Array.from(new Map(allAssets.map((item) => [item.id, item])).values());
};

const fetchAgentSessionsList = async (): Promise<WorkspaceSnapshot["agentSessions"]> => {
  try {
    return mapAgentSessions(await agentApi.listSessions());
  } catch (err) {
    console.warn("Agent sessions unavailable:", err);
    return [];
  }
};

const fetchReviewsList = async (): Promise<WorkspaceSnapshot["reviews"]> => {
  try {
    return mapReviews(await reviewsApi.list());
  } catch (err) {
    console.warn("Reviews unavailable:", err);
    return [];
  }
};

const applySlicePatch = async (
  current: WorkspaceSnapshot,
  slice: SnapshotSlice,
): Promise<Partial<WorkspaceSnapshot>> => {
  switch (slice) {
    case "workspaceTree":
      return { workspaceRoot: await fetchWorkspaceTree() };
    case "projectsList":
      return { projects: await fetchProjectsList() };
    case "experimentsTree":
      return await fetchExperimentsTree(current.projects);
    case "assets":
      return { assets: await fetchAllAssets(current.projects) };
    case "agentSessions":
      return { agentSessions: await fetchAgentSessionsList() };
    case "reviews":
      return { reviews: await fetchReviewsList() };
  }
};

// Apply slice patches in order, threading the in-progress snapshot so a slice
// that depends on a freshly fetched predecessor (e.g. assets after projects)
// sees the new value within the same fetch cycle.
const fetchSlices = async (
  current: WorkspaceSnapshot,
  slices: readonly SnapshotSlice[],
): Promise<WorkspaceSnapshot> => {
  let next = current;
  for (const slice of slices) {
    const patch = await applySlicePatch(next, slice);
    next = { ...next, ...patch };
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

  const runFetch = useCallback((slices: readonly SnapshotSlice[], silent: boolean): void => {
    if (slices.length === 0) return;
    // Coalesce overlapping fetches — a slow snapshot fetch must never queue
    // additional requests behind it.
    if (inflightRef.current) return;
    inflightRef.current = true;
    if (!silent) setStatus("loading");

    fetchSlices(snapshotRef.current, slices)
      .then((nextSnapshot: WorkspaceSnapshot) => {
        setSnapshot(nextSnapshot);
        setStatus("ready");
        setError(null);
      })
      .catch((err: Error) => {
        setError(err);
        setStatus("error");
      })
      .finally(() => {
        inflightRef.current = false;
      });
  }, []);

  const refresh = useCallback((): void => {
    runFetch(ALL_SLICES, false);
  }, [runFetch]);

  // Bootstrap once: every view depends on the snapshot for navigation/inspector
  // even if it doesn't poll, so we hydrate the full thing on mount.
  useEffect(() => {
    runFetch(ALL_SLICES, false);
  }, [runFetch]);

  // View-scoped polling: refresh only what the active view reads. Switching
  // views tears down the prior interval and starts the new one.
  useEffect(() => {
    if (activeView === undefined) return;
    const slices = VIEW_POLL_SLICES[activeView];
    if (slices.length === 0) return;
    const id = setInterval(() => runFetch(slices, true), SNAPSHOT_POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [activeView, runFetch]);

  return { snapshot, status, error, refresh };
};

import { useCallback, useEffect, useState } from "react";
import type { WorkspaceSnapshot } from "@/app/types";
import {
  buildEmptySnapshot,
  emptyConsoleEntries,
  mapAssets,
  mapExperiments,
  mapProjects,
  mapRuns,
  mapWorkflows,
  mapWorkspaceTree,
  workspaceApi,
} from "@/app/state/api";

export type WorkspaceStatus = "idle" | "loading" | "ready" | "error";

export interface WorkspaceState {
  snapshot: WorkspaceSnapshot;
  status: WorkspaceStatus;
  error: Error | null;
  refresh: () => void;
}

const buildSnapshot = async (): Promise<WorkspaceSnapshot> => {
  let explorerRoot: WorkspaceSnapshot["explorerRoot"] = null;

  try {
    const workspaceTree = await workspaceApi.getWorkspaceTree("/");
    explorerRoot = mapWorkspaceTree("/", workspaceTree);
  } catch (err) {
    console.warn("Workspace explorer tree unavailable:", err);
  }

  const projectsResponse = await workspaceApi.getProjects();
  const projectSummaries = mapProjects(projectsResponse);

  const experimentsByProject = await Promise.all(
    projectsResponse.map(async (project) => {
      const experiments = await workspaceApi.getExperiments(project.id);
      return { projectId: project.id, experiments };
    }),
  );

  const experimentSummaries = experimentsByProject.flatMap((item) =>
    mapExperiments(item.projectId, item.experiments),
  );

  const runsByExperiment = await Promise.all(
    experimentsByProject.flatMap((item) =>
      item.experiments.map(async (experiment) => {
        const runs = await workspaceApi.getRuns(item.projectId, experiment.id);
        return { projectId: item.projectId, experimentId: experiment.id, runs };
      }),
    ),
  );

  const runSummaries = runsByExperiment.flatMap((item) =>
    mapRuns(item.projectId, item.experimentId, item.runs),
  );


  const projectAssets = await Promise.all(
    projectsResponse.map(async (project) => {
      try {
        const assets = await workspaceApi.getProjectAssets(project.id);
        return mapAssets(assets, project.id);
      } catch (err) {
        console.warn(`Failed to fetch assets for project ${project.id}:`, err);
        return [];
      }
    }),
  );

  const assetSummaries = [
    ...mapAssets(await workspaceApi.getAssets()),
    ...projectAssets.flat(),
  ];
  const rawExperiments = experimentsByProject.flatMap((item) => item.experiments);
  const workflowSummaries = mapWorkflows(experimentSummaries, rawExperiments);

  return {
    projects: projectSummaries,
    experiments: experimentSummaries,
    runs: runSummaries,
    assets: assetSummaries,
    workflows: workflowSummaries,
    explorerRoot,
    consoleEntries: emptyConsoleEntries(),
  };
};

export const useWorkspaceState = (): WorkspaceState => {
  const [snapshot, setSnapshot] = useState<WorkspaceSnapshot>(buildEmptySnapshot());
  const [status, setStatus] = useState<WorkspaceStatus>("idle");
  const [error, setError] = useState<Error | null>(null);

  const refresh = useCallback((): void => {
    setStatus("loading");

    buildSnapshot()
      .then((nextSnapshot: WorkspaceSnapshot) => {
        setSnapshot(nextSnapshot);
        setStatus("ready");
        setError(null);
      })
      .catch((err: Error) => {
        setError(err);
        setStatus("error");
      });
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { snapshot, status, error, refresh };
};

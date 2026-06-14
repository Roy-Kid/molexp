import { STATUS_GROUPS } from "@/app/runs/statusGroups";
import type {
  AssetSummary,
  ExperimentSummary,
  RunSummary,
  WorkflowSummary,
  WorkspaceSnapshot,
} from "@/app/types";
import type { TaskGraphJson } from "@/components/workflow/task-graph-ir";
import { countRunStatuses, type RunStatusCounts } from "./dashboardData";

export interface WorkflowRollup {
  exists: boolean;
  taskCount: number;
  linkCount: number;
  parallelGroupCount: number;
}

export interface ExperimentRollup {
  experiment: ExperimentSummary;
  runs: RunSummary[];
  counts: RunStatusCounts;
  workflow: WorkflowSummary | undefined;
  workflowSummary: WorkflowRollup;
}

export interface AttentionItem {
  experiment: ExperimentSummary;
  reason: "failed" | "running" | "missing-workflow" | "empty";
  count: number;
}

export interface ProjectWorkbenchData {
  experiments: ExperimentRollup[];
  runs: RunSummary[];
  counts: RunStatusCounts;
  attention: AttentionItem[];
  recentRuns: RunSummary[];
  assetCount: number;
}

export interface ParameterAxisSummary {
  key: string;
  values: string[];
  count: number;
}

export interface RunGroupSummary {
  key: string;
  label: string;
  runs: RunSummary[];
  counts: RunStatusCounts;
}

export interface ExperimentWorkbenchData {
  counts: RunStatusCounts;
  parameterAxes: ParameterAxisSummary[];
  runGroups: RunGroupSummary[];
  workflowSummary: WorkflowRollup;
}

const latestRunTime = (run: RunSummary): number =>
  Date.parse(run.finishedAt ?? run.startedAt ?? run.updatedAt ?? "") || 0;

const stringifyAxisValue = (value: unknown): string => {
  if (Array.isArray(value)) return value.map(stringifyAxisValue).join(", ");
  if (value === null || value === undefined) return "-";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

export const summarizeWorkflowGraph = (
  workflow: Pick<WorkflowSummary, "graph"> | { graph?: TaskGraphJson } | undefined,
): WorkflowRollup => {
  const graph = workflow?.graph;
  const links = graph?.links ?? [];
  return {
    exists: Boolean(graph),
    taskCount: graph?.task_configs.length ?? 0,
    linkCount: links.length,
    parallelGroupCount: links.filter((link) => link.kind === "parallel").length,
  };
};

export const buildProjectWorkbenchData = (
  projectId: string,
  snapshot: WorkspaceSnapshot,
  projectAssets: Pick<AssetSummary, "id">[] = snapshot.assets.filter(
    (asset) => asset.projectId === projectId,
  ),
): ProjectWorkbenchData => {
  const experiments = snapshot.experiments.filter(
    (experiment) => experiment.projectId === projectId,
  );
  const runs = snapshot.runs.filter((run) => run.projectId === projectId);
  const rollups = experiments.map((experiment) => {
    const experimentRuns = runs.filter((run) => run.experimentId === experiment.id);
    const workflow = snapshot.workflows.find((item) => item.experimentId === experiment.id);
    return {
      experiment,
      runs: experimentRuns,
      counts: countRunStatuses(experimentRuns),
      workflow,
      workflowSummary: summarizeWorkflowGraph(workflow),
    } satisfies ExperimentRollup;
  });

  const attention: AttentionItem[] = [];
  for (const item of rollups) {
    if (item.counts.failed > 0) {
      attention.push({ experiment: item.experiment, reason: "failed", count: item.counts.failed });
    }
    if (item.counts.running > 0) {
      attention.push({
        experiment: item.experiment,
        reason: "running",
        count: item.counts.running,
      });
    }
    if (!item.workflowSummary.exists) {
      attention.push({ experiment: item.experiment, reason: "missing-workflow", count: 1 });
    }
    if (item.counts.total === 0) {
      attention.push({ experiment: item.experiment, reason: "empty", count: 0 });
    }
  }

  const anomalous = runs.filter((run) => run.status === "failed" || run.status === "running");
  const recentRuns = [...new Map([...anomalous, ...runs].map((run) => [run.id, run])).values()]
    .sort((a, b) => latestRunTime(b) - latestRunTime(a))
    .slice(0, 8);

  return {
    experiments: rollups,
    runs,
    counts: countRunStatuses(runs),
    attention,
    recentRuns,
    assetCount: projectAssets.length,
  };
};

export const buildParameterAxes = (
  experiment: Pick<ExperimentSummary, "parameterSpace">,
  runs: Pick<RunSummary, "parameters">[],
): ParameterAxisSummary[] => {
  const valuesByKey = new Map<string, Set<string>>();
  const ensure = (key: string): Set<string> => {
    const existing = valuesByKey.get(key);
    if (existing) return existing;
    const next = new Set<string>();
    valuesByKey.set(key, next);
    return next;
  };

  for (const [key, value] of Object.entries(experiment.parameterSpace ?? {})) {
    const bucket = ensure(key);
    if (Array.isArray(value)) {
      for (const item of value) bucket.add(stringifyAxisValue(item));
    } else {
      bucket.add(stringifyAxisValue(value));
    }
  }

  for (const run of runs) {
    for (const [key, value] of Object.entries(run.parameters ?? {})) {
      ensure(key).add(stringifyAxisValue(value));
    }
  }

  return [...valuesByKey.entries()].map(([key, values]) => ({
    key,
    values: [...values].filter(Boolean),
    count: values.size,
  }));
};

const chooseGroupingAxis = (axes: ParameterAxisSummary[]): string | null => {
  const usable = axes.find((axis) => axis.count > 1 && axis.count <= 12);
  return usable?.key ?? null;
};

export const buildRunGroups = (
  runs: RunSummary[],
  axes: ParameterAxisSummary[],
): RunGroupSummary[] => {
  const axis = chooseGroupingAxis(axes);
  const groups = new Map<string, RunSummary[]>();

  for (const run of runs) {
    const value = axis ? stringifyAxisValue(run.parameters?.[axis]) : run.status;
    const label = axis
      ? `${axis}: ${value}`
      : (STATUS_GROUPS.find((g) => g.aliases.includes(run.status))?.label ?? run.status);
    const list = groups.get(label) ?? [];
    list.push(run);
    groups.set(label, list);
  }

  return [...groups.entries()]
    .map(([label, groupedRuns]) => ({
      key: label,
      label,
      runs: groupedRuns,
      counts: countRunStatuses(groupedRuns),
    }))
    .sort((a, b) => b.runs.length - a.runs.length || a.label.localeCompare(b.label));
};

export const buildExperimentWorkbenchData = (
  experiment: ExperimentSummary,
  runs: RunSummary[],
  workflow: Pick<WorkflowSummary, "graph"> | undefined,
): ExperimentWorkbenchData => {
  const parameterAxes = buildParameterAxes(experiment, runs);
  return {
    counts: countRunStatuses(runs),
    parameterAxes,
    runGroups: buildRunGroups(runs, parameterAxes),
    workflowSummary: summarizeWorkflowGraph(workflow),
  };
};

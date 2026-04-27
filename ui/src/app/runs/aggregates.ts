import { groupForStatus, type StatusGroupId } from "./statusGroups";
import type {
  RunsQuickView,
  WorkspaceExecutionRow,
  WorkspaceRunRow,
  WorkspaceRunsFilters,
  WorkspaceRunsStats,
} from "./types";

const HOUR_MS = 60 * 60 * 1000;

const isInGroup = (status: string, groupId: StatusGroupId): boolean =>
  groupForStatus(status) === groupId;

const ACTIVE_GROUPS: ReadonlySet<StatusGroupId> = new Set(["running", "pending"]);

const safeDate = (value: string | null | undefined): Date | null => {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

const includesValue = (filter: string[] | undefined, value: string | null): boolean => {
  if (!filter || filter.length === 0) return true;
  if (!value) return false;
  return filter.includes(value);
};

const matchesQuickView = (run: WorkspaceRunRow, view: RunsQuickView, now: number): boolean => {
  const group = groupForStatus(run.status);
  switch (view) {
    case "active":
      return group !== null && ACTIVE_GROUPS.has(group);
    case "failed24h": {
      if (group !== "failed") return false;
      const finished = safeDate(run.finishedAt);
      if (!finished) return false;
      return now - finished.getTime() <= 24 * HOUR_MS;
    }
    case "longRunning": {
      if (group !== "running") return false;
      const earliestStart = run.executions.reduce<number | null>((min, exec) => {
        const start = safeDate(exec.startedAt);
        if (!start) return min;
        return min === null ? start.getTime() : Math.min(min, start.getTime());
      }, null);
      if (earliestStart === null) return false;
      return now - earliestStart >= HOUR_MS;
    }
    default:
      return true;
  }
};

interface FilterPredicates {
  status: (run: WorkspaceRunRow) => boolean;
  projectId: (run: WorkspaceRunRow) => boolean;
  experimentId: (run: WorkspaceRunRow) => boolean;
  backend: (run: WorkspaceRunRow) => boolean;
  cluster: (run: WorkspaceRunRow) => boolean;
  quickView: (run: WorkspaceRunRow) => boolean;
}

const buildPredicates = (filters: WorkspaceRunsFilters, now: number): FilterPredicates => ({
  status: (run) => includesValue(filters.status, run.status),
  projectId: (run) => includesValue(filters.projectId, run.projectId),
  experimentId: (run) => includesValue(filters.experimentId, run.experimentId),
  backend: (run) => includesValue(filters.backend, run.backend),
  cluster: (run) => includesValue(filters.cluster, run.cluster),
  quickView: (run) => {
    const views = filters.quickView;
    if (!views || views.length === 0) return true;
    return views.some((view) => matchesQuickView(run, view, now));
  },
});

export const applyFilters = (
  runs: WorkspaceRunRow[],
  filters: WorkspaceRunsFilters,
  now: number = Date.now(),
): WorkspaceRunRow[] => {
  const predicates = buildPredicates(filters, now);
  return runs.filter(
    (run) =>
      predicates.status(run) &&
      predicates.projectId(run) &&
      predicates.experimentId(run) &&
      predicates.backend(run) &&
      predicates.cluster(run) &&
      predicates.quickView(run),
  );
};

export interface FacetCount {
  value: string;
  label: string;
  count: number;
}

export interface FacetSnapshot {
  status: FacetCount[];
  backend: FacetCount[];
  cluster: FacetCount[];
  projectId: FacetCount[];
  experimentId: FacetCount[];
  quickView: Record<RunsQuickView, number>;
}

const tally = (
  runs: WorkspaceRunRow[],
  pick: (run: WorkspaceRunRow) => Array<{ value: string; label: string } | null>,
): FacetCount[] => {
  const map = new Map<string, FacetCount>();
  for (const run of runs) {
    for (const entry of pick(run)) {
      if (!entry) continue;
      const existing = map.get(entry.value);
      if (existing) existing.count += 1;
      else map.set(entry.value, { value: entry.value, label: entry.label, count: 1 });
    }
  }
  return Array.from(map.values()).sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count;
    return a.label.localeCompare(b.label);
  });
};

const omit = <K extends keyof FilterPredicates>(
  predicates: FilterPredicates,
  excluded: K,
): ((run: WorkspaceRunRow) => boolean) => {
  return (run) => {
    for (const key of Object.keys(predicates) as Array<keyof FilterPredicates>) {
      if (key === excluded) continue;
      if (!predicates[key](run)) return false;
    }
    return true;
  };
};

export const computeFacetCounts = (
  allRuns: WorkspaceRunRow[],
  filters: WorkspaceRunsFilters,
  now: number = Date.now(),
): FacetSnapshot => {
  const predicates = buildPredicates(filters, now);

  const filterFor = <K extends keyof FilterPredicates>(excluded: K): WorkspaceRunRow[] =>
    allRuns.filter(omit(predicates, excluded));

  const statusRuns = filterFor("status");
  const backendRuns = filterFor("backend");
  const clusterRuns = filterFor("cluster");
  const projectRuns = filterFor("projectId");
  const experimentRuns = filterFor("experimentId");
  const quickRuns = filterFor("quickView");

  return {
    status: tally(statusRuns, (run) => [{ value: run.status, label: run.status }]),
    backend: tally(backendRuns, (run) =>
      run.backend ? [{ value: run.backend, label: run.backend }] : [],
    ),
    cluster: tally(clusterRuns, (run) =>
      run.cluster ? [{ value: run.cluster, label: run.cluster }] : [],
    ),
    projectId: tally(projectRuns, (run) => [{ value: run.projectId, label: run.projectName }]),
    experimentId: tally(experimentRuns, (run) => [
      { value: run.experimentId, label: run.experimentName },
    ]),
    quickView: {
      active: quickRuns.filter((run) => matchesQuickView(run, "active", now)).length,
      failed24h: quickRuns.filter((run) => matchesQuickView(run, "failed24h", now)).length,
      longRunning: quickRuns.filter((run) => matchesQuickView(run, "longRunning", now)).length,
    },
  };
};

export const computeKpiStats = (runs: WorkspaceRunRow[]): WorkspaceRunsStats => {
  const stats: WorkspaceRunsStats = {
    total: runs.length,
    running: 0,
    pending: 0,
    failed: 0,
    succeeded: 0,
  };
  for (const run of runs) {
    const group = groupForStatus(run.status);
    if (group === "running") stats.running += 1;
    else if (group === "pending") stats.pending += 1;
    else if (group === "failed") stats.failed += 1;
    else if (group === "succeeded") stats.succeeded += 1;
  }
  return stats;
};

const latestExecution = (run: WorkspaceRunRow): WorkspaceExecutionRow | null => {
  if (run.executions.length === 0) return null;
  let latest: WorkspaceExecutionRow | null = null;
  let latestTs = -Infinity;
  for (const exec of run.executions) {
    const start = safeDate(exec.startedAt);
    if (!start) continue;
    if (start.getTime() > latestTs) {
      latestTs = start.getTime();
      latest = exec;
    }
  }
  return latest ?? run.executions[run.executions.length - 1];
};

export const computeAvgWaitSeconds = (
  runs: WorkspaceRunRow[],
  windowHours: number = 24,
  now: number = Date.now(),
): number | null => {
  const cutoff = now - windowHours * HOUR_MS;
  let total = 0;
  let count = 0;
  for (const run of runs) {
    const exec = latestExecution(run);
    if (!exec) continue;
    const start = safeDate(exec.startedAt);
    const submitted = safeDate(run.createdAt);
    if (!start || !submitted) continue;
    if (start.getTime() < cutoff) continue;
    const waitMs = start.getTime() - submitted.getTime();
    if (waitMs < 0) continue;
    total += waitMs;
    count += 1;
  }
  if (count === 0) return null;
  return total / count / 1000;
};

export interface BackendDistributionEntry {
  backend: string;
  cluster: string | null;
  count: number;
}

export const computeBackendDistribution = (
  runs: WorkspaceRunRow[],
): BackendDistributionEntry[] => {
  const map = new Map<string, BackendDistributionEntry>();
  for (const run of runs) {
    if (!run.backend) continue;
    const key = `${run.backend}::${run.cluster ?? ""}`;
    const existing = map.get(key);
    if (existing) existing.count += 1;
    else map.set(key, { backend: run.backend, cluster: run.cluster, count: 1 });
  }
  return Array.from(map.values()).sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count;
    return a.backend.localeCompare(b.backend);
  });
};

export interface FailingExperimentEntry {
  experimentId: string;
  experimentName: string;
  projectId: string;
  projectName: string;
  failedCount: number;
  totalCount: number;
}

export const computeTopFailingExperiments = (
  runs: WorkspaceRunRow[],
  topN: number = 5,
): FailingExperimentEntry[] => {
  const map = new Map<string, FailingExperimentEntry>();
  for (const run of runs) {
    const existing = map.get(run.experimentId);
    const isFailed = isInGroup(run.status, "failed");
    if (existing) {
      existing.totalCount += 1;
      if (isFailed) existing.failedCount += 1;
    } else {
      map.set(run.experimentId, {
        experimentId: run.experimentId,
        experimentName: run.experimentName,
        projectId: run.projectId,
        projectName: run.projectName,
        failedCount: isFailed ? 1 : 0,
        totalCount: 1,
      });
    }
  }
  return Array.from(map.values())
    .filter((entry) => entry.failedCount > 0)
    .sort((a, b) => {
      if (b.failedCount !== a.failedCount) return b.failedCount - a.failedCount;
      return a.experimentName.localeCompare(b.experimentName);
    })
    .slice(0, topN);
};

export interface ActivityBucket {
  hour: Date;
  started: number;
  succeeded: number;
  failed: number;
  cancelled: number;
  finished: number;
}

const outcomeFor = (status: string): "succeeded" | "failed" | "cancelled" | null => {
  const group = groupForStatus(status);
  if (group === "succeeded" || group === "failed" || group === "cancelled") return group;
  return null;
};

export const computeActivityBuckets = (
  runs: WorkspaceRunRow[],
  hours: number = 24,
  now: number = Date.now(),
): ActivityBucket[] => {
  const bucketStart = (timestamp: number): number => {
    const date = new Date(timestamp);
    date.setMinutes(0, 0, 0);
    return date.getTime();
  };

  const currentBucket = bucketStart(now);
  const earliestBucket = currentBucket - (hours - 1) * HOUR_MS;
  const buckets = new Map<number, ActivityBucket>();
  for (let t = earliestBucket; t <= currentBucket; t += HOUR_MS) {
    buckets.set(t, {
      hour: new Date(t),
      started: 0,
      succeeded: 0,
      failed: 0,
      cancelled: 0,
      finished: 0,
    });
  }

  for (const run of runs) {
    for (const exec of run.executions) {
      const start = safeDate(exec.startedAt);
      if (start) {
        const bucket = bucketStart(start.getTime());
        const entry = buckets.get(bucket);
        if (entry) entry.started += 1;
      }
      const end = safeDate(exec.finishedAt);
      if (end) {
        const bucket = bucketStart(end.getTime());
        const entry = buckets.get(bucket);
        if (entry) {
          entry.finished += 1;
          const outcome = outcomeFor(exec.status);
          if (outcome) entry[outcome] += 1;
        }
      }
    }
  }

  return Array.from(buckets.values());
};

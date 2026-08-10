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
  target: (run: WorkspaceRunRow) => boolean;
  quickView: (run: WorkspaceRunRow) => boolean;
}

const buildPredicates = (filters: WorkspaceRunsFilters, now: number): FilterPredicates => ({
  status: (run) => includesValue(filters.status, run.status),
  projectId: (run) => includesValue(filters.projectId, run.projectId),
  experimentId: (run) => includesValue(filters.experimentId, run.experimentId),
  backend: (run) => includesValue(filters.backend, run.backend),
  cluster: (run) => includesValue(filters.cluster, run.cluster),
  target: (run) => includesValue(filters.target, run.target),
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
      predicates.target(run) &&
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
  target: FacetCount[];
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
  const targetRuns = filterFor("target");
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
    target: tally(targetRuns, (run) =>
      run.target ? [{ value: run.target, label: run.target }] : [],
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

export const computeBackendDistribution = (runs: WorkspaceRunRow[]): BackendDistributionEntry[] => {
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

export interface KpiSparkline {
  /** Sparkline buckets ordered chronologically (oldest → newest). */
  series: number[];
  /** Count for the most recent bucket. */
  current: number;
  /** Count for the bucket immediately before the most recent one. */
  previous: number;
  /** current − previous. */
  delta: number;
}

export interface KpiSparklines {
  running: KpiSparkline;
  pending: KpiSparkline;
  failed: KpiSparkline;
  succeeded: KpiSparkline;
  /** "submitted" series — count of runs created in each bucket (any status). */
  submitted: KpiSparkline;
}

const emptyKpiSparkline = (buckets: number): KpiSparkline => ({
  series: new Array(buckets).fill(0),
  current: 0,
  previous: 0,
  delta: 0,
});

const finalizeKpiSparkline = (series: number[]): KpiSparkline => {
  const n = series.length;
  const current = n >= 1 ? series[n - 1] : 0;
  const previous = n >= 2 ? series[n - 2] : 0;
  return { series, current, previous, delta: current - previous };
};

/**
 * Bucket the supplied runs into `buckets` evenly-spaced bins covering the
 * trailing `hours` window (default 24 / 24 ≈ one bucket per hour). Produces
 * one series per status group plus a `submitted` series counting newly-created
 * runs. Used by the KPI strip sparklines — derived purely from the existing
 * run list response, no extra backend calls required.
 */
export const computeKpiSparklines = (
  runs: WorkspaceRunRow[],
  hours: number = 24,
  buckets: number = 24,
  now: number = Date.now(),
): KpiSparklines => {
  if (buckets <= 0 || hours <= 0) {
    return {
      running: emptyKpiSparkline(0),
      pending: emptyKpiSparkline(0),
      failed: emptyKpiSparkline(0),
      succeeded: emptyKpiSparkline(0),
      submitted: emptyKpiSparkline(0),
    };
  }

  const totalMs = hours * HOUR_MS;
  const bucketMs = totalMs / buckets;
  const windowStart = now - totalMs;

  const running = new Array<number>(buckets).fill(0);
  const pending = new Array<number>(buckets).fill(0);
  const failed = new Array<number>(buckets).fill(0);
  const succeeded = new Array<number>(buckets).fill(0);
  const submitted = new Array<number>(buckets).fill(0);

  const indexFor = (ts: number): number | null => {
    if (ts < windowStart || ts > now) return null;
    const raw = Math.floor((ts - windowStart) / bucketMs);
    if (raw < 0) return null;
    return Math.min(raw, buckets - 1);
  };

  for (const run of runs) {
    const created = safeDate(run.createdAt);
    if (created) {
      const i = indexFor(created.getTime());
      if (i !== null) submitted[i] += 1;
    }

    const group = groupForStatus(run.status);
    const finished = safeDate(run.finishedAt);

    if (group === "running" || group === "pending") {
      const earliestStart = run.executions
        .map((exec) => safeDate(exec.startedAt))
        .filter((d): d is Date => d !== null)
        .map((d) => d.getTime())
        .sort((a, b) => a - b)[0];
      const ts = earliestStart ?? created?.getTime() ?? null;
      if (ts !== null) {
        const i = indexFor(ts);
        if (i !== null) {
          if (group === "running") running[i] += 1;
          else pending[i] += 1;
        }
      }
    } else if (group === "failed" && finished) {
      const i = indexFor(finished.getTime());
      if (i !== null) failed[i] += 1;
    } else if (group === "succeeded" && finished) {
      const i = indexFor(finished.getTime());
      if (i !== null) succeeded[i] += 1;
    }
  }

  return {
    running: finalizeKpiSparkline(running),
    pending: finalizeKpiSparkline(pending),
    failed: finalizeKpiSparkline(failed),
    succeeded: finalizeKpiSparkline(succeeded),
    submitted: finalizeKpiSparkline(submitted),
  };
};

export type RecentEventKind = "submitted" | "started" | "finished";

export interface RecentEvent {
  kind: RecentEventKind;
  at: string;
  executionId?: string;
  /** For finished events, mirrors execution.status (succeeded/failed/cancelled). */
  outcome?: string;
}

/**
 * Derives a "Recent events" list for a single run strictly from data the
 * backend already exposes — `createdAt`, plus per-execution `startedAt` /
 * `finishedAt`. Emits ONLY `submitted | started | finished`; never invents
 * synthetic events ("checkpoint saved", "stdout streaming", etc.).
 */
export const computeRecentEventsForRun = (run: WorkspaceRunRow): RecentEvent[] => {
  const events: RecentEvent[] = [];

  if (run.createdAt) {
    events.push({ kind: "submitted", at: run.createdAt });
  }

  for (const exec of run.executions) {
    if (exec.startedAt) {
      events.push({ kind: "started", at: exec.startedAt, executionId: exec.executionId });
    }
    if (exec.finishedAt) {
      events.push({
        kind: "finished",
        at: exec.finishedAt,
        executionId: exec.executionId,
        outcome: exec.status,
      });
    }
  }

  return events.sort((a, b) => {
    const ta = new Date(a.at).getTime();
    const tb = new Date(b.at).getTime();
    if (Number.isNaN(ta) && Number.isNaN(tb)) return 0;
    if (Number.isNaN(ta)) return 1;
    if (Number.isNaN(tb)) return -1;
    return tb - ta;
  });
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

/**
 * Pure-function tests for the Runs aggregate / facet derivations.
 */

import { describe, expect, it } from "@rstest/core";

import {
  applyFilters,
  computeActivityBuckets,
  computeAvgWaitSeconds,
  computeBackendDistribution,
  computeFacetCounts,
  computeKpiStats,
  computeTopFailingExperiments,
} from "@/app/runs/aggregates";
import type {
  WorkspaceExecutionRow,
  WorkspaceRunRow,
  WorkspaceRunsFilters,
} from "@/app/runs/types";

const NOW = Date.parse("2026-04-27T12:00:00Z");
const MIN = 60_000;
const HOUR = 60 * MIN;

const exec = (over: Partial<WorkspaceExecutionRow>): WorkspaceExecutionRow => ({
  executionId: "exec-1",
  runId: "run-1",
  status: "succeeded",
  startedAt: new Date(NOW - 30 * MIN).toISOString(),
  finishedAt: new Date(NOW - 5 * MIN).toISOString(),
  durationSeconds: 1500,
  schedulerJobId: null,
  backend: "local",
  metadata: {},
  backendMetadata: {},
  ...over,
});

const run = (over: Partial<WorkspaceRunRow>): WorkspaceRunRow => ({
  id: "run-1",
  name: "Run 1",
  projectId: "proj-A",
  projectName: "Project A",
  experimentId: "exp-1",
  experimentName: "Experiment 1",
  status: "succeeded",
  backend: "local",
  cluster: null,
  scheduler: null,
  profile: "default",
  parameters: {},
  createdAt: new Date(NOW - 60 * MIN).toISOString(),
  finishedAt: new Date(NOW - 5 * MIN).toISOString(),
  executionCount: 1,
  latestSchedulerJobId: null,
  executions: [exec({})],
  ...over,
});

describe("applyFilters", () => {
  it("returns all runs when filters are empty", () => {
    const runs = [run({ id: "a" }), run({ id: "b", status: "running" })];
    expect(applyFilters(runs, {}, NOW)).toHaveLength(2);
  });

  it("ANDs across array filters and ORs within a single filter", () => {
    const runs = [
      run({ id: "a", status: "running", backend: "slurm" }),
      run({ id: "b", status: "running", backend: "local" }),
      run({ id: "c", status: "failed", backend: "slurm" }),
    ];
    const filters: WorkspaceRunsFilters = {
      status: ["running", "failed"],
      backend: ["slurm"],
    };
    const result = applyFilters(runs, filters, NOW);
    expect(result.map((r) => r.id)).toEqual(["a", "c"]);
  });

  it("supports the active quick view", () => {
    const runs = [
      run({ id: "a", status: "running" }),
      run({ id: "b", status: "pending" }),
      run({ id: "c", status: "succeeded" }),
    ];
    const result = applyFilters(runs, { quickView: ["active"] }, NOW);
    expect(result.map((r) => r.id).sort()).toEqual(["a", "b"]);
  });

  it("excludes failed runs older than 24h from failed24h quick view", () => {
    const fresh = run({
      id: "fresh",
      status: "failed",
      finishedAt: new Date(NOW - 2 * HOUR).toISOString(),
    });
    const stale = run({
      id: "stale",
      status: "failed",
      finishedAt: new Date(NOW - 30 * HOUR).toISOString(),
    });
    const result = applyFilters([fresh, stale], { quickView: ["failed24h"] }, NOW);
    expect(result.map((r) => r.id)).toEqual(["fresh"]);
  });

  it("treats a running run started >1h ago as long-running", () => {
    const long = run({
      id: "long",
      status: "running",
      executions: [exec({ startedAt: new Date(NOW - 90 * MIN).toISOString(), finishedAt: null })],
    });
    const short = run({
      id: "short",
      status: "running",
      executions: [exec({ startedAt: new Date(NOW - 10 * MIN).toISOString(), finishedAt: null })],
    });
    const result = applyFilters([long, short], { quickView: ["longRunning"] }, NOW);
    expect(result.map((r) => r.id)).toEqual(["long"]);
  });

  it("ORs multiple selected quick views", () => {
    const runs = [
      run({ id: "active", status: "running" }),
      run({
        id: "failed",
        status: "failed",
        finishedAt: new Date(NOW - 2 * HOUR).toISOString(),
      }),
      run({ id: "stale", status: "succeeded" }),
    ];
    const result = applyFilters(runs, { quickView: ["active", "failed24h"] }, NOW);
    expect(result.map((r) => r.id).sort()).toEqual(["active", "failed"]);
  });
});

describe("computeKpiStats", () => {
  it("returns zeros for an empty list", () => {
    expect(computeKpiStats([])).toEqual({
      total: 0,
      running: 0,
      pending: 0,
      failed: 0,
      succeeded: 0,
    });
  });

  it("buckets statuses including aliases (timed_out → failed, queued → pending)", () => {
    const runs = [
      run({ id: "1", status: "running" }),
      run({ id: "2", status: "queued" }),
      run({ id: "3", status: "timed_out" }),
      run({ id: "4", status: "succeeded" }),
      run({ id: "5", status: "succeeded" }),
    ];
    expect(computeKpiStats(runs)).toEqual({
      total: 5,
      running: 1,
      pending: 1,
      failed: 1,
      succeeded: 2,
    });
  });
});

describe("computeAvgWaitSeconds", () => {
  it("returns null when no run has both timestamps", () => {
    const r = run({ executions: [] });
    expect(computeAvgWaitSeconds([r], 24, NOW)).toBeNull();
  });

  it("averages submit→start over runs in the window", () => {
    const a = run({
      id: "a",
      createdAt: new Date(NOW - 70 * MIN).toISOString(),
      executions: [exec({ startedAt: new Date(NOW - 60 * MIN).toISOString() })],
    });
    const b = run({
      id: "b",
      createdAt: new Date(NOW - 35 * MIN).toISOString(),
      executions: [exec({ startedAt: new Date(NOW - 30 * MIN).toISOString() })],
    });
    const avg = computeAvgWaitSeconds([a, b], 24, NOW);
    expect(avg).toBe(((10 * MIN + 5 * MIN) / 2) / 1000);
  });
});

describe("computeBackendDistribution", () => {
  it("groups by backend+cluster and sorts by count descending", () => {
    const runs = [
      run({ id: "1", backend: "slurm", cluster: "cluster-a" }),
      run({ id: "2", backend: "slurm", cluster: "cluster-a" }),
      run({ id: "3", backend: "slurm", cluster: "cluster-b" }),
      run({ id: "4", backend: "local", cluster: null }),
    ];
    const dist = computeBackendDistribution(runs);
    expect(dist).toEqual([
      { backend: "slurm", cluster: "cluster-a", count: 2 },
      { backend: "local", cluster: null, count: 1 },
      { backend: "slurm", cluster: "cluster-b", count: 1 },
    ]);
  });

  it("skips runs with no backend", () => {
    const runs = [run({ backend: null })];
    expect(computeBackendDistribution(runs)).toHaveLength(0);
  });
});

describe("computeTopFailingExperiments", () => {
  it("ranks experiments by failed count, omits zero-failure experiments", () => {
    const runs = [
      run({ id: "a", experimentId: "x1", experimentName: "X1", status: "failed" }),
      run({ id: "b", experimentId: "x1", experimentName: "X1", status: "failed" }),
      run({ id: "c", experimentId: "x1", experimentName: "X1", status: "succeeded" }),
      run({ id: "d", experimentId: "x2", experimentName: "X2", status: "failed" }),
      run({ id: "e", experimentId: "x3", experimentName: "X3", status: "succeeded" }),
    ];
    const top = computeTopFailingExperiments(runs, 5);
    expect(top.map((entry) => entry.experimentId)).toEqual(["x1", "x2"]);
    expect(top[0]).toMatchObject({ failedCount: 2, totalCount: 3 });
    expect(top[1]).toMatchObject({ failedCount: 1, totalCount: 1 });
  });
});

describe("computeActivityBuckets", () => {
  it("creates one bucket per hour over the window", () => {
    const buckets = computeActivityBuckets([], 24, NOW);
    expect(buckets).toHaveLength(24);
    expect(buckets.every((b) => b.started === 0 && b.finished === 0)).toBe(true);
  });

  it("counts started/finished and per-outcome events per hourly bucket", () => {
    const r = run({
      id: "x",
      executions: [
        exec({
          status: "succeeded",
          startedAt: new Date(NOW - 90 * MIN).toISOString(),
          finishedAt: new Date(NOW - 30 * MIN).toISOString(),
        }),
      ],
    });
    const buckets = computeActivityBuckets([r], 24, NOW);
    const totals = buckets.reduce(
      (acc, b) => ({
        started: acc.started + b.started,
        finished: acc.finished + b.finished,
        succeeded: acc.succeeded + b.succeeded,
        failed: acc.failed + b.failed,
        cancelled: acc.cancelled + b.cancelled,
      }),
      { started: 0, finished: 0, succeeded: 0, failed: 0, cancelled: 0 },
    );
    expect(totals).toEqual({
      started: 1,
      finished: 1,
      succeeded: 1,
      failed: 0,
      cancelled: 0,
    });
  });
});

describe("computeFacetCounts", () => {
  it("counts each facet ignoring its own filter (faceted-search semantics)", () => {
    const runs = [
      run({ id: "1", status: "running", backend: "slurm" }),
      run({ id: "2", status: "failed", backend: "slurm" }),
      run({ id: "3", status: "running", backend: "local" }),
      run({ id: "4", status: "succeeded", backend: "local" }),
    ];
    const facets = computeFacetCounts(runs, { backend: ["slurm"] }, NOW);
    const statusCounts = Object.fromEntries(facets.status.map((c) => [c.value, c.count]));
    expect(statusCounts).toEqual({ running: 1, failed: 1 });
    const backendCounts = Object.fromEntries(facets.backend.map((c) => [c.value, c.count]));
    expect(backendCounts).toEqual({ slurm: 2, local: 2 });
  });
});

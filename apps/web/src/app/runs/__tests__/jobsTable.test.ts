import { describe, expect, it } from "@rstest/core";

import {
  compareJobs,
  computeRunDurationSeconds,
  DEFAULT_JOBS_SORT,
  formatJobsSort,
  nextJobsSort,
  paginate,
  parseJobsSort,
  parsePage,
  parsePageSize,
  sortJobs,
} from "../jobsTable";
import type { WorkspaceExecutionRow, WorkspaceRunRow } from "../types";

const exec = (over: Partial<WorkspaceExecutionRow> = {}): WorkspaceExecutionRow => ({
  executionId: "exec-1",
  runId: "r1",
  status: "succeeded",
  startedAt: "2026-01-01T00:00:00Z",
  finishedAt: "2026-01-01T00:01:00Z",
  durationSeconds: 60,
  schedulerJobId: null,
  backend: "local",
  metadata: {},
  backendMetadata: {},
  ...over,
});

const baseRun = (overrides: Partial<WorkspaceRunRow> = {}): WorkspaceRunRow => ({
  id: "r1",
  name: "run-a",
  projectId: "p1",
  projectName: "Project",
  experimentId: "e1",
  experimentName: "Exp",
  status: "succeeded",
  backend: "local",
  cluster: null,
  scheduler: null,
  target: null,
  profile: null,
  parameters: {},
  createdAt: "2026-01-01T00:00:00Z",
  finishedAt: "2026-01-01T00:01:00Z",
  executionCount: 1,
  latestSchedulerJobId: null,
  executions: [exec()],
  ...overrides,
});

describe("computeRunDurationSeconds", () => {
  it("returns null when no execution has startedAt", () => {
    const run = baseRun({
      executions: [exec({ startedAt: "", finishedAt: null })],
    });
    expect(computeRunDurationSeconds(run)).toBeNull();
  });

  it("uses earliest start and finishedAt", () => {
    const run = baseRun({
      finishedAt: "2026-01-01T00:02:00Z",
      executions: [exec({ startedAt: "2026-01-01T00:00:00Z" })],
    });
    expect(computeRunDurationSeconds(run)).toBe(120);
  });
});

describe("sortJobs", () => {
  const rows = [
    baseRun({ id: "a", name: "alpha", status: "failed", createdAt: "2026-01-02T00:00:00Z" }),
    baseRun({ id: "b", name: "beta", status: "running", createdAt: "2026-01-03T00:00:00Z" }),
    baseRun({ id: "c", name: "gamma", status: "succeeded", createdAt: "2026-01-01T00:00:00Z" }),
  ];

  it("sorts by name ascending", () => {
    const sorted = sortJobs(rows, { key: "name", dir: "asc" });
    expect(sorted.map((r) => r.id)).toEqual(["a", "b", "c"]);
  });

  it("sorts by submitted descending (default)", () => {
    const sorted = sortJobs(rows, DEFAULT_JOBS_SORT);
    expect(sorted.map((r) => r.id)).toEqual(["b", "a", "c"]);
  });

  it("uses id as stable secondary key", () => {
    const twins = [
      baseRun({ id: "z", name: "same", createdAt: "2026-01-01T00:00:00Z" }),
      baseRun({ id: "a", name: "same", createdAt: "2026-01-01T00:00:00Z" }),
    ];
    const sorted = sortJobs(twins, { key: "name", dir: "asc" });
    expect(sorted.map((r) => r.id)).toEqual(["a", "z"]);
  });

  it("compareJobs flips with dir", () => {
    const a = rows[0];
    const b = rows[1];
    expect(Math.sign(compareJobs(a, b, { key: "submitted", dir: "asc" }))).toBe(
      -Math.sign(compareJobs(a, b, { key: "submitted", dir: "desc" })),
    );
  });
});

describe("parse / format jobs sort", () => {
  it("round-trips", () => {
    expect(parseJobsSort(formatJobsSort({ key: "status", dir: "asc" }))).toEqual({
      key: "status",
      dir: "asc",
    });
  });

  it("falls back on garbage", () => {
    expect(parseJobsSort("nope")).toEqual(DEFAULT_JOBS_SORT);
    expect(parseJobsSort(null)).toEqual(DEFAULT_JOBS_SORT);
  });
});

describe("nextJobsSort", () => {
  it("flips dir on same key", () => {
    expect(nextJobsSort({ key: "name", dir: "asc" }, "name")).toEqual({
      key: "name",
      dir: "desc",
    });
  });

  it("defaults time-like keys to desc", () => {
    expect(nextJobsSort({ key: "name", dir: "asc" }, "submitted").dir).toBe("desc");
  });
});

describe("paginate", () => {
  const items = Array.from({ length: 12 }, (_, i) => i);

  it("slices the requested page", () => {
    const page = paginate(items, 2, 5);
    expect(page.items).toEqual([5, 6, 7, 8, 9]);
    expect(page.totalPages).toBe(3);
    expect(page.page).toBe(2);
  });

  it("clamps page past the end", () => {
    const page = paginate(items, 99, 5);
    expect(page.page).toBe(3);
    expect(page.items).toEqual([10, 11]);
  });

  it("handles empty", () => {
    const page = paginate([], 3, 10);
    expect(page).toEqual({
      items: [],
      page: 1,
      pageSize: 10,
      totalItems: 0,
      totalPages: 1,
    });
  });
});

describe("parsePage / parsePageSize", () => {
  it("parses valid values", () => {
    expect(parsePage("3")).toBe(3);
    expect(parsePageSize("100")).toBe(100);
  });

  it("rejects invalid page size", () => {
    expect(parsePageSize("17")).toBe(50);
    expect(parsePage("0")).toBe(1);
  });
});

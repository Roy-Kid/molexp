/**
 * Tests for the molq plugin module — formatters, the API client, and the
 * execution column / detail contribution registration surface.
 */

import { afterEach, beforeEach, describe, expect, it, rs } from "@rstest/core";

import {
  listExecutionColumns,
  listExecutionDetails,
  registerExecutionColumn,
  registerExecutionDetail,
} from "@/app/registry";
import { resetContributionRuntimeForTests } from "@/plugins/contribution-runtime";
import { molqApi } from "@/plugins/molq/api";
import { formatDuration, formatRelative } from "@/plugins/molq/format";
import type { ExecutionColumnContribution, ExecutionDetailContribution } from "@/plugins/types";

beforeEach(() => {
  resetContributionRuntimeForTests();
});

afterEach(() => {
  rs.restoreAllMocks();
});

describe("formatDuration", () => {
  it("returns dash for null/negative/NaN", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(-5)).toBe("—");
    expect(formatDuration(Number.NaN)).toBe("—");
  });

  it("formats seconds, minutes, and hours separately", () => {
    expect(formatDuration(45)).toBe("45s");
    expect(formatDuration(125)).toBe("2m 05s");
    expect(formatDuration(3725)).toBe("1h 02m 05s");
  });
});

describe("formatRelative", () => {
  it("returns dash when input is null", () => {
    expect(formatRelative(null)).toBe("—");
  });

  it("falls back to the raw string when not a date", () => {
    expect(formatRelative("not-a-date")).toBe("not-a-date");
  });

  it("returns 'just now' for recent timestamps", () => {
    const recent = new Date(Date.now() - 5_000).toISOString();
    expect(formatRelative(recent)).toBe("just now");
  });

  it("returns minutes when delta is < 1 hour", () => {
    const t = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    expect(formatRelative(t)).toBe("5m ago");
  });
});

describe("execution column registry", () => {
  const Cell = (() => null) as ExecutionColumnContribution["Cell"];

  it("filters by backend so molq columns only appear for molq executions", () => {
    registerExecutionColumn({
      id: "molq:column:cluster",
      backend: "molq",
      columnId: "cluster",
      header: "Cluster",
      Cell,
    });

    expect(listExecutionColumns("molq").map((c) => c.columnId)).toEqual(["cluster"]);
    expect(listExecutionColumns("local")).toEqual([]);
    expect(listExecutionColumns(null)).toEqual([]);
  });

  it("dedupes by id and orders by priority desc", () => {
    registerExecutionColumn({
      id: "a",
      backend: "molq",
      columnId: "scheduler-job",
      header: "Job",
      priority: 10,
      Cell,
    });
    registerExecutionColumn({
      id: "a",
      backend: "molq",
      columnId: "scheduler-job",
      header: "Duplicate (skipped)",
      priority: 999,
      Cell,
    });
    registerExecutionColumn({
      id: "b",
      backend: "molq",
      columnId: "cluster",
      header: "Cluster",
      priority: 100,
      Cell,
    });

    const cols = listExecutionColumns("molq");
    expect(cols.map((c) => c.id)).toEqual(["b", "a"]);
  });
});

describe("execution detail registry", () => {
  it("returns backend-scoped detail contributions", () => {
    const Component = (() => null) as ExecutionDetailContribution["Component"];
    registerExecutionDetail({
      id: "molq:detail:submission",
      backend: "molq",
      title: "Molq submission",
      Component,
    });

    expect(listExecutionDetails("molq")).toHaveLength(1);
    expect(listExecutionDetails("local")).toHaveLength(0);
  });
});

describe("molqApi.listTargets", () => {
  it("returns the targets array from the response envelope", async () => {
    const fetchSpy = rs.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ targets: [{ name: "demo" }], total: 1 }), {
        status: 200,
      }),
    );

    const targets = await molqApi.listTargets();

    expect(fetchSpy).toHaveBeenCalledWith("/api/plugins/molq/targets");
    expect(targets).toEqual([{ name: "demo" }]);
  });

  it("throws a descriptive error on non-OK response", async () => {
    rs.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "boom" }), { status: 500 }),
    );

    await expect(molqApi.listTargets()).rejects.toThrow(/List targets failed \(500\): boom/);
  });
});

describe("molqApi.listJobs", () => {
  it("encodes target and limit as query params", async () => {
    const fetchSpy = rs.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          jobs: [],
          stats: { running: 0, pending: 0, failed: 0, succeeded: 0, avgWaitSeconds: null },
          total: 0,
        }),
        { status: 200 },
      ),
    );

    await molqApi.listJobs("demo", 50);

    const url = (fetchSpy.mock.calls[0]?.[0] as string) ?? "";
    expect(url).toContain("target=demo");
    expect(url).toContain("limit=50");
  });
});

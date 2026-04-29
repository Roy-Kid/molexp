/**
 * Round-trip tests for filterParams: URL → WorkspaceRunsFilters → URL.
 */

import { describe, expect, it } from "@rstest/core";

import {
  hasActiveFilters,
  parseFilterParams,
  toggleArrayFilter,
  toggleQuickView,
  writeFilterParams,
} from "@/app/runs/filterParams";

describe("parseFilterParams", () => {
  it("decodes comma-separated multi-value params", () => {
    const params = new URLSearchParams("status=running,failed&backend=slurm");
    expect(parseFilterParams(params)).toEqual({
      status: ["running", "failed"],
      backend: ["slurm"],
    });
  });

  it("ignores unknown quickView values", () => {
    const params = new URLSearchParams("quickView=bogus");
    expect(parseFilterParams(params)).toEqual({});
  });

  it("preserves valid quickView values and drops invalid ones", () => {
    const params = new URLSearchParams("quickView=active,bogus,failed24h");
    expect(parseFilterParams(params)).toEqual({ quickView: ["active", "failed24h"] });
  });
});

describe("writeFilterParams", () => {
  it("clears keys when their array is empty", () => {
    const prev = new URLSearchParams("status=running&backend=slurm");
    const next = writeFilterParams(prev, { backend: ["slurm"] });
    expect(next.get("status")).toBeNull();
    expect(next.get("backend")).toBe("slurm");
  });

  it("preserves unrelated params", () => {
    const prev = new URLSearchParams("runId=abc");
    const next = writeFilterParams(prev, { status: ["running"] });
    expect(next.get("runId")).toBe("abc");
    expect(next.get("status")).toBe("running");
  });
});

describe("toggleArrayFilter", () => {
  it("adds a value not present", () => {
    const next = toggleArrayFilter({}, "status", "running");
    expect(next.status).toEqual(["running"]);
  });

  it("removes a value already present", () => {
    const next = toggleArrayFilter({ status: ["running", "failed"] }, "status", "running");
    expect(next.status).toEqual(["failed"]);
  });

  it("clears experimentId when projectId changes", () => {
    const next = toggleArrayFilter({ projectId: ["A"], experimentId: ["e1"] }, "projectId", "B");
    expect(next.projectId).toEqual(["A", "B"]);
    expect(next.experimentId).toBeUndefined();
  });
});

describe("toggleQuickView", () => {
  it("adds a view not present", () => {
    const next = toggleQuickView({}, "active");
    expect(next.quickView).toEqual(["active"]);
  });
  it("removes a view already present", () => {
    const next = toggleQuickView({ quickView: ["active", "failed24h"] }, "active");
    expect(next.quickView).toEqual(["failed24h"]);
  });
  it("clears the field entirely when the last view is removed", () => {
    const next = toggleQuickView({ quickView: ["active"] }, "active");
    expect(next.quickView).toBeUndefined();
  });
});

describe("hasActiveFilters", () => {
  it("is false for empty filters", () => {
    expect(hasActiveFilters({})).toBe(false);
  });
  it("is true once any filter is set", () => {
    expect(hasActiveFilters({ status: ["running"] })).toBe(true);
    expect(hasActiveFilters({ quickView: ["active"] })).toBe(true);
  });
});

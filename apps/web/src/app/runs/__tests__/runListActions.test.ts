import { describe, expect, it } from "@rstest/core";
import { buildRunListActions, primaryRunVerb } from "@/app/runs/runListActions";
import type { RunSummary } from "@/app/types";

const baseRun = (status: string): RunSummary =>
  ({
    id: "abc12345",
    name: "run",
    status,
    summary: "",
    updatedAt: "",
    projectId: "p",
    experimentId: "e",
    executorInfo: {},
    profile: null,
    configHash: null,
    parameters: {},
    results: {},
    workflowSource: null,
    workflowSnapshot: null,
    startedAt: null,
    finishedAt: null,
    executionHistory: [],
    errorMessage: null,
  }) as RunSummary;

const noopHandlers = {
  copyId: () => undefined,
  copyPath: () => undefined,
};

describe("primaryRunVerb", () => {
  it("returns the one primary control per phase", () => {
    expect(primaryRunVerb("pending")?.kind).toBe("start");
    expect(primaryRunVerb("running")?.kind).toBe("cancel");
    expect(primaryRunVerb("failed")?.kind).toBe("resume");
    expect(primaryRunVerb("succeeded")).toBeNull();
  });
});

describe("buildRunListActions", () => {
  it("is copy-only regardless of status", () => {
    for (const status of ["pending", "running", "failed", "succeeded", "cancelled"]) {
      const ids = buildRunListActions(baseRun(status), noopHandlers).map((a) => a.id);
      expect(ids).toEqual(["copy-id", "copy-path"]);
    }
  });

  it("invokes copy handlers", () => {
    const seen: string[] = [];
    const run = baseRun("succeeded");
    const actions = buildRunListActions(run, {
      copyId: (r) => {
        seen.push(`id:${r.id}`);
      },
      copyPath: (r) => {
        seen.push(`path:${r.id}`);
      },
    });
    actions.find((a) => a.id === "copy-id")?.onSelect();
    actions.find((a) => a.id === "copy-path")?.onSelect();
    expect(seen).toEqual(["id:abc12345", "path:abc12345"]);
  });
});

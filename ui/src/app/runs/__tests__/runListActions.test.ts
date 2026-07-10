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
  open: () => undefined,
  cancel: () => undefined,
  resume: () => undefined,
  rerun: () => undefined,
  copyId: () => undefined,
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
  it("shows Start only for pending, never a disabled Cancel", () => {
    const ids = buildRunListActions(baseRun("pending"), noopHandlers).map((a) => a.id);
    expect(ids).toContain("start");
    expect(ids).not.toContain("cancel");
    expect(ids).not.toContain("resume");
  });

  it("shows Cancel only for running", () => {
    const ids = buildRunListActions(baseRun("running"), noopHandlers).map((a) => a.id);
    expect(ids).toContain("cancel");
    expect(ids).not.toContain("start");
    expect(ids).not.toContain("resume");
  });

  it("shows resume/rerun/fresh/harvest for failed", () => {
    const ids = buildRunListActions(baseRun("failed"), noopHandlers).map((a) => a.id);
    expect(ids).toContain("resume");
    expect(ids).toContain("rerun");
    expect(ids).toContain("rerun-fresh");
    expect(ids).toContain("harvest");
    expect(ids).toContain("copy-id");
    expect(ids).not.toContain("cancel");
    expect(ids).not.toContain("start");
  });

  it("shows harvest for succeeded, no lifecycle verbs", () => {
    const ids = buildRunListActions(baseRun("succeeded"), noopHandlers).map((a) => a.id);
    expect(ids).toContain("harvest");
    expect(ids).not.toContain("resume");
    expect(ids).not.toContain("cancel");
  });
});

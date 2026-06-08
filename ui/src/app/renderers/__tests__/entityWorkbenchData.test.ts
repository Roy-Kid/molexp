import { describe, expect, it } from "@rstest/core";
import {
  buildExperimentWorkbenchData,
  buildProjectWorkbenchData,
} from "@/app/renderers/entityWorkbenchData";
import type {
  ExperimentSummary,
  ProjectSummary,
  RunSummary,
  WorkflowSummary,
  WorkspaceSnapshot,
} from "@/app/types";

const project: ProjectSummary = {
  id: "matrix",
  name: "Matrix",
  status: "active",
  summary: "",
  updatedAt: "2026-06-01T00:00:00Z",
};

const experiment = (
  id: string,
  parameterSpace: Record<string, unknown> = {},
): ExperimentSummary => ({
  id,
  name: id,
  status: "active",
  summary: "",
  workflowFile: "workflow.json",
  updatedAt: "2026-06-02T00:00:00Z",
  projectId: project.id,
  parameterSpace,
  workflowSource: null,
});

const run = (
  id: string,
  status: RunSummary["status"],
  parameters: Record<string, unknown>,
): RunSummary => ({
  id,
  name: id,
  status,
  summary: "",
  updatedAt: `2026-06-02T00:0${id.slice(-1)}:00Z`,
  projectId: project.id,
  experimentId: "series",
  executorInfo: { backend: "local" },
  profile: null,
  configHash: null,
  parameters,
  results: {},
  workflowSource: null,
  workflowSnapshot: null,
  startedAt: null,
  finishedAt: null,
  executionHistory: [],
  errorMessage: null,
});

const workflow: WorkflowSummary = {
  id: "wf-series",
  name: "series workflow",
  status: "active",
  summary: "",
  updatedAt: "2026-06-02T00:00:00Z",
  projectId: project.id,
  experimentId: "series",
  graph: {
    task_configs: [
      { id: "param", type: "task" },
      { id: "box", type: "task" },
    ],
    links: [{ from: "param", to: "box", kind: "parallel" }],
  },
};

const snapshot: WorkspaceSnapshot = {
  workspaces: [],
  projects: [project],
  experiments: [
    experiment("series", { mode: ["block", "random"], ratio: [1, 2], salt: [54, 108] }),
    experiment("empty"),
  ],
  runs: [
    run("run-1", "succeeded", { mode: "block", ratio: 1, salt: 54 }),
    run("run-2", "failed", { mode: "random", ratio: 2, salt: 108 }),
    run("run-3", "running", { mode: "block", ratio: 2, salt: 54 }),
  ],
  assets: [
    {
      id: "asset-1",
      name: "out.dat",
      kind: "data",
      status: "active",
      summary: "",
      updatedAt: "2026-06-02T00:00:00Z",
      sizeBytes: 12,
      projectId: project.id,
    },
  ],
  workflows: [workflow],
  agentSessions: [],
  workspaceRoot: null,
  consoleEntries: [],
};

describe("buildProjectWorkbenchData", () => {
  it("rolls up project inventory and run statuses", () => {
    const data = buildProjectWorkbenchData(project.id, snapshot);
    expect(data.experiments).toHaveLength(2);
    expect(data.counts).toMatchObject({ total: 3, succeeded: 1, failed: 1, running: 1 });
    expect(data.assetCount).toBe(1);
  });

  it("flags failed, running, empty, and missing-workflow experiments", () => {
    const reasons = buildProjectWorkbenchData(project.id, snapshot).attention.map((item) => [
      item.experiment.id,
      item.reason,
    ]);
    expect(reasons).toContainEqual(["series", "failed"]);
    expect(reasons).toContainEqual(["series", "running"]);
    expect(reasons).toContainEqual(["empty", "missing-workflow"]);
    expect(reasons).toContainEqual(["empty", "empty"]);
  });
});

describe("buildExperimentWorkbenchData", () => {
  it("derives parameter axes from declared space and run parameters", () => {
    const data = buildExperimentWorkbenchData(snapshot.experiments[0], snapshot.runs, workflow);
    expect(data.parameterAxes.map((axis) => axis.key)).toEqual(["mode", "ratio", "salt"]);
    expect(data.parameterAxes.find((axis) => axis.key === "mode")?.values).toEqual([
      "block",
      "random",
    ]);
  });

  it("summarizes workflow graph and groups runs by the primary parameter axis", () => {
    const data = buildExperimentWorkbenchData(snapshot.experiments[0], snapshot.runs, workflow);
    expect(data.workflowSummary).toMatchObject({
      exists: true,
      taskCount: 2,
      linkCount: 1,
      parallelGroupCount: 1,
    });
    expect(data.runGroups.map((group) => group.label)).toEqual(["mode: block", "mode: random"]);
  });
});

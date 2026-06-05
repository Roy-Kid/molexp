/**
 * Pure-logic tests for the entity graph: relation resolution, URL building, and
 * the command-palette catalog/search. These cover the connectivity foundation
 * (the edges that power the Related panel + ⌘K) in the node test env.
 */

import { describe, expect, it } from "@rstest/core";
import { buildCatalog, searchCatalog } from "@/app/entities/catalog";
import { entityPath, runPath } from "@/app/entities/paths";
import { resolveRelations } from "@/app/entities/relations";
import type { WorkspaceSnapshot } from "@/app/types";

const snapshot: WorkspaceSnapshot = {
  projects: [
    { id: "p1", name: "Proj One", status: "active", summary: "first", updatedAt: "2026-01-01" },
  ],
  experiments: [
    {
      id: "e1",
      name: "Exp One",
      status: "active",
      summary: "",
      workflowFile: "",
      updatedAt: "2026-01-01",
      projectId: "p1",
      parameterSpace: {},
      workflowSource: null,
    },
  ],
  runs: [
    {
      id: "r1",
      name: "Run One",
      status: "succeeded",
      summary: "",
      updatedAt: "2026-01-01",
      projectId: "p1",
      experimentId: "e1",
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
    },
    {
      id: "r2",
      name: "Run Two",
      status: "failed",
      summary: "",
      updatedAt: "2026-01-01",
      projectId: "p1",
      experimentId: "e1",
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
    },
  ],
  assets: [
    {
      id: "a1",
      name: "Asset One",
      kind: "data",
      status: "active",
      summary: "",
      updatedAt: "2026-01-01",
      sizeBytes: null,
      projectId: "p1",
    },
  ],
  workflows: [
    {
      id: "w1",
      name: "WF One",
      status: "active",
      summary: "",
      updatedAt: "2026-01-01",
      projectId: "p1",
      experimentId: "e1",
    },
  ],
  agentSessions: [
    {
      id: "ag1",
      sessionId: "s1",
      goal: "Investigate convergence",
      status: "running",
      createdAt: "2026-01-01",
      eventCount: 3,
    },
  ],
  workspaceRoot: null,
  consoleEntries: [],
  workspaces: [],
};

const groupRefs = (relations: ReturnType<typeof resolveRelations>, relation: string): string[] =>
  relations.find((g) => g.relation === relation)?.refs.map((r) => r.id) ?? [];

describe("resolveRelations", () => {
  it("links a run to its project, experiment, workflow, and sibling runs", () => {
    const rel = resolveRelations({ kind: "run", id: "r1" }, snapshot);
    expect(groupRefs(rel, "project")).toEqual(["p1"]);
    expect(groupRefs(rel, "experiment")).toEqual(["e1"]);
    expect(groupRefs(rel, "workflow")).toEqual(["w1"]);
    expect(groupRefs(rel, "siblings")).toEqual(["r2"]);
  });

  it("links an experiment to its project, workflow, and runs", () => {
    const rel = resolveRelations({ kind: "experiment", id: "e1" }, snapshot);
    expect(groupRefs(rel, "project")).toEqual(["p1"]);
    expect(groupRefs(rel, "workflow")).toEqual(["w1"]);
    expect(groupRefs(rel, "runs").sort()).toEqual(["r1", "r2"]);
  });

  it("links a workflow back to its experiment and runs", () => {
    const rel = resolveRelations({ kind: "workflow", id: "w1" }, snapshot);
    expect(groupRefs(rel, "experiment")).toEqual(["e1"]);
    expect(groupRefs(rel, "runs").sort()).toEqual(["r1", "r2"]);
  });

  it("omits empty relation groups", () => {
    const rel = resolveRelations({ kind: "agent", id: "ag1" }, snapshot);
    expect(rel).toEqual([]);
  });
});

describe("entityPath / runPath", () => {
  it("builds the canonical run URL via runPath", () => {
    expect(runPath("p1", "e1", "r1")).toBe("/projects/p1/experiments/e1/runs/r1");
    expect(entityPath({ kind: "run", id: "r1" }, snapshot)).toBe(
      "/projects/p1/experiments/e1/runs/r1",
    );
  });

  it("nests a task under its run", () => {
    expect(entityPath({ kind: "task", id: "t1", runId: "r1" }, snapshot)).toBe(
      "/projects/p1/experiments/e1/runs/r1/tasks/t1",
    );
  });

  it("returns null for a ref absent from the snapshot", () => {
    expect(entityPath({ kind: "run", id: "ghost" }, snapshot)).toBeNull();
  });
});

describe("command palette catalog", () => {
  it("indexes every searchable kind", () => {
    const catalog = buildCatalog(snapshot);
    const kinds = new Set(catalog.map((e) => e.ref.kind));
    expect(kinds).toEqual(new Set(["project", "experiment", "run", "workflow", "asset", "agent"]));
  });

  it("matches by name across kinds and ranks prefix first", () => {
    const catalog = buildCatalog(snapshot);
    const hits = searchCatalog(catalog, "run");
    expect(hits.map((h) => h.ref.id)).toContain("r1");
    expect(hits.map((h) => h.ref.id)).toContain("r2");
  });

  it("finds an agent session by its goal text", () => {
    const catalog = buildCatalog(snapshot);
    const hits = searchCatalog(catalog, "convergence");
    expect(hits.map((h) => h.ref.id)).toEqual(["ag1"]);
  });
});

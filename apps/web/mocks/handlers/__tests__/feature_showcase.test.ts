import { describe, expect, it } from "@rstest/core";
import { getResponse } from "msw";

import { handlers } from "../index";

Reflect.set(globalThis, "location", new URL("http://localhost/"));

const getJson = async <T,>(path: string): Promise<T> => {
  const response = await getResponse(handlers, new Request(`http://localhost${path}`));
  expect(response?.status).toBe(200);
  return (await response?.json()) as T;
};

describe("feature showcase mock", () => {
  it("keeps the entity tree and workspace runs dashboard on one fixture set", async () => {
    const experiments = await getJson<Array<{ id: string; runCount: number }>>(
      "/api/projects/protein-folding/experiments",
    );
    const alphaFoldRuns = await getJson<Array<{ id: string; status: string }>>(
      "/api/projects/protein-folding/experiments/exp-001/runs",
    );
    const workspaceRuns = await getJson<{
      total: number;
      runs: Array<{ id: string; status: string }>;
    }>("/api/workspace/runs");

    expect(experiments.find((row) => row.id === "exp-001")?.runCount).toBe(
      alphaFoldRuns.length,
    );
    expect(workspaceRuns.runs.some((row) => row.id === "run-001")).toBe(true);
    expect(new Set(workspaceRuns.runs.map((row) => row.status))).toEqual(
      new Set(["succeeded", "running", "failed", "cancelled", "pending"]),
    );
  });

  it("populates the cross-feature navigation surfaces", async () => {
    const workspaces = await getJson<unknown[]>("/api/workspaces");
    const knowledge = await getJson<{ total: number }>("/api/knowledge");
    const events = await getJson<unknown[]>("/api/events");
    const plans = await getJson<{ total: number }>("/api/plans");
    const approvals = await getJson<{ total: number }>("/api/approvals");

    expect(workspaces.length).toBeGreaterThan(0);
    expect(knowledge.total).toBeGreaterThan(0);
    expect(events.length).toBeGreaterThan(0);
    expect(plans.total).toBeGreaterThan(0);
    expect(approvals.total).toBeGreaterThan(0);
  });

  it("exposes renderable MolPlot and MolVis products on the showcase run", async () => {
    const files = await getJson<{
      nodes: Array<{ relPath: string; children?: Array<{ relPath: string }> }>;
    }>("/api/projects/protein-folding/experiments/exp-001/runs/run-001/files");
    const paths = files.nodes.flatMap((node) => [
      node.relPath,
      ...(node.children ?? []).map((child) => child.relPath),
    ]);

    expect(paths).toContain("observables/energy.mlp.vl.json");
    expect(paths).toContain("trajectory.xyz");

    const plot = await getJson<{ content: string }>(
      "/api/projects/protein-folding/experiments/exp-001/runs/run-001/file/text?path=observables%2Fenergy.mlp.vl.json",
    );
    const plotSpec = JSON.parse(plot.content) as { title?: string };
    expect(plotSpec.title).toBe("Potential energy convergence");

    const trajectory = await getJson<{ content: string }>(
      "/api/projects/protein-folding/experiments/exp-001/runs/run-001/file/text?path=trajectory.xyz",
    );
    expect(trajectory.content).toContain("frame 1");
  });
});

import { describe, expect, it } from "@rstest/core";

import { buildElements, type WorkflowIR } from "../WorkflowGraph";

const linear: WorkflowIR = {
  task_configs: [
    { task_id: "fetch", task_type: "fetcher" },
    { task_id: "process", task_type: "processor" },
    { task_id: "report", task_type: "reporter" },
  ],
  links: [
    { source: "fetch", target: "process" },
    { source: "process", target: "report" },
  ],
};

const diamond: WorkflowIR = {
  task_configs: [
    { task_id: "ingest", task_type: "ingester" },
    { task_id: "branchA", task_type: "filterer" },
    { task_id: "branchB", task_type: "filterer" },
    { task_id: "merge", task_type: "merger" },
  ],
  links: [
    { source: "ingest", target: "branchA" },
    { source: "ingest", target: "branchB" },
    { source: "branchA", target: "merge" },
    { source: "branchB", target: "merge" },
  ],
};

describe("buildElements", () => {
  it("emits one node per task_config and preserves task_id/task_type as data", () => {
    const { nodes } = buildElements(linear);
    expect(nodes).toHaveLength(3);
    const dataById = new Map(nodes.map((n) => [n.id, n.data]));
    expect(dataById.get("fetch")).toMatchObject({ taskId: "fetch", taskType: "fetcher" });
    expect(dataById.get("report")).toMatchObject({ taskId: "report", taskType: "reporter" });
  });

  it("emits one edge per valid link with source/target preserved", () => {
    const { edges } = buildElements(linear);
    expect(edges).toHaveLength(2);
    expect(edges.some((e) => e.source === "fetch" && e.target === "process")).toBe(true);
    expect(edges.some((e) => e.source === "process" && e.target === "report")).toBe(true);
  });

  it("filters out links pointing to unknown task ids and surfaces them via invalidLinks", () => {
    const ir: WorkflowIR = {
      task_configs: [{ task_id: "a", task_type: "t" }],
      links: [
        { source: "a", target: "ghost" },
        { source: "phantom", target: "a" },
        { source: "a", target: "a" }, // self-loop on a known node IS valid
      ],
    };
    const { edges, invalidLinks } = buildElements(ir);
    expect(invalidLinks).toHaveLength(2);
    expect(edges).toHaveLength(1);
    expect(edges[0]).toMatchObject({ source: "a", target: "a" });
  });

  it("renders parallel branches with distinct positions", () => {
    const { nodes } = buildElements(diamond);
    const byId = new Map(nodes.map((n) => [n.id, n.position]));
    // The two parallel branches must not collapse onto the same point.
    const a = byId.get("branchA");
    const b = byId.get("branchB");
    expect(a).toBeTruthy();
    expect(b).toBeTruthy();
    expect(a).not.toEqual(b);
  });

  it("returns empty elements for an empty IR", () => {
    const { nodes, edges, invalidLinks } = buildElements({ task_configs: [], links: [] });
    expect(nodes).toHaveLength(0);
    expect(edges).toHaveLength(0);
    expect(invalidLinks).toHaveLength(0);
  });

  it("does not crash on a cyclic IR — every node still gets a position", () => {
    const ir: WorkflowIR = {
      task_configs: [
        { task_id: "a", task_type: "t" },
        { task_id: "b", task_type: "t" },
      ],
      links: [
        { source: "a", target: "b" },
        { source: "b", target: "a" },
      ],
    };
    const { nodes } = buildElements(ir);
    expect(nodes.map((n) => n.id).sort()).toEqual(["a", "b"]);
    for (const n of nodes) {
      expect(typeof n.position.x).toBe("number");
      expect(typeof n.position.y).toBe("number");
    }
  });
});

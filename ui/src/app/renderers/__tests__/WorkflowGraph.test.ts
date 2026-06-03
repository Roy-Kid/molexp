import { describe, expect, it } from "@rstest/core";
import type { TaskGraphJson } from "@/types/task_graph_ir";
import { buildFlowgramDocument } from "../flowgram-document";

const linear: TaskGraphJson = {
  task_configs: [
    { id: "fetch", type: "fetcher", position: { x: 0, y: 0 } },
    { id: "process", type: "processor", position: { x: 0, y: 140 } },
    { id: "report", type: "reporter", position: { x: 0, y: 280 } },
  ],
  links: [
    { from: "fetch", to: "process", kind: "data" },
    { from: "process", to: "report", kind: "data" },
  ],
};

const diamond: TaskGraphJson = {
  task_configs: [
    { id: "ingest", type: "ingester" },
    { id: "branchA", type: "filterer" },
    { id: "branchB", type: "filterer" },
    { id: "merge", type: "merger" },
  ],
  links: [
    { from: "ingest", to: "branchA", kind: "data" },
    { from: "ingest", to: "branchB", kind: "data" },
    { from: "branchA", to: "merge", kind: "data" },
    { from: "branchB", to: "merge", kind: "data" },
  ],
};

describe("buildFlowgramDocument", () => {
  it("emits one document node per task_config carrying position + data", () => {
    const doc = buildFlowgramDocument(linear);
    expect(doc.nodes).toHaveLength(3);
    const byId = new Map(doc.nodes.map((n) => [n.id, n]));
    // explicit positions are preserved
    expect(byId.get("fetch")?.meta.position).toEqual({ x: 0, y: 0 });
    expect(byId.get("process")?.meta.position).toEqual({ x: 0, y: 140 });
    // data carries task identity
    expect(byId.get("fetch")?.data).toMatchObject({ taskId: "fetch", taskType: "fetcher" });
    expect(byId.get("report")?.data).toMatchObject({ taskId: "report", taskType: "reporter" });
  });

  it("emits one edge per valid link with source/target node ids preserved", () => {
    const doc = buildFlowgramDocument(linear);
    expect(doc.edges).toHaveLength(2);
    expect(doc.edges.some((e) => e.sourceNodeID === "fetch" && e.targetNodeID === "process")).toBe(
      true,
    );
    expect(doc.edges.some((e) => e.sourceNodeID === "process" && e.targetNodeID === "report")).toBe(
      true,
    );
  });

  it("drops links pointing to unknown task ids (self-loop on a known node is valid)", () => {
    const ir: TaskGraphJson = {
      task_configs: [{ id: "a", type: "t" }],
      links: [
        { from: "a", to: "ghost" },
        { from: "phantom", to: "a" },
        { from: "a", to: "a" }, // self-loop on a known node IS valid
      ],
    };
    const doc = buildFlowgramDocument(ir);
    expect(doc.edges).toHaveLength(1);
    expect(doc.edges[0]).toMatchObject({ sourceNodeID: "a", targetNodeID: "a" });
  });

  it("renders parallel branches with distinct fallback positions", () => {
    const doc = buildFlowgramDocument(diamond);
    const byId = new Map(doc.nodes.map((n) => [n.id, n.meta.position]));
    const a = byId.get("branchA");
    const b = byId.get("branchB");
    expect(a).toBeTruthy();
    expect(b).toBeTruthy();
    // The two parallel branches must not collapse onto the same point.
    expect(a).not.toEqual(b);
  });

  it("returns an empty document for an empty IR", () => {
    const doc = buildFlowgramDocument({ task_configs: [], links: [] });
    expect(doc.nodes).toHaveLength(0);
    expect(doc.edges).toHaveLength(0);
  });

  it("survives a cyclic IR — every node still has a numeric position", () => {
    const ir: TaskGraphJson = {
      task_configs: [
        { id: "a", type: "t" },
        { id: "b", type: "t" },
      ],
      links: [
        { from: "a", to: "b" },
        { from: "b", to: "a" },
      ],
    };
    const doc = buildFlowgramDocument(ir);
    expect(doc.nodes.map((n) => n.id).sort()).toEqual(["a", "b"]);
    for (const n of doc.nodes) {
      expect(typeof n.meta.position.x).toBe("number");
      expect(typeof n.meta.position.y).toBe("number");
    }
    // cyclic edges still produced (both endpoints are known)
    expect(doc.edges).toHaveLength(2);
  });
});

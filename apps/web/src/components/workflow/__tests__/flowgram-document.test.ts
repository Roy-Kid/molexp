import { describe, expect, it } from "@rstest/core";
import type { TaskGraphJson } from "@/components/workflow/task-graph-ir";
import { buildFlowgramDocument, parseTaskGraphIr } from "../flowgram-document";

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

describe("parseTaskGraphIr — full-graph IR + subworkflow", () => {
  // The shape `to_graph_ir()` emits and `entry.py` persists as workflow_source:
  // `{tasks: [{name, task_type, depends_on, subworkflow}], edges: [{source,target,kind}]}`.
  const graphIr = JSON.stringify({
    name: "outer",
    tasks: [
      { name: "items", task_type: "lister", depends_on: [], subworkflow: null },
      {
        name: "sub",
        task_type: "molexp.SubWorkflow",
        depends_on: [],
        subworkflow: {
          name: "inner",
          tasks: [
            { name: "load", task_type: "loader", depends_on: [] },
            { name: "scale", task_type: "scaler", depends_on: ["load"] },
          ],
          edges: [{ source: "load", target: "scale", kind: "data" }],
        },
      },
      { name: "collect", task_type: "collector", depends_on: [] },
    ],
    edges: [
      { source: "items", target: "sub", kind: "parallel" },
      { source: "sub", target: "collect", kind: "parallel" },
    ],
  });

  it("accepts the graph-IR shape (tasks/edges), not just wire (task_configs/links)", () => {
    const ir = parseTaskGraphIr(graphIr);
    expect(ir).not.toBeNull();
    expect(ir?.task_configs.map((t) => t.id).sort()).toEqual(["collect", "items", "sub"]);
    expect(ir?.links).toHaveLength(2);
  });

  it("carries a SubWorkflow node's inner graph through recursive normalization", () => {
    const ir = parseTaskGraphIr(graphIr) as TaskGraphJson;
    const sub = ir.task_configs.find((t) => t.id === "sub");
    expect(sub?.subworkflow).toBeDefined();
    expect(sub?.subworkflow?.name).toBe("inner");
    expect(sub?.subworkflow?.task_configs.map((t) => t.id).sort()).toEqual(["load", "scale"]);
    // Plain nodes carry no inner graph.
    expect(ir.task_configs.find((t) => t.id === "items")?.subworkflow).toBeUndefined();
  });

  it("buildFlowgramDocument exposes subworkflow + parallel flag on the node data", () => {
    const ir = parseTaskGraphIr(graphIr) as TaskGraphJson;
    const doc = buildFlowgramDocument(ir);
    const sub = doc.nodes.find((n) => n.id === "sub");
    expect(sub?.data.parallel).toBe(true); // target of a kind="parallel" edge
    expect(sub?.data.subworkflow?.name).toBe("inner");
    const items = doc.nodes.find((n) => n.id === "items");
    expect(items?.data.subworkflow).toBeUndefined();
  });
});

describe("execution status + error propagation", () => {
  // The backend execution workflow.json shape: tasks keyed by `task_id`, each
  // carrying a runtime `status` and (on failure) an `error` message.
  const execWorkflowJson = JSON.stringify({
    task_configs: [
      { task_id: "task_a", task_type: "fn", status: "completed" },
      {
        task_id: "task_b",
        task_type: "fn",
        status: "failed",
        error: "RuntimeError: boom in task_b: x=1",
      },
      { task_id: "task_c", task_type: "fn", status: "pending" },
    ],
    links: [
      { source: "task_a", target: "task_b", status: "failed" },
      { source: "task_b", target: "task_c", status: "failed" },
    ],
  });

  it("carries per-task status + failure error from the execution IR to the node data", () => {
    const ir = parseTaskGraphIr(execWorkflowJson) as TaskGraphJson;
    const b = ir.task_configs.find((t) => t.id === "task_b");
    expect(b?.status).toBe("failed");
    expect(b?.error).toBe("RuntimeError: boom in task_b: x=1");

    const doc = buildFlowgramDocument(ir);
    const byId = new Map(doc.nodes.map((n) => [n.id, n]));
    expect(byId.get("task_a")?.data.status).toBe("completed");
    expect(byId.get("task_b")?.data.status).toBe("failed");
    expect(byId.get("task_b")?.data.error).toBe("RuntimeError: boom in task_b: x=1");
    expect(byId.get("task_c")?.data.status).toBe("pending");
    // A succeeded/pending node carries no error.
    expect(byId.get("task_a")?.data.error).toBeUndefined();
    expect(byId.get("task_c")?.data.error).toBeUndefined();
  });
});

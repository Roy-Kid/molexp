import { describe, expect, it } from "@rstest/core";
import type { TaskGraphJson } from "@/components/workflow/task-graph-ir";
import {
  buildFlowgramDocument,
  flowgramDocToTaskGraphJson,
  taskGraphToWireDocument,
} from "../flowgram-document";

const tg: TaskGraphJson = {
  name: "demo",
  task_configs: [
    { id: "a", type: "core.constant", position: { x: 0, y: 0 }, config: { value: 1 } },
    { id: "b", type: "core.add", position: { x: 0, y: 140 }, config: {} },
  ],
  links: [{ from: "a", to: "b", kind: "data" }],
};

const pickNode = (t: { id: string; type: string; config?: unknown; position?: unknown }) => ({
  id: t.id,
  type: t.type,
  config: t.config ?? {},
  position: t.position,
});
const pickEdge = (l: { from: string; to: string; kind?: string }) => ({
  from: l.from,
  to: l.to,
  kind: l.kind,
});

describe("flowgram reverse serializer", () => {
  it("round-trips buildFlowgramDocument ∘ reverse on id/type/config/position/edge kind", () => {
    const back = flowgramDocToTaskGraphJson(buildFlowgramDocument(tg));
    expect(back.task_configs.map(pickNode)).toEqual(tg.task_configs.map(pickNode));
    expect(back.links.map(pickEdge)).toEqual([{ from: "a", to: "b", kind: "data" }]);
  });

  it("returns the canonical TaskGraphJson shape (sole IR type)", () => {
    const back = flowgramDocToTaskGraphJson(buildFlowgramDocument(tg));
    expect(Array.isArray(back.task_configs)).toBe(true);
    expect(Array.isArray(back.links)).toBe(true);
  });

  it("taskGraphToWireDocument emits the backend wire field names", () => {
    const wire = taskGraphToWireDocument(tg) as {
      task_configs: Record<string, unknown>[];
      links: Record<string, unknown>[];
      entries: unknown[];
    };
    expect(wire.task_configs[0]).toMatchObject({
      task_id: "a",
      task_type: "core.constant",
      status: "pending",
      position: { x: 0, y: 0 },
    });
    expect(wire.links[0]).toMatchObject({ source: "a", target: "b", kind: "data" });
    expect(wire.entries).toEqual([]);
  });
});

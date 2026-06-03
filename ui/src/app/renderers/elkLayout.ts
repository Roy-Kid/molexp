import type { Edge, Node } from "@xyflow/react";
import ELK from "elkjs/lib/elk.bundled.js";

/**
 * ELK (Eclipse Layout Kernel) layered layout — the layouter xyflow's own
 * examples use for complex DAGs. Unlike dagre (node positions only), ELK does
 * proper ORTHOGONAL edge routing, returning bend points so fan-in edges route
 * cleanly *around* nodes instead of crossing through them.
 */
const elk = new ELK();

export interface ElkPoint {
  x: number;
  y: number;
}

export interface ElkLayoutResult<N extends Node> {
  nodes: N[];
  /** edge id → polyline points (start, …bends, end) in flow coordinates. */
  edgePoints: Record<string, ElkPoint[]>;
}

const LAYOUT_OPTIONS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.direction": "DOWN",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
  "elk.layered.spacing.nodeNodeBetweenLayers": "70",
  "elk.layered.spacing.edgeNodeBetweenLayers": "30",
  "elk.spacing.nodeNode": "45",
  "elk.spacing.edgeEdge": "18",
  "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
};

// Fixed node footprint — MUST match the rendered node size so ELK's edge
// endpoints land exactly on the node borders (otherwise arrows float).
export const ELK_NODE_WIDTH = 200;
export const ELK_NODE_HEIGHT = 88;

export const layoutWithElk = async <N extends Node>(
  nodes: N[],
  edges: Edge[],
  options: { nodeWidth?: number; nodeHeight?: number } = {},
): Promise<ElkLayoutResult<N>> => {
  const { nodeWidth = ELK_NODE_WIDTH, nodeHeight = ELK_NODE_HEIGHT } = options;
  if (nodes.length === 0) return { nodes, edgePoints: {} };

  const graph = {
    id: "root",
    layoutOptions: LAYOUT_OPTIONS,
    children: nodes.map((n) => ({ id: n.id, width: nodeWidth, height: nodeHeight })),
    edges: edges
      .filter((e) => e.source && e.target)
      .map((e) => ({ id: e.id, sources: [e.source], targets: [e.target] })),
  };

  const laidOut = await elk.layout(graph);

  const posById = new Map((laidOut.children ?? []).map((c) => [c.id, c]));
  const positionedNodes = nodes.map((node) => {
    const c = posById.get(node.id);
    return c ? { ...node, position: { x: c.x ?? 0, y: c.y ?? 0 } } : node;
  });

  // Full routed polyline: start → bends → end. With node sizes matched to the
  // rendered nodes, ELK's start/end land on the node borders, so the line and
  // its arrowhead connect cleanly.
  const edgePoints: Record<string, ElkPoint[]> = {};
  for (const e of laidOut.edges ?? []) {
    const section = e.sections?.[0];
    if (!section) continue;
    edgePoints[e.id] = [
      section.startPoint,
      ...(section.bendPoints ?? []),
      section.endPoint,
    ].map((p) => ({ x: p.x, y: p.y }));
  }

  return { nodes: positionedNodes, edgePoints };
};

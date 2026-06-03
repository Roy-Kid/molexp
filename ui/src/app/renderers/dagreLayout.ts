import dagre from "@dagrejs/dagre";
import type { Edge, Node } from "@xyflow/react";

/**
 * Lay out a DAG with dagre so workflow graphs read as a clean top-down flow
 * (proper layering + crossing minimization) instead of the hand-rolled
 * level-grid that tangled fan-in edges into a "net".
 */
export const layoutWithDagre = <N extends Node>(
  nodes: N[],
  edges: Edge[],
  options: { nodeWidth?: number; nodeHeight?: number; rankdir?: "TB" | "LR" } = {},
): N[] => {
  const { nodeWidth = 210, nodeHeight = 104, rankdir = "TB" } = options;
  const g = new dagre.graphlib.Graph();
  // Generous separation + tight-tree ranking so long fan-in edges (residues →
  // both tleap passes) get routed through real gaps instead of crossing nodes.
  g.setGraph({
    rankdir,
    nodesep: 70,
    ranksep: 120,
    edgesep: 30,
    ranker: "tight-tree",
    marginx: 24,
    marginy: 24,
  });
  g.setDefaultEdgeLabel(() => ({}));

  for (const node of nodes) {
    g.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  }
  for (const edge of edges) {
    if (g.hasNode(edge.source) && g.hasNode(edge.target)) {
      g.setEdge(edge.source, edge.target);
    }
  }

  dagre.layout(g);

  return nodes.map((node) => {
    const pos = g.node(node.id);
    if (!pos) return node;
    // dagre returns node centers; ReactFlow wants top-left.
    return { ...node, position: { x: pos.x - nodeWidth / 2, y: pos.y - nodeHeight / 2 } };
  });
};

import type { Edge, Node } from "@xyflow/react";
import type { EdgeJson, TaskGraphJson, TaskNodeJson } from "@/types/task_graph_ir";

/**
 * Converts TaskGraph JSON IR to React Flow elements with auto-layout.
 * Uses a simple layer-based layout algorithm (Top-Down).
 */
export const getLayoutedElements = (nodes: TaskNodeJson[], edges: EdgeJson[]) => {
  // Convert TaskNodeJson to generic format for layout
  const genericNodes = nodes.map((n) => ({ id: n.id, type: n.type }));
  const genericEdges = edges.map((e) => ({ from: e.from, to: e.to }));

  const layout = calculateLayout(genericNodes, genericEdges);

  // Map back to React Flow elements
  const flowNodes: Node[] = nodes.map((node) => {
    const pos = layout[node.id] || { x: 0, y: 0 };

    // Map type to React Flow type
    let type = "process";
    const lowerType = node.type.toLowerCase();
    if (lowerType.includes("start") || lowerType.includes("load")) type = "start";
    else if (lowerType.includes("end") || lowerType.includes("save")) type = "end";

    return {
      id: node.id,
      type,
      position: pos,
      data: {
        label: node.label || node.type,
        ...node.params,
      },
    };
  });

  const flowEdges: Edge[] = edges.map((edge) => ({
    id: `e_${edge.from}_${edge.to}`,
    source: edge.from,
    target: edge.to,
    animated: true,
    type: "smoothstep",
  }));

  return { nodes: flowNodes, edges: flowEdges };
};

/**
 * Auto-layouts existing React Flow nodes and edges.
 */
export const autoLayoutNodes = (nodes: Node[], edges: Edge[]) => {
  const genericNodes = nodes.map((n) => ({ id: n.id, type: n.type || "process" }));
  const genericEdges = edges.map((e) => ({ from: e.source, to: e.target }));

  const layout = calculateLayout(genericNodes, genericEdges);

  const layoutedNodes = nodes.map((node) => ({
    ...node,
    position: layout[node.id] || node.position,
  }));

  return { nodes: layoutedNodes, edges };
};

// Shared layout logic
const calculateLayout = (nodes: { id: string }[], edges: { from: string; to: string }[]) => {
  const rankSep = 250; // Increased from 150
  const nodeSep = 300; // Increased from 200

  const adj: Record<string, string[]> = {};
  const inDegree: Record<string, number> = {};

  nodes.forEach((n) => {
    adj[n.id] = [];
    inDegree[n.id] = 0;
  });

  edges.forEach((e) => {
    if (adj[e.from]) adj[e.from].push(e.to);
    inDegree[e.to] = (inDegree[e.to] || 0) + 1;
  });

  const levels: Record<string, number> = {};
  const queue: string[] = nodes.filter((n) => inDegree[n.id] === 0).map((n) => n.id);

  queue.forEach((id) => {
    levels[id] = 0;
  });

  const tempInDegree = { ...inDegree };
  const processQueue = [...queue];

  while (processQueue.length > 0) {
    const u = processQueue.shift();
    if (u === undefined) break;
    if (adj[u]) {
      adj[u].forEach((v) => {
        levels[v] = Math.max(levels[v] || 0, (levels[u] || 0) + 1);
        tempInDegree[v]--;
        if (tempInDegree[v] === 0) {
          processQueue.push(v);
        }
      });
    }
  }

  // Handle disconnected/cycles
  nodes.forEach((n) => {
    if (levels[n.id] === undefined) levels[n.id] = 0;
  });

  const levelGroups: Record<number, string[]> = {};
  let maxLevel = 0;
  Object.entries(levels).forEach(([id, level]) => {
    if (!levelGroups[level]) levelGroups[level] = [];
    levelGroups[level].push(id);
    maxLevel = Math.max(maxLevel, level);
  });

  const maxNodesInLevel = Math.max(...Object.values(levelGroups).map((g) => g.length));
  const totalWidth = maxNodesInLevel * nodeSep;

  const layout: Record<string, { x: number; y: number }> = {};

  nodes.forEach((node) => {
    const level = levels[node.id];
    const indexInLevel = levelGroups[level].indexOf(node.id);
    const nodesInThisLevel = levelGroups[level].length;

    const levelWidth = nodesInThisLevel * nodeSep;
    const xOffset = (totalWidth - levelWidth) / 2;

    const x = xOffset + indexInLevel * nodeSep + 50;
    const y = level * rankSep + 50;

    layout[node.id] = { x, y };
  });

  return layout;
};

/**
 * Converts React Flow nodes and edges to TaskGraph JSON IR.
 */
export const toTaskGraphJson = (
  nodes: Node[],
  edges: Edge[],
  name: string = "Workflow",
): TaskGraphJson => {
  const taskNodes: TaskNodeJson[] = nodes.map((node) => {
    // Extract config from node data (static configuration)
    const { config, label, category, isOutput, status, plannedColor, ...otherData } = node.data;
    const configRecord =
      typeof config === "object" && config !== null && !Array.isArray(config)
        ? (config as Record<string, unknown>)
        : ({} as Record<string, unknown>);

    return {
      id: node.id,
      type:
        node.type === "process"
          ? category
            ? `${category}.${label}`
            : (label as string)
          : node.type || "process",
      label: label as string,
      config: configRecord, // Static configuration for the node
      params: otherData, // Other runtime parameters (if any)
      metadata: {
        position: node.position,
        isOutput,
        ...otherData,
      },
    };
  });

  const taskEdges: EdgeJson[] = edges.map((edge) => ({
    from: edge.source,
    to: edge.target,
    kind: "dependency",
  }));

  const targets = nodes.filter((n) => n.data.isOutput).map((n) => n.id);

  return {
    name,
    nodes: taskNodes,
    edges: taskEdges,
    targets: targets.length > 0 ? targets : undefined,
    metadata: {
      createdAt: new Date().toISOString(),
    },
  };
};

/**
 * Infers the minimal subgraph required to execute the given targets.
 * Returns a topologically sorted list of node IDs.
 */
export const planExecution = (nodes: Node[], edges: Edge[], targets: string[]): string[] => {
  // 1. Build dependency graph (reverse adjacency list)
  // node -> [dependencies]
  const deps: Record<string, string[]> = {};
  for (const n of nodes) {
    deps[n.id] = [];
  }
  edges.forEach((e) => {
    if (deps[e.target]) {
      deps[e.target].push(e.source);
    }
  });

  // 2. Collect all required nodes (transitive closure of dependencies for targets)
  const required = new Set<string>();
  const visit = (nodeId: string) => {
    if (required.has(nodeId)) return;
    required.add(nodeId);
    (deps[nodeId] || []).forEach(visit);
  };
  targets.forEach(visit);

  // 3. Topological Sort of the subgraph
  const sorted: string[] = [];
  const visited = new Set<string>();
  const tempVisited = new Set<string>();

  const sortVisit = (nodeId: string) => {
    if (tempVisited.has(nodeId)) {
      throw new Error(`Cycle detected involving node ${nodeId}`);
    }
    if (visited.has(nodeId)) return;

    tempVisited.add(nodeId);
    (deps[nodeId] || []).forEach((depId) => {
      if (required.has(depId)) {
        sortVisit(depId);
      }
    });
    tempVisited.delete(nodeId);
    visited.add(nodeId);
    sorted.push(nodeId);
  };

  // Sort only the required nodes
  required.forEach((nodeId) => {
    if (!visited.has(nodeId)) {
      sortVisit(nodeId);
    }
  });

  return sorted;
};

/**
 * Analyzes paths for each target to support multi-color visualization.
 * Returns a map of Node/Edge ID -> List of Target IDs that depend on it.
 */
export const analyzePaths = (
  nodes: Node[],
  edges: Edge[],
  targets: string[],
): Record<string, string[]> => {
  // Build dependency graph
  const deps: Record<string, string[]> = {};
  const edgeDeps: Record<string, string[]> = {}; // node -> incoming edges

  for (const n of nodes) {
    deps[n.id] = [];
  }
  edges.forEach((e) => {
    if (!deps[e.target]) deps[e.target] = [];
    deps[e.target].push(e.source);

    if (!edgeDeps[e.target]) edgeDeps[e.target] = [];
    edgeDeps[e.target].push(e.id);
  });

  const itemToTargets: Record<string, string[]> = {};

  // Helper to add target to item
  const add = (itemId: string, targetId: string) => {
    if (!itemToTargets[itemId]) itemToTargets[itemId] = [];
    if (!itemToTargets[itemId].includes(targetId)) {
      itemToTargets[itemId].push(targetId);
    }
  };

  // For each target, traverse backwards
  targets.forEach((targetId) => {
    const visited = new Set<string>();
    const queue = [targetId];

    while (queue.length > 0) {
      const curr = queue.shift();
      if (curr === undefined) break;
      if (visited.has(curr)) continue;
      visited.add(curr);

      add(curr, targetId);

      // Add incoming edges to this path
      for (const edgeId of edgeDeps[curr] || []) {
        add(edgeId, targetId);
      }

      // Continue to dependencies
      for (const dep of deps[curr] || []) {
        queue.push(dep);
      }
    }
  });

  return itemToTargets;
};

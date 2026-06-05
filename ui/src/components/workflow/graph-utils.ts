import type { EdgeJson, TaskGraphJson, TaskNodeJson } from "@/components/workflow/task-graph-ir";

/**
 * workflow-utils — pure graph algorithms over the canonical task-graph IR.
 *
 * Decoupled from any rendering library: callers pass plain `{id, type, data,
 * position}` nodes and `{source, target}` edges (the minimal shape every graph
 * surface can produce) rather than xyflow `Node`/`Edge`.
 */

/** Minimal node shape consumed by the graph utilities (no rendering deps). */
export interface GraphNodeInput {
  id: string;
  type?: string;
  position: { x: number; y: number };
  data: Record<string, unknown>;
}

/** Minimal edge shape consumed by the graph utilities (no rendering deps). */
export interface GraphEdgeInput {
  id: string;
  source: string;
  target: string;
  /** Typed edge kind; defaults to "dependency" when absent. */
  kind?: string;
}

/**
 * Converts graph nodes and edges to the canonical TaskGraph JSON IR
 * (`task_configs` + typed `links` + per-node `position`).
 */
export const toTaskGraphJson = (
  nodes: GraphNodeInput[],
  edges: GraphEdgeInput[],
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
            ? `${String(category)}.${String(label)}`
            : (label as string)
          : node.type || "process",
      label: label as string,
      position: node.position,
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
    kind: edge.kind ?? "dependency",
  }));

  const targets = nodes.filter((n) => n.data.isOutput).map((n) => n.id);

  return {
    name,
    task_configs: taskNodes,
    links: taskEdges,
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
export const planExecution = (
  nodes: GraphNodeInput[],
  edges: GraphEdgeInput[],
  targets: string[],
): string[] => {
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
  nodes: GraphNodeInput[],
  edges: GraphEdgeInput[],
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

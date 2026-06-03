/**
 * flowgram-document — the SOLE forward builder turning the canonical task-graph
 * IR ({@link TaskGraphJson}) into a flowgram free-layout document.
 *
 * This module is pure (no React / no flowgram runtime import) so it can be unit
 * tested and imported by both the canvas components and the data layer without
 * pulling the editor into a node test runner.
 *
 * Document shape (matches `@flowgram.ai/free-layout-editor`'s `WorkflowJSON`):
 *   - node: `{ id, type, meta: { position: { x, y } }, data: { ... } }`
 *   - edge: `{ sourceNodeID, targetNodeID }`
 *
 * Rules:
 *   - one document node per `task_config`;
 *   - one document edge per VALID link (both endpoints known) — links to unknown
 *     ids are dropped;
 *   - when a node lacks an explicit `position`, fall back to a deterministic
 *     layered layout (longest-path level by in-degree);
 *   - cyclic IR never throws — residual nodes still receive a position.
 */

import type { TaskGraphJson, TaskNodeJson } from "@/types/task_graph_ir";

export interface FlowgramNodeData extends Record<string, unknown> {
  /** Display title (the task id). */
  title: string;
  /** Secondary line (the task type). */
  subtitle: string;
  taskId: string;
  taskType: string;
  config?: Record<string, unknown>;
}

export interface FlowgramNode {
  id: string;
  type: string;
  meta: { position: { x: number; y: number } };
  data: FlowgramNodeData;
}

export interface FlowgramEdge {
  sourceNodeID: string;
  targetNodeID: string;
}

export interface FlowgramDocument {
  nodes: FlowgramNode[];
  edges: FlowgramEdge[];
}

// Deterministic fallback-layout spacing (top→bottom DAG flow).
const RANK_SEP = 140;
const NODE_SEP = 220;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * Parse a serialized workflow IR string (an experiment's `workflow_source` / a
 * run's `workflowSource`, produced by `Workflow.to_dict()`) into the canonical
 * {@link TaskGraphJson}. Returns `null` when the string is absent, not JSON, or
 * not a `{task_configs, links}` payload — so callers fall back to the raw text.
 *
 * The backend emits tasks as `{task_id, task_type, config, ...}` and links as
 * `{source, target, ...}`; this normalizes them to the client IR field names
 * (`id` / `type`, `from` / `to`).
 */
export const parseTaskGraphIr = (source: string | null | undefined): TaskGraphJson | null => {
  if (!source) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(source);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  if (!Array.isArray(parsed.task_configs) || !Array.isArray(parsed.links)) return null;
  return normalizeTaskGraph(parsed);
};

/**
 * Normalize a raw `{task_configs, links}` object (backend or client field
 * names) into a {@link TaskGraphJson}. Tolerant: it accepts both `id`/`task_id`
 * and `type`/`task_type` for nodes and both `from`/`to` and `source`/`target`
 * for links.
 */
export const normalizeTaskGraph = (raw: Record<string, unknown>): TaskGraphJson => {
  const rawTasks = Array.isArray(raw.task_configs) ? raw.task_configs : [];
  const rawLinks = Array.isArray(raw.links) ? raw.links : [];

  const task_configs: TaskNodeJson[] = [];
  for (const item of rawTasks) {
    if (!isRecord(item)) continue;
    const id =
      typeof item.id === "string"
        ? item.id
        : typeof item.task_id === "string"
          ? item.task_id
          : null;
    if (!id) continue;
    const type =
      typeof item.type === "string"
        ? item.type
        : typeof item.task_type === "string"
          ? item.task_type
          : id;
    const position =
      isRecord(item.position) &&
      typeof item.position.x === "number" &&
      typeof item.position.y === "number"
        ? { x: item.position.x, y: item.position.y }
        : undefined;
    task_configs.push({
      id,
      type,
      label: typeof item.label === "string" ? item.label : undefined,
      position,
      config: isRecord(item.config) ? item.config : undefined,
    });
  }

  const links = [];
  for (const item of rawLinks) {
    if (!isRecord(item)) continue;
    const from =
      typeof item.from === "string"
        ? item.from
        : typeof item.source === "string"
          ? item.source
          : null;
    const to =
      typeof item.to === "string" ? item.to : typeof item.target === "string" ? item.target : null;
    if (!from || !to) continue;
    links.push({ from, to, kind: typeof item.kind === "string" ? item.kind : undefined });
  }

  return {
    name: typeof raw.name === "string" ? raw.name : undefined,
    task_configs,
    links,
    metadata: isRecord(raw.metadata) ? raw.metadata : undefined,
  };
};

/**
 * Deterministic layered layout by longest-path level (in-degree). Cycles leave
 * residual nodes at level 0 — never throws. Returns one position per node id.
 */
const fallbackLayout = (
  ids: string[],
  links: { from: string; to: string }[],
): Map<string, { x: number; y: number }> => {
  const idSet = new Set(ids);
  const adjacency = new Map<string, string[]>(ids.map((id) => [id, []]));
  const indegree = new Map<string, number>(ids.map((id) => [id, 0]));
  for (const link of links) {
    if (!idSet.has(link.from) || !idSet.has(link.to)) continue;
    adjacency.get(link.from)?.push(link.to);
    indegree.set(link.to, (indegree.get(link.to) ?? 0) + 1);
  }

  const level = new Map<string, number>();
  const pending = new Map(indegree);
  const queue = ids.filter((id) => (pending.get(id) ?? 0) === 0);
  for (const id of queue) level.set(id, 0);
  while (queue.length) {
    const id = queue.shift() as string;
    const lv = level.get(id) ?? 0;
    for (const next of adjacency.get(id) ?? []) {
      level.set(next, Math.max(level.get(next) ?? 0, lv + 1));
      pending.set(next, (pending.get(next) ?? 0) - 1);
      if ((pending.get(next) ?? 0) === 0) queue.push(next);
    }
  }

  const perLevel = new Map<number, number>();
  const positions = new Map<string, { x: number; y: number }>();
  for (const id of ids) {
    const lv = level.get(id) ?? 0; // residual (cyclic) nodes stay at level 0
    const col = perLevel.get(lv) ?? 0;
    perLevel.set(lv, col + 1);
    positions.set(id, { x: col * NODE_SEP, y: lv * RANK_SEP });
  }
  return positions;
};

/**
 * Build a flowgram free-layout document from the canonical task-graph IR.
 * Never throws.
 */
export const buildFlowgramDocument = (ir: TaskGraphJson): FlowgramDocument => {
  const tasks = Array.isArray(ir.task_configs) ? ir.task_configs : [];
  const links = Array.isArray(ir.links) ? ir.links : [];

  // Dedup by id (first wins), preserving order.
  const byId = new Map<string, TaskNodeJson>();
  const ids: string[] = [];
  for (const task of tasks) {
    if (typeof task?.id !== "string" || byId.has(task.id)) continue;
    byId.set(task.id, task);
    ids.push(task.id);
  }

  const fallback = fallbackLayout(ids, links);

  const nodes: FlowgramNode[] = ids.map((id) => {
    const task = byId.get(id) as TaskNodeJson;
    const position = task.position ?? fallback.get(id) ?? { x: 0, y: 0 };
    return {
      id,
      type: task.type || "task",
      meta: { position: { x: position.x, y: position.y } },
      data: {
        title: id,
        subtitle: task.label ?? task.type ?? id,
        taskId: id,
        taskType: task.type ?? id,
        config: task.config,
      },
    };
  });

  const edges: FlowgramEdge[] = [];
  for (const link of links) {
    if (!byId.has(link.from) || !byId.has(link.to)) continue; // drop invalid links
    edges.push({ sourceNodeID: link.from, targetNodeID: link.to });
  }

  return { nodes, edges };
};

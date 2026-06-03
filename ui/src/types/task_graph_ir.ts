/**
 * task_graph_ir — the SOLE client-side workflow IR type.
 *
 * Aligned with the backend serialized-IR shape (spec 01): a workflow is a flat
 * list of `task_configs` plus `links` carrying a `kind`, with each task able to
 * declare an explicit canvas `position`. Every workflow surface in the UI
 * (canvas viewers, the inline plan preview, the forward flowgram-document
 * builder) consumes this one type — there is no second `{nodes, edges}` IR.
 */

export interface TaskNodePosition {
  x: number;
  y: number;
}

export interface TaskNodeJson {
  /** Stable task identifier (the DAG node id). */
  id: string;
  /** Task type / kind (e.g. `molcrafts.typify`). */
  type: string;
  /** Optional human-facing label; defaults to `type` when absent. */
  label?: string;
  /** Explicit canvas position; absent → deterministic fallback layout. */
  position?: TaskNodePosition;
  /** Static node configuration / inputs carried from the IR. */
  config?: Record<string, unknown>;
  /** Execution status (`pending` / `running` / `completed` / `failed` / `skipped`). */
  status?: string;
  /** Runtime parameters (if any). */
  params?: Record<string, unknown>;
  /** Free-form per-node metadata. */
  metadata?: Record<string, unknown>;
}

export interface EdgeJson {
  /** Source task id. */
  from: string;
  /** Target task id. */
  to: string;
  /** Link semantics — e.g. `data` / `control` / `dependency`. */
  kind?: string;
}

export interface TaskGraphJson {
  name?: string;
  task_configs: TaskNodeJson[];
  links: EdgeJson[];
  targets?: string[];
  metadata?: Record<string, unknown>;
}

/**
 * WorkflowGraph — inline ReactFlow renderer for a molexp workflow IR.
 *
 * Renders a workflow IR as the visual half of an ``exit_plan_mode``
 * handoff: the agent submits the IR, this component lays it out (via the
 * project's existing :func:`getLayoutedElements` helper) and draws it with
 * xyflow so the user can see the proposed topology.
 *
 * Why xyflow rather than custom SVG: ``WorkflowGraphViewer`` and
 * ``WorkflowFileViewer`` already use ``@xyflow/react`` for every other
 * workflow surface in molexp. Reusing the same library keeps the bundle
 * small (xyflow is already vendored) and gives the plan preview the
 * same look + interactions (zoom, pan, fit-to-view) users see in the
 * full workflow viewer.
 */

import type { Edge, Node, NodeProps, NodeTypes } from "@xyflow/react";
import { Background, Controls, Handle, MarkerType, Position, ReactFlow } from "@xyflow/react";
// xyflow's stylesheet is imported once at the app entry (see index.tsx).
import { type JSX, useMemo } from "react";
import { layoutWithDagre } from "@/app/renderers/dagreLayout";

interface TaskConfig {
  task_id: string;
  task_type: string;
  config?: Record<string, unknown>;
}

interface Link {
  source: string;
  target: string;
}

export interface WorkflowIR {
  name?: string;
  task_configs: TaskConfig[];
  links: Link[];
  metadata?: Record<string, unknown>;
}

/**
 * Parse a serialized workflow IR string (an experiment's `workflow_source` /
 * a run's `workflowSource`, produced by `Workflow.to_dict()`) into a
 * {@link WorkflowIR}. Returns `null` when the string is absent, not JSON, or
 * not a `{task_configs, links}` payload — so callers fall back to showing the
 * raw text instead of a broken graph.
 */
export const parseWorkflowIr = (source: string | null | undefined): WorkflowIR | null => {
  if (!source) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(source);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) return null;
  const ir = parsed as Partial<WorkflowIR>;
  if (!Array.isArray(ir.task_configs) || !Array.isArray(ir.links)) return null;
  return {
    name: typeof ir.name === "string" ? ir.name : undefined,
    task_configs: ir.task_configs,
    links: ir.links,
    metadata: ir.metadata,
  };
};

interface PlanNodeData extends Record<string, unknown> {
  taskId: string;
  taskType: string;
}

type PlanFlowNode = Node<PlanNodeData, "planNode">;

const PlanNode = ({ data }: NodeProps<PlanFlowNode>): JSX.Element => (
  <div className="rounded-md border border-violet-300 bg-card px-3 py-2 shadow-sm min-w-[140px] cursor-pointer transition-colors hover:border-violet-500 dark:border-violet-700">
    <Handle type="target" position={Position.Top} className="h-2 w-2 bg-violet-400" />
    <p className="text-xs font-semibold text-foreground truncate">{data.taskId}</p>
    <p className="font-mono text-[10px] text-muted-foreground truncate">[{data.taskType}]</p>
    <Handle type="source" position={Position.Bottom} className="h-2 w-2 bg-violet-400" />
  </div>
);

const NODE_TYPES: NodeTypes = { planNode: PlanNode };

interface WorkflowGraphProps {
  ir: WorkflowIR;
  /** Optional fixed height for the inline render area (default 280px). */
  height?: number;
  className?: string;
  /** Called with the task id when a node is clicked. */
  onNodeClick?: (taskId: string) => void;
}

export const WorkflowGraph = ({
  ir,
  height = 280,
  className,
  onNodeClick,
}: WorkflowGraphProps): JSX.Element => {
  const { nodes, edges, invalidLinks } = useMemo(() => buildElements(ir), [ir]);

  if (nodes.length === 0) {
    return (
      <p
        className={
          "rounded-md border border-dashed border-border/60 bg-muted/10 px-3 py-2 text-xs italic text-muted-foreground " +
          (className ?? "")
        }
      >
        Empty workflow — no tasks declared.
      </p>
    );
  }

  return (
    <div className={className}>
      <div className="overflow-hidden rounded-md border border-border/60" style={{ height }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.2}
          maxZoom={1.5}
          proOptions={{ hideAttribution: true }}
          onNodeClick={onNodeClick ? (_event, node) => onNodeClick(node.id) : undefined}
        >
          <Background gap={16} size={1} />
          <Controls
            showInteractive={false}
            position="bottom-right"
            className="scale-75 origin-bottom-right"
          />
        </ReactFlow>
      </div>
      {invalidLinks.length > 0 && (
        <p className="mt-1 text-[10px] text-amber-600 dark:text-amber-400">
          {invalidLinks.length} link
          {invalidLinks.length === 1 ? "" : "s"} reference {invalidLinks.length === 1 ? "an" : ""}{" "}
          unknown task {invalidLinks.length === 1 ? "id" : "ids"} and were skipped.
        </p>
      )}
    </div>
  );
};

export interface BuiltElements {
  nodes: PlanFlowNode[];
  edges: Edge[];
  invalidLinks: Link[];
}

export const buildElements = (ir: WorkflowIR): BuiltElements => {
  const taskById = new Map<string, TaskConfig>();
  for (const task of ir.task_configs) {
    if (!taskById.has(task.task_id)) {
      taskById.set(task.task_id, task);
    }
  }

  const validLinks: Link[] = [];
  const invalidLinks: Link[] = [];
  for (const link of ir.links) {
    if (taskById.has(link.source) && taskById.has(link.target)) {
      validLinks.push(link);
    } else {
      invalidLinks.push(link);
    }
  }

  const rawNodes: PlanFlowNode[] = ir.task_configs.map((t) => ({
    id: t.task_id,
    type: "planNode",
    position: { x: 0, y: 0 },
    data: {
      taskId: t.task_id,
      taskType: t.task_type,
    },
  }));

  const edges: Edge[] = validLinks.map((link, idx) => ({
    id: `e-${link.source}-${link.target}-${idx}`,
    source: link.source,
    target: link.target,
    type: "smoothstep",
    style: { stroke: "#a78bfa", strokeWidth: 1.5 },
    markerEnd: { type: MarkerType.ArrowClosed, color: "#a78bfa" },
  }));

  const nodes = layoutWithDagre(rawNodes, edges, { nodeWidth: 150, nodeHeight: 64 });
  return { nodes, edges, invalidLinks };
};

// Layout is delegated to dagre (see :func:`layoutWithDagre`).

import type { Edge, Node, NodeProps, NodeTypes } from "@xyflow/react";
import {
  Background,
  ControlButton,
  Controls,
  MarkerType,
  Position,
  ReactFlow,
} from "@xyflow/react";
import { useEffect, useMemo, useState } from "react";
import "@xyflow/react/dist/style.css";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps, SemanticStatus } from "@/app/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

interface WorkflowFileNode {
  task_id: string;
  task_type: string;
  config: Record<string, unknown>;
  status: SemanticStatus;
}

interface WorkflowFileLink {
  source: string;
  target: string;
  status: SemanticStatus;
}

interface WorkflowFilePayload {
  workflow_id: string;
  name?: string | null;
  task_configs: WorkflowFileNode[];
  links: WorkflowFileLink[];
}

interface WorkflowNodeData extends Record<string, unknown> {
  label: string;
  nodeType: "task";
  status: SemanticStatus;
  description: string;
}

type WorkflowFlowNode = Node<WorkflowNodeData, "workflowNode">;

const statusStyles: Record<SemanticStatus, { border: string; background: string; edge: string }> = {
  active: { border: "#10b981", background: "#ecfdf5", edge: "#10b981" },
  archived: { border: "#64748b", background: "#f1f5f9", edge: "#64748b" },
  draft: { border: "#f59e0b", background: "#fffbeb", edge: "#f59e0b" },
  pending: { border: "#94a3b8", background: "#f8fafc", edge: "#94a3b8" },
  running: { border: "#3b82f6", background: "#eff6ff", edge: "#3b82f6" },
  succeeded: { border: "#16a34a", background: "#f0fdf4", edge: "#16a34a" },
  failed: { border: "#ef4444", background: "#fef2f2", edge: "#ef4444" },
  cancelled: { border: "#6b7280", background: "#f3f4f6", edge: "#6b7280" },
  skipped: { border: "#d97706", background: "#fffbeb", edge: "#d97706" },
};

const WorkflowNode = ({ data }: NodeProps<WorkflowFlowNode>): JSX.Element => {
  const style = statusStyles[data.status] ?? statusStyles.pending;
  return (
    <div
      className="rounded-md border px-3 py-2 shadow-sm"
      style={{ borderColor: style.border, backgroundColor: style.background }}
    >
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {data.nodeType}
      </p>
      <p className="text-sm font-semibold text-foreground">{data.label}</p>
      <p className="text-xs text-muted-foreground">{data.description}</p>
    </div>
  );
};

const layoutNodes = (tasks: WorkflowFileNode[], links: WorkflowFileLink[]): WorkflowFlowNode[] => {
  const columnWidth = 260;
  const rowHeight = 150;

  const inDegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();

  tasks.forEach((task) => {
    inDegree.set(task.task_id, 0);
    adjacency.set(task.task_id, []);
  });

  links.forEach((link) => {
    if (!adjacency.has(link.source)) {
      adjacency.set(link.source, []);
    }
    adjacency.get(link.source)?.push(link.target);
    inDegree.set(link.target, (inDegree.get(link.target) ?? 0) + 1);
  });

  const queue = tasks
    .filter((task) => (inDegree.get(task.task_id) ?? 0) === 0)
    .map((task) => task.task_id);
  const levels = new Map<string, number>();
  for (const id of queue) {
    levels.set(id, 0);
  }

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current) {
      break;
    }
    const currentLevel = levels.get(current) ?? 0;
    for (const neighbor of adjacency.get(current) ?? []) {
      const nextLevel = Math.max(levels.get(neighbor) ?? 0, currentLevel + 1);
      levels.set(neighbor, nextLevel);
      inDegree.set(neighbor, (inDegree.get(neighbor) ?? 0) - 1);
      if ((inDegree.get(neighbor) ?? 0) <= 0) {
        queue.push(neighbor);
      }
    }
  }

  const layers = new Map<number, WorkflowFileNode[]>();
  tasks.forEach((task) => {
    const level = levels.get(task.task_id) ?? 0;
    const layer = layers.get(level) ?? [];
    layer.push(task);
    layers.set(level, layer);
  });

  return Array.from(layers.entries())
    .sort(([a], [b]) => a - b)
    .flatMap(([level, layerTasks]) =>
      layerTasks.map((task, index) => ({
        id: task.task_id,
        type: "workflowNode",
        position: { x: index * columnWidth, y: level * rowHeight },
        sourcePosition: Position.Bottom,
        targetPosition: Position.Top,
        data: {
          label: task.task_type || task.task_id,
          nodeType: "task",
          status: task.status,
          description: task.task_id,
        },
      })),
    );
};

export const WorkflowFileViewer = ({ selection }: RendererProps): JSX.Element => {
  const [payload, setPayload] = useState<WorkflowFilePayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [_layoutVersion, setLayoutVersion] = useState(0);
  const [reactFlowInstance, setReactFlowInstance] = useState<{
    fitView: () => void;
  } | null>(null);

  useEffect(() => {
    if (selection.objectType !== "workspace-file") {
      return;
    }

    workspaceApi
      .getWorkspaceFileText(selection.objectId)
      .then((content) => {
        const parsed = JSON.parse(content) as WorkflowFilePayload;
        if (!parsed.task_configs || !parsed.links) {
          throw new Error("Invalid workflow.json payload");
        }
        const missingTaskStatus = parsed.task_configs.some((task) => !task.status);
        const missingLinkStatus = parsed.links.some((link) => !link.status);
        if (missingTaskStatus || missingLinkStatus) {
          throw new Error("workflow.json is missing status fields for nodes or links");
        }
        setPayload(parsed);
        setError(null);
        setLayoutVersion((prev) => prev + 1);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load workflow");
        setPayload(null);
      });
  }, [selection]);

  const nodes = useMemo<WorkflowFlowNode[]>(() => {
    if (!payload) {
      return [];
    }
    return layoutNodes(payload.task_configs, payload.links);
  }, [payload]);

  const edges = useMemo<Array<Edge>>(() => {
    if (!payload) {
      return [];
    }
    return payload.links.map((link) => {
      const style = statusStyles[link.status] ?? statusStyles.pending;
      return {
        id: `${link.source}:${link.target}`,
        source: link.source,
        target: link.target,
        animated: link.status === "running",
        style: { stroke: style.edge, strokeWidth: 2 },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: style.edge,
        },
      };
    });
  }, [payload]);

  const nodeTypes = useMemo<NodeTypes>(() => ({ workflowNode: WorkflowNode }), []);

  return (
    <Card className="flex h-full flex-col border-border/60 bg-background">
      <CardHeader className="space-y-2">
        <CardTitle className="text-lg font-semibold">
          {payload?.name ?? "Workflow Preview"}
        </CardTitle>
        <p className="text-sm text-muted-foreground">{selection.objectId}</p>
      </CardHeader>
      <Separator />
      <CardContent className="flex-1 pt-4">
        {error && <div className="text-sm text-destructive">{error}</div>}
        {!error && (
          <div className="flex h-full min-h-64 flex-1 flex-col rounded-md border border-border">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              fitView
              className="flex-1"
              onInit={(instance) => {
                setReactFlowInstance(instance);
              }}
            >
              <Background />
              <Controls>
                <ControlButton
                  onClick={() => {
                    setLayoutVersion((prev) => prev + 1);
                    reactFlowInstance?.fitView();
                  }}
                  title="Refresh layout"
                >
                  ⟳
                </ControlButton>
              </Controls>
            </ReactFlow>
            {nodes.length === 0 && (
              <div className="border-t border-border px-4 py-3 text-xs text-muted-foreground">
                No workflow nodes found in workflow.json.
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

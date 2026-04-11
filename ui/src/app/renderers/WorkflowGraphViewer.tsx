import type { Edge, Node, NodeProps, NodeTypes } from "@xyflow/react";
import { Background, Controls, Handle, MarkerType, Position, ReactFlow } from "@xyflow/react";
import { useEffect, useMemo, useState } from "react";
import "@xyflow/react/dist/style.css";
import type {
  RendererProps,
  SemanticStatus,
  WorkflowGraph,
  WorkflowNodeMetadata,
} from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface WorkflowNodeData extends Record<string, unknown> {
  label: string;
  nodeType: "task" | "input" | "output";
  status: SemanticStatus;
  description: string;
}

interface WorkflowEdgeWithStatus {
  id: string;
  source: string;
  target: string;
  label: string;
  status: SemanticStatus;
  animated?: boolean;
}

interface DisplayWorkflowGraph {
  nodes: WorkflowNodeMetadata[];
  edges: WorkflowEdgeWithStatus[];
}

interface FileWorkflowData {
  id: string;
  name?: string;
  graph: DisplayWorkflowGraph | null;
}

type WorkflowFlowNode = Node<WorkflowNodeData, "workflowNode">;

const STATUS_VALUES: readonly SemanticStatus[] = [
  "active",
  "archived",
  "draft",
  "pending",
  "running",
  "succeeded",
  "failed",
  "cancelled",
  "skipped",
];

const getSemanticStatus = (value: unknown): SemanticStatus => {
  if (typeof value === "string" && STATUS_VALUES.includes(value as SemanticStatus)) {
    return value as SemanticStatus;
  }
  return "pending";
};

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
};

const getStatusColor = (status: SemanticStatus): string => {
  switch (status) {
    case "succeeded":
      return "border-blue-500 text-blue-700";
    case "failed":
      return "border-red-500 text-red-700";
    case "running":
      return "border-green-500 text-green-700";
    case "skipped":
    case "cancelled":
      return "border-yellow-500 text-yellow-700";
    default:
      return "border-border text-foreground";
  }
};

const getEdgeColor = (status: SemanticStatus): string => {
  switch (status) {
    case "succeeded":
      return "#3b82f6";
    case "failed":
      return "#ef4444";
    case "running":
      return "#22c55e";
    case "skipped":
    case "cancelled":
      return "#eab308";
    default:
      return "#94a3b8";
  }
};

const WorkflowNode = ({ data }: NodeProps<WorkflowFlowNode>): JSX.Element => {
  const colorClass = getStatusColor(data.status);

  return (
    <div
      className={`rounded-md border-2 bg-background px-3 py-2 shadow-sm min-w-[150px] ${colorClass}`}
    >
      <Handle type="target" position={Position.Top} className="h-3 w-3 bg-muted-foreground" />
      <p className="text-xs font-semibold uppercase tracking-wide opacity-70">{data.nodeType}</p>
      <p className="text-sm font-bold">{data.label}</p>
      <p className="text-xs opacity-70 capitalize">{data.status}</p>
      <Handle type="source" position={Position.Bottom} className="h-3 w-3 bg-muted-foreground" />
    </div>
  );
};

const normalizeGraph = (graph: WorkflowGraph): DisplayWorkflowGraph => {
  return {
    nodes: graph.nodes.map((node) => ({
      nodeId: node.nodeId,
      label: node.label,
      nodeType: node.nodeType,
      status: node.status,
      description: node.description,
      position: node.position,
    })),
    edges: graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      status: "pending",
    })),
  };
};

const normalizeGraphFromUnknown = (value: unknown): DisplayWorkflowGraph | null => {
  const graph = asRecord(value);
  if (!graph) {
    return null;
  }

  const rawNodes = Array.isArray(graph.nodes) ? graph.nodes : null;
  if (!rawNodes) {
    return null;
  }

  const nodes: WorkflowNodeMetadata[] = rawNodes
    .map((item) => {
      const node = asRecord(item);
      if (!node) {
        return null;
      }
      const nodeId = typeof node.nodeId === "string" ? node.nodeId : null;
      if (!nodeId) {
        return null;
      }

      const nodeType =
        node.nodeType === "task" || node.nodeType === "input" || node.nodeType === "output"
          ? node.nodeType
          : "task";
      const positionRecord = asRecord(node.position);
      const x = typeof positionRecord?.x === "number" ? positionRecord.x : 0;
      const y = typeof positionRecord?.y === "number" ? positionRecord.y : 0;

      return {
        nodeId,
        label: typeof node.label === "string" ? node.label : nodeId,
        nodeType,
        status: getSemanticStatus(node.status),
        description: typeof node.description === "string" ? node.description : "",
        position: { x, y },
      };
    })
    .filter((node): node is WorkflowNodeMetadata => node !== null);

  const rawEdges = Array.isArray(graph.edges) ? graph.edges : [];
  const edges: WorkflowEdgeWithStatus[] = rawEdges.reduce<WorkflowEdgeWithStatus[]>(
    (acc, item, index) => {
      const edge = asRecord(item);
      if (!edge || typeof edge.source !== "string" || typeof edge.target !== "string") {
        return acc;
      }
      acc.push({
        id: typeof edge.id === "string" ? edge.id : `edge-${index}`,
        source: edge.source,
        target: edge.target,
        label: typeof edge.label === "string" ? edge.label : "",
        status: getSemanticStatus(edge.status),
        animated: edge.animated === true,
      });
      return acc;
    },
    [],
  );

  return { nodes, edges };
};

const graphFromTaskConfigPayload = (
  value: Record<string, unknown>,
): DisplayWorkflowGraph | null => {
  const tasks = Array.isArray(value.task_configs) ? value.task_configs : null;
  const links = Array.isArray(value.links) ? value.links : null;
  if (!tasks || !links) {
    return null;
  }

  const nodes: WorkflowNodeMetadata[] = tasks
    .map((item, index) => {
      const task = asRecord(item);
      if (!task || typeof task.task_id !== "string") {
        return null;
      }

      return {
        nodeId: task.task_id,
        label: typeof task.task_type === "string" ? task.task_type : task.task_id,
        nodeType: "task",
        status: getSemanticStatus(task.status),
        description: task.task_id,
        position: { x: (index % 4) * 220, y: Math.floor(index / 4) * 140 },
      };
    })
    .filter((node): node is WorkflowNodeMetadata => node !== null);

  const edges: WorkflowEdgeWithStatus[] = links.reduce<WorkflowEdgeWithStatus[]>(
    (acc, item, index) => {
      const link = asRecord(item);
      if (!link || typeof link.source !== "string" || typeof link.target !== "string") {
        return acc;
      }
      acc.push({
        id: typeof link.id === "string" ? link.id : `${link.source}:${link.target}:${index}`,
        source: link.source,
        target: link.target,
        label: typeof link.label === "string" ? link.label : "",
        status: getSemanticStatus(link.status),
        animated: link.animated === true,
      });
      return acc;
    },
    [],
  );

  return { nodes, edges };
};

const extractWorkflowFromFile = async (filePath: string): Promise<FileWorkflowData | null> => {
  try {
    const response = await fetch(`/api/workspace/files?path=${encodeURIComponent(filePath)}`);
    if (!response.ok) {
      return null;
    }

    const data = (await response.json()) as unknown;
    const root = asRecord(data);
    if (!root) {
      return null;
    }

    const context = asRecord(root.context);
    const contextWorkflow = asRecord(context?.workflow);
    if (contextWorkflow) {
      return {
        id:
          typeof contextWorkflow.workflow_id === "string"
            ? contextWorkflow.workflow_id
            : typeof contextWorkflow.id === "string"
              ? contextWorkflow.id
              : filePath,
        name: typeof contextWorkflow.name === "string" ? contextWorkflow.name : undefined,
        graph:
          normalizeGraphFromUnknown(contextWorkflow.graph) ??
          graphFromTaskConfigPayload(contextWorkflow),
      };
    }

    return {
      id: typeof root.workflow_id === "string" ? root.workflow_id : filePath,
      name: typeof root.name === "string" ? root.name : undefined,
      graph: normalizeGraphFromUnknown(root.graph) ?? graphFromTaskConfigPayload(root),
    };
  } catch (error) {
    console.error("Failed to extract workflow:", error);
    return null;
  }
};

const parallelGraph: DisplayWorkflowGraph = {
  nodes: [
    {
      nodeId: "start",
      label: "Input Data",
      nodeType: "input",
      status: "succeeded",
      position: { x: 250, y: 0 },
      description: "Raw sequence",
    },
    {
      nodeId: "preprocess",
      label: "MSA Search",
      nodeType: "task",
      status: "succeeded",
      position: { x: 250, y: 100 },
      description: "Multiple Sequence Alignment",
    },
    {
      nodeId: "model_1",
      label: "Model 1",
      nodeType: "task",
      status: "succeeded",
      position: { x: 50, y: 250 },
      description: "Prediction Model 1",
    },
    {
      nodeId: "model_2",
      label: "Model 2",
      nodeType: "task",
      status: "failed",
      position: { x: 250, y: 250 },
      description: "Prediction Model 2 (Failed)",
    },
    {
      nodeId: "model_3",
      label: "Model 3",
      nodeType: "task",
      status: "skipped",
      position: { x: 450, y: 250 },
      description: "Prediction Model 3 (Skipped)",
    },
    {
      nodeId: "consensus",
      label: "Consensus",
      nodeType: "task",
      status: "running",
      position: { x: 250, y: 400 },
      description: "Ensemble voting",
    },
    {
      nodeId: "output",
      label: "PDB Structure",
      nodeType: "output",
      status: "pending",
      position: { x: 250, y: 500 },
      description: "Final structure",
    },
  ],
  edges: [
    { id: "e1", source: "start", target: "preprocess", label: "seq", status: "succeeded" },
    { id: "e2", source: "preprocess", target: "model_1", label: "msa", status: "succeeded" },
    { id: "e3", source: "preprocess", target: "model_2", label: "msa", status: "failed" },
    { id: "e4", source: "preprocess", target: "model_3", label: "msa", status: "skipped" },
    { id: "e5", source: "model_1", target: "consensus", label: "pdb", status: "running" },
    { id: "e6", source: "model_2", target: "consensus", label: "pdb", status: "skipped" },
    { id: "e7", source: "model_3", target: "consensus", label: "pdb", status: "skipped" },
    { id: "e8", source: "consensus", target: "output", label: "best", status: "pending" },
  ],
};

const loopGraph: DisplayWorkflowGraph = {
  nodes: [
    {
      nodeId: "init",
      label: "Init Params",
      nodeType: "input",
      status: "succeeded",
      position: { x: 200, y: 0 },
      description: "Initial configuration",
    },
    {
      nodeId: "sim",
      label: "Simulation",
      nodeType: "task",
      status: "succeeded",
      position: { x: 200, y: 150 },
      description: "Run dynamics",
    },
    {
      nodeId: "eval",
      label: "Evaluate Energy",
      nodeType: "task",
      status: "succeeded",
      position: { x: 200, y: 300 },
      description: "Check stability",
    },
    {
      nodeId: "check",
      label: "Converged?",
      nodeType: "task",
      status: "running",
      position: { x: 200, y: 450 },
      description: "Decision gate",
    },
    {
      nodeId: "refine",
      label: "Refine",
      nodeType: "task",
      status: "pending",
      position: { x: 450, y: 225 },
      description: "Adjust parameters",
    },
    {
      nodeId: "final",
      label: "Report",
      nodeType: "output",
      status: "pending",
      position: { x: 200, y: 600 },
      description: "Analysis report",
    },
  ],
  edges: [
    { id: "e1", source: "init", target: "sim", label: "start", status: "succeeded" },
    { id: "e2", source: "sim", target: "eval", label: "traj", status: "succeeded" },
    { id: "e3", source: "eval", target: "check", label: "score", status: "succeeded" },
    { id: "e4", source: "check", target: "final", label: "yes", status: "pending" },
    { id: "e5", source: "check", target: "refine", label: "no", status: "running", animated: true },
    {
      id: "e6",
      source: "refine",
      target: "sim",
      label: "retry",
      status: "pending",
      animated: true,
    },
  ],
};

const asyncGraph: DisplayWorkflowGraph = {
  nodes: [
    {
      nodeId: "trigger",
      label: "Job Submission",
      nodeType: "input",
      status: "succeeded",
      position: { x: 100, y: 100 },
      description: "Submit to cluster",
    },
    {
      nodeId: "remote",
      label: "Remote Cluster",
      nodeType: "task",
      status: "running",
      position: { x: 350, y: 100 },
      description: "External computation",
    },
    {
      nodeId: "monitor",
      label: "Status Poller",
      nodeType: "task",
      status: "running",
      position: { x: 350, y: 250 },
      description: "Check status",
    },
    {
      nodeId: "notify",
      label: "Notification",
      nodeType: "task",
      status: "skipped",
      position: { x: 100, y: 250 },
      description: "Slack alert (Skipped)",
    },
    {
      nodeId: "download",
      label: "Fetch Results",
      nodeType: "task",
      status: "pending",
      position: { x: 600, y: 100 },
      description: "Download artifacts",
    },
  ],
  edges: [
    { id: "e1", source: "trigger", target: "remote", label: "submit", status: "succeeded" },
    { id: "e2", source: "trigger", target: "notify", label: "started", status: "skipped" },
    {
      id: "e3",
      source: "remote",
      target: "monitor",
      label: "heartbeat",
      status: "running",
      animated: true,
    },
    { id: "e4", source: "remote", target: "download", label: "done", status: "pending" },
  ],
};

const getDemoGraph = (workflowId: string): DisplayWorkflowGraph => {
  if (workflowId.includes("exp-002")) {
    return loopGraph;
  }
  if (workflowId.includes("exp-101")) {
    return asyncGraph;
  }
  return parallelGraph;
};

export const WorkflowGraphViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const snapshotWorkflow =
    snapshot.workflows.find((item) => item.id === selection.objectId) ?? null;
  const [fileWorkflow, setFileWorkflow] = useState<FileWorkflowData | null>(null);
  const [isLoadingFile, setIsLoadingFile] = useState(false);

  useEffect(() => {
    let cancelled = false;

    if (selection.objectType !== "workspace-file") {
      setFileWorkflow(null);
      setIsLoadingFile(false);
      return;
    }

    setIsLoadingFile(true);
    extractWorkflowFromFile(selection.objectId)
      .then((extracted) => {
        if (!cancelled) {
          setFileWorkflow(extracted);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingFile(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selection.objectId, selection.objectType]);

  const workflowId = snapshotWorkflow?.id ?? fileWorkflow?.id ?? selection.objectId;
  const workflowName = snapshotWorkflow?.name ?? fileWorkflow?.name;

  const graph = useMemo<DisplayWorkflowGraph | null>(() => {
    if (snapshotWorkflow?.graph) {
      return normalizeGraph(snapshotWorkflow.graph);
    }
    if (fileWorkflow?.graph) {
      return fileWorkflow.graph;
    }
    if (snapshotWorkflow || fileWorkflow) {
      return getDemoGraph(workflowId);
    }
    return null;
  }, [fileWorkflow, snapshotWorkflow, workflowId]);

  const nodes = useMemo<WorkflowFlowNode[]>(() => {
    if (!graph) {
      return [];
    }
    return graph.nodes.map((node) => ({
      id: node.nodeId,
      type: "workflowNode",
      position: node.position,
      data: {
        label: node.label,
        nodeType: node.nodeType,
        status: node.status,
        description: node.description,
      },
    }));
  }, [graph]);

  const edges = useMemo<Edge[]>(() => {
    if (!graph) {
      return [];
    }
    return graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: "smoothstep",
      animated: edge.animated || edge.status === "running",
      style: {
        stroke: getEdgeColor(edge.status),
        strokeWidth: 2,
      },
      label: edge.label,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: getEdgeColor(edge.status),
      },
    }));
  }, [graph]);

  const nodeTypes = useMemo<NodeTypes>(() => ({ workflowNode: WorkflowNode }), []);

  if (isLoadingFile) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Workflow Graph</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-[400px] w-full" />
        </CardContent>
      </Card>
    );
  }

  if (!snapshotWorkflow && !fileWorkflow) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Workflow Graph</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No workflow data found in this file.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Workflow Graph</CardTitle>
          {workflowName && <Badge variant="outline">{workflowName}</Badge>}
        </div>
      </CardHeader>
      <CardContent className="flex-1 p-0">
        <div className="h-full w-full">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            fitView
            attributionPosition="bottom-right"
          >
            <Background />
            <Controls />
          </ReactFlow>
        </div>
      </CardContent>
    </Card>
  );
};

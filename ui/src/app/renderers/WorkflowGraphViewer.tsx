import { useEffect, useMemo, useState } from "react";
import type { Node, Edge, NodeProps } from "@xyflow/react";
import { ReactFlow, Background, Controls, Handle, Position, MarkerType } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { RendererProps, SemanticStatus } from "@/app/types";

interface WorkflowNodeData extends Record<string, unknown> {
  label: string;
  nodeType: "task" | "input" | "output";
  status: SemanticStatus;
  description: string;
}

const getStatusColor = (status: SemanticStatus): string => {
  switch (status) {
    case "succeeded": return "border-blue-500 text-blue-700"; // Blue
    case "failed": return "border-red-500 text-red-700"; // Red
    case "running": return "border-green-500 text-green-700"; // Green
    case "skipped": return "border-yellow-500 text-yellow-700"; // Yellow
    case "cancelled": return "border-yellow-500 text-yellow-700"; // Yellow
    default: return "border-border text-foreground";
  }
};

const getEdgeColor = (status: SemanticStatus): string => {
  switch (status) {
    case "succeeded": return "#3b82f6"; // blue-500
    case "failed": return "#ef4444"; // red-500
    case "running": return "#22c55e"; // green-500
    case "skipped": return "#eab308"; // yellow-500
    case "cancelled": return "#eab308"; // yellow-500
    default: return "#94a3b8"; // slate-400
  }
};

const WorkflowNode = ({ data }: NodeProps<WorkflowNodeData>): JSX.Element => {
  const colorClass = getStatusColor(data.status);
  
  return (
    <div className={`rounded-md border-2 bg-background px-3 py-2 shadow-sm min-w-[150px] ${colorClass}`}>
      <Handle type="target" position={Position.Top} className="w-3 h-3 bg-muted-foreground" />
      <p className="text-xs font-semibold uppercase tracking-wide opacity-70">
        {data.nodeType}
      </p>
      <p className="text-sm font-bold">{data.label}</p>
      <p className="text-xs opacity-70 capitalize">{data.status}</p>
      <Handle type="source" position={Position.Bottom} className="w-3 h-3 bg-muted-foreground" />
    </div>
  );
};

/**
 * Extract workflow data from run.json or ckpt.json
 */
const extractWorkflowFromFile = async (filePath: string): Promise<any | null> => {
  try {
    const response = await fetch(`/api/workspace/files?path=${encodeURIComponent(filePath)}`);
    if (!response.ok) return null;
    
    const data = await response.json();
    
    // Check if this is run.json or ckpt.json with context.workflow
    if (data.context?.workflow) {
      return data.context.workflow;
    }
    
    // Check if this is a direct workflow file
    if (data.task_configs && data.links) {
      return data;
    }
    
    return null;
  } catch (error) {
    console.error("Failed to extract workflow:", error);
    return null;
  }
};

export const WorkflowGraphViewer = ({
  selection,
  snapshot,
}: RendererProps): JSX.Element => {
  // Try to find workflow from snapshot first (for workflow objects)
  let workflow = snapshot.workflows.find(item => item.id === selection.objectId) ?? null;
  
  // For workspace files (run.json, ckpt.json), we need to extract workflow from the file
  const [fileWorkflow, setFileWorkflow] = useState<any | null>(null);
  const [isLoadingFile, setIsLoadingFile] = useState(false);
  
  useEffect(() => {
    if (selection.objectType === "workspace-file" && !workflow) {
      setIsLoadingFile(true);
      extractWorkflowFromFile(selection.objectId).then(extracted => {
        setFileWorkflow(extracted);
        setIsLoadingFile(false);
      });
    }
  }, [selection.objectId, selection.objectType]);
  
  // Use file workflow if available
  if (selection.objectType === "workspace-file" && fileWorkflow) {
    workflow = fileWorkflow;
  }
  
  const isLoading = isLoadingFile;

  // 1. Parallel Pattern (e.g. AlphaFold - Split/Merge)
  const parallelGraph = {
    nodes: [
      { nodeId: "start", label: "Input Data", nodeType: "input", status: "succeeded", position: { x: 250, y: 0 }, description: "Raw sequence" },
      { nodeId: "preprocess", label: "MSA Search", nodeType: "task", status: "succeeded", position: { x: 250, y: 100 }, description: "Multiple Sequence Alignment" },
      { nodeId: "model_1", label: "Model 1", nodeType: "task", status: "succeeded", position: { x: 50, y: 250 }, description: "Prediction Model 1" },
      { nodeId: "model_2", label: "Model 2", nodeType: "task", status: "failed", position: { x: 250, y: 250 }, description: "Prediction Model 2 (Failed)" },
      { nodeId: "model_3", label: "Model 3", nodeType: "task", status: "skipped", position: { x: 450, y: 250 }, description: "Prediction Model 3 (Skipped)" },
      { nodeId: "consensus", label: "Consensus", nodeType: "task", status: "running", position: { x: 250, y: 400 }, description: "Ensemble voting" },
      { nodeId: "output", label: "PDB Structure", nodeType: "output", status: "pending", position: { x: 250, y: 500 }, description: "Final structure" },
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
    ]
  };

  // 2. Loop Pattern (e.g. Structure Sweep - Optimization)
  const loopGraph = {
    nodes: [
      { nodeId: "init", label: "Init Params", nodeType: "input", status: "succeeded", position: { x: 200, y: 0 }, description: "Initial configuration" },
      { nodeId: "sim", label: "Simulation", nodeType: "task", status: "succeeded", position: { x: 200, y: 150 }, description: "Run dynamics" },
      { nodeId: "eval", label: "Evaluate Energy", nodeType: "task", status: "succeeded", position: { x: 200, y: 300 }, description: "Check stability" },
      { nodeId: "check", label: "Converged?", nodeType: "task", status: "running", position: { x: 200, y: 450 }, description: "Decision gate" },
      { nodeId: "refine", label: "Refine", nodeType: "task", status: "pending", position: { x: 450, y: 225 }, description: "Adjust parameters" },
      { nodeId: "final", label: "Report", nodeType: "output", status: "pending", position: { x: 200, y: 600 }, description: "Analysis report" },
    ],
    edges: [
      { id: "e1", source: "init", target: "sim", label: "start", status: "succeeded" },
      { id: "e2", source: "sim", target: "eval", label: "traj", status: "succeeded" },
      { id: "e3", source: "eval", target: "check", label: "score", status: "succeeded" },
      { id: "e4", source: "check", target: "final", label: "yes", status: "pending" },
      { id: "e5", source: "check", target: "refine", label: "no", animated: true, status: "running" },
      { id: "e6", source: "refine", target: "sim", label: "retry", animated: true, status: "pending" },
    ]
  };

  // 3. Async Pattern (e.g. Catalyst - Remote Job)
  const asyncGraph = {
    nodes: [
      { nodeId: "trigger", label: "Job Submission", nodeType: "input", status: "succeeded", position: { x: 100, y: 100 }, description: "Submit to cluster" },
      { nodeId: "remote", label: "Remote Cluster", nodeType: "task", status: "running", position: { x: 350, y: 100 }, description: "External computation" },
      { nodeId: "monitor", label: "Status Poller", nodeType: "task", status: "running", position: { x: 350, y: 250 }, description: "Check status" },
      { nodeId: "notify", label: "Notification", nodeType: "task", status: "skipped", position: { x: 100, y: 250 }, description: "Slack alert (Skipped)" },
      { nodeId: "download", label: "Fetch Results", nodeType: "task", status: "pending", position: { x: 600, y: 100 }, description: "Download artifacts" },
    ],
    edges: [
      { id: "e1", source: "trigger", target: "remote", label: "submit", status: "succeeded" },
      { id: "e2", source: "trigger", target: "notify", label: "started", status: "skipped" },
      { id: "e3", source: "remote", target: "monitor", label: "heartbeat", animated: true, status: "running" },
      { id: "e4", source: "remote", target: "download", label: "done", status: "pending" },
    ]
  };

  const nodes = useMemo<Array<Node<WorkflowNodeData>>>(() => {
    // Select mock graph based on workflow ID pattern
    let graph = workflow?.graph;
    
    if (!graph && workflow) {
      if (workflow.id.includes("exp-002")) {
        graph = loopGraph;
      } else if (workflow.id.includes("exp-101")) {
        graph = asyncGraph;
      } else {
        graph = parallelGraph; // Default to parallel (exp-001)
      }
    }

    if (!graph && !workflow) {
      return [];
    }

    // @ts-ignore - graph structure mismatch workaround for demo
    const nodeList = graph.nodes || graph; 
    
    if (!Array.isArray(nodeList)) return [];

    return nodeList.map((node: any) => ({
      id: node.nodeId || "unknown",
      type: "workflowNode",
      position: node.position || { x: 0, y: 0 },
      data: {
        label: node.label || "Node",
        nodeType: node.nodeType || "task",
        status: node.status || "pending",
        description: node.description || "",
      },
    }));
  }, [workflow]);

  const edges = useMemo<Edge[]>(() => {
    let graph = workflow?.graph;

    if (!graph && workflow) {
      if (workflow.id.includes("exp-002")) {
        graph = loopGraph;
      } else if (workflow.id.includes("exp-101")) {
        graph = asyncGraph;
      } else {
        graph = parallelGraph;
      }
    }

    if (!graph) return [];

    // @ts-ignore
    const edgeList = graph.edges || [];

    return edgeList.map((edge: any) => ({
      id: edge.id || `e-${Math.random()}`,
      source: edge.source,
      target: edge.target,
      type: "smoothstep",
      animated: edge.animated || edge.status === "running",
      style: {
        stroke: getEdgeColor(edge.status || "pending"),
        strokeWidth: 2,
      },
      label: edge.label, // Pass label to edge
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: getEdgeColor(edge.status || "pending"),
      },
    }));
  }, [workflow]);

  if (isLoading) {
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

  if (!workflow) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Workflow Graph</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No workflow data found in this file.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Workflow Graph</CardTitle>
          {workflow.name && (
            <Badge variant="outline">{workflow.name}</Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="flex-1 p-0">
        <div className="h-full w-full">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={{ workflowNode: WorkflowNode }}
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

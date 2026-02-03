import { useEffect, useMemo, useState } from "react";
import type { Node, Edge, NodeProps } from "@xyflow/react";
import { ReactFlow, Background, Controls } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { RendererProps, SemanticStatus } from "@/app/types";

interface WorkflowNodeData {
  label: string;
  nodeType: "task" | "input" | "output";
  status: SemanticStatus;
  description: string;
}

const WorkflowNode = ({ data }: NodeProps<WorkflowNodeData>): JSX.Element => {
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {data.nodeType}
      </p>
      <p className="text-sm font-semibold text-foreground">{data.label}</p>
      <p className="text-xs text-muted-foreground">{data.status}</p>
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
  onInspectorTargetChange,
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

  const nodes = useMemo<Array<Node<WorkflowNodeData>>>(() => {
    if (!workflow || !workflow.graph) {
      return [];
    }

    return workflow.graph.nodes.map(node => ({
      id: node.nodeId,
      type: "workflowNode",
      position: node.position,
      data: {
        label: node.label,
        nodeType: node.nodeType,
        status: node.status,
        description: node.description || "",
      },
    }));
  }, [workflow]);

  const edges = useMemo<Edge[]>(() => {
    if (!workflow || !workflow.graph) {
      return [];
    }

    return workflow.graph.edges.map(edge => ({
      id: edge.edgeId,
      source: edge.source,
      target: edge.target,
      type: "smoothstep",
      animated: edge.status === "running",
      style: {
        stroke: edge.status === "failed" ? "#ef4444" : "#94a3b8",
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
          {workflow.metadata?.name && (
            <Badge variant="outline">{workflow.metadata.name}</Badge>
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

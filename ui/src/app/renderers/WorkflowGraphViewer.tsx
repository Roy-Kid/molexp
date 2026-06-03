/**
 * WorkflowGraphViewer — the workflow "Graph" tab, rendered on the read-only
 * flowgram free-layout canvas. Resolves the workflow's task-graph IR either from
 * the in-memory snapshot (`workflow.graph`) or by fetching the selected
 * workspace file, lowers it via {@link buildFlowgramDocument}, and draws it.
 * Clicking a node opens the right-panel TaskViewer via `inspectedTask`.
 */

import { useEffect, useMemo, useState } from "react";
import { FlowgramCanvas } from "@/app/renderers/FlowgramCanvas";
import {
  buildFlowgramDocument,
  type FlowgramDocument,
  normalizeTaskGraph,
} from "@/app/renderers/flowgram-document";
import { useInspectedTask } from "@/app/state/inspectedTask";
import type { RendererProps } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { TaskGraphJson } from "@/types/task_graph_ir";

interface FileWorkflowData {
  id: string;
  name?: string;
  graph: TaskGraphJson | null;
}

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
};

const graphFromPayload = (value: Record<string, unknown>): TaskGraphJson | null => {
  if (!Array.isArray(value.task_configs) || !Array.isArray(value.links)) {
    return null;
  }
  return normalizeTaskGraph(value);
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
        graph: graphFromPayload(contextWorkflow),
      };
    }

    return {
      id: typeof root.workflow_id === "string" ? root.workflow_id : filePath,
      name: typeof root.name === "string" ? root.name : undefined,
      graph: graphFromPayload(root),
    };
  } catch (error) {
    console.error("Failed to extract workflow:", error);
    return null;
  }
};

export const WorkflowGraphViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const { inspectTask } = useInspectedTask();
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

  const workflowName = snapshotWorkflow?.name ?? fileWorkflow?.name;

  const graph = useMemo<TaskGraphJson | null>(() => {
    if (snapshotWorkflow?.graph) {
      return snapshotWorkflow.graph;
    }
    if (fileWorkflow?.graph) {
      return fileWorkflow.graph;
    }
    return null;
  }, [fileWorkflow, snapshotWorkflow]);

  const document = useMemo<FlowgramDocument | null>(
    () => (graph ? buildFlowgramDocument(graph) : null),
    [graph],
  );

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
          {!document || document.nodes.length === 0 ? (
            <div className="flex h-full items-center justify-center">
              <p className="text-sm text-muted-foreground">No tasks to display.</p>
            </div>
          ) : (
            <FlowgramCanvas document={document} onNodeClick={(taskId) => inspectTask(taskId, "")} />
          )}
        </div>
      </CardContent>
    </Card>
  );
};

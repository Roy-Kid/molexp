/**
 * WorkflowGraphViewer — the workflow "Graph" tab. Renders the workflow entity's
 * task-graph IR (from the snapshot, or the just-saved draft) on the editable
 * flowgram free-layout canvas, and saves edits back via {@link workflowApi}.
 * Clicking a node opens the right-panel TaskViewer via `inspectedTask`.
 *
 * Raw workspace-file `workflow.json` previews are a different source + format
 * and are handled by {@link WorkflowFileViewer}; this viewer only ever receives
 * ``workflow`` entity selections (it is mounted solely by WorkflowViewer).
 */

import { useEffect, useMemo, useState } from "react";
import { FlowgramCanvas } from "@/components/workflow/flowgram-canvas";
import { FlowgramCanvasToolbar } from "@/components/workflow/flowgram-canvas-toolbar";
import {
  buildFlowgramDocument,
  type FlowgramDocument,
  flowgramDocToTaskGraphJson,
  normalizeTaskGraph,
  taskGraphToWireDocument,
} from "@/components/workflow/flowgram-document";
import { workflowApi } from "@/app/state/api";
import { useInspectedTask } from "@/app/state/inspectedTask";
import type { RendererProps } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { TaskGraphJson } from "@/components/workflow/task-graph-ir";

export const WorkflowGraphViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const { inspectTask } = useInspectedTask();
  const workflow = snapshot.workflows.find((item) => item.id === selection.objectId) ?? null;

  const [savedGraph, setSavedGraph] = useState<TaskGraphJson | null>(null);
  const [draft, setDraft] = useState<FlowgramDocument | null>(null);
  const [saving, setSaving] = useState(false);

  // Reset edit state whenever the selected workflow changes. objectId is a
  // trigger, not a read — biome's exhaustive-deps can't see that and would
  // strip it, which would leave stale edits when switching workflows.
  // biome-ignore lint/correctness/useExhaustiveDependencies: objectId is a reset trigger
  useEffect(() => {
    setSavedGraph(null);
    setDraft(null);
  }, [selection.objectId]);

  // Prefer the freshly-saved graph, else the snapshot's IR.
  const graph = savedGraph ?? workflow?.graph ?? null;
  const document = useMemo<FlowgramDocument | null>(
    () => (graph ? buildFlowgramDocument(graph) : null),
    [graph],
  );

  const handleSave = async (): Promise<void> => {
    if (!workflow || !draft) return;
    setSaving(true);
    try {
      const wire = taskGraphToWireDocument(
        flowgramDocToTaskGraphJson(draft, workflow.name ?? "Workflow"),
      );
      const persisted = await workflowApi.save(workflow.projectId, workflow.experimentId, wire);
      // Reload from the server-normalized document so the canvas reflects
      // exactly what was persisted.
      setSavedGraph(normalizeTaskGraph(persisted));
      setDraft(null);
    } finally {
      setSaving(false);
    }
  };

  if (!workflow) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Workflow Graph</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No workflow data found.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Workflow Graph</CardTitle>
          <div className="flex items-center gap-2">
            {workflow.name && <Badge variant="outline">{workflow.name}</Badge>}
            <FlowgramCanvasToolbar onSave={handleSave} saving={saving} dirty={draft !== null} />
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-1 p-0">
        <div className="h-full w-full">
          {!document || document.nodes.length === 0 ? (
            <div className="flex h-full items-center justify-center">
              <p className="text-sm text-muted-foreground">No tasks to display.</p>
            </div>
          ) : (
            <FlowgramCanvas
              document={document}
              editable
              onChange={setDraft}
              onNodeClick={(taskId) => inspectTask(taskId, "")}
            />
          )}
        </div>
      </CardContent>
    </Card>
  );
};

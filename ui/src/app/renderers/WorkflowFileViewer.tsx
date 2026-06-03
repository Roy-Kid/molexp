/**
 * WorkflowFileViewer — preview of a workspace `workflow.json` on the read-only
 * flowgram canvas. Loads the file text, validates it carries per-node /
 * per-link status, then lowers the IR via {@link buildFlowgramDocument}.
 */

import { useEffect, useMemo, useState } from "react";
import { FlowgramCanvas } from "@/app/renderers/FlowgramCanvas";
import {
  buildFlowgramDocument,
  type FlowgramDocument,
  normalizeTaskGraph,
} from "@/app/renderers/flowgram-document";
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

export const WorkflowFileViewer = ({ selection }: RendererProps): JSX.Element => {
  const [payload, setPayload] = useState<WorkflowFilePayload | null>(null);
  const [error, setError] = useState<string | null>(null);

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
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load workflow");
        setPayload(null);
      });
  }, [selection]);

  const document = useMemo<FlowgramDocument | null>(() => {
    if (!payload) {
      return null;
    }
    return buildFlowgramDocument(normalizeTaskGraph(payload as unknown as Record<string, unknown>));
  }, [payload]);

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
            {document && document.nodes.length > 0 ? (
              <FlowgramCanvas document={document} className="flex-1" />
            ) : (
              <div className="px-4 py-3 text-xs text-muted-foreground">
                No workflow nodes found in workflow.json.
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

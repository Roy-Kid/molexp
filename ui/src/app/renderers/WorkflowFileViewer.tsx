/**
 * WorkflowFileViewer — preview of a workspace `workflow.json` on the read-only
 * flowgram canvas. Loads the file text, validates it carries per-node /
 * per-link status, then lowers the IR via {@link buildFlowgramDocument}.
 */

import { useEffect, useState } from "react";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps, SemanticStatus } from "@/app/types";
import { WorkbenchAction, WorkbenchOperationState } from "@/components/workbench";
import { FlowgramCanvas } from "@/components/workflow/flowgram-canvas";
import {
  buildFlowgramDocument,
  type FlowgramDocument,
  normalizeTaskGraph,
} from "@/components/workflow/flowgram-document";

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
  const [document, setDocument] = useState<FlowgramDocument | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [settledObjectId, setSettledObjectId] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  useEffect(() => {
    void requestVersion;
    if (selection.objectType !== "workspace-file") {
      setPayload(null);
      setDocument(null);
      setLoading(false);
      setError(null);
      setSettledObjectId(null);
      return;
    }

    let cancelled = false;
    setPayload(null);
    setDocument(null);
    setLoading(true);
    setError(null);
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
        const nextDocument = buildFlowgramDocument(
          normalizeTaskGraph(parsed as unknown as Record<string, unknown>),
        );
        if (cancelled) return;
        setPayload(parsed);
        setDocument(nextDocument);
        setError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load workflow");
        setPayload(null);
        setDocument(null);
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
          setSettledObjectId(selection.objectId);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [requestVersion, selection.objectId, selection.objectType]);

  const requestPending =
    selection.objectType === "workspace-file" &&
    (loading || settledObjectId !== selection.objectId);

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas" aria-busy={requestPending}>
      <header className="flex h-10 flex-none flex-col justify-center gap-1 border-b border-border px-3">
        <p className="truncate text-body font-medium text-foreground">
          {payload?.name ?? "Workflow preview"}
        </p>
        <p className="truncate font-mono text-micro text-muted-foreground tabular-nums">
          {selection.objectId}
        </p>
      </header>
      <div className="min-h-0 flex-1">
        {requestPending ? (
          <WorkbenchOperationState
            kind="loading"
            title="Loading workflow preview…"
            skeletonRows={4}
          />
        ) : error ? (
          <WorkbenchOperationState
            kind="error"
            title="Could not load workflow preview"
            detail={error}
            action={
              <WorkbenchAction
                kind="secondary"
                size="compact"
                onClick={() => {
                  setLoading(true);
                  setRequestVersion((version) => version + 1);
                }}
              >
                Retry
              </WorkbenchAction>
            }
          />
        ) : document && document.nodes.length > 0 ? (
          <FlowgramCanvas document={document} className="h-full" />
        ) : (
          <WorkbenchOperationState
            kind="empty"
            title="No workflow nodes"
            detail="workflow.json loaded successfully but contains no runnable nodes."
          />
        )}
      </div>
    </div>
  );
};

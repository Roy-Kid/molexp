import { lazy, Suspense, useEffect, useState } from "react";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";
import { WorkbenchAction, WorkbenchOperationState } from "@/components/workbench";

// Lazy-loaded so `@monaco-editor/react` stays out of the initial page-load
// bundle. This is the second Monaco consumer alongside the `editor` plugin's
// TextEditor; both must lazy-import for Monaco to remain an async chunk.
const Editor = lazy(() => import("@monaco-editor/react"));

export const WorkflowSourceViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const [content, setContent] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [unavailableReason, setUnavailableReason] = useState<string | null>(null);
  const [settledWorkflowId, setSettledWorkflowId] = useState<string | null>(null);
  const [requestVersion, setRequestVersion] = useState(0);

  const workflow = snapshot.workflows.find((w) => w.id === selection.objectId);
  // The summary often contains the file path, or we can try to derive it via ID if formatted "workflow:<id>"
  // However, looking at api.ts: mapWorkflows: summary: workflowPath ("workflow" or actual path)
  // And id: `workflow:${experiment.id}`.
  // We need to know where the workflow file is.
  // In `mapExperiments`: workflowFile: experiment.workflow.
  // So we can find the experiment by ID (experimentId is in workflow summary)

  useEffect(() => {
    void requestVersion;
    if (!workflow) {
      setContent("");
      setIsLoading(false);
      setError(null);
      setUnavailableReason(null);
      setSettledWorkflowId(null);
      return;
    }

    // Use summary as path for now, as consistent with api.ts mapping
    const path = workflow.summary;

    // If it looks like a path (contains .yaml, .json, or /), try deciding if absolute or relative?
    // Based on api.ts mapWorkflows, it's just `workflowPath` string.

    setIsLoading(true);
    setError(null);
    setUnavailableReason(null);
    setContent("");
    // If it's a relative path, we might need to know the project path.
    // Usually these paths are relative to workspace root or project?
    // Let's assume relative to workspace root for now or try to fetch.
    // NOTE: experiment.workflow is usually just filename if in project root, or path.
    // api.ts calls `getWorkspaceFileText(path)`.

    // Actually, `workflow.summary` might be just a description.
    // Let's check `workflow.experimentId`.
    const experiment = snapshot.experiments.find((e) => e.id === workflow?.experimentId);
    // experiment.workflowFile seems to be the path.
    const actualPath = experiment?.workflowFile || path;
    if (!actualPath) {
      setUnavailableReason("No source path is configured for this workflow.");
      setIsLoading(false);
      setSettledWorkflowId(workflow.id);
      return;
    }

    let cancelled = false;
    workspaceApi
      .getWorkspaceFileText(actualPath)
      .then((text) => {
        if (cancelled) return;
        setContent(text);
        setError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        console.warn("Failed to fetch workflow source", err);
        setError(err instanceof Error ? err.message : "Source code could not be loaded.");
        setContent("");
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoading(false);
          setSettledWorkflowId(workflow.id);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [requestVersion, workflow, snapshot.experiments]);

  if (!workflow) {
    return (
      <WorkbenchOperationState
        kind="empty"
        title="Workflow source unavailable"
        detail="The selected workflow is not present in the current workspace snapshot."
      />
    );
  }

  const sourcePending = isLoading || settledWorkflowId !== workflow.id;

  if (sourcePending) {
    return <WorkbenchOperationState kind="loading" title="Loading workflow source…" />;
  }

  if (unavailableReason) {
    return (
      <WorkbenchOperationState
        kind="disabled"
        title="Workflow source unavailable"
        detail={unavailableReason}
      />
    );
  }

  if (error) {
    return (
      <WorkbenchOperationState
        kind="error"
        title="Could not load workflow source"
        detail={error}
        action={
          <WorkbenchAction
            kind="secondary"
            size="compact"
            onClick={() => {
              setIsLoading(true);
              setRequestVersion((version) => version + 1);
            }}
          >
            Retry
          </WorkbenchAction>
        }
      />
    );
  }

  if (content.length === 0) {
    return (
      <WorkbenchOperationState
        kind="empty"
        title="Workflow source is empty"
        detail="The source file loaded successfully but contains no text."
      />
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <Suspense
        fallback={<WorkbenchOperationState kind="loading" title="Loading source editor…" />}
      >
        <Editor
          height="100%"
          language="yaml"
          value={content}
          theme="light"
          options={{
            readOnly: true,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            lineNumbers: "on",
          }}
        />
      </Suspense>
    </div>
  );
};

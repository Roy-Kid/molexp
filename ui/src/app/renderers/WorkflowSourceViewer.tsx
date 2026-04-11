import Editor from "@monaco-editor/react";
import { useEffect, useState } from "react";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";
import { Card, CardContent } from "@/components/ui/card";

export const WorkflowSourceViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const [content, setContent] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const workflow = snapshot.workflows.find((w) => w.id === selection.objectId);
  // The summary often contains the file path, or we can try to derive it via ID if formatted "workflow:<id>"
  // However, looking at api.ts: mapWorkflows: summary: workflowPath ("workflow" or actual path)
  // And id: `workflow:${experiment.id}`.
  // We need to know where the workflow file is.
  // In `mapExperiments`: workflowFile: experiment.workflow.
  // So we can find the experiment by ID (experimentId is in workflow summary)

  useEffect(() => {
    if (!workflow) return;

    // Use summary as path for now, as consistent with api.ts mapping
    const path = workflow.summary;

    // If it looks like a path (contains .yaml, .json, or /), try deciding if absolute or relative?
    // Based on api.ts mapWorkflows, it's just `workflowPath` string.

    setIsLoading(true);
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

    workspaceApi
      .getWorkspaceFileText(actualPath)
      .then((text) => {
        setContent(text);
        setError(null);
      })
      .catch((err) => {
        console.warn("Failed to fetch workflow source", err);
        setError("Source code not available or could not be loaded.");
        setContent("# Source not available");
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [workflow, snapshot.experiments]);

  if (isLoading) {
    return (
      <Card className="h-full border-0 shadow-none">
        <CardContent className="h-full flex items-center justify-center text-muted-foreground text-sm">
          Loading source...
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full border-0 shadow-none">
      <CardContent className="h-full p-0">
        {error && (
          <div className="border-b border-border px-4 py-2 text-sm text-destructive">{error}</div>
        )}
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
      </CardContent>
    </Card>
  );
};

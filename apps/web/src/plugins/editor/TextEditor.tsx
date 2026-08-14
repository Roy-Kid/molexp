import { Save } from "lucide-react";
import { Suspense, useEffect, useMemo, useState } from "react";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { WorkbenchIconAction, WorkbenchOperationState } from "@/components/workbench";
import { filePreviewPluginRegistry } from "@/lib/file-preview-plugins";
import { MonacoEditor } from "./MonacoEditor";

/**
 * Monaco-backed text editor for workspace files.
 *
 * Lazy-loaded so `@monaco-editor/react` / `monaco-editor` (a large
 * dependency) is split into an async chunk fetched only when the editor
 * actually mounts, instead of riding in the initial page-load bundle.
 *
 * Preview-host contract: this component is the host of the `editor` panel
 * slot (see `./index.ts`). It renders an Edit/Preview tab pair and, in the
 * Preview tab, delegates to whichever {@link FilePreviewPlugin} the
 * `filePreviewPluginRegistry` resolves for the current file. The preview
 * *content* is supplied by other plugins (core markdown/workflow, molvis,
 * …); this editor only owns the hosting surface.
 */
export const TextEditor = ({ selection }: RendererProps): JSX.Element => {
  const [value, setValue] = useState<string>("");
  const [status, setStatus] = useState<"idle" | "loading" | "saving" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const previewPlugin = useMemo(() => {
    if (selection.objectType !== "workspace-file") {
      return null;
    }

    const name = selection.filePath.split("/").pop() ?? selection.filePath;
    return filePreviewPluginRegistry.getPluginForFile(name, selection.filePath, {
      hasPreviewSidecar: selection.hasPreviewSidecar,
    });
  }, [selection]);

  const language = useMemo(() => {
    if (selection.objectType !== "workspace-file") {
      return "plaintext";
    }
    const kind = selection.fileKind;
    if (kind === "json") return "json";
    if (kind === "yaml") return "yaml";
    if (kind === "python") return "python";
    if (kind === "markdown") return "markdown";
    if (kind === "text") return "plaintext";
    return "plaintext";
  }, [selection]);

  useEffect(() => {
    if (selection.objectType !== "workspace-file") {
      return;
    }

    let isMounted = true;
    setStatus("loading");
    setError(null);
    workspaceApi
      .getWorkspaceFileText(selection.objectId)
      .then((content) => {
        if (isMounted) {
          setValue(content);
          setStatus("idle");
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Failed to load file");
          setStatus("error");
        }
      });

    return () => {
      isMounted = false;
    };
  }, [selection]);

  const handleSave = async () => {
    if (selection.objectType !== "workspace-file") return;

    setStatus("saving");
    try {
      await workspaceApi.writeFile(selection.objectId, value);
      setStatus("idle");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save file");
      setStatus("error");
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <header className="flex h-10 flex-none items-center justify-between gap-2 border-b border-border px-3">
        <p className="min-w-0 truncate font-mono text-label text-muted-foreground tabular-nums">
          {selection.objectId}
        </p>
        <WorkbenchIconAction
          label={status === "saving" ? "Saving file" : "Save file"}
          onClick={handleSave}
          disabled={status === "loading" || status === "saving"}
        >
          <Save className="h-3.5 w-3.5" />
        </WorkbenchIconAction>
      </header>
      <div className="min-h-0 flex-1">
        {status === "loading" && !value ? (
          <WorkbenchOperationState kind="loading" title="Loading file…" skeletonRows={6} />
        ) : status === "error" ? (
          <WorkbenchOperationState
            kind="error"
            title="Could not load file"
            detail={error ?? undefined}
          />
        ) : (
          <Tabs defaultValue="edit" className="flex h-full flex-col gap-0">
            {previewPlugin ? (
              <div className="flex-none border-b border-border px-3 py-2">
                <TabsList className="h-control-compact w-fit rounded-control bg-muted p-1">
                  <TabsTrigger value="edit" className="h-6 text-label">
                    Edit
                  </TabsTrigger>
                  <TabsTrigger value="preview" className="h-6 text-label">
                    Preview
                  </TabsTrigger>
                </TabsList>
              </div>
            ) : null}

            <TabsContent value="edit" className="m-0 min-h-0 flex-1">
              <Suspense
                fallback={
                  <div className="p-3 text-label text-muted-foreground">Loading editor…</div>
                }
              >
                <MonacoEditor
                  height="100%"
                  language={language}
                  value={value}
                  theme="light"
                  onChange={(nextValue) => {
                    setValue(nextValue ?? "");
                  }}
                  options={{
                    minimap: { enabled: false },
                    wordWrap: "on",
                    scrollBeyondLastLine: false,
                  }}
                />
              </Suspense>
            </TabsContent>

            {previewPlugin && selection.objectType === "workspace-file" ? (
              <TabsContent value="preview" className="m-0 min-h-0 flex-1 overflow-auto">
                <previewPlugin.Component
                  content={value}
                  name={selection.filePath.split("/").pop() ?? selection.filePath}
                  path={selection.filePath}
                  folderId="workspace"
                  assetId={
                    selection.objectType === "workspace-file" ? selection.assetId : undefined
                  }
                />
              </TabsContent>
            ) : null}
          </Tabs>
        )}
      </div>
    </div>
  );
};

import Editor from "@monaco-editor/react";
import { Save } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { filePreviewPluginRegistry } from "@/lib/file-preview-plugins";

export const TextEditor = ({ selection }: RendererProps): JSX.Element => {
  const [value, setValue] = useState<string>("");
  const [status, setStatus] = useState<"idle" | "loading" | "saving" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const previewPlugin = useMemo(() => {
    if (selection.objectType !== "workspace-file") {
      return null;
    }

    const name = selection.filePath.split("/").pop() ?? selection.filePath;
    return filePreviewPluginRegistry.getPluginForFile(name, selection.filePath);
  }, [selection]);

  const language = useMemo(() => {
    // If not a file, default to text. Though logic suggests it handles files usually.
    if (selection.objectType !== "workspace-file") {
      // Could be 'project' json metadata if inspector is not used?
      // But typically TextEditor is for files.
      return "plaintext";
    }
    // Mapping file extensions/kinds to Monaco languages
    const kind = selection.fileKind;
    if (kind === "json") return "json";
    if (kind === "yaml") return "yaml";
    if (kind === "python") return "python";
    if (kind === "markdown") return "markdown";
    if (kind === "text") return "plaintext";
    return "plaintext";
  }, [selection]);

  useEffect(() => {
    // Only load if it's a file
    if (selection.objectType !== "workspace-file") {
      return;
    }

    let isMounted = true;
    setStatus("loading");
    setError(null);
    workspaceApi
      .getWorkspaceFileText(selection.objectId) // objectId is the path for workspace-file
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
  }, [selection]); // Re-fetch when selection changes

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
    <Card className="flex h-full flex-col border-border/60 bg-background">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div className="space-y-1">
          <CardTitle className="text-lg font-semibold">Text Editor</CardTitle>
          <p className="text-sm text-muted-foreground break-all">{selection.objectId}</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleSave}
          disabled={status === "loading" || status === "saving"}
          className="gap-2"
        >
          <Save className="h-4 w-4" />
          {status === "saving" ? "Saving..." : "Save"}
        </Button>
      </CardHeader>
      <Separator />
      <CardContent className="flex-1 pt-4 p-0 min-h-0">
        {status === "error" ? (
          <div className="p-4 text-sm text-destructive">{error}</div>
        ) : (
          <Tabs defaultValue="edit" className="flex h-full flex-col">
            {previewPlugin ? (
              <div className="border-b border-border/60 px-4 py-2">
                <TabsList className="h-auto w-fit rounded-md bg-muted/30 p-1">
                  <TabsTrigger value="edit">Edit</TabsTrigger>
                  <TabsTrigger value="preview">Preview</TabsTrigger>
                </TabsList>
              </div>
            ) : null}

            <TabsContent value="edit" className="m-0 flex-1 min-h-0">
              <Editor
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
            </TabsContent>

            {previewPlugin && selection.objectType === "workspace-file" ? (
              <TabsContent value="preview" className="m-0 flex-1 overflow-auto">
                <previewPlugin.Component
                  content={value}
                  name={selection.filePath.split("/").pop() ?? selection.filePath}
                  path={selection.filePath}
                  folderId="workspace"
                />
              </TabsContent>
            ) : null}
          </Tabs>
        )}
      </CardContent>
    </Card>
  );
};

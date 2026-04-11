import type { FilePreviewContentProps } from "@/lib/file-preview-plugins";

export const WorkflowPreview = ({ content, name, path }: FilePreviewContentProps): JSX.Element => {
  return (
    <div className="space-y-2 p-3">
      <div className="text-xs text-muted-foreground">Workflow preview</div>
      <div className="rounded border border-border bg-muted/20 px-3 py-2 text-xs">
        <p>
          <strong>Name:</strong> {name}
        </p>
        <p>
          <strong>Path:</strong> {path}
        </p>
      </div>
      <pre className="max-h-[420px] overflow-auto rounded border border-border bg-background p-3 text-xs">
        {content}
      </pre>
    </div>
  );
};

import type { FilePreviewContentProps } from "@/lib/file-preview-plugins";

export const WorkflowPreview = ({ content, name, path }: FilePreviewContentProps): JSX.Element => {
  return (
    <div className="space-y-2 p-3">
      <div className="text-label text-muted-foreground">Workflow preview</div>
      <div className="rounded-control border border-border bg-muted/20 px-3 py-2 text-label">
        <p>
          <strong>Name:</strong> {name}
        </p>
        <p>
          <strong>Path:</strong> {path}
        </p>
      </div>
      <pre className="max-h-chart-lg overflow-auto rounded-control border border-border bg-background p-3 text-label">
        {content}
      </pre>
    </div>
  );
};

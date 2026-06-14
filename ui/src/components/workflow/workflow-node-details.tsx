/**
 * WorkflowNodeDetails — a self-contained, read-only detail card for one task
 * node of the canonical {@link TaskGraphJson} IR.
 *
 * Unlike the app's `WorkflowInspector` (which reads the workspace snapshot /
 * selection store), this component is pure: it takes a plain {@link TaskNodeJson}
 * (or `null`) and renders its id / type / status / static config with molexp's
 * shadcn-ui chrome. It composes anywhere the workflow components are reused —
 * the in-app canvas, the `workflow.json` preview, the VSCode extension webview.
 */

import type { JSX } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type { TaskNodeJson } from "@/components/workflow/task-graph-ir";

export interface WorkflowNodeDetailsProps {
  /** The selected task node, or `null` when nothing is selected. */
  node: TaskNodeJson | null;
  className?: string;
}

const Field = ({ label, value }: { label: string; value: string }): JSX.Element => (
  <div className="space-y-1">
    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{label}</p>
    <p className="break-words text-sm font-medium text-foreground">{value}</p>
  </div>
);

export const WorkflowNodeDetails = ({ node, className }: WorkflowNodeDetailsProps): JSX.Element => {
  const hasConfig = node?.config && Object.keys(node.config).length > 0;
  return (
    <Card className={`h-full border-border/60 bg-muted/30 ${className ?? ""}`}>
      <CardHeader className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-base font-semibold">Node</CardTitle>
          {node && (
            <Badge variant="secondary" className="uppercase tracking-wide">
              {node.status ?? "pending"}
            </Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          {node ? "Read-only task configuration." : "Select a node to inspect it."}
        </p>
      </CardHeader>
      <Separator />
      <CardContent className="space-y-4 pt-4">
        {!node && <p className="text-sm text-muted-foreground">No node selected.</p>}
        {node && (
          <div className="space-y-4">
            <Field label="Node" value={node.label ?? node.id} />
            <Field label="Node ID" value={node.id} />
            <Field label="Type" value={node.type} />
            {hasConfig && (
              <div className="space-y-1 border-t border-border/60 pt-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Config
                </p>
                <ScrollArea className="max-h-60 rounded border border-border/60 bg-background">
                  <pre className="p-2 font-mono text-[11px] leading-relaxed text-foreground">
                    {JSON.stringify(node.config, null, 2)}
                  </pre>
                </ScrollArea>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

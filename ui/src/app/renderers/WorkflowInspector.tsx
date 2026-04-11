import { useMemo } from "react";
import { buildMetadataFields, type MetadataField } from "@/app/renderers/metadata";
import type { RendererProps, WorkflowNodeMetadata } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";

const buildNodeFields = (node: WorkflowNodeMetadata | null): MetadataField[] => {
  if (!node) {
    return [{ label: "Node", value: "No node metadata available" }];
  }

  return [
    { label: "Node", value: node.label },
    { label: "Node ID", value: node.nodeId },
    { label: "Type", value: node.nodeType },
    { label: "Status", value: node.status },
    { label: "Description", value: node.description },
  ];
};

export const WorkflowInspector = ({
  selection,
  snapshot,
  inspectorTarget,
}: RendererProps): JSX.Element => {
  const isLoading = false;
  const workflow = snapshot.workflows.find((item) => item.id === selection.objectId) ?? null;

  const workflowFields = useMemo<MetadataField[]>(() => {
    return buildMetadataFields(selection, snapshot);
  }, [selection, snapshot]);

  const nodeFields = useMemo<MetadataField[]>(() => {
    if (inspectorTarget.kind !== "workflow-node" || !workflow?.graph) {
      return [];
    }
    const node =
      workflow.graph.nodes.find((item) => item.nodeId === inspectorTarget.nodeId) ?? null;
    return buildNodeFields(node);
  }, [inspectorTarget, workflow]);

  const showingNode = inspectorTarget.kind === "workflow-node";

  return (
    <Card className="h-full border-border/60 bg-muted/30">
      <CardHeader className="space-y-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-semibold">Inspector</CardTitle>
          <Badge variant="secondary" className="uppercase tracking-wide">
            {showingNode ? "node" : selection.objectType}
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground">
          Metadata is read-only and synchronized from the backend.
        </p>
      </CardHeader>
      <Separator />
      <CardContent className="space-y-4 pt-4">
        {isLoading && (
          <div className="space-y-2">
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-4 w-1/2" />
            <Skeleton className="h-4 w-3/5" />
          </div>
        )}
        {!isLoading && (
          <div className="space-y-4">
            <div className="space-y-2">
              {workflowFields.map((field) => (
                <div key={field.label} className="space-y-1">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    {field.label}
                  </p>
                  <p className="text-sm font-medium text-foreground">{field.value}</p>
                </div>
              ))}
            </div>
            {showingNode && nodeFields.length > 0 && (
              <div className="space-y-2 border-t border-border/60 pt-4">
                {nodeFields.map((field) => (
                  <div key={field.label} className="space-y-1">
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      {field.label}
                    </p>
                    <p className="text-sm font-medium text-foreground">{field.value}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

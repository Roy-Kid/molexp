import { useMemo } from "react";
import { buildMetadataFields, type MetadataField } from "@/app/renderers/metadata";
import type { RendererProps } from "@/app/types";
import { Skeleton } from "@/components/ui/skeleton";
import { NodeInspector, NodeInspectorRow, NodeInspectorSection } from "@/components/workbench";
import type { TaskNodeJson } from "@/components/workflow/task-graph-ir";

const buildNodeFields = (node: TaskNodeJson | null): MetadataField[] => {
  if (!node) {
    return [{ label: "Node", value: "No node metadata available" }];
  }

  return [
    { label: "Node", value: node.label ?? node.id },
    { label: "Node ID", value: node.id },
    { label: "Type", value: node.type },
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
      workflow.graph.task_configs.find((item) => item.id === inspectorTarget.nodeId) ?? null;
    return buildNodeFields(node);
  }, [inspectorTarget, workflow]);

  const showingNode = inspectorTarget.kind === "workflow-node";

  return (
    <NodeInspector
      title={showingNode ? "Node" : "Workflow"}
      subtitle={showingNode ? inspectorTarget.nodeId : selection.objectId}
      identity={selection.objectType}
    >
      {isLoading && (
        <div className="space-y-2">
          <Skeleton className="h-4 w-2/3" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-4 w-3/5" />
        </div>
      )}
      {!isLoading && (
        <>
          <NodeInspectorSection title="Metadata">
            {workflowFields.map((field) => (
              <NodeInspectorRow
                key={field.label}
                label={field.label}
                value={field.value}
                mono={field.label.toLowerCase().includes("id")}
              />
            ))}
          </NodeInspectorSection>
          {showingNode && nodeFields.length > 0 && (
            <NodeInspectorSection title="Node">
              {nodeFields.map((field) => (
                <NodeInspectorRow
                  key={field.label}
                  label={field.label}
                  value={field.value}
                  mono={field.label.toLowerCase().includes("id")}
                />
              ))}
            </NodeInspectorSection>
          )}
        </>
      )}
    </NodeInspector>
  );
};

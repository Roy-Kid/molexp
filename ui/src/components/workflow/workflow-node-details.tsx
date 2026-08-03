/**
 * WorkflowNodeDetails — a self-contained, read-only detail surface for one
 * task node of the canonical {@link TaskGraphJson} IR.
 *
 * Unlike the app's `WorkflowInspector` (which reads the workspace snapshot /
 * selection store), this component is pure: it takes a plain {@link TaskNodeJson}
 * (or `null`) and renders its id / type / status / static config.
 */

import type { JSX } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  NodeInspector,
  NodeInspectorRow,
  NodeInspectorSection,
  RunStatusBadge,
} from "@/components/workbench";
import type { TaskNodeJson } from "@/components/workflow/task-graph-ir";
import { cn } from "@/lib/utils";

export interface WorkflowNodeDetailsProps {
  /** The selected task node, or `null` when nothing is selected. */
  node: TaskNodeJson | null;
  className?: string;
}

export const WorkflowNodeDetails = ({ node, className }: WorkflowNodeDetailsProps): JSX.Element => {
  const hasConfig = node?.config && Object.keys(node.config).length > 0;

  if (!node) {
    return (
      <NodeInspector
        className={className}
        title="Node"
        empty
        emptyHint="Select a node to inspect it."
      />
    );
  }

  return (
    <NodeInspector
      className={cn(className)}
      title={node.label ?? node.id}
      subtitle={node.type}
      identity={node.id}
      status={node.status ?? "pending"}
      footer={node.status ? <RunStatusBadge status={node.status} /> : undefined}
    >
      <NodeInspectorSection title="Identity">
        <NodeInspectorRow label="Node" value={node.label ?? node.id} />
        <NodeInspectorRow label="Node ID" value={node.id} mono />
        <NodeInspectorRow label="Type" value={node.type} mono />
      </NodeInspectorSection>
      {hasConfig && (
        <NodeInspectorSection title="Config">
          <ScrollArea className="max-h-60 rounded-[var(--radius-control)] border border-border bg-background">
            <pre className="p-2 font-mono text-micro leading-relaxed text-foreground">
              {JSON.stringify(node.config, null, 2)}
            </pre>
          </ScrollArea>
        </NodeInspectorSection>
      )}
    </NodeInspector>
  );
};

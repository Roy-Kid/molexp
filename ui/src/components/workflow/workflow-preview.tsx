/**
 * WorkflowPreview — the composable, app-decoupled read-only workflow surface.
 *
 * Give it either a parsed {@link TaskGraphJson} (`ir`) or a serialized IR string
 * (`source`, e.g. the contents of a `workflow.json` file) and it renders the
 * {@link WorkflowGraph} canvas next to a {@link WorkflowNodeDetails} panel,
 * wiring node-click selection between them with internal state.
 *
 * It depends on nothing from the app shell (no store / router / API), so it is
 * the single mount point every reuse site shares: the in-app file viewer and the
 * molexp VSCode extension webview both render `<WorkflowPreview source={...} />`.
 */

import { type JSX, useMemo, useState } from "react";
import { parseTaskGraphIr } from "@/components/workflow/flowgram-document";
import type { TaskGraphJson } from "@/components/workflow/task-graph-ir";
import { WorkflowGraph } from "@/components/workflow/workflow-graph";
import { WorkflowNodeDetails } from "@/components/workflow/workflow-node-details";

export interface WorkflowPreviewProps {
  /** Parsed IR. Mutually exclusive with `source` — `ir` wins when both given. */
  ir?: TaskGraphJson | null;
  /** Serialized IR string (a `workflow.json` payload); parsed internally. */
  source?: string | null;
  /** Canvas height in px (default 420). */
  height?: number;
  /** Hide the side details panel, showing only the graph. */
  hideDetails?: boolean;
  className?: string;
}

export const WorkflowPreview = ({
  ir,
  source,
  height = 420,
  hideDetails = false,
  className,
}: WorkflowPreviewProps): JSX.Element => {
  const graph = useMemo<TaskGraphJson | null>(
    () => ir ?? parseTaskGraphIr(source ?? null),
    [ir, source],
  );
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const selectedNode = useMemo(
    () => graph?.task_configs.find((task) => task.id === selectedId) ?? null,
    [graph, selectedId],
  );

  if (!graph) {
    return (
      <p
        className={`rounded-md border border-dashed border-border/60 bg-muted/10 px-3 py-2 text-xs italic text-muted-foreground ${className ?? ""}`}
      >
        Not a workflow IR document.
      </p>
    );
  }

  if (hideDetails) {
    return (
      <WorkflowGraph ir={graph} height={height} onNodeClick={setSelectedId} className={className} />
    );
  }

  return (
    <div className={`flex gap-3 ${className ?? ""}`} style={{ height }}>
      <WorkflowGraph
        ir={graph}
        height={height}
        onNodeClick={setSelectedId}
        className="min-w-0 flex-1"
      />
      <div className="w-72 shrink-0 overflow-hidden">
        <WorkflowNodeDetails node={selectedNode} />
      </div>
    </div>
  );
};

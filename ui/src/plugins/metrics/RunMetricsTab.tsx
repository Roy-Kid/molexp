import { AlertTriangle } from "lucide-react";
import type { JSX } from "react";
import { EmptyState } from "@/app/components/entity";
import { RunMetricsView } from "@/app/runs/metrics/RunMetricsView";
import type { RendererProps } from "@/app/types";
import type { DiscoveredFile } from "@/plugins/types";

type RunMetricsTabProps = RendererProps & { discoveredFiles?: DiscoveredFile[] };

/**
 * Workspace-explorer metrics tab: a thin wrapper that resolves the run from
 * the workspace `snapshot` and delegates rendering to the coord-driven
 * {@link RunMetricsView}. The metrics rendering core lives in RunMetricsView
 * so the runs-dashboard `RunInspector` can reuse it without `snapshot.runs`.
 */
export const RunMetricsTab = ({ selection, snapshot }: RunMetricsTabProps): JSX.Element => {
  const run = snapshot.runs.find((item) => item.id === selection.objectId) ?? null;

  if (!run) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState
          icon={<AlertTriangle className="h-6 w-6" />}
          title="Run not found"
          description="It may have been deleted or not yet synced."
        />
      </div>
    );
  }

  return (
    <RunMetricsView
      key={run.id}
      projectId={run.projectId}
      experimentId={run.experimentId}
      runId={run.id}
    />
  );
};

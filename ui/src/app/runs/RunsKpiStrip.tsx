import type { JSX } from "react";

import { OverviewHighlight } from "@/app/components/entity";
import { formatDuration } from "@/lib/format-time";

import type { WorkspaceRunsStats } from "./types";

interface RunsKpiStripProps {
  stats: WorkspaceRunsStats;
  avgWaitSeconds: number | null;
}

export const RunsKpiStrip = ({ stats, avgWaitSeconds }: RunsKpiStripProps): JSX.Element => (
  <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
    <OverviewHighlight label="Running" value={stats.running} />
    <OverviewHighlight label="Pending" value={stats.pending} />
    <OverviewHighlight label="Failed" value={stats.failed} />
    <OverviewHighlight label="Succeeded" value={stats.succeeded} />
    <OverviewHighlight
      label="Avg wait (24h)"
      value={formatDuration(avgWaitSeconds)}
      detail="submit → start"
    />
  </div>
);

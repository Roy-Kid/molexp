import type { JSX } from "react";

import { formatDuration } from "@/lib/format-time";

import type { KpiSparklines } from "./aggregates";
import { RunsKpiCard } from "./RunsKpiCard";
import type { WorkspaceRunsStats } from "./types";

interface RunsKpiStripProps {
  stats: WorkspaceRunsStats;
  avgWaitSeconds: number | null;
  sparklines?: KpiSparklines;
}

export const RunsKpiStrip = ({
  stats,
  avgWaitSeconds,
  sparklines,
}: RunsKpiStripProps): JSX.Element => (
  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
    <RunsKpiCard
      label="Running"
      value={stats.running}
      tone="running"
      sparkline={sparklines?.running.series}
      delta={sparklines?.running.delta ?? null}
      deltaSuffix="in last hour"
      muted={stats.running === 0}
    />
    <RunsKpiCard
      label="Pending"
      value={stats.pending}
      tone="pending"
      sparkline={sparklines?.pending.series}
      delta={sparklines?.pending.delta ?? null}
      deltaSuffix="in last hour"
      muted={stats.pending === 0}
    />
    <RunsKpiCard
      label="Failed"
      value={stats.failed}
      tone="failed"
      invertDelta
      sparkline={sparklines?.failed.series}
      delta={sparklines?.failed.delta ?? null}
      deltaSuffix="in last hour"
      muted={stats.failed === 0}
    />
    <RunsKpiCard
      label="Succeeded"
      value={stats.succeeded}
      tone="succeeded"
      sparkline={sparklines?.succeeded.series}
      delta={sparklines?.succeeded.delta ?? null}
      deltaSuffix="in last hour"
      muted={stats.succeeded === 0}
    />
    <RunsKpiCard
      label="Avg wait (24h)"
      value={formatDuration(avgWaitSeconds)}
      tone="neutral"
      detail="submit → start"
      muted={avgWaitSeconds == null}
    />
  </div>
);

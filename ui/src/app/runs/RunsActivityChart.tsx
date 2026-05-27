import type { JSX } from "react";
import { useMemo } from "react";

import { MolvisBarChart } from "@/lib/charts";

import type { ActivityBucket } from "./aggregates";

interface RunsActivityChartProps {
  buckets: ActivityBucket[];
}

const SERIES_COLORS = {
  succeeded: "#10b981",
  failed: "#ef4444",
  cancelled: "#71717a",
  started: "#3b82f6",
};

const formatHour = (date: Date): string =>
  `${date.getMonth() + 1}/${date.getDate()} ${date.getHours().toString().padStart(2, "0")}:00`;

export const RunsActivityChart = ({ buckets }: RunsActivityChartProps): JSX.Element => {
  const config = useMemo(() => {
    const xs = buckets.map((bucket) => bucket.hour.toISOString());
    const labels = buckets.map((bucket) => formatHour(bucket.hour));

    const mkPoints = (sel: (b: ActivityBucket) => number) =>
      buckets.map((bucket, i) => ({ x: xs[i], y: sel(bucket), text: labels[i] }));

    return {
      mode: "stack" as const,
      hovermode: "x unified" as const,
      showLegend: false,
      bargap: 0.1,
      modebar: false,
      xAxis: {
        dtype: "date" as const,
        tickformat: "%H:00",
        nticks: 12,
      },
      yAxis: {
        label: "runs",
        rangemode: "nonnegative" as const,
      },
      series: [
        {
          id: "succeeded",
          label: "Succeeded",
          color: SERIES_COLORS.succeeded,
          points: mkPoints((b) => b.succeeded),
          hovertemplate: "<b>%{text}</b><br>Succeeded: %{y}<extra></extra>",
        },
        {
          id: "failed",
          label: "Failed",
          color: SERIES_COLORS.failed,
          points: mkPoints((b) => b.failed),
          hovertemplate: "<b>%{text}</b><br>Failed: %{y}<extra></extra>",
        },
        {
          id: "cancelled",
          label: "Cancelled",
          color: SERIES_COLORS.cancelled,
          points: mkPoints((b) => b.cancelled),
          hovertemplate: "<b>%{text}</b><br>Cancelled: %{y}<extra></extra>",
        },
        {
          id: "started",
          label: "Started",
          type: "line" as const,
          color: SERIES_COLORS.started,
          points: mkPoints((b) => b.started),
          hovertemplate: "<b>%{text}</b><br>Started: %{y}<extra></extra>",
        },
      ],
      theme: "auto" as const,
    };
  }, [buckets]);

  const totals = useMemo(
    () =>
      buckets.reduce(
        (acc, bucket) => ({
          started: acc.started + bucket.started,
          succeeded: acc.succeeded + bucket.succeeded,
          failed: acc.failed + bucket.failed,
          cancelled: acc.cancelled + bucket.cancelled,
        }),
        { started: 0, succeeded: 0, failed: 0, cancelled: 0 },
      ),
    [buckets],
  );

  return (
    <div className="rounded border border-border bg-background p-3">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          Activity · last 24h
        </div>
        <div className="flex flex-wrap gap-x-3 text-[10px] text-muted-foreground">
          <LegendDot color={SERIES_COLORS.started} label={`Started ${totals.started}`} />
          <LegendDot color={SERIES_COLORS.succeeded} label={`Succeeded ${totals.succeeded}`} />
          <LegendDot color={SERIES_COLORS.failed} label={`Failed ${totals.failed}`} />
          <LegendDot color={SERIES_COLORS.cancelled} label={`Cancelled ${totals.cancelled}`} />
        </div>
      </div>
      <MolvisBarChart config={config} style={{ width: "100%", height: "200px" }} />
    </div>
  );
};

interface LegendDotProps {
  color: string;
  label: string;
}

const LegendDot = ({ color, label }: LegendDotProps): JSX.Element => (
  <span className="inline-flex items-center gap-1">
    <span
      aria-hidden="true"
      className="inline-block h-2 w-2 rounded-full"
      style={{ backgroundColor: color }}
    />
    {label}
  </span>
);

import { useMemo } from "react";
import type { JSX } from "react";

import { Plot } from "@/lib/plot";

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

const CHART_LAYOUT = {
  barmode: "stack" as const,
  bargap: 0.1,
  height: 200,
  margin: { l: 36, r: 12, t: 8, b: 32 },
  xaxis: {
    type: "date" as const,
    showgrid: false,
    tickfont: { size: 10 },
    tickformat: "%H:00",
    nticks: 12,
  },
  yaxis: {
    showgrid: true,
    gridcolor: "rgba(125,125,125,0.15)",
    tickfont: { size: 10 },
    rangemode: "nonnegative" as const,
    title: { text: "runs", font: { size: 10 }, standoff: 6 },
  },
  showlegend: false,
  hovermode: "x unified" as const,
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
};

const CHART_CONFIG = { displayModeBar: false, responsive: true };
const CHART_STYLE = { width: "100%" };

const formatHour = (date: Date): string =>
  `${date.getMonth() + 1}/${date.getDate()} ${date.getHours().toString().padStart(2, "0")}:00`;

export const RunsActivityChart = ({ buckets }: RunsActivityChartProps): JSX.Element => {
  const traces = useMemo(() => {
    if (buckets.length === 0) return [];
    const hours = buckets.map((bucket) => bucket.hour.toISOString());
    const labels = buckets.map((bucket) => formatHour(bucket.hour));
    return [
      {
        type: "bar",
        name: "Succeeded",
        x: hours,
        y: buckets.map((bucket) => bucket.succeeded),
        text: labels,
        marker: { color: SERIES_COLORS.succeeded },
        hovertemplate: "<b>%{text}</b><br>Succeeded: %{y}<extra></extra>",
      },
      {
        type: "bar",
        name: "Failed",
        x: hours,
        y: buckets.map((bucket) => bucket.failed),
        text: labels,
        marker: { color: SERIES_COLORS.failed },
        hovertemplate: "<b>%{text}</b><br>Failed: %{y}<extra></extra>",
      },
      {
        type: "bar",
        name: "Cancelled",
        x: hours,
        y: buckets.map((bucket) => bucket.cancelled),
        text: labels,
        marker: { color: SERIES_COLORS.cancelled },
        hovertemplate: "<b>%{text}</b><br>Cancelled: %{y}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "Started",
        x: hours,
        y: buckets.map((bucket) => bucket.started),
        text: labels,
        line: { color: SERIES_COLORS.started, width: 2 },
        marker: { size: 4, color: SERIES_COLORS.started },
        hovertemplate: "<b>%{text}</b><br>Started: %{y}<extra></extra>",
      },
    ];
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
      <Plot
        data={traces}
        layout={CHART_LAYOUT}
        config={CHART_CONFIG}
        style={CHART_STYLE}
        useResizeHandler
      />
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

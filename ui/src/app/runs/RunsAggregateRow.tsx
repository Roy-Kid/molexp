import { useMemo } from "react";
import type { JSX } from "react";

import { Plot } from "@/lib/plot";
import { cn } from "@/lib/utils";

import type {
  BackendDistributionEntry,
  FailingExperimentEntry,
} from "./aggregates";

interface RunsAggregateRowProps {
  backendDistribution: BackendDistributionEntry[];
  topFailing: FailingExperimentEntry[];
  onSelectBackend: (backend: string) => void;
  onSelectExperiment: (entry: FailingExperimentEntry) => void;
}

export const RunsAggregateRow = ({
  backendDistribution,
  topFailing,
  onSelectBackend,
  onSelectExperiment,
}: RunsAggregateRowProps): JSX.Element => (
  <div className="grid gap-4 lg:grid-cols-2">
    <BackendDistributionChart
      distribution={backendDistribution}
      onSelectBackend={onSelectBackend}
    />
    <TopFailingList entries={topFailing} onSelect={onSelectExperiment} />
  </div>
);

interface BackendDistributionChartProps {
  distribution: BackendDistributionEntry[];
  onSelectBackend: (backend: string) => void;
}

const CLUSTER_PALETTE = [
  "#3b82f6",
  "#8b5cf6",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#06b6d4",
  "#a855f7",
  "#84cc16",
];

const BACKEND_CHART_LAYOUT = {
  barmode: "stack" as const,
  height: 160,
  margin: { l: 64, r: 12, t: 8, b: 28 },
  xaxis: { showgrid: true, gridcolor: "rgba(125,125,125,0.15)" },
  yaxis: { automargin: true, tickfont: { size: 11 } },
  showlegend: true,
  legend: { orientation: "h" as const, y: -0.3, x: 0, font: { size: 10 } },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
};

const BACKEND_CHART_CONFIG = { displayModeBar: false, responsive: true };
const BACKEND_CHART_STYLE = { width: "100%" };

const BackendDistributionChart = ({
  distribution,
  onSelectBackend,
}: BackendDistributionChartProps): JSX.Element => {
  const traces = useMemo(() => {
    if (distribution.length === 0) return [];
    const backends = Array.from(new Set(distribution.map((entry) => entry.backend)));
    const clusters = Array.from(
      new Set(distribution.map((entry) => entry.cluster ?? "—")),
    );
    return clusters.map((clusterName, index) => {
      const x = backends.map((backend) => {
        const match = distribution.find(
          (entry) => entry.backend === backend && (entry.cluster ?? "—") === clusterName,
        );
        return match?.count ?? 0;
      });
      return {
        type: "bar",
        orientation: "h",
        name: clusterName === "—" ? "(no cluster)" : clusterName,
        y: backends,
        x,
        customdata: backends,
        hovertemplate: `<b>%{y}</b> · ${
          clusterName === "—" ? "(no cluster)" : clusterName
        }<br>%{x} runs<extra></extra>`,
        marker: { color: CLUSTER_PALETTE[index % CLUSTER_PALETTE.length] },
      };
    });
  }, [distribution]);

  if (distribution.length === 0) {
    return (
      <PanelShell title="Backend / cluster distribution">
        <EmptyMessage>No active runs to break down.</EmptyMessage>
      </PanelShell>
    );
  }

  const handleClick = (event: { points: Array<{ customdata?: unknown }> }): void => {
    const point = event.points?.[0];
    const backend = point?.customdata;
    if (typeof backend === "string") onSelectBackend(backend);
  };

  return (
    <PanelShell title="Backend / cluster distribution">
      <Plot
        data={traces}
        layout={BACKEND_CHART_LAYOUT}
        config={BACKEND_CHART_CONFIG}
        style={BACKEND_CHART_STYLE}
        useResizeHandler
        onClick={handleClick}
      />
    </PanelShell>
  );
};

interface TopFailingListProps {
  entries: FailingExperimentEntry[];
  onSelect: (entry: FailingExperimentEntry) => void;
}

const TopFailingList = ({ entries, onSelect }: TopFailingListProps): JSX.Element => {
  if (entries.length === 0) {
    return (
      <PanelShell title="Top failing experiments">
        <EmptyMessage>No failed runs in the current view.</EmptyMessage>
      </PanelShell>
    );
  }
  const maxFailed = entries[0]?.failedCount ?? 1;
  return (
    <PanelShell title="Top failing experiments">
      <ul className="divide-y divide-border/40">
        {entries.map((entry) => {
          const failedRatio = entry.failedCount / Math.max(entry.totalCount, 1);
          return (
            <li key={entry.experimentId}>
              <button
                type="button"
                onClick={() => onSelect(entry)}
                className="group flex w-full items-center gap-2 px-1 py-1.5 text-left transition-colors hover:bg-muted/40"
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate text-xs font-medium text-foreground">
                    {entry.experimentName}
                  </div>
                  <div className="truncate text-[10px] text-muted-foreground">
                    {entry.projectName}
                  </div>
                </div>
                <div className="flex w-16 flex-col items-end">
                  <div className="text-xs font-semibold text-destructive">
                    {entry.failedCount}/{entry.totalCount}
                  </div>
                  <div className="mt-0.5 h-1 w-full overflow-hidden rounded bg-muted">
                    <div
                      className={cn("h-full bg-destructive/70")}
                      style={{ width: `${(entry.failedCount / maxFailed) * 100}%` }}
                    />
                    <span className="sr-only">{Math.round(failedRatio * 100)}% failure</span>
                  </div>
                </div>
              </button>
            </li>
          );
        })}
      </ul>
    </PanelShell>
  );
};

interface PanelShellProps {
  title: string;
  children: JSX.Element | JSX.Element[];
}

const PanelShell = ({ title, children }: PanelShellProps): JSX.Element => (
  <div className="rounded border border-border bg-background p-3">
    <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
      {title}
    </div>
    {children}
  </div>
);

const EmptyMessage = ({ children }: { children: string }): JSX.Element => (
  <div className="py-6 text-center text-xs italic text-muted-foreground">{children}</div>
);

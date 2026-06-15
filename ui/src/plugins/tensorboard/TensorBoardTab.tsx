import { AlertTriangle, BarChart3, Download } from "lucide-react";
import { type JSX, useEffect, useMemo, useState } from "react";
import { EmptyState, OverviewSection } from "@/app/components/entity";
import type { TensorboardScalarSeries, TensorboardScalarsResponse } from "@/app/state/api";
import { TensorboardScalarsError, workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";
import { MolplotLineChart } from "@/plugins/molplot";
import type { DiscoveredFile } from "@/plugins/types";

type TensorBoardTabProps = RendererProps & { discoveredFiles?: DiscoveredFile[] };

type LoadState =
  | { kind: "loading" }
  | { kind: "missingDep"; message: string }
  | { kind: "error"; message: string }
  | { kind: "ready"; data: TensorboardScalarsResponse };

const formatValue = (value: number): string => {
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1000 || (abs > 0 && abs < 0.001)) return value.toExponential(3);
  return value.toPrecision(4);
};

interface SeriesCardProps {
  series: TensorboardScalarSeries;
  xMode: "step" | "wall";
  yScale: "linear" | "log";
}

const SeriesCard = ({ series, xMode, yScale }: SeriesCardProps): JSX.Element => {
  const config = useMemo(
    () => ({
      series: [
        {
          id: series.tag,
          label: series.tag,
          initialPoints: series.points.map((p) => ({
            x: xMode === "step" ? p.step : p.wallTime * 1000,
            y: p.value,
          })),
        },
      ],
      xAxis: {
        label: xMode === "step" ? "step" : "wall time",
        type: xMode === "wall" ? ("linear" as const) : ("linear" as const),
      },
      yAxis: {
        label: series.tag,
        type: yScale,
      },
      modebar: true,
      theme: "auto" as const,
    }),
    [series, xMode, yScale],
  );

  const latest = series.points[series.points.length - 1]?.value ?? 0;
  return (
    <section className="min-w-0 rounded-md border border-border bg-background p-3">
      <div className="flex items-baseline justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium text-foreground">{series.tag}</div>
          <div className="font-mono text-[11px] text-muted-foreground">
            {series.points.length} pts · {series.logdir || "."}
          </div>
        </div>
        <div className="font-mono text-xs text-muted-foreground">{formatValue(latest)}</div>
      </div>
      <MolplotLineChart config={config} style={{ width: "100%", height: "220px" }} />
    </section>
  );
};

const groupByPrefix = (
  series: TensorboardScalarSeries[],
): Array<[string, TensorboardScalarSeries[]]> => {
  const buckets = new Map<string, TensorboardScalarSeries[]>();
  for (const s of series) {
    const slash = s.tag.indexOf("/");
    const group = slash > 0 ? s.tag.slice(0, slash) : "";
    const list = buckets.get(group) ?? [];
    list.push(s);
    buckets.set(group, list);
  }
  return [...buckets.entries()].sort(([a], [b]) => a.localeCompare(b));
};

export const TensorBoardTab = ({ selection, snapshot }: TensorBoardTabProps): JSX.Element => {
  const run = useMemo(
    () => snapshot.runs.find((r) => r.id === selection.objectId) ?? null,
    [snapshot.runs, selection.objectId],
  );
  const [state, setState] = useState<LoadState>({ kind: "loading" });
  const [search, setSearch] = useState("");
  const [xMode, setXMode] = useState<"step" | "wall">("step");
  const [yScale, setYScale] = useState<"linear" | "log">("linear");

  useEffect(() => {
    if (!run) {
      setState({ kind: "error", message: "Run not found" });
      return;
    }
    let cancelled = false;
    setState({ kind: "loading" });
    workspaceApi
      .getRunTensorboardScalars(run.projectId, run.experimentId, run.id)
      .then((data) => {
        if (cancelled) return;
        setState({ kind: "ready", data });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof TensorboardScalarsError) {
          setState({
            kind: err.status === 503 ? "missingDep" : "error",
            message: err.message,
          });
        } else {
          setState({
            kind: "error",
            message: err instanceof Error ? err.message : "Failed to load scalars",
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [run]);

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

  if (state.kind === "loading") {
    return (
      <div className="flex h-full items-center justify-center bg-background text-sm text-muted-foreground">
        Loading TensorBoard scalars…
      </div>
    );
  }

  if (state.kind === "missingDep") {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState
          icon={<Download className="h-6 w-6" />}
          title="TensorBoard not installed"
          description={state.message}
        />
      </div>
    );
  }

  if (state.kind === "error") {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState
          icon={<AlertTriangle className="h-6 w-6" />}
          title="Cannot read scalars"
          description={state.message}
        />
      </div>
    );
  }

  const { data } = state;
  const filtered = search
    ? data.series.filter((s) => s.tag.toLowerCase().includes(search.toLowerCase()))
    : data.series;
  const grouped = groupByPrefix(filtered);

  return (
    <div className="flex-1 overflow-auto bg-background">
      <div className="mx-auto flex max-w-6xl flex-col gap-5 px-4 py-4 md:px-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
            <div className="text-sm font-medium text-foreground">TensorBoard</div>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
            <span>{data.logdirs.length} logdir</span>
            <span>{data.series.length} series</span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 rounded-md border border-border bg-muted/30 px-3 py-2 text-xs">
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter tags"
            className="h-7 w-56 rounded border border-border bg-background px-2 text-xs"
          />
          <div className="flex items-center gap-1">
            <span className="font-medium text-foreground">X</span>
            {(["step", "wall"] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setXMode(mode)}
                className={`rounded px-2 py-0.5 transition-colors ${
                  xMode === mode
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                }`}
              >
                {mode === "step" ? "Step" : "Wall"}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1">
            <span className="font-medium text-foreground">Y</span>
            {(["linear", "log"] as const).map((scale) => (
              <button
                key={scale}
                type="button"
                onClick={() => setYScale(scale)}
                className={`rounded px-2 py-0.5 transition-colors ${
                  yScale === scale
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                }`}
              >
                {scale === "linear" ? "Linear" : "Log"}
              </button>
            ))}
          </div>
        </div>

        {data.series.length === 0 ? (
          <EmptyState
            icon={<BarChart3 className="h-6 w-6" />}
            title="No scalars found"
            description={`Discovered ${data.logdirs.length} logdir(s) but no scalar tags.`}
          />
        ) : grouped.length === 0 ? (
          <OverviewSection title="Scalars">
            <div className="rounded-md border border-dashed border-border px-3 py-6 text-center text-sm text-muted-foreground">
              No tags match the current filter.
            </div>
          </OverviewSection>
        ) : (
          grouped.map(([groupName, items]) => (
            <OverviewSection key={groupName || "_root"} title={groupName || "Scalars"}>
              <div className="grid gap-3 lg:grid-cols-2">
                {items.map((series) => (
                  <SeriesCard
                    key={`${series.logdir}::${series.tag}`}
                    series={series}
                    xMode={xMode}
                    yScale={yScale}
                  />
                ))}
              </div>
            </OverviewSection>
          ))
        )}
      </div>
    </div>
  );
};

import { Activity, AlertTriangle, BarChart3, Maximize2, Wrench } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { EmptyState, OverviewSection } from "@/app/components/entity";
import type { MetricRecord } from "@/app/state/api";
import { workspaceApi } from "@/app/state/api";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { smoothEma } from "@/plugins/metrics/smoothing";
import { MolplotLineChart } from "@/plugins/molplot";

/**
 * Coord-driven run-metrics view: polls `getRunMetrics` for the run named by
 * its explicit `{projectId, experimentId, runId}` props and renders a molplot
 * line chart per scalar series. It deliberately takes explicit coordinates
 * rather than resolving the run from a workspace tree — so every consumer (the
 * workspace-explorer `RunMetricsTab` wrapper and the runs-dashboard
 * `RunInspector`) supplies the run identity it already holds directly.
 *
 * The pure builders (`buildScalarSeries`, `groupSeries`, `buildLineChartConfig`)
 * are exported for unit testing under the repo's node test environment.
 */
const POLL_INTERVAL_MS = 1500;

type XMode = "step" | "wall";
type YScale = "linear" | "log";

interface ScalarPoint {
  step: number;
  wall: number;
  y: number;
}

export interface ScalarSeries {
  key: string;
  group: string;
  points: ScalarPoint[];
  latest: number;
}

export interface RunMetricsViewProps {
  projectId: string;
  experimentId: string;
  runId: string;
}

const isFiniteNumber = (value: unknown): value is number => {
  return typeof value === "number" && Number.isFinite(value);
};

const formatValue = (value: number): string => {
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(3);
  }
  return value.toPrecision(4);
};

const parseWall = (raw?: string): number => {
  if (!raw) return Number.NaN;
  const t = Date.parse(raw);
  return Number.isFinite(t) ? t : Number.NaN;
};

export const buildScalarSeries = (records: MetricRecord[]): ScalarSeries[] => {
  const grouped = new Map<string, ScalarPoint[]>();

  records.forEach((record, index) => {
    if (record.t !== "scalar" || !isFiniteNumber(record.v)) {
      return;
    }
    const points = grouped.get(record.k) ?? [];
    const wall = parseWall(record.w);
    points.push({
      step: isFiniteNumber(record.s) ? record.s : index,
      wall: Number.isFinite(wall) ? wall : index,
      y: record.v,
    });
    grouped.set(record.k, points);
  });

  return Array.from(grouped.entries())
    .map(([key, points]) => {
      points.sort((a, b) => a.step - b.step);
      const slash = key.indexOf("/");
      const group = slash > 0 ? key.slice(0, slash) : "";
      return {
        key,
        group,
        points,
        latest: points[points.length - 1]?.y ?? 0,
      };
    })
    .sort((left, right) => left.key.localeCompare(right.key));
};

export const groupSeries = (series: ScalarSeries[]): Array<[string, ScalarSeries[]]> => {
  const buckets = new Map<string, ScalarSeries[]>();
  for (const s of series) {
    const list = buckets.get(s.group) ?? [];
    list.push(s);
    buckets.set(s.group, list);
  }
  return Array.from(buckets.entries()).sort(([a], [b]) => a.localeCompare(b));
};

// Exported so the multi-run aggregation view (aggregateSeries.ts) cycles the
// same 8 colors across runs that the single-run view cycles across series.
export const PALETTE = [
  "#2563eb", // blue
  "#dc2626", // red
  "#16a34a", // green
  "#d97706", // amber
  "#7c3aed", // violet
  "#0891b2", // cyan
  "#db2777", // pink
  "#65a30d", // lime
];

interface ChartConfigOptions {
  xMode: XMode;
  yScale: YScale;
  smoothing: number;
  color: string;
  /** Show the per-chart plotly modebar. Default false (hidden until toggled). */
  showToolbar?: boolean;
}

// Opacity applied to the raw signal trace when a smoothed overlay is drawn on
// top of it, so the smoothed curve reads as the primary line and the raw noise
// recedes into the background.
const RAW_TRACE_OPACITY = 0.3;

export const buildLineChartConfig = (series: ScalarSeries, options: ChartConfigOptions) => {
  const { xMode, yScale, smoothing, color, showToolbar = false } = options;
  const xs = series.points.map((p) => (xMode === "step" ? p.step : p.wall));
  const ys = series.points.map((p) => p.y);
  const smoothed = smoothing > 0 ? smoothEma(ys, smoothing) : null;
  // Restore the two-trace overlay (faded raw behind the smoothed curve) — the
  // migration to a single series hid noise that the smoothing slider is meant
  // to reveal vs. the raw signal.
  const seriesList = smoothed
    ? [
        {
          id: `${series.key}::raw`,
          label: "raw",
          color,
          width: 1,
          opacity: RAW_TRACE_OPACITY,
          initialPoints: ys.map((y, i) => ({ x: xs[i], y })),
        },
        {
          id: series.key,
          label: series.key,
          color,
          width: 2,
          initialPoints: smoothed.map((y, i) => ({ x: xs[i], y })),
        },
      ]
    : [
        {
          id: series.key,
          label: series.key,
          color,
          mode: "lines+markers" as const,
          initialPoints: ys.map((y, i) => ({ x: xs[i], y })),
        },
      ];
  return {
    series: seriesList,
    xAxis: {
      label: xMode === "step" ? "step" : "wall time",
      type: "linear" as const,
    },
    yAxis: { type: yScale, label: series.key },
    hovertemplate: "%{y:.6g}<extra></extra>",
    hovermode: "x unified" as const,
    modebar: showToolbar,
    modebarRemove: [
      "lasso2d",
      "select2d",
      "toggleSpikelines",
      "hoverClosestCartesian",
      "hoverCompareCartesian",
    ],
    theme: "auto" as const,
  };
};

interface ChartProps {
  series: ScalarSeries;
  xMode: XMode;
  yScale: YScale;
  smoothing: number;
  color: string;
  showToolbar: boolean;
  height: string;
}

const MetricChart = ({
  series,
  xMode,
  yScale,
  smoothing,
  color,
  showToolbar,
  height,
}: ChartProps): JSX.Element => {
  const config = useMemo(
    () => buildLineChartConfig(series, { xMode, yScale, smoothing, color, showToolbar }),
    [series, xMode, yScale, smoothing, color, showToolbar],
  );

  return <MolplotLineChart config={config} style={{ width: "100%", height }} />;
};

interface MetricPanelProps {
  series: ScalarSeries;
  xMode: XMode;
  yScale: YScale;
  smoothing: number;
  color: string;
}

/**
 * One scalar series tile. Owns two pieces of local view state that are
 * deliberately per-panel rather than global: whether the plotly modebar is
 * revealed (hidden by default to keep the grid calm) and whether the panel is
 * blown up into a focus dialog. The enlarge dialog renders a second, taller
 * chart instance with the toolbar always on.
 */
const MetricPanel = ({
  series,
  xMode,
  yScale,
  smoothing,
  color,
}: MetricPanelProps): JSX.Element => {
  const [showToolbar, setShowToolbar] = useState(false);
  const [enlarged, setEnlarged] = useState(false);

  // A one-shot scalar (a single recorded point, no step progression) is a VALUE,
  // not a time series — show the number, not an empty one-point chart.
  if (series.points.length <= 1) {
    return (
      <section className="min-w-0 rounded-md border border-border bg-background p-3">
        <div className="truncate text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          {series.key}
        </div>
        <div className="mt-1 font-mono text-2xl font-semibold tabular-nums text-foreground">
          {formatValue(series.latest)}
        </div>
      </section>
    );
  }

  return (
    <section className="min-w-0 rounded-md border border-border bg-background p-3">
      <div className="flex items-baseline justify-between gap-3">
        <div className="min-w-0 truncate text-sm font-medium text-foreground">{series.key}</div>
        <div className="flex shrink-0 items-center gap-2">
          <span className="font-mono text-xs text-muted-foreground">
            {formatValue(series.latest)}
          </span>
          <button
            type="button"
            onClick={() => setShowToolbar((value) => !value)}
            aria-pressed={showToolbar}
            title={showToolbar ? "Hide chart toolbar" : "Show chart toolbar"}
            aria-label={showToolbar ? "Hide chart toolbar" : "Show chart toolbar"}
            className={`rounded p-1 transition-colors ${
              showToolbar
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            }`}
          >
            <Wrench className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={() => setEnlarged(true)}
            title="Enlarge chart"
            aria-label="Enlarge chart"
            className="rounded p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
      <MetricChart
        series={series}
        xMode={xMode}
        yScale={yScale}
        smoothing={smoothing}
        color={color}
        showToolbar={showToolbar}
        height="220px"
      />
      <Dialog open={enlarged} onOpenChange={setEnlarged}>
        <DialogContent className="max-w-5xl">
          <DialogHeader>
            <DialogTitle className="truncate font-mono text-sm">{series.key}</DialogTitle>
          </DialogHeader>
          <MetricChart
            series={series}
            xMode={xMode}
            yScale={yScale}
            smoothing={smoothing}
            color={color}
            showToolbar
            height="70vh"
          />
        </DialogContent>
      </Dialog>
    </section>
  );
};

const OtherRecords = ({ records }: { records: MetricRecord[] }): JSX.Element | null => {
  const nonScalar = records.filter((record) => record.t !== "scalar").slice(-20);
  if (nonScalar.length === 0) {
    return null;
  }

  return (
    <OverviewSection title="Other Events">
      <div className="divide-y divide-border rounded-md border border-border">
        {nonScalar.map((record) => (
          <div
            key={`${record.k}:${record.t}:${record.s ?? ""}:${record.w ?? ""}`}
            className="grid gap-1 px-3 py-2 text-sm md:grid-cols-4"
          >
            <div className="min-w-0 font-medium text-foreground">{record.k}</div>
            <div className="text-muted-foreground">{record.t}</div>
            <div className="text-muted-foreground">{record.s ?? "-"}</div>
            <pre className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap font-mono text-xs text-muted-foreground">
              {JSON.stringify(record.v)}
            </pre>
          </div>
        ))}
      </div>
    </OverviewSection>
  );
};

interface ControlsProps {
  smoothing: number;
  xMode: XMode;
  yScale: YScale;
  onSmoothingChange: (value: number) => void;
  onXModeChange: (value: XMode) => void;
  onYScaleChange: (value: YScale) => void;
}

// Vertical control stack for the left sidebar. Each control is a labelled block
// laid out top-to-bottom (rather than the old horizontal toolbar) so it reads
// naturally in a narrow rail.
const ChartControls = ({
  smoothing,
  xMode,
  yScale,
  onSmoothingChange,
  onXModeChange,
  onYScaleChange,
}: ControlsProps): JSX.Element => (
  <div className="flex flex-col gap-4 text-xs">
    <div className="flex flex-col gap-1.5">
      <span className="font-medium text-foreground">Smoothing</span>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={0}
          max={0.99}
          step={0.01}
          value={smoothing}
          onChange={(event) => onSmoothingChange(Number(event.target.value))}
          className="h-1 flex-1 cursor-pointer accent-primary"
          aria-label="EMA smoothing weight"
        />
        <span className="w-8 text-right font-mono tabular-nums text-muted-foreground">
          {smoothing.toFixed(2)}
        </span>
      </div>
    </div>
    <div className="flex flex-col gap-1.5">
      <span className="font-medium text-foreground">X axis</span>
      <div className="grid grid-cols-2 gap-1">
        {(["step", "wall"] as const).map((mode) => (
          <button
            key={mode}
            type="button"
            onClick={() => onXModeChange(mode)}
            className={`rounded px-2 py-1 transition-colors ${
              xMode === mode
                ? "bg-primary text-primary-foreground"
                : "bg-muted/40 text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            }`}
          >
            {mode === "step" ? "Step" : "Wall"}
          </button>
        ))}
      </div>
    </div>
    <div className="flex flex-col gap-1.5">
      <span className="font-medium text-foreground">Y axis</span>
      <div className="grid grid-cols-2 gap-1">
        {(["linear", "log"] as const).map((scale) => (
          <button
            key={scale}
            type="button"
            onClick={() => onYScaleChange(scale)}
            className={`rounded px-2 py-1 transition-colors ${
              yScale === scale
                ? "bg-primary text-primary-foreground"
                : "bg-muted/40 text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            }`}
          >
            {scale === "linear" ? "Linear" : "Log"}
          </button>
        ))}
      </div>
    </div>
  </div>
);

export const RunMetricsView = ({
  projectId,
  experimentId,
  runId,
}: RunMetricsViewProps): JSX.Element => {
  const [records, setRecords] = useState<MetricRecord[]>([]);
  const [nextLine, setNextLine] = useState(0);
  const [parseErrors, setParseErrors] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [smoothing, setSmoothing] = useState(0.6);
  const [xMode, setXMode] = useState<XMode>("step");
  const [yScale, setYScale] = useState<YScale>("linear");

  const scalarSeries = useMemo(() => buildScalarSeries(records), [records]);
  const grouped = useMemo(() => groupSeries(scalarSeries), [scalarSeries]);

  // Track the latest tail position in a ref so the polling effect doesn't list
  // it as a dependency — otherwise every successful fetch (which advances
  // nextLine) tears down the interval and re-fires an immediate fetch,
  // collapsing POLL_INTERVAL_MS into a tight loop.
  const nextLineRef = useRef(nextLine);
  useEffect(() => {
    nextLineRef.current = nextLine;
  }, [nextLine]);

  // The fetch effect re-keys on the run coords; callers also mount this view
  // with `key={runId}` so a run switch remounts it with fresh state (cursor +
  // accumulator), rather than threading a manual reset effect.
  useEffect(() => {
    let cancelled = false;

    const fetchMetrics = async (): Promise<void> => {
      const sinceLine = nextLineRef.current;
      try {
        const response = await workspaceApi.getRunMetrics(projectId, experimentId, runId, {
          sinceLine,
        });
        if (cancelled) {
          return;
        }
        setRecords((current) =>
          sinceLine === 0 ? response.records : [...current, ...response.records],
        );
        setNextLine(response.nextLine);
        setParseErrors((current) => current + response.parseErrors);
        setError(null);
      } catch (metricsError) {
        if (!cancelled) {
          setError(metricsError instanceof Error ? metricsError.message : "Failed to load metrics");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void fetchMetrics();
    const intervalId = window.setInterval(() => {
      void fetchMetrics();
    }, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [projectId, experimentId, runId]);

  if (loading && records.length === 0) {
    return (
      <div className="flex h-full items-center justify-center bg-background text-sm text-muted-foreground">
        Loading metrics...
      </div>
    );
  }

  if (error && records.length === 0) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState
          icon={<AlertTriangle className="h-6 w-6" />}
          title="Metrics unavailable"
          description={error}
        />
      </div>
    );
  }

  if (records.length === 0) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState
          icon={<Activity className="h-6 w-6" />}
          title="No metrics recorded"
          description="This run has not written metrics yet."
        />
      </div>
    );
  }

  return (
    <div className="flex flex-1 overflow-hidden bg-background">
      <aside className="flex w-56 shrink-0 flex-col gap-4 overflow-y-auto border-r border-border bg-muted/20 px-4 py-4">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-muted-foreground" />
          <div className="text-sm font-medium text-foreground">Run Metrics</div>
        </div>

        {scalarSeries.length > 0 && (
          <ChartControls
            smoothing={smoothing}
            xMode={xMode}
            yScale={yScale}
            onSmoothingChange={setSmoothing}
            onXModeChange={setXMode}
            onYScaleChange={setYScale}
          />
        )}

        <div className="mt-auto flex flex-col gap-0.5 border-t border-border pt-3 text-xs text-muted-foreground">
          <span>{records.length} records</span>
          <span>{scalarSeries.length} scalar series</span>
          {parseErrors > 0 && <span>{parseErrors} parse errors</span>}
        </div>
      </aside>

      <div className="flex-1 overflow-auto">
        <div className="mx-auto flex max-w-6xl flex-col gap-5 px-4 py-4 md:px-6">
          {grouped.length > 0 ? (
            grouped.map(([groupName, items]) => (
              <OverviewSection key={groupName || "_root"} title={groupName ? groupName : "Scalars"}>
                <div className="grid gap-3 lg:grid-cols-2">
                  {items.map((series, index) => (
                    <MetricPanel
                      key={series.key}
                      series={series}
                      xMode={xMode}
                      yScale={yScale}
                      smoothing={smoothing}
                      color={PALETTE[index % PALETTE.length]}
                    />
                  ))}
                </div>
              </OverviewSection>
            ))
          ) : (
            <OverviewSection title="Scalars">
              <div className="rounded-md border border-dashed border-border px-3 py-6 text-center text-sm text-muted-foreground">
                No scalar metrics recorded.
              </div>
            </OverviewSection>
          )}

          <OtherRecords records={records} />
        </div>
      </div>
    </div>
  );
};

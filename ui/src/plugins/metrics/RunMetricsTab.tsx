import { Activity, AlertTriangle, BarChart3 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { EmptyState, OverviewSection } from "@/app/components/entity";
import type { MetricRecord } from "@/app/state/api";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";
import type { DiscoveredFile } from "@/plugins/types";
import { Plot } from "@/lib/plot";
import { smoothEma } from "./smoothing";

type RunMetricsTabProps = RendererProps & { discoveredFiles?: DiscoveredFile[] };

const POLL_INTERVAL_MS = 1500;

type XMode = "step" | "wall";
type YScale = "linear" | "log";

interface ScalarPoint {
  step: number;
  wall: number;
  y: number;
}

interface ScalarSeries {
  key: string;
  group: string;
  points: ScalarPoint[];
  latest: number;
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

const buildScalarSeries = (records: MetricRecord[]): ScalarSeries[] => {
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

const groupSeries = (series: ScalarSeries[]): Array<[string, ScalarSeries[]]> => {
  const buckets = new Map<string, ScalarSeries[]>();
  for (const s of series) {
    const list = buckets.get(s.group) ?? [];
    list.push(s);
    buckets.set(s.group, list);
  }
  return Array.from(buckets.entries()).sort(([a], [b]) => a.localeCompare(b));
};

const PALETTE = [
  "#2563eb", // blue
  "#dc2626", // red
  "#16a34a", // green
  "#d97706", // amber
  "#7c3aed", // violet
  "#0891b2", // cyan
  "#db2777", // pink
  "#65a30d", // lime
];

interface ChartProps {
  series: ScalarSeries;
  xMode: XMode;
  yScale: YScale;
  smoothing: number;
  color: string;
}

const MetricChart = ({ series, xMode, yScale, smoothing, color }: ChartProps): JSX.Element => {
  const xs = series.points.map((p) => (xMode === "step" ? p.step : p.wall));
  const ys = series.points.map((p) => p.y);
  const smoothed = smoothing > 0 ? smoothEma(ys, smoothing) : null;

  const data: Record<string, unknown>[] = smoothed
    ? [
        {
          type: "scatter",
          mode: "lines",
          name: "raw",
          x: xs,
          y: ys,
          line: { color, width: 1 },
          opacity: 0.25,
          hoverinfo: "skip",
          showlegend: false,
        },
        {
          type: "scatter",
          mode: "lines",
          name: "smoothed",
          x: xs,
          y: smoothed,
          line: { color, width: 2, shape: "spline", smoothing: 0.6 },
          hovertemplate: "%{y:.6g}<extra></extra>",
        },
      ]
    : [
        {
          type: "scatter",
          mode: "lines+markers",
          name: series.key,
          x: xs,
          y: ys,
          line: { color, width: 2 },
          marker: { size: 4, color },
          hovertemplate: "%{y:.6g}<extra></extra>",
        },
      ];

  const layout: Record<string, unknown> = {
    autosize: true,
    margin: { l: 48, r: 16, t: 8, b: 36 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "ui-sans-serif, system-ui, sans-serif", size: 11, color: "#64748b" },
    xaxis: {
      type: xMode === "wall" ? "date" : "linear",
      gridcolor: "rgba(148,163,184,0.18)",
      zerolinecolor: "rgba(148,163,184,0.3)",
      tickfont: { size: 10 },
    },
    yaxis: {
      type: yScale,
      gridcolor: "rgba(148,163,184,0.18)",
      zerolinecolor: "rgba(148,163,184,0.3)",
      tickfont: { size: 10 },
    },
    hovermode: "x unified",
    showlegend: false,
  };

  const config: Record<string, unknown> = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: [
      "lasso2d",
      "select2d",
      "toggleSpikelines",
      "hoverClosestCartesian",
      "hoverCompareCartesian",
    ],
    displayModeBar: "hover",
  };

  return (
    <Plot
      data={data}
      layout={layout}
      config={config}
      useResizeHandler
      style={{ width: "100%", height: "220px" }}
    />
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

const ChartControls = ({
  smoothing,
  xMode,
  yScale,
  onSmoothingChange,
  onXModeChange,
  onYScaleChange,
}: ControlsProps): JSX.Element => (
  <div className="flex flex-wrap items-center gap-4 rounded-md border border-border bg-muted/30 px-3 py-2 text-xs">
    <label className="flex items-center gap-2">
      <span className="font-medium text-foreground">Smoothing</span>
      <input
        type="range"
        min={0}
        max={0.99}
        step={0.01}
        value={smoothing}
        onChange={(event) => onSmoothingChange(Number(event.target.value))}
        className="h-1 w-32 cursor-pointer accent-primary"
        aria-label="EMA smoothing weight"
      />
      <span className="font-mono tabular-nums text-muted-foreground">{smoothing.toFixed(2)}</span>
    </label>
    <div className="flex items-center gap-1">
      <span className="font-medium text-foreground">X</span>
      {(["step", "wall"] as const).map((mode) => (
        <button
          key={mode}
          type="button"
          onClick={() => onXModeChange(mode)}
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
          onClick={() => onYScaleChange(scale)}
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
);

export const RunMetricsTab = ({ selection, snapshot }: RunMetricsTabProps): JSX.Element => {
  const run = snapshot.runs.find((item) => item.id === selection.objectId) ?? null;
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

  useEffect(() => {
    if (!run) {
      setLoading(false);
      return;
    }

    let cancelled = false;

    const fetchMetrics = async (sinceLine: number): Promise<void> => {
      try {
        const response = await workspaceApi.getRunMetrics(run.projectId, run.experimentId, run.id, {
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

    void fetchMetrics(nextLine);
    const intervalId = window.setInterval(() => {
      void fetchMetrics(nextLine);
    }, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [nextLine, run]);

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
    <div className="flex-1 overflow-auto bg-background">
      <div className="mx-auto flex max-w-6xl flex-col gap-5 px-4 py-4 md:px-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
            <div className="text-sm font-medium text-foreground">Run Metrics</div>
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <span>{records.length} records</span>
            <span>{scalarSeries.length} scalar series</span>
            {parseErrors > 0 && <span>{parseErrors} parse errors</span>}
          </div>
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

        {grouped.length > 0 ? (
          grouped.map(([groupName, items]) => (
            <OverviewSection key={groupName || "_root"} title={groupName ? groupName : "Scalars"}>
              <div className="grid gap-3 lg:grid-cols-2">
                {items.map((series, index) => (
                  <section
                    key={series.key}
                    className="min-w-0 rounded-md border border-border bg-background p-3"
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <div className="min-w-0 truncate text-sm font-medium text-foreground">
                        {series.key}
                      </div>
                      <div className="font-mono text-xs text-muted-foreground">
                        {formatValue(series.latest)}
                      </div>
                    </div>
                    <MetricChart
                      series={series}
                      xMode={xMode}
                      yScale={yScale}
                      smoothing={smoothing}
                      color={PALETTE[index % PALETTE.length]}
                    />
                  </section>
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
  );
};

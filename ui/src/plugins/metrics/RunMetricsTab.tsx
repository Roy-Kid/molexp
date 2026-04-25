import { Activity, AlertTriangle, BarChart3 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { EmptyState, OverviewSection } from "@/app/components/entity";
import type { MetricRecord } from "@/app/state/api";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";

const POLL_INTERVAL_MS = 1500;

interface ScalarPoint {
  x: number;
  y: number;
}

interface ScalarSeries {
  key: string;
  points: ScalarPoint[];
  latest: number;
}

const isFiniteNumber = (value: unknown): value is number => {
  return typeof value === "number" && Number.isFinite(value);
};

const formatValue = (value: number): string => {
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) {
    return value.toExponential(3);
  }
  return value.toPrecision(4);
};

const buildScalarSeries = (records: MetricRecord[]): ScalarSeries[] => {
  const grouped = new Map<string, ScalarPoint[]>();

  records.forEach((record, index) => {
    if (record.t !== "scalar" || !isFiniteNumber(record.v)) {
      return;
    }
    const points = grouped.get(record.k) ?? [];
    points.push({
      x: isFiniteNumber(record.s) ? record.s : index,
      y: record.v,
    });
    grouped.set(record.k, points);
  });

  return Array.from(grouped.entries())
    .map(([key, points]) => ({
      key,
      points,
      latest: points[points.length - 1]?.y ?? 0,
    }))
    .sort((left, right) => left.key.localeCompare(right.key));
};

const Sparkline = ({ points }: { points: ScalarPoint[] }): JSX.Element => {
  const width = 320;
  const height = 90;
  const padding = 8;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;

  const line = points
    .map((point) => {
      const x = padding + ((point.x - minX) / xSpan) * (width - padding * 2);
      const y = height - padding - ((point.y - minY) / ySpan) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg className="h-24 w-full" viewBox={`0 0 ${width} ${height}`} role="img">
      <title>Metric trend</title>
      <line
        x1={padding}
        x2={width - padding}
        y1={height - padding}
        y2={height - padding}
        className="stroke-border"
        strokeWidth="1"
      />
      <polyline
        points={line}
        fill="none"
        className="stroke-foreground"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
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

export const RunMetricsTab = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const run = snapshot.runs.find((item) => item.id === selection.objectId) ?? null;
  const [records, setRecords] = useState<MetricRecord[]>([]);
  const [nextLine, setNextLine] = useState(0);
  const [parseErrors, setParseErrors] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const scalarSeries = useMemo(() => buildScalarSeries(records), [records]);

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

        <OverviewSection title="Scalars">
          {scalarSeries.length > 0 ? (
            <div className="grid gap-3 lg:grid-cols-2">
              {scalarSeries.map((series) => (
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
                  <Sparkline points={series.points} />
                </section>
              ))}
            </div>
          ) : (
            <div className="rounded-md border border-dashed border-border px-3 py-6 text-center text-sm text-muted-foreground">
              No scalar metrics recorded.
            </div>
          )}
        </OverviewSection>

        <OtherRecords records={records} />
      </div>
    </div>
  );
};

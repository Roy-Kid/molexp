import type { LineChartConfig } from "@molcrafts/molplot";
import { AlertTriangle, Layers, Settings2 } from "lucide-react";
import { type JSX, useEffect, useMemo, useState } from "react";

import { EmptyState } from "@/app/components/entity";
import { type MetricRecord, workspaceApi } from "@/app/state/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { MolplotLineChart } from "@/plugins/molplot";

import {
  type AggregateOptions,
  buildErrorbandConfig,
  buildMeanConfig,
  buildOverlayConfig,
  type RunSeries,
  type XMode,
  type YScale,
} from "./aggregateSeries";
import { buildScalarSeries, type ScalarSeries } from "./RunMetricsView";

/**
 * Experiment-scoped multi-run metrics aggregation (Mode A). A sibling of the
 * single-run `RunMetricsView` — it does NOT extend that component's single
 * poll-cursor / remount contract. Instead it takes the set of selected run ids,
 * snapshot-fetches each run's metrics in parallel, and folds the chosen metric
 * key across runs into one chart via the pure builders in `aggregateSeries.ts`.
 *
 * The data-path logic (`collectRunSeries`, `pickKeySeries`, `availableKeys`,
 * `selectAggregateConfig`) is exported as pure, dependency-injected functions so
 * it is exercised directly under the repo's node test environment, independent
 * of the React render — see MultiRunMetricsView.test.tsx.
 */

export type { RunSeries } from "./aggregateSeries";

export type AggregateOp = "overlay" | "mean" | "errorbar";

/** Injectable metrics source so the orchestration logic is testable. */
export type MetricsFetcher = (
  projectId: string,
  experimentId: string,
  runId: string,
) => Promise<MetricRecord[]>;

export interface RunAllSeries {
  runId: string;
  /** Every scalar series the run logged (all keys). */
  series: ScalarSeries[];
}

export interface CollectResult {
  perRunAll: RunAllSeries[];
  /** Run ids whose metrics fetch failed — surfaced, never fatal. */
  failures: string[];
}

/**
 * Fetch every selected run's metrics in parallel, isolating per-run failures:
 * one rejecting run lands in `failures` and the rest still resolve. Each
 * successful run's records are folded into scalar series via `buildScalarSeries`.
 */
export const collectRunSeries = async (
  fetcher: MetricsFetcher,
  projectId: string,
  experimentId: string,
  runIds: string[],
): Promise<CollectResult> => {
  const settled = await Promise.all(
    runIds.map(async (runId) => {
      try {
        const records = await fetcher(projectId, experimentId, runId);
        return { runId, series: buildScalarSeries(records), ok: true as const };
      } catch {
        return { runId, series: [] as ScalarSeries[], ok: false as const };
      }
    }),
  );

  const perRunAll: RunAllSeries[] = [];
  const failures: string[] = [];
  for (const result of settled) {
    if (result.ok) {
      perRunAll.push({ runId: result.runId, series: result.series });
    } else {
      failures.push(result.runId);
    }
  }
  return { perRunAll, failures };
};

/** Union of scalar metric keys across all runs, sorted. */
export const availableKeys = (perRunAll: RunAllSeries[]): string[] => {
  const keys = new Set<string>();
  for (const { series } of perRunAll) {
    for (const s of series) {
      keys.add(s.key);
    }
  }
  return Array.from(keys).sort();
};

/** For the chosen key, take each run's matching series; skip runs lacking it. */
export const pickKeySeries = (perRunAll: RunAllSeries[], metricKey: string): RunSeries[] => {
  const out: RunSeries[] = [];
  for (const { runId, series } of perRunAll) {
    const match = series.find((s) => s.key === metricKey);
    if (match) {
      out.push({ runId, series: match });
    }
  }
  return out;
};

/** Dispatch the selected aggregation op onto the pure builders. */
export const selectAggregateConfig = (
  op: AggregateOp,
  perRun: RunSeries[],
  options: AggregateOptions,
): { config: LineChartConfig; dropped: number } => {
  if (op === "overlay") {
    return { config: buildOverlayConfig(perRun, options), dropped: 0 };
  }
  const series = perRun.map((r) => r.series);
  if (op === "mean") {
    return buildMeanConfig(series, options);
  }
  return buildErrorbandConfig(series, options);
};

const realFetcher: MetricsFetcher = async (projectId, experimentId, runId) => {
  const response = await workspaceApi.getRunMetrics(projectId, experimentId, runId, {
    type: "scalar",
    sinceLine: 0,
    limit: 100000,
  });
  return response.records;
};

const OP_LABELS: Record<AggregateOp, string> = {
  overlay: "Overlay",
  mean: "Mean",
  errorbar: "Errorbar (±std)",
};

export interface MultiRunMetricsViewProps {
  projectId: string;
  experimentId: string;
  /** Ordered ids of the runs selected for aggregation. */
  runIds: string[];
}

export const MultiRunMetricsView = ({
  projectId,
  experimentId,
  runIds,
}: MultiRunMetricsViewProps): JSX.Element => {
  const [perRunAll, setPerRunAll] = useState<RunAllSeries[]>([]);
  const [failures, setFailures] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [metricKey, setMetricKey] = useState("");
  const [op, setOp] = useState<AggregateOp>("overlay");
  const [smoothing, setSmoothing] = useState(0);
  const [xMode, setXMode] = useState<XMode>("step");
  const [yScale, setYScale] = useState<YScale>("linear");
  const [configOpen, setConfigOpen] = useState(false);

  // One-shot snapshot fetch (not a live poll). Re-runs when the selected run set
  // changes — `runIds` is memoised by the caller (selectedRunIds), so depending
  // on it directly is stable across renders.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    void (async () => {
      try {
        const result = await collectRunSeries(realFetcher, projectId, experimentId, runIds);
        if (cancelled) {
          return;
        }
        setPerRunAll(result.perRunAll);
        setFailures(result.failures);
        if (result.perRunAll.length === 0 && result.failures.length > 0) {
          setError("All selected runs failed to load metrics.");
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError instanceof Error ? fetchError.message : "Failed to load metrics");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // realFetcher is a module constant.
  }, [projectId, experimentId, runIds]);

  const keys = useMemo(() => availableKeys(perRunAll), [perRunAll]);

  // Default / repair the selected metric key whenever the available set shifts.
  useEffect(() => {
    if (keys.length > 0 && !keys.includes(metricKey)) {
      setMetricKey(keys[0]);
    }
  }, [keys, metricKey]);

  const picked = useMemo(() => pickKeySeries(perRunAll, metricKey), [perRunAll, metricKey]);

  const { config, dropped } = useMemo(() => {
    if (picked.length === 0) {
      return { config: { series: [] } as LineChartConfig, dropped: 0 };
    }
    return selectAggregateConfig(op, picked, { xMode, yScale, smoothing, metricKey });
  }, [picked, op, xMode, yScale, smoothing, metricKey]);

  if (runIds.length === 0) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState
          icon={<Layers className="h-6 w-6" />}
          title="No runs selected"
          description="Select runs in the Runs tab, then aggregate their metrics here."
        />
      </div>
    );
  }

  if (loading && perRunAll.length === 0) {
    return (
      <div className="flex h-full items-center justify-center bg-background text-sm text-muted-foreground">
        Loading metrics for {runIds.length} runs…
      </div>
    );
  }

  if (error && perRunAll.length === 0) {
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

  if (keys.length === 0) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState
          icon={<Layers className="h-6 w-6" />}
          title="No scalar metrics"
          description="None of the selected runs recorded scalar metrics."
        />
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden bg-background">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2">
        <Layers className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium text-foreground">
          {OP_LABELS[op]} · {metricKey}
        </span>
        <span className="text-xs text-muted-foreground">
          {picked.length}/{runIds.length} runs
        </span>

        <Dialog open={configOpen} onOpenChange={setConfigOpen}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm" className="ml-auto h-7 gap-1.5">
              <Settings2 className="h-3.5 w-3.5" />
              Aggregation
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Aggregate metrics</DialogTitle>
            </DialogHeader>
            <div className="flex flex-col gap-4 py-2 text-sm">
              <div className="flex flex-col gap-1.5">
                <span className="font-medium text-foreground">Operation</span>
                <Select value={op} onValueChange={(value) => setOp(value as AggregateOp)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="overlay">Overlay (one line per run)</SelectItem>
                    <SelectItem value="mean">Mean (per-step average)</SelectItem>
                    <SelectItem value="errorbar">Errorbar (mean ± std)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex flex-col gap-1.5">
                <span className="font-medium text-foreground">Metric</span>
                <Select value={metricKey} onValueChange={setMetricKey}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {keys.map((key) => (
                      <SelectItem key={key} value={key}>
                        {key}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex flex-col gap-1.5">
                <span className="font-medium text-foreground">Smoothing</span>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={0}
                    max={0.99}
                    step={0.01}
                    value={smoothing}
                    onChange={(event) => setSmoothing(Number(event.target.value))}
                    className="h-1 flex-1 cursor-pointer accent-primary"
                    aria-label="EMA smoothing weight"
                  />
                  <span className="w-9 text-right font-mono tabular-nums text-muted-foreground">
                    {smoothing.toFixed(2)}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="flex flex-col gap-1.5">
                  <span className="font-medium text-foreground">X axis</span>
                  <Select value={xMode} onValueChange={(value) => setXMode(value as XMode)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="step">Step</SelectItem>
                      <SelectItem value="wall">Wall time</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex flex-col gap-1.5">
                  <span className="font-medium text-foreground">Y axis</span>
                  <Select value={yScale} onValueChange={(value) => setYScale(value as YScale)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="linear">Linear</SelectItem>
                      <SelectItem value="log">Log</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <div className="min-h-0 flex-1 p-4">
        <MolplotLineChart config={config} style={{ width: "100%", height: "100%" }} />
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-0.5 border-t border-border px-4 py-2 text-xs text-muted-foreground">
        <span>{picked.length} runs aggregated</span>
        {dropped > 0 && <span>{dropped} unaligned steps dropped</span>}
        {failures.length > 0 && <span>{failures.length} runs failed to load</span>}
      </div>
    </div>
  );
};

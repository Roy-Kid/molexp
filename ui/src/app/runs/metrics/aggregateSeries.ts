import type { LineChartConfig, LineSeriesConfig } from "@molcrafts/molplot";

import { smoothEma } from "@/plugins/metrics/smoothing";

import { PALETTE, type ScalarSeries } from "./RunMetricsView";

/**
 * Pure builders for Mode-A multi-run metrics aggregation: take per-run scalar
 * series for a single metric key and fold them into a molplot LineChartConfig
 * three ways — overlay (one line per run), mean (per-step arithmetic mean), and
 * errorbar (mean ± sample std).
 *
 * Replicas are assumed to be logged on identical steps; alignment is therefore
 * a strict common-step intersection with NO interpolation or binning. Steps that
 * are not shared by every run are dropped and counted (`dropped`), mirroring the
 * `parseErrors` counter convention of the single-run view.
 *
 * All functions are pure (no fetch, no React) so they run directly under the
 * repo's node test environment — see aggregateSeries.test.ts.
 */

export type XMode = "step" | "wall";
export type YScale = "linear" | "log";

export interface AggregateOptions {
  xMode: XMode;
  yScale: YScale;
  /** EMA weight in [0, 1); 0 disables smoothing. */
  smoothing: number;
  /** The metric key being aggregated — used for axis + series labels. */
  metricKey: string;
}

/** One selected run's scalar series for the chosen metric key. */
export interface RunSeries {
  runId: string;
  series: ScalarSeries;
}

type ScalarPoint = ScalarSeries["points"][number];

interface AlignedStep {
  step: number;
  /** Mean wall-clock across runs at this step (runs may differ slightly). */
  wall: number;
  /** y value from each run at this step, in run order. */
  ys: number[];
}

export interface AlignResult {
  points: AlignedStep[];
  /** Distinct steps present in some run but not all — excluded from aggregation. */
  dropped: number;
}

/**
 * Intersect the step axes of every series. A step survives only if every run
 * has a sample at it; everything else is dropped (and counted). Points are
 * returned sorted by step ascending.
 */
export const alignSteps = (series: ScalarSeries[]): AlignResult => {
  if (series.length === 0) {
    return { points: [], dropped: 0 };
  }

  const byStep = series.map((s) => {
    const m = new Map<number, ScalarPoint>();
    for (const p of s.points) {
      m.set(p.step, p);
    }
    return m;
  });

  const unionSteps = new Set<number>();
  for (const m of byStep) {
    for (const step of m.keys()) {
      unionSteps.add(step);
    }
  }

  const common = Array.from(unionSteps)
    .filter((step) => byStep.every((m) => m.has(step)))
    .sort((a, b) => a - b);

  const points: AlignedStep[] = common.map((step) => {
    const perRun = byStep.map((m) => m.get(step) as ScalarPoint);
    const wall = perRun.reduce((acc, p) => acc + p.wall, 0) / perRun.length;
    return { step, wall, ys: perRun.map((p) => p.y) };
  });

  return { points, dropped: unionSteps.size - common.length };
};

const mean = (xs: number[]): number => xs.reduce((acc, x) => acc + x, 0) / xs.length;

/** Sample standard deviation (n-1 denominator); 0 for a single sample. */
const sampleStd = (xs: number[]): number => {
  if (xs.length < 2) {
    return 0;
  }
  const m = mean(xs);
  const variance = xs.reduce((acc, x) => acc + (x - m) ** 2, 0) / (xs.length - 1);
  return Math.sqrt(variance);
};

const MODEBAR_REMOVE = [
  "lasso2d",
  "select2d",
  "toggleSpikelines",
  "hoverClosestCartesian",
  "hoverCompareCartesian",
];

/** Shared axis / modebar / theme block, mirroring RunMetricsView's chart config. */
const baseConfig = (
  series: LineSeriesConfig[],
  options: Pick<AggregateOptions, "xMode" | "yScale" | "metricKey">,
): LineChartConfig => ({
  series,
  xAxis: {
    label: options.xMode === "step" ? "step" : "wall time",
    type: "linear",
  },
  yAxis: { type: options.yScale, label: options.metricKey },
  hovertemplate: "%{y:.6g}<extra></extra>",
  hovermode: "x unified",
  showLegend: true,
  modebar: false,
  modebarRemove: MODEBAR_REMOVE,
  theme: "auto",
});

const xValue = (point: ScalarPoint, xMode: XMode): number =>
  xMode === "step" ? point.step : point.wall;

const applySmoothing = (ys: number[], smoothing: number): number[] =>
  smoothing > 0 ? smoothEma(ys, smoothing) : ys;

/**
 * Overlay: one line per run, palette-cycled in run order. Each run keeps its own
 * x-axis (steps need not be aligned for a pure overlay).
 */
export const buildOverlayConfig = (
  perRun: RunSeries[],
  options: AggregateOptions,
): LineChartConfig => {
  const series: LineSeriesConfig[] = perRun.map(({ runId, series: s }, index) => {
    const xs = s.points.map((p) => xValue(p, options.xMode));
    const ys = applySmoothing(
      s.points.map((p) => p.y),
      options.smoothing,
    );
    return {
      id: runId,
      label: runId,
      color: PALETTE[index % PALETTE.length],
      width: 2,
      mode: "lines+markers",
      initialPoints: ys.map((y, i) => ({ x: xs[i], y })),
    };
  });
  return baseConfig(series, options);
};

/**
 * Per-step arithmetic mean over the common steps. Returns a single synthetic
 * ScalarSeries plus the dropped-step count, so callers can both render it and
 * surface alignment loss.
 */
export const buildMeanSeries = (
  series: ScalarSeries[],
): { mean: ScalarSeries; dropped: number } => {
  const { points, dropped } = alignSteps(series);
  const meanPoints: ScalarPoint[] = points.map((p) => ({
    step: p.step,
    wall: p.wall,
    y: mean(p.ys),
  }));
  const key = series[0]?.key ?? "";
  return {
    mean: {
      key: `${key} (mean)`,
      group: series[0]?.group ?? "",
      points: meanPoints,
      latest: meanPoints[meanPoints.length - 1]?.y ?? 0,
    },
    dropped,
  };
};

/** Mean line as a single-series config (the `mean` aggregation op). */
export const buildMeanConfig = (
  series: ScalarSeries[],
  options: AggregateOptions,
): { config: LineChartConfig; dropped: number } => {
  const { mean: meanSeries, dropped } = buildMeanSeries(series);
  const xs = meanSeries.points.map((p) => xValue(p, options.xMode));
  const ys = applySmoothing(
    meanSeries.points.map((p) => p.y),
    options.smoothing,
  );
  const line: LineSeriesConfig = {
    id: "mean",
    label: `${options.metricKey} (mean)`,
    color: PALETTE[0],
    width: 2,
    mode: "lines+markers",
    initialPoints: ys.map((y, i) => ({ x: xs[i], y })),
  };
  return { config: baseConfig([line], options), dropped };
};

// molplot's LineChart has no native filled-band / error_y trace — LineSeriesConfig
// exposes only id/label/color/width/opacity/mode/initialPoints (see
// @molcrafts/molplot types). The ±std band is therefore drawn as two faint
// boundary lines (upper/lower) around the solid mean line.
const BAND_OPACITY = 0.25;

/**
 * Errorbar: per-step mean ± sample std, rendered as a mean line plus faded
 * upper/lower boundary lines. Smoothing is applied to the mean before offsetting
 * by the (unsmoothed) per-step std so the band tracks the displayed mean.
 */
export const buildErrorbandConfig = (
  series: ScalarSeries[],
  options: AggregateOptions,
): { config: LineChartConfig; dropped: number } => {
  const { points, dropped } = alignSteps(series);
  const xs = points.map((p) => xValue(p, options.xMode));
  const means = applySmoothing(
    points.map((p) => mean(p.ys)),
    options.smoothing,
  );
  const stds = points.map((p) => sampleStd(p.ys));
  const color = PALETTE[0];

  const upper: LineSeriesConfig = {
    id: "upper",
    label: "+std",
    color,
    width: 1,
    opacity: BAND_OPACITY,
    initialPoints: means.map((m, i) => ({ x: xs[i], y: m + stds[i] })),
  };
  const lower: LineSeriesConfig = {
    id: "lower",
    label: "-std",
    color,
    width: 1,
    opacity: BAND_OPACITY,
    initialPoints: means.map((m, i) => ({ x: xs[i], y: m - stds[i] })),
  };
  const meanLine: LineSeriesConfig = {
    id: "mean",
    label: `${options.metricKey} (mean)`,
    color,
    width: 2,
    mode: "lines+markers",
    initialPoints: means.map((m, i) => ({ x: xs[i], y: m })),
  };

  return { config: baseConfig([upper, lower, meanLine], options), dropped };
};

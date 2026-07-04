/**
 * Tests for the pure metric-series builders behind the shared RunMetricsView
 * component (node env, no jsdom — the builders are exercised directly).
 */

import { describe, expect, it } from "@rstest/core";

import type { MetricRecord } from "@/app/state/api";

import { buildLineChartConfig, buildScalarSeries, groupSeries } from "./RunMetricsView";

// Two distinct scalar keys across several (intentionally out-of-order) steps,
// plus one non-scalar record that must be ignored by buildScalarSeries.
const SAMPLE_RECORDS: MetricRecord[] = [
  { t: "scalar", k: "train/loss", s: 2, w: "2026-06-15T00:00:02Z", v: 0.5 },
  { t: "scalar", k: "eval/loss", s: 1, w: "2026-06-15T00:00:01Z", v: 0.9 },
  { t: "scalar", k: "train/loss", s: 0, w: "2026-06-15T00:00:00Z", v: 1.0 },
  { t: "scalar", k: "train/loss", s: 1, w: "2026-06-15T00:00:01Z", v: 0.7 },
  { t: "scalar", k: "eval/loss", s: 0, w: "2026-06-15T00:00:00Z", v: 1.2 },
  { t: "histogram", k: "weights/layer0", s: 0, w: "2026-06-15T00:00:00Z", v: [1, 2, 3] },
];

interface ScalarPointShape {
  step: number;
  wall: number;
  y: number;
}

interface ScalarSeriesShape {
  key: string;
  group: string;
  points: ScalarPointShape[];
  latest: number;
}

describe("buildScalarSeries (ac-001)", () => {
  it("returns exactly one series per distinct scalar key, ignoring non-scalar records", () => {
    const series = buildScalarSeries(SAMPLE_RECORDS) as ScalarSeriesShape[];
    expect(series).toHaveLength(2);
    const keys = series.map((s) => s.key).sort();
    expect(keys).toEqual(["eval/loss", "train/loss"]);
  });

  it("sorts each series' points by step ascending", () => {
    const series = buildScalarSeries(SAMPLE_RECORDS) as ScalarSeriesShape[];
    const train = series.find((s) => s.key === "train/loss");
    expect(train).toBeDefined();
    const steps = (train as ScalarSeriesShape).points.map((p) => p.step);
    expect(steps).toEqual([0, 1, 2]);
    const ys = (train as ScalarSeriesShape).points.map((p) => p.y);
    expect(ys).toEqual([1.0, 0.7, 0.5]);
  });
});

describe("groupSeries (ac-001)", () => {
  it("buckets each series under the prefix before the first slash", () => {
    const series = buildScalarSeries(SAMPLE_RECORDS) as ScalarSeriesShape[];
    const grouped = groupSeries(series) as Array<[string, ScalarSeriesShape[]]>;
    const groupNames = grouped.map(([name]) => name).sort();
    expect(groupNames).toEqual(["eval", "train"]);

    const byGroup = new Map(grouped);
    const evalSeries = byGroup.get("eval");
    const trainSeries = byGroup.get("train");
    expect(evalSeries?.map((s) => s.key)).toEqual(["eval/loss"]);
    expect(trainSeries?.map((s) => s.key)).toEqual(["train/loss"]);
  });
});

describe("buildLineChartConfig (ac-001)", () => {
  it("emits at least one series whose initialPoints mirror the scalar points (x=step, y=value)", () => {
    const series = buildScalarSeries(SAMPLE_RECORDS) as ScalarSeriesShape[];
    const train = series.find((s) => s.key === "train/loss") as ScalarSeriesShape;

    const config = buildLineChartConfig(train, {
      xMode: "step",
      yScale: "linear",
      smoothing: 0,
      color: "#2563eb",
    }) as { series: Array<{ initialPoints: Array<{ x: number; y: number }> }> };

    expect(Array.isArray(config.series)).toBe(true);
    expect(config.series.length).toBeGreaterThanOrEqual(1);

    const primary = config.series[0];
    expect(primary.initialPoints).toEqual([
      { x: 0, y: 1.0 },
      { x: 1, y: 0.7 },
      { x: 2, y: 0.5 },
    ]);
  });
});

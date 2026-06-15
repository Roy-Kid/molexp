/**
 * RED tests for the extracted shared RunMetricsView component.
 *
 * Rstest runs in a node environment without jsdom (see rstest.config.ts:
 * testEnvironment: "node"). We therefore cannot use @testing-library /
 * render(). Two complementary strategies are used here, mirroring the repo
 * convention (see app/settings/__tests__/AddRemoteWorkspaceForm.test.tsx):
 *
 *   1. The pure builder functions (buildScalarSeries / groupSeries /
 *      buildLineChartConfig) are imported and exercised directly. These
 *      imports FAIL today because RunMetricsView.tsx does not exist yet
 *      (RED for ac-001 logic).
 *
 *   2. The component's JSX wiring is asserted by parsing the .tsx source
 *      with node:fs. readFileSync throws today (file absent) -> RED for the
 *      ac-001 render + ac-002 coord-driven contract.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

import type { MetricRecord } from "@/app/state/api";

import { buildLineChartConfig, buildScalarSeries, groupSeries } from "./RunMetricsView";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const SOURCE_PATH = resolve(__dirname, "./RunMetricsView.tsx");

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

  it("excludes the non-scalar (histogram) record entirely", () => {
    const series = buildScalarSeries(SAMPLE_RECORDS) as ScalarSeriesShape[];
    expect(series.some((s) => s.key === "weights/layer0")).toBe(false);
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

describe("RunMetricsView source — render + coord-driven (ac-001, ac-002)", () => {
  // readFileSync throws today because RunMetricsView.tsx does not exist -> RED.
  const source = readFileSync(SOURCE_PATH, "utf8");

  it("renders MolplotLineChart per scalar series", () => {
    expect(source).toContain("<MolplotLineChart");
    expect(source).toContain("@/plugins/molplot");
  });

  it("is driven by projectId / experimentId / runId props", () => {
    expect(source).toContain("projectId");
    expect(source).toContain("experimentId");
    expect(source).toContain("runId");
  });

  it("calls getRunMetrics with the coordinate props", () => {
    expect(source).toMatch(/getRunMetrics\(/);
    expect(source).toMatch(/getRunMetrics\([\s\S]*projectId[\s\S]*experimentId[\s\S]*runId/);
  });

  it("contains no reference to snapshot (coord-driven, not snapshot-coupled)", () => {
    expect(source).not.toContain("snapshot");
  });
});

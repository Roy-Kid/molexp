/**
 * RED tests for the pure multi-run aggregation builders (ac-001..ac-005).
 *
 * Rstest runs in a node environment without jsdom (see rstest.config.ts), so
 * these exercise the pure builders directly — no DOM, no render(). The imports
 * fail today because aggregateSeries.ts does not exist yet (RED).
 *
 * Aligns with the repo idiom in RunMetricsView.test.tsx: pure transforms are
 * imported and called against fixtures; PALETTE is consumed from the (newly
 * exported) RunMetricsView surface.
 */

import { describe, expect, it } from "@rstest/core";

import {
  type AggregateOptions,
  buildErrorbandConfig,
  buildMeanSeries,
  buildOverlayConfig,
  type RunSeries,
} from "./aggregateSeries";
import { PALETTE, type ScalarSeries } from "./RunMetricsView";

const OPTS: AggregateOptions = {
  xMode: "step",
  yScale: "linear",
  smoothing: 0,
  metricKey: "loss",
};

// A scalar series whose points sit on consecutive integer steps (step === wall
// === index), matching the strict same-step replica assumption.
const makeSeries = (key: string, ys: number[], startStep = 0): ScalarSeries => ({
  key,
  group: key.includes("/") ? key.slice(0, key.indexOf("/")) : "",
  points: ys.map((y, i) => ({ step: startStep + i, wall: startStep + i, y })),
  latest: ys[ys.length - 1] ?? 0,
});

const RUN_A: RunSeries = { runId: "run-a", series: makeSeries("loss", [1.0, 0.8, 0.6]) };
const RUN_B: RunSeries = { runId: "run-b", series: makeSeries("loss", [2.0, 1.0, 0.4]) };

interface ConfigShape {
  series: Array<{ id: string; color?: string; initialPoints?: Array<{ x: number; y: number }> }>;
}

describe("buildOverlayConfig (ac-001)", () => {
  it("emits one series per run, palette-cycled in run order", () => {
    const config = buildOverlayConfig([RUN_A, RUN_B], OPTS) as ConfigShape;
    expect(config.series).toHaveLength(2);
    expect(config.series[0].color).toBe(PALETTE[0]);
    expect(config.series[1].color).toBe(PALETTE[1]);
  });

  it("maps each run's scalar points onto its series (x=step, y=value)", () => {
    const config = buildOverlayConfig([RUN_A, RUN_B], OPTS) as ConfigShape;
    expect(config.series[0].initialPoints).toEqual([
      { x: 0, y: 1.0 },
      { x: 1, y: 0.8 },
      { x: 2, y: 0.6 },
    ]);
  });

  it("cycles the palette past its length for > PALETTE.length runs", () => {
    const many: RunSeries[] = Array.from({ length: PALETTE.length + 1 }, (_v, i) => ({
      runId: `run-${i}`,
      series: makeSeries("loss", [i]),
    }));
    const config = buildOverlayConfig(many, OPTS) as ConfigShape;
    expect(config.series[PALETTE.length].color).toBe(PALETTE[0]);
  });
});

describe("buildMeanSeries (ac-002)", () => {
  it("computes the per-step arithmetic mean over identical-step replicas", () => {
    const { mean, dropped } = buildMeanSeries([RUN_A.series, RUN_B.series]);
    expect(dropped).toBe(0);
    expect(mean.points.map((p) => p.y)).toEqual([1.5, 0.9, 0.5]);
    expect(mean.points.map((p) => p.step)).toEqual([0, 1, 2]);
  });
});

describe("buildErrorbandConfig (ac-003)", () => {
  it("produces a mean line plus ±sample-std boundary series", () => {
    const { config, dropped } = buildErrorbandConfig([RUN_A.series, RUN_B.series], OPTS);
    expect(dropped).toBe(0);
    const c = config as ConfigShape;
    const mean = c.series.find((s) => s.id === "mean");
    const upper = c.series.find((s) => s.id === "upper");
    const lower = c.series.find((s) => s.id === "lower");
    expect(mean).toBeDefined();
    expect(upper).toBeDefined();
    expect(lower).toBeDefined();

    // sample std at step 0 over {1.0, 2.0}: sqrt(((1-1.5)^2+(2-1.5)^2)/(2-1)) = sqrt(0.5)
    const expectedStd0 = Math.sqrt(0.5);
    expect(mean?.initialPoints?.[0].y).toBeCloseTo(1.5, 10);
    expect(upper?.initialPoints?.[0].y).toBeCloseTo(1.5 + expectedStd0, 10);
    expect(lower?.initialPoints?.[0].y).toBeCloseTo(1.5 - expectedStd0, 10);
  });
});

describe("strict same-step alignment (ac-004)", () => {
  it("uses only the common-step intersection and counts dropped steps", () => {
    const a = makeSeries("loss", [1.0, 0.8, 0.6], 0); // steps 0,1,2
    const b = makeSeries("loss", [2.0, 1.0, 0.4], 1); // steps 1,2,3
    const { mean, dropped } = buildMeanSeries([a, b]);
    // union {0,1,2,3} minus intersection {1,2} = 2 dropped
    expect(dropped).toBe(2);
    expect(mean.points.map((p) => p.step)).toEqual([1, 2]);
    // step 1: mean(0.8, 2.0) = 1.4 ; step 2: mean(0.6, 1.0) = 0.8
    expect(mean.points.map((p) => p.y)).toEqual([1.4, 0.8]);
  });
});

describe("single-run degradation (ac-005)", () => {
  it("overlay → one series; mean → that run; errorband std → 0", () => {
    const overlay = buildOverlayConfig([RUN_A], OPTS) as ConfigShape;
    expect(overlay.series).toHaveLength(1);

    const { mean, dropped } = buildMeanSeries([RUN_A.series]);
    expect(dropped).toBe(0);
    expect(mean.points.map((p) => p.y)).toEqual([1.0, 0.8, 0.6]);

    const { config } = buildErrorbandConfig([RUN_A.series], OPTS);
    const c = config as ConfigShape;
    const meanS = c.series.find((s) => s.id === "mean");
    const upper = c.series.find((s) => s.id === "upper");
    // std = 0 with a single sample → band collapses onto the mean line
    expect(upper?.initialPoints?.map((p) => p.y)).toEqual(meanS?.initialPoints?.map((p) => p.y));
  });
});

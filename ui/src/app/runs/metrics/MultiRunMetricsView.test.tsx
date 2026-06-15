/**
 * RED tests for the multi-run aggregation orchestrator (ac-006, ac-007).
 *
 * Rstest runs in node without jsdom, so we cannot render the component. Instead
 * we test the exported, dependency-injected logic directly:
 *   - collectRunSeries(fetcher, …)  — parallel fetch + per-run failure isolation
 *   - selectAggregateConfig(op, …)  — op/key dispatch into the pure builders
 * plus source-text assertions for the component's Modal / Select / chart wiring
 * (the repo convention from RunMetricsView.test.tsx).
 *
 * Imports fail today because MultiRunMetricsView.tsx does not exist yet (RED).
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

import type { MetricRecord } from "@/app/state/api";

import {
  collectRunSeries,
  type MetricsFetcher,
  pickKeySeries,
  type RunSeries,
  selectAggregateConfig,
} from "./MultiRunMetricsView";
import type { ScalarSeries } from "./RunMetricsView";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const SOURCE_PATH = resolve(__dirname, "./MultiRunMetricsView.tsx");

const recordsFor = (loss: number[], acc: number[]): MetricRecord[] => [
  ...loss.map((v, i) => ({ t: "scalar", k: "loss", s: i, v }) as MetricRecord),
  ...acc.map((v, i) => ({ t: "scalar", k: "acc", s: i, v }) as MetricRecord),
];

const OPTS = { xMode: "step" as const, yScale: "linear" as const, smoothing: 0, metricKey: "loss" };

describe("collectRunSeries (ac-006)", () => {
  it("fetches each selected run exactly once and builds its scalar series", async () => {
    const calls: string[] = [];
    const fetcher: MetricsFetcher = async (_p, _e, runId) => {
      calls.push(runId);
      return recordsFor([1, 0.8], [0.5, 0.6]);
    };

    const { perRunAll, failures } = await collectRunSeries(fetcher, "p", "e", [
      "run-a",
      "run-b",
      "run-c",
    ]);

    expect(calls.sort()).toEqual(["run-a", "run-b", "run-c"]);
    expect(failures).toEqual([]);
    expect(perRunAll).toHaveLength(3);
    // each run yields two scalar keys
    expect(perRunAll[0].series.map((s) => s.key).sort()).toEqual(["acc", "loss"]);
  });
});

describe("selectAggregateConfig op/key dispatch (ac-006)", () => {
  const makeSeries = (key: string, ys: number[]): ScalarSeries => ({
    key,
    group: "",
    points: ys.map((y, i) => ({ step: i, wall: i, y })),
    latest: ys[ys.length - 1] ?? 0,
  });
  const perRun: RunSeries[] = [
    { runId: "a", series: makeSeries("loss", [1, 0.8, 0.6]) },
    { runId: "b", series: makeSeries("loss", [2, 1.0, 0.4]) },
  ];

  it("returns a distinct config shape per op", () => {
    const overlay = selectAggregateConfig("overlay", perRun, OPTS).config as {
      series: unknown[];
    };
    const meanCfg = selectAggregateConfig("mean", perRun, OPTS).config as { series: unknown[] };
    const errCfg = selectAggregateConfig("errorbar", perRun, OPTS).config as {
      series: { id: string }[];
    };

    expect(overlay.series).toHaveLength(2); // one per run
    expect(meanCfg.series).toHaveLength(1); // single mean line
    expect(errCfg.series.map((s) => s.id).sort()).toEqual(["lower", "mean", "upper"]);
  });
});

describe("pickKeySeries (ac-006)", () => {
  it("selects each run's series for the chosen key and skips runs missing it", () => {
    const perRunAll = [
      {
        runId: "a",
        series: [
          { key: "loss", group: "", points: [{ step: 0, wall: 0, y: 1 }], latest: 1 },
          { key: "acc", group: "", points: [{ step: 0, wall: 0, y: 9 }], latest: 9 },
        ] as ScalarSeries[],
      },
      {
        runId: "b",
        series: [
          { key: "acc", group: "", points: [{ step: 0, wall: 0, y: 8 }], latest: 8 },
        ] as ScalarSeries[],
      },
    ];
    const loss = pickKeySeries(perRunAll, "loss");
    expect(loss.map((r) => r.runId)).toEqual(["a"]); // b has no loss
    const acc = pickKeySeries(perRunAll, "acc");
    expect(acc.map((r) => r.runId)).toEqual(["a", "b"]);
  });
});

describe("partial-failure tolerance (ac-007)", () => {
  it("one run's fetch rejection does not blank the others", async () => {
    const fetcher: MetricsFetcher = async (_p, _e, runId) => {
      if (runId === "run-bad") {
        throw new Error("boom");
      }
      return recordsFor([1, 0.8], [0.5, 0.6]);
    };

    const { perRunAll, failures } = await collectRunSeries(fetcher, "p", "e", [
      "run-a",
      "run-bad",
      "run-c",
    ]);

    expect(failures).toEqual(["run-bad"]);
    expect(perRunAll.map((r) => r.runId).sort()).toEqual(["run-a", "run-c"]);
    expect(perRunAll.length).toBeGreaterThan(0);
  });
});

describe("MultiRunMetricsView source — wiring (ac-006, ac-007)", () => {
  const source = readFileSync(SOURCE_PATH, "utf8");

  it("renders the molplot chart for the aggregated config", () => {
    expect(source).toContain("<MolplotLineChart");
    expect(source).toContain("@/plugins/molplot");
  });

  it("offers an operation chooser modal (Dialog) and selectors", () => {
    expect(source).toContain("Dialog");
    expect(source).toMatch(/overlay|mean|errorbar/);
  });

  it("fetches via workspaceApi.getRunMetrics through collectRunSeries", () => {
    expect(source).toContain("collectRunSeries");
    expect(source).toContain("getRunMetrics");
  });

  it("is driven by selected runIds", () => {
    expect(source).toContain("runIds");
  });
});

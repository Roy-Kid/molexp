/**
 * Source-level wiring assertions for the multi-run aggregation feature (ac-009).
 *
 * Rstest runs in node without jsdom, so — following the repo convention in
 * RunMetricsView.test.tsx — the ExperimentViewer wiring is asserted by parsing
 * its source: the ephemeral selection hook, the lazy "aggregate" tab, and that
 * selection lives in React state (the useRunMultiSelect hook) rather than the
 * Zustand store. Behavioural correctness of the selection reducer itself is
 * covered by useRunMultiSelect.test.ts.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const viewer = readFileSync(resolve(__dirname, "./ExperimentViewer.tsx"), "utf8");
const hook = readFileSync(resolve(__dirname, "../runs/useRunMultiSelect.ts"), "utf8");

describe("ExperimentViewer aggregate wiring (ac-009)", () => {
  it("drives selection through the ephemeral useRunMultiSelect hook", () => {
    expect(viewer).toContain("useRunMultiSelect");
    expect(viewer).toContain("selectedRunIds");
  });

  it("exposes a lazy-mounted aggregate tab rendering MultiRunMetricsView", () => {
    expect(viewer).toContain('value: "aggregate"');
    expect(viewer).toContain('activeTab === "aggregate"');
    expect(viewer).toContain("<MultiRunMetricsView");
  });

  it("keeps selection in local React state, not a Zustand store", () => {
    // selection must not be threaded through the app's zustand store
    expect(viewer).not.toMatch(/useWorkspaceStore|zustand/);
  });

  it("holds the run-id selection as a Set in React state in the hook", () => {
    expect(hook).toContain("useState");
    expect(hook).toMatch(/Set<string>/);
  });
});

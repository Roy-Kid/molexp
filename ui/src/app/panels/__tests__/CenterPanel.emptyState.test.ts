/**
 * Source-contract tests for the CenterPanel empty-selection placeholder
 * (rstest node env, no jsdom — see app/runs/metrics/RunMetricsView.test.tsx;
 * importing CenterPanel would drag the whole renderer registry into a DOM-less
 * process, so the JSX wiring is asserted on the source text instead).
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(resolve(__dirname, "../CenterPanel.tsx"), "utf8");

describe("CenterPanel empty-selection copy", () => {
  it("has agent-specific copy instead of the Experiments wording", () => {
    expect(src).toContain("Select an agent task from the left, or start a new one.");
    expect(src).toContain("No agent task selected");
  });

  it("resolves copy per left-panel view", () => {
    expect(src).toContain("EMPTY_SELECTION_COPY");
    expect(src).toMatch(/emptySelectionCopy\s*=\s*\(view\?: LeftPanelView\)/);
    expect(src).toContain("<EmptySelectionPlaceholder view={leftPanelView} />");
  });

  it("keeps the generic fallback for the experiment-flow views", () => {
    expect(src).toContain("Pick a project, experiment, run, or workflow from the left navigation");
  });
});

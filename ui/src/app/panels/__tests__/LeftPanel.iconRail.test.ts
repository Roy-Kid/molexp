/**
 * Source-contract tests for the LeftPanel icon rail (rstest node env, no
 * jsdom — see app/runs/metrics/RunMetricsView.test.tsx; LeftPanel's import
 * graph is too heavy to load in a DOM-less process, so the JSX wiring is
 * asserted on the source text).
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(resolve(__dirname, "../LeftPanel.tsx"), "utf8");

describe("LeftPanel icon rail labels", () => {
  it("every view button carries a native title and an aria-label", () => {
    expect(src).toContain("title={option.label}");
    expect(src).toContain("aria-label={option.label}");
    // The bottom settings button is labelled too.
    expect(src).toContain('title="Settings"');
    expect(src).toContain('aria-label="Settings"');
  });

  it("labels match the route section names (plural, per breadcrumbTrail)", () => {
    for (const label of [
      '{ id: "projects", label: "Experiments"',
      '{ id: "runs", label: "Runs"',
      '{ id: "workflow", label: "Workflows"',
      '{ id: "workspace", label: "Workspace"',
      '{ id: "asset", label: "Assets"',
      '{ id: "agent", label: "Agent Tasks"',
      '{ id: "knowledge", label: "Knowledge"',
    ]) {
      expect(src).toContain(label);
    }
  });

  it("keeps the styled Radix tooltip alongside the native title", () => {
    expect(src).toContain('<TooltipContent side="right">{option.label}</TooltipContent>');
  });
});

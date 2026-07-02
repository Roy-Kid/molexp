/**
 * Tests for the countLabel pluralization helper plus a source contract
 * that the LeftPanel experiment/run count chips use it (rstest node env,
 * no jsdom — see app/runs/metrics/RunMetricsView.test.tsx).
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

import { countLabel } from "@/lib/count-label";

const __dirname = dirname(fileURLToPath(import.meta.url));

describe("countLabel", () => {
  it("uses the singular for exactly one", () => {
    expect(countLabel(1, "run")).toBe("1 run");
    expect(countLabel(1, "exp")).toBe("1 exp");
  });

  it("pluralizes zero and many", () => {
    expect(countLabel(0, "run")).toBe("0 runs");
    expect(countLabel(3, "exp")).toBe("3 exps");
  });

  it("accepts an explicit plural", () => {
    expect(countLabel(2, "study", "studies")).toBe("2 studies");
  });
});

describe("LeftPanel count chips (source contract)", () => {
  it("experiment and run counts are pluralized via countLabel", () => {
    const src = readFileSync(resolve(__dirname, "../app/panels/LeftPanel.tsx"), "utf8");
    expect(src).toContain('countLabel(project.experiments.length, "exp")');
    expect(src).toContain('countLabel(experiment.runs.length, "run")');
    // The old hardcoded plural strings are gone.
    expect(src).not.toContain("} exp</CompactCount>");
    expect(src).not.toContain("} runs</CompactCount>");
  });
});

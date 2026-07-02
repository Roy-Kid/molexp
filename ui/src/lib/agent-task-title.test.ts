/**
 * Tests for the agent-task display-title helpers, plus source-contract
 * assertions (rstest runs in node without jsdom — see
 * app/runs/metrics/RunMetricsView.test.tsx) that the display surfaces
 * (sidebar, breadcrumb, session header) actually route through them.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

import {
  agentTaskDisplayTitle,
  isAutoDerivedTitle,
  stripMarkdownLine,
} from "@/lib/agent-task-title";

const __dirname = dirname(fileURLToPath(import.meta.url));
const read = (rel: string): string => readFileSync(resolve(__dirname, rel), "utf8");

describe("stripMarkdownLine", () => {
  it("strips heading markers", () => {
    expect(stripMarkdownLine("# Coarse-grained zwitterion study")).toBe(
      "Coarse-grained zwitterion study",
    );
    expect(stripMarkdownLine("### Deep heading")).toBe("Deep heading");
  });

  it("strips list bullets, blockquotes and numbered prefixes", () => {
    expect(stripMarkdownLine("- item one")).toBe("item one");
    expect(stripMarkdownLine("* starred")).toBe("starred");
    expect(stripMarkdownLine("> quoted")).toBe("quoted");
    expect(stripMarkdownLine("1. first step")).toBe("first step");
  });

  it("strips emphasis, inline code, and keeps link labels", () => {
    expect(stripMarkdownLine("**bold** and _em_ and `code`")).toBe("bold and em and code");
    expect(stripMarkdownLine("[molexp docs](https://example.com)")).toBe("molexp docs");
  });

  it("collapses internal whitespace", () => {
    expect(stripMarkdownLine("a   b\t c")).toBe("a b c");
  });
});

describe("isAutoDerivedTitle", () => {
  it("recognizes the server's compacted-goal title as auto-derived", () => {
    const goal = "# Plan\n\nrun the baseline";
    expect(isAutoDerivedTitle("# Plan run the baseline", goal)).toBe(true);
  });

  it("recognizes the truncated ('...') variant as auto-derived", () => {
    const goal = "# A very long draft goal that keeps going";
    expect(isAutoDerivedTitle("# A very long draft...", goal)).toBe(true);
  });

  it("treats an unrelated curated title as NOT auto-derived", () => {
    expect(isAutoDerivedTitle("Zwitterion CG Simulation Plan", "# draft text")).toBe(false);
  });
});

describe("agentTaskDisplayTitle", () => {
  it("strips markdown from a draft goal's first line", () => {
    const task = { title: "", goal: "# CG zwitterion study\n\nDetails follow." };
    expect(agentTaskDisplayTitle(task)).toBe("CG zwitterion study");
  });

  it("ignores the auto-derived title and uses the goal's first line", () => {
    const goal = "# CG study\nmore text";
    const task = { title: "# CG study more text", goal };
    expect(agentTaskDisplayTitle(task)).toBe("CG study");
  });

  it("prefers a curated (plan report) title over the goal draft", () => {
    const task = { title: "Zwitterion CG Simulation Plan", goal: "# messy draft\nstuff" };
    expect(agentTaskDisplayTitle(task)).toBe("Zwitterion CG Simulation Plan");
  });

  it("skips leading blank lines in the goal", () => {
    expect(agentTaskDisplayTitle({ goal: "\n\n  \n## Actual title" })).toBe("Actual title");
  });

  it("clamps to maxLength with an ellipsis", () => {
    const long = "x".repeat(100);
    const result = agentTaskDisplayTitle({ goal: long }, 32);
    expect(result.length).toBeLessThanOrEqual(32);
    expect(result.endsWith("…")).toBe(true);
  });

  it("falls back to a default for empty input", () => {
    expect(agentTaskDisplayTitle({})).toBe("Untitled agent task");
    expect(agentTaskDisplayTitle({ title: "", goal: "  \n " })).toBe("Untitled agent task");
  });
});

describe("display-surface wiring (source contract)", () => {
  it("LeftPanel sidebar labels route through agentTaskDisplayTitle and expose the full goal on hover", () => {
    const src = read("../app/panels/LeftPanel.tsx");
    expect(src).toContain('from "@/lib/agent-task-title"');
    expect(src).toContain("agentTaskDisplayTitle(session");
    expect(src).toContain("hoverTitle: session.goal");
    expect(src).not.toContain("shortenGoal(");
  });

  it("breadcrumb trail routes agent crumbs through agentTaskDisplayTitle", () => {
    const src = read("../app/entities/breadcrumbTrail.ts");
    expect(src).toContain('from "@/lib/agent-task-title"');
    expect(src).toContain("agentTaskDisplayTitle(session");
    expect(src).not.toContain("session?.goal");
  });

  it("AgentViewer session header shows the cleaned title with the raw goal as tooltip", () => {
    const src = read("../app/renderers/AgentViewer.tsx");
    expect(src).toContain("title={agentTaskDisplayTitle(session)}");
    expect(src).toContain("titleTooltip={session.goal}");
  });

  it("mapAgentSessions carries the server title into AgentSessionSummary", () => {
    const src = read("../app/state/api.ts");
    expect(src).toContain('title: s.title ?? ""');
  });
});

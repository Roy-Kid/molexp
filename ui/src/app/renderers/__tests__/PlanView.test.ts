import { describe, expect, it } from "@rstest/core";

import { parsePlan } from "../PlanView";

describe("parsePlan", () => {
  it("returns no steps for empty input", () => {
    expect(parsePlan("")).toEqual([]);
  });

  it("parses a numbered list with tool calls", () => {
    const md = [
      "1. list_projects() — survey what is available",
      "2. get_run_status(run_id=run-1) — check status before plotting",
      "3. run_python(code=...) — render Plotly scatter",
    ].join("\n");
    const steps = parsePlan(md);
    expect(steps).toHaveLength(3);
    expect(steps[0]).toMatchObject({ index: 1, toolName: "list_projects", args: null });
    expect(steps[1]).toMatchObject({
      index: 2,
      toolName: "get_run_status",
      args: "run_id=run-1",
    });
    expect(steps[2].toolName).toBe("run_python");
  });

  it("ignores non-step lines between steps", () => {
    const md = [
      "Here is the plan:",
      "",
      "1. list_runs(experiment_id=e1) — pull recent runs",
      "Some prose explanation.",
      "2. wait_for_run(run_id=r1) — wait for completion",
    ].join("\n");
    const steps = parsePlan(md);
    expect(steps.map((s) => s.toolName)).toEqual(["list_runs", "wait_for_run"]);
  });

  it("falls back to raw line when there is no parens nor rationale", () => {
    const md = "1. just-a-token";
    const steps = parsePlan(md);
    expect(steps).toHaveLength(1);
    expect(steps[0].raw).toBe("1. just-a-token");
  });
});

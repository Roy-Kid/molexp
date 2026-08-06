import { describe, expect, it } from "@rstest/core";
import type { NextActionResponse } from "@/api/generated/models/NextActionResponse";
import { pathForNextAction } from "@/app/components/copilotPaths";
import type { WorkspaceSnapshot } from "@/app/types";

const snapshot = {
  runs: [
    {
      id: "run1",
      projectId: "p1",
      experimentId: "e1",
      status: "failed",
    },
  ],
} as unknown as WorkspaceSnapshot;

describe("pathForNextAction", () => {
  it("maps diagnose_failed_run to the run entity route", () => {
    const action: NextActionResponse = {
      kind: "diagnose_failed_run",
      target: "run1",
      rationale: "failed",
      advisory: true,
      requiresProposal: false,
    };
    expect(pathForNextAction(action, snapshot)).toBe("/projects/p1/experiments/e1/runs/run1");
  });

  it("flags retry as navigable without executing", () => {
    const action: NextActionResponse = {
      kind: "retry_failed_run",
      target: "run1",
      rationale: "retry",
      advisory: true,
      requiresProposal: true,
    };
    const path = pathForNextAction(action, snapshot);
    expect(path).toContain("/runs/run1");
  });

  it("maps open questions toward knowledge", () => {
    const action: NextActionResponse = {
      kind: "answer_open_question",
      target: "notes/q1",
      rationale: "q",
      advisory: true,
      requiresProposal: false,
    };
    expect(pathForNextAction(action, snapshot)).toContain("/knowledge/");
  });
});

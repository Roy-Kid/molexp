import { describe, expect, it } from "@rstest/core";

import type { ApiSessionEvent } from "@/app/types";
import { derivePendingUserRequest, EVENT_META, groupEventsIntoTurns } from "../agentEvents";

const ev = (type: string, payload: Record<string, unknown> = {}): ApiSessionEvent => ({
  type,
  ts: "2026-04-28T00:00:00Z",
  payload,
});

describe("EVENT_META", () => {
  // ac-001 — agentEvents map matches §6.5 new names; legacy entries removed.
  it("exports an entry for every kept harness event name", () => {
    const kept = [
      "PlanCreated",
      "PlanDecided",
      "ToolCallRequested",
      "ToolCallCompleted",
      "ToolApprovalRequested",
      "UserMessageRequested",
      "UserMessageReceived",
      "SessionStarted",
      "SessionCompleted",
      "TurnStarted",
      "ContextBuilt",
      "ModelRequested",
      "ModelResponded",
      "FailureRecorded",
    ];
    for (const name of kept) {
      expect(EVENT_META[name]).toBeDefined();
      expect(typeof EVENT_META[name].label).toBe("string");
    }
  });

  it("does NOT export entries for events dropped in §6.5", () => {
    // ObservationEvent and ReplanEvent are dropped outright.
    // ResultArtifactEvent and WorkflowStartedEvent fold into ToolCallCompleted.
    const dropped = [
      "ObservationEvent",
      "ReplanEvent",
      "ResultArtifactEvent",
      "WorkflowStartedEvent",
    ];
    for (const name of dropped) {
      expect(EVENT_META[name]).toBeUndefined();
    }
  });
});

describe("derivePendingUserRequest", () => {
  it("returns null when there are no events", () => {
    expect(derivePendingUserRequest([])).toBeNull();
  });

  it("returns null when the last event is unrelated", () => {
    expect(
      derivePendingUserRequest([
        ev("ToolCallRequested", { tool_name: "list_runs" }),
        ev("ToolCallCompleted", { tool_name: "list_runs" }),
      ]),
    ).toBeNull();
  });

  it("returns the latest pending request when no reply followed", () => {
    const result = derivePendingUserRequest([
      ev("ToolCallRequested", { tool_name: "list_task_types" }),
      ev("UserMessageRequested", { request_id: "req-1", prompt: "Which scope?" }),
      ev("ToolCallCompleted", { tool_name: "list_task_types" }),
    ]);
    expect(result).toEqual({ requestId: "req-1", prompt: "Which scope?" });
  });

  it("clears the request once a UserMessageReceived answers it", () => {
    expect(
      derivePendingUserRequest([
        ev("UserMessageRequested", { request_id: "req-1", prompt: "?" }),
        ev("UserMessageReceived", { content: "scope=project", request_id: "req-1" }),
      ]),
    ).toBeNull();
  });

  it("treats a request without a string request_id as no pending request", () => {
    expect(
      derivePendingUserRequest([ev("UserMessageRequested", { prompt: "no-id question" })]),
    ).toBeNull();
  });

  it("treats a missing prompt as null prompt rather than failure", () => {
    expect(derivePendingUserRequest([ev("UserMessageRequested", { request_id: "r" })])).toEqual({
      requestId: "r",
      prompt: null,
    });
  });

  it("handles multiple request/reply cycles, returning only the latest unanswered one", () => {
    const result = derivePendingUserRequest([
      ev("UserMessageRequested", { request_id: "r1", prompt: "first" }),
      ev("UserMessageReceived", { content: "answer 1", request_id: "r1" }),
      ev("ToolCallCompleted", { tool_name: "noop" }),
      ev("UserMessageRequested", { request_id: "r2", prompt: "second" }),
    ]);
    expect(result).toEqual({ requestId: "r2", prompt: "second" });
  });

  it("ignores a legacy event type without crashing (§6.5 mixed-log compatibility)", () => {
    // Edge case 1 from the spec — logs that mix old + new must not break the UI.
    const result = derivePendingUserRequest([
      ev("ObservationEvent", { content: "thinking…" }),
      ev("UserMessageRequested", { request_id: "req-1", prompt: "?" }),
    ]);
    expect(result).toEqual({ requestId: "req-1", prompt: "?" });
  });
});

describe("groupEventsIntoTurns", () => {
  it("returns a single in-progress goal turn when there are no events", () => {
    const turns = groupEventsIntoTurns([], "Plot energy");
    expect(turns).toHaveLength(1);
    expect(turns[0]).toMatchObject({
      question: "Plot energy",
      source: "goal",
      result: null,
      inProgress: true,
    });
    expect(turns[0].steps).toEqual([]);
  });

  it("collects intermediate events as steps and promotes a SessionCompleted to the result", () => {
    const events: ApiSessionEvent[] = [
      ev("ToolCallRequested", { tool_name: "list_runs" }),
      ev("ToolCallCompleted", { tool_name: "list_runs", result: { ok: true } }),
      ev("SessionCompleted", { summary: "Found 3 runs." }),
    ];
    const turns = groupEventsIntoTurns(events, "List runs");
    expect(turns).toHaveLength(1);
    expect(turns[0].source).toBe("goal");
    expect(turns[0].steps).toHaveLength(2);
    expect(turns[0].result?.type).toBe("SessionCompleted");
    expect(turns[0].inProgress).toBe(false);
  });

  it("opens a new turn when a UserMessageReceived appears mid-stream", () => {
    const events: ApiSessionEvent[] = [
      ev("ToolCallCompleted", { tool_name: "first_answer" }),
      ev("UserMessageReceived", { content: "follow-up question" }),
      ev("ToolCallRequested", { tool_name: "list_runs" }),
    ];
    const turns = groupEventsIntoTurns(events, "Initial goal");
    expect(turns).toHaveLength(2);
    expect(turns[0]).toMatchObject({
      question: "Initial goal",
      source: "goal",
      inProgress: false,
    });
    // ToolCallCompleted is intermediate — it should land as a step, not a result.
    expect(turns[0].steps.some((s) => s.type === "ToolCallCompleted")).toBe(true);
    expect(turns[0].result).toBeNull();
    expect(turns[1]).toMatchObject({
      question: "follow-up question",
      source: "user",
      inProgress: true,
    });
    expect(turns[1].steps).toHaveLength(1);
    expect(turns[1].result).toBeNull();
  });

  it("folds inline artifacts into a ToolCallCompleted instead of a separate event (§6.5)", () => {
    // ResultArtifactEvent / WorkflowStartedEvent merge into ToolCallCompleted —
    // artifacts ride on result.artifacts and metadata carries run_id / workflow_id.
    const events: ApiSessionEvent[] = [
      ev("ToolCallRequested", { tool_name: "execute_run" }),
      ev("ToolCallCompleted", {
        tool_name: "execute_run",
        result: {
          ok: true,
          value: { status: "running" },
          artifacts: [{ kind: "text", payload: { body: "started" } }],
          metadata: { run_id: "r-1", workflow_id: "w-1" },
        },
      }),
      ev("SessionCompleted", { summary: "Done." }),
    ];
    const turns = groupEventsIntoTurns(events, "Run something");
    expect(turns).toHaveLength(1);
    expect(turns[0].result?.type).toBe("SessionCompleted");
    // The completed tool call carries the artifact inline; no standalone artifact event.
    expect(turns[0].steps.some((s) => s.type === "ToolCallCompleted")).toBe(true);
    expect(turns[0].steps.some((s) => s.type === "ResultArtifactEvent")).toBe(false);
  });

  it("promotes a PlanCreated to the turn result", () => {
    // Every PlanCreated IS the agent's answer for the plan-mode turn —
    // the user reviews + approves it as the headline.
    const events: ApiSessionEvent[] = [
      ev("ToolCallRequested", { tool_name: "list_task_types" }),
      ev("PlanCreated", {
        request_id: "plan-1",
        plan_markdown: "1. inspect\n2. plan\n",
        workflow_preview: {
          workflow_ir: {
            task_configs: [{ task_id: "t1", task_type: "noop", config: {} }],
            links: [],
          },
        },
      }),
    ];
    const turns = groupEventsIntoTurns(events, "Plan something");
    expect(turns).toHaveLength(1);
    expect(turns[0].result?.type).toBe("PlanCreated");
    expect(turns[0].inProgress).toBe(false);
    // The earlier tool call stays as a step.
    expect(turns[0].steps.some((s) => s.type === "ToolCallRequested")).toBe(true);
  });

  it("treats an investigation-heavy plan as the turn result", () => {
    // Investigation steps are nodes in the IR; the plan event still
    // wins the turn-result slot exactly like any other workflow plan.
    const events: ApiSessionEvent[] = [
      ev("ToolCallRequested", { tool_name: "list_projects" }),
      ev("PlanCreated", {
        request_id: "plan-investigate-1",
        plan_markdown: "1. inspect_dataset\n2. summarize",
        workflow_preview: {
          workflow_ir: {
            task_configs: [
              { task_id: "inspect", task_type: "inspect_dataset", config: {} },
              { task_id: "summary", task_type: "summarize_dataset", config: {} },
            ],
            links: [{ source: "inspect", target: "summary" }],
          },
        },
      }),
    ];
    const turns = groupEventsIntoTurns(events, "Survey datasets");
    expect(turns).toHaveLength(1);
    expect(turns[0].result?.type).toBe("PlanCreated");
    const payload = turns[0].result?.payload as {
      workflow_preview?: { workflow_ir?: { task_configs?: unknown[] } };
    };
    expect(payload?.workflow_preview?.workflow_ir?.task_configs).toHaveLength(2);
  });

  it("a SessionCompleted after a plan demotes the plan to a step", () => {
    // Post-approval the agent finishes the workflow and emits a
    // SessionCompleted. That becomes the new headline; the prior
    // plan event is still inspectable via the steps drawer.
    const events: ApiSessionEvent[] = [
      ev("PlanCreated", {
        request_id: "plan-1",
        plan_markdown: "1. step",
        workflow_preview: {
          workflow_ir: {
            task_configs: [{ task_id: "t1", task_type: "noop", config: {} }],
            links: [],
          },
        },
      }),
      ev("ToolCallRequested", { tool_name: "execute_run" }),
      ev("SessionCompleted", { summary: "Done." }),
    ];
    const turns = groupEventsIntoTurns(events, "Goal");
    expect(turns[0].result?.type).toBe("SessionCompleted");
    expect(turns[0].steps.some((s) => s.type === "PlanCreated")).toBe(true);
    expect(turns[0].steps.some((s) => s.type === "ToolCallRequested")).toBe(true);
  });
});

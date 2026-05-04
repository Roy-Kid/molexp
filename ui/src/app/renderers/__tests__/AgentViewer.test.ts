import { describe, expect, it } from "@rstest/core";

import type { ApiSessionEvent } from "@/app/types";
import { derivePendingUserRequest, groupEventsIntoTurns } from "../agentEvents";

const ev = (type: string, payload: Record<string, unknown> = {}): ApiSessionEvent => ({
  type,
  ts: "2026-04-28T00:00:00Z",
  payload,
});

describe("derivePendingUserRequest", () => {
  it("returns null when there are no events", () => {
    expect(derivePendingUserRequest([])).toBeNull();
  });

  it("returns null when the last event is unrelated", () => {
    expect(
      derivePendingUserRequest([
        ev("ToolCallEvent", { tool_name: "list_runs" }),
        ev("ObservationEvent", { content: "thinking…" }),
      ]),
    ).toBeNull();
  });

  it("returns the latest pending request when no reply followed", () => {
    const result = derivePendingUserRequest([
      ev("ToolCallEvent", { tool_name: "list_task_types" }),
      ev("UserMessageRequestEvent", { request_id: "req-1", prompt: "Which scope?" }),
      ev("ObservationEvent", { content: "thinking…" }),
    ]);
    expect(result).toEqual({ requestId: "req-1", prompt: "Which scope?" });
  });

  it("clears the request once a UserMessageEvent answers it", () => {
    expect(
      derivePendingUserRequest([
        ev("UserMessageRequestEvent", { request_id: "req-1", prompt: "?" }),
        ev("UserMessageEvent", { content: "scope=project", request_id: "req-1" }),
      ]),
    ).toBeNull();
  });

  it("treats a request without a string request_id as no pending request", () => {
    expect(
      derivePendingUserRequest([ev("UserMessageRequestEvent", { prompt: "no-id question" })]),
    ).toBeNull();
  });

  it("treats a missing prompt as null prompt rather than failure", () => {
    expect(derivePendingUserRequest([ev("UserMessageRequestEvent", { request_id: "r" })])).toEqual({
      requestId: "r",
      prompt: null,
    });
  });

  it("handles multiple request/reply cycles, returning only the latest unanswered one", () => {
    const result = derivePendingUserRequest([
      ev("UserMessageRequestEvent", { request_id: "r1", prompt: "first" }),
      ev("UserMessageEvent", { content: "answer 1", request_id: "r1" }),
      ev("ObservationEvent", { content: "thinking…" }),
      ev("UserMessageRequestEvent", { request_id: "r2", prompt: "second" }),
    ]);
    expect(result).toEqual({ requestId: "r2", prompt: "second" });
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

  it("collects intermediate events as steps and promotes a SessionCompletedEvent to the result", () => {
    const events: ApiSessionEvent[] = [
      ev("ToolCallEvent", { tool_name: "list_runs" }),
      ev("ObservationEvent", { content: "got 3 runs" }),
      ev("SessionCompletedEvent", { summary: "Found 3 runs." }),
    ];
    const turns = groupEventsIntoTurns(events, "List runs");
    expect(turns).toHaveLength(1);
    expect(turns[0].source).toBe("goal");
    expect(turns[0].steps).toHaveLength(2);
    expect(turns[0].result?.type).toBe("SessionCompletedEvent");
    expect(turns[0].inProgress).toBe(false);
  });

  it("opens a new turn when a UserMessageEvent appears mid-stream", () => {
    const events: ApiSessionEvent[] = [
      ev("ResultArtifactEvent", { kind: "text", payload: { body: "first answer" } }),
      ev("UserMessageEvent", { content: "follow-up question" }),
      ev("ToolCallEvent", { tool_name: "list_runs" }),
    ];
    const turns = groupEventsIntoTurns(events, "Initial goal");
    expect(turns).toHaveLength(2);
    expect(turns[0]).toMatchObject({
      question: "Initial goal",
      source: "goal",
      inProgress: false,
    });
    expect(turns[0].result?.type).toBe("ResultArtifactEvent");
    expect(turns[1]).toMatchObject({
      question: "follow-up question",
      source: "user",
      inProgress: true,
    });
    expect(turns[1].steps).toHaveLength(1);
    expect(turns[1].result).toBeNull();
  });

  it("keeps the last result event as headline when multiple results appear in one turn", () => {
    const events: ApiSessionEvent[] = [
      ev("ResultArtifactEvent", { kind: "text", payload: { body: "intermediate" } }),
      ev("SessionCompletedEvent", { summary: "Done." }),
    ];
    const turns = groupEventsIntoTurns(events, "Goal");
    expect(turns).toHaveLength(1);
    expect(turns[0].result?.type).toBe("SessionCompletedEvent");
    // The earlier ResultArtifactEvent is still inspectable via steps.
    expect(turns[0].steps.some((s) => s.type === "ResultArtifactEvent")).toBe(true);
  });

  it("promotes a PlanCreatedEvent to the turn result", () => {
    // Every PlanCreatedEvent IS the agent's answer for the plan-mode
    // turn — the user reviews + approves it as the headline.
    const events: ApiSessionEvent[] = [
      ev("ToolCallEvent", { tool_name: "list_task_types" }),
      ev("PlanCreatedEvent", {
        request_id: "plan-1",
        plan_markdown: "1. inspect\n2. plan\n",
        workflow_preview: {
          workflow_ir: {
            task_configs: [
              { task_id: "t1", task_type: "noop", config: {} },
            ],
            links: [],
          },
        },
      }),
    ];
    const turns = groupEventsIntoTurns(events, "Plan something");
    expect(turns).toHaveLength(1);
    expect(turns[0].result?.type).toBe("PlanCreatedEvent");
    expect(turns[0].inProgress).toBe(false);
    // The earlier tool call stays as a step.
    expect(turns[0].steps.some((s) => s.type === "ToolCallEvent")).toBe(true);
  });

  it("treats an investigation-heavy plan as the turn result", () => {
    // Investigation steps are nodes in the IR; the plan event still
    // wins the turn-result slot exactly like any other workflow plan.
    const events: ApiSessionEvent[] = [
      ev("ToolCallEvent", { tool_name: "list_projects" }),
      ev("PlanCreatedEvent", {
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
    expect(turns[0].result?.type).toBe("PlanCreatedEvent");
    const payload = turns[0].result?.payload as {
      workflow_preview?: { workflow_ir?: { task_configs?: unknown[] } };
    };
    expect(payload?.workflow_preview?.workflow_ir?.task_configs).toHaveLength(2);
  });

  it("a SessionCompletedEvent after a plan demotes the plan to a step", () => {
    // Post-approval the agent finishes the workflow and emits a
    // SessionCompletedEvent. That becomes the new headline; the prior
    // plan event is still inspectable via the steps drawer.
    const events: ApiSessionEvent[] = [
      ev("PlanCreatedEvent", {
        request_id: "plan-1",
        plan_markdown: "1. step",
        workflow_preview: {
          workflow_ir: {
            task_configs: [
              { task_id: "t1", task_type: "noop", config: {} },
            ],
            links: [],
          },
        },
      }),
      ev("ToolCallEvent", { tool_name: "execute_run" }),
      ev("SessionCompletedEvent", { summary: "Done." }),
    ];
    const turns = groupEventsIntoTurns(events, "Goal");
    expect(turns[0].result?.type).toBe("SessionCompletedEvent");
    expect(turns[0].steps.some((s) => s.type === "PlanCreatedEvent")).toBe(true);
    expect(turns[0].steps.some((s) => s.type === "ToolCallEvent")).toBe(true);
  });
});

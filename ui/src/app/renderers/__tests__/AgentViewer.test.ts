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
        ev("PlanCreatedEvent", { plan_steps: ["a"] }),
        ev("ToolCallEvent", { tool_name: "list_runs" }),
      ]),
    ).toBeNull();
  });

  it("returns the latest pending request when no reply followed", () => {
    const result = derivePendingUserRequest([
      ev("PlanCreatedEvent", { plan_steps: ["a"] }),
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
      ev("PlanCreatedEvent", { plan_steps: ["a", "b"] }),
      ev("ToolCallEvent", { tool_name: "list_runs" }),
      ev("ObservationEvent", { content: "got 3 runs" }),
      ev("SessionCompletedEvent", { summary: "Found 3 runs." }),
    ];
    const turns = groupEventsIntoTurns(events, "List runs");
    expect(turns).toHaveLength(1);
    expect(turns[0].source).toBe("goal");
    expect(turns[0].steps).toHaveLength(3);
    expect(turns[0].result?.type).toBe("SessionCompletedEvent");
    expect(turns[0].inProgress).toBe(false);
  });

  it("opens a new turn when a UserMessageEvent appears mid-stream", () => {
    const events: ApiSessionEvent[] = [
      ev("PlanCreatedEvent", { plan_steps: ["a"] }),
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
});

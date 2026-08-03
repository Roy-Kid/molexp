import { describe, expect, it } from "@rstest/core";

import type { ApiSessionEvent } from "@/app/types";
import { nextAgentMode } from "../agent/agentMode";
import { DEFAULT_PLAN_STAGE, PLAN_STAGES, planStage } from "../agent/planStages";
import {
  collectArtifacts,
  completedStageKinds,
  derivePendingUserRequest,
  derivePlanRef,
  EVENT_META,
  groupEventsIntoTurns,
  normalizeStreamFrame,
} from "../agentEvents";

const ev = (type: string, payload: Record<string, unknown> = {}): ApiSessionEvent => ({
  type,
  ts: "2026-05-30T00:00:00Z",
  payload,
});

const KINDS = [
  "loop_started",
  "loop_completed",
  "stage_started",
  "stage_completed",
  "plan_emitted",
  "approval_requested",
  "approval_decided",
  "tool_call_started",
  "tool_call_completed",
  "artifact_written",
  "preflight_failed",
  "repair_proposed",
  "clarification_required",
  "compaction_performed",
  "error",
  "thinking_delta",
  "token_delta",
  "llm_call",
];

describe("agent mode switching", () => {
  it("cycles the single task composer between Chat and Plan", () => {
    expect(nextAgentMode("chat")).toBe("plan");
    expect(nextAgentMode("plan")).toBe("chat");
  });
});

describe("EVENT_META", () => {
  it("defines an entry for every snake_case AgentEvent kind", () => {
    for (const kind of KINDS) {
      expect(EVENT_META[kind]).toBeDefined();
      expect(typeof EVENT_META[kind].label).toBe("string");
      expect(EVENT_META[kind].label.length).toBeGreaterThan(0);
      expect(EVENT_META[kind].icon).toBeDefined();
      expect(typeof EVENT_META[kind].colorClass).toBe("string");
    }
  });

  it("drops every old PascalCase key", () => {
    const dropped = [
      "SessionStarted",
      "PlanCreated",
      "PlanDecided",
      "ToolApprovalRequested",
      "UserMessageRequested",
      "UserMessageReceived",
      "ModelResponded",
      "ModelRequested",
      "TurnStarted",
      "ContextBuilt",
      "FailureRecorded",
      "SessionCompleted",
      "ToolCallRequested",
      "ToolCallCompleted",
    ];
    for (const name of dropped) {
      expect(EVENT_META[name]).toBeUndefined();
    }
  });
});

describe("normalizeStreamFrame", () => {
  it("maps an AgentEvent frame to snapshot wire shape (no timestamp in payload)", () => {
    const frame = { kind: "token_delta", timestamp: "2026-05-30T00:00:01Z", text: "hi" };
    expect(normalizeStreamFrame(frame)).toEqual({
      type: "token_delta",
      ts: "2026-05-30T00:00:01Z",
      // timestamp stripped so SSE rows dedupe against getSession snapshots
      payload: { kind: "token_delta", text: "hi" },
    });
  });

  it("returns null for a done control frame", () => {
    expect(normalizeStreamFrame({ type: "done" })).toBeNull();
  });

  it("returns null for a waiting control frame", () => {
    expect(normalizeStreamFrame({ type: "waiting" })).toBeNull();
  });

  it("passes through an already-normalized {type, ts, payload} envelope", () => {
    const env = { type: "loop_completed", ts: "2026-05-30T00:00:02Z", payload: { text: "done" } };
    expect(normalizeStreamFrame(env)).toEqual(env);
  });
});

describe("derivePendingUserRequest", () => {
  it("returns null when there are no events", () => {
    expect(derivePendingUserRequest([])).toBeNull();
  });

  it("returns null when the last event is unrelated", () => {
    expect(
      derivePendingUserRequest([
        ev("tool_call_started", { tool_name: "list_runs" }),
        ev("tool_call_completed", { tool_name: "list_runs" }),
      ]),
    ).toBeNull();
  });

  it("returns a pending request for a trailing clarification_required (no request_id dep)", () => {
    const result = derivePendingUserRequest([
      ev("tool_call_started", { tool_name: "list_task_types" }),
      ev("clarification_required", { gate: "scope", questions: "Which scope?" }),
    ]);
    expect(result).toMatchObject({ requestId: "scope", prompt: "Which scope?" });
  });

  it("clears once a later loop_completed supersedes the clarification", () => {
    expect(
      derivePendingUserRequest([
        ev("clarification_required", { gate: "scope", questions: "?" }),
        ev("loop_completed", { text: "resolved" }),
      ]),
    ).toBeNull();
  });

  it("clears once a later loop_started supersedes the clarification", () => {
    expect(
      derivePendingUserRequest([
        ev("clarification_required", { gate: "scope", questions: "Which scope?" }),
        ev("loop_started", { user_input: "proj / exp" }),
      ]),
    ).toBeNull();
  });

  it("ignores an unknown/legacy event without crashing (mixed-log compatibility)", () => {
    const result = derivePendingUserRequest([
      ev("ObservationEvent", { content: "thinking…" }),
      ev("clarification_required", { gate: "g", questions: "pick one" }),
    ]);
    expect(result).toMatchObject({ requestId: "g", prompt: "pick one" });
  });

  it("carries experiment catalog for the in-bubble scope form", () => {
    const result = derivePendingUserRequest([
      ev("clarification_required", {
        request_id: "context-1",
        questions: "Choose scope",
        context_kind: "experiment",
        allow_create: true,
        catalog: [{ project_id: "p", experiment_id: "e", label: "p / e" }],
      }),
    ]);
    expect(result?.contextKind).toBe("experiment");
    expect(result?.catalog).toEqual([{ project_id: "p", experiment_id: "e", label: "p / e" }]);
    expect(result?.allowCreate).toBe(true);
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

  it("absorbs the leading loop_started into the goal turn and promotes loop_completed", () => {
    const events: ApiSessionEvent[] = [
      ev("loop_started", { user_input: "List runs" }),
      ev("tool_call_started", { tool_name: "list_runs" }),
      ev("tool_call_completed", { tool_name: "list_runs" }),
      ev("loop_completed", { text: "Found 3 runs." }),
    ];
    const turns = groupEventsIntoTurns(events, "List runs");
    expect(turns).toHaveLength(1);
    expect(turns[0].source).toBe("goal");
    // loop_started is a boundary, not a step; the two tool events are steps.
    expect(turns[0].steps).toHaveLength(2);
    expect(turns[0].steps.every((s) => s.type !== "loop_started")).toBe(true);
    expect(turns[0].result?.type).toBe("loop_completed");
    expect(turns[0].inProgress).toBe(false);
  });

  it("opens a new turn on a subsequent loop_started using its user_input", () => {
    const events: ApiSessionEvent[] = [
      ev("loop_started", { user_input: "Initial goal" }),
      ev("tool_call_completed", { tool_name: "first" }),
      ev("loop_completed", { text: "first answer" }),
      ev("loop_started", { user_input: "follow-up question" }),
      ev("tool_call_started", { tool_name: "list_runs" }),
    ];
    const turns = groupEventsIntoTurns(events, "Initial goal");
    expect(turns).toHaveLength(2);
    expect(turns[0]).toMatchObject({ question: "Initial goal", source: "goal", inProgress: false });
    expect(turns[0].result?.type).toBe("loop_completed");
    expect(turns[1]).toMatchObject({
      question: "follow-up question",
      source: "user",
      inProgress: true,
    });
    expect(turns[1].steps).toHaveLength(1);
    expect(turns[1].result).toBeNull();
  });

  it("collapses a duplicate loop_started for the same open prompt (SSE+snapshot)", () => {
    const events: ApiSessionEvent[] = [
      ev("loop_started", { user_input: "Plot Rg", kind: "loop_started" }),
      ev("tool_call_started", { tool_name: "code_run" }),
      // Same prompt, still in flight — stream/snapshot double delivery.
      ev("loop_started", { user_input: "Plot Rg", kind: "loop_started" }),
      ev("tool_call_completed", { tool_name: "code_run" }),
    ];
    const turns = groupEventsIntoTurns(events, "Plot Rg");
    expect(turns).toHaveLength(1);
    expect(turns[0].source).toBe("goal");
    expect(turns[0].inProgress).toBe(true);
    expect(turns[0].steps).toHaveLength(2);
  });

  it("promotes a plan_emitted to the turn result and keeps tool calls as steps", () => {
    const events: ApiSessionEvent[] = [
      ev("loop_started", { user_input: "Plan something" }),
      ev("tool_call_started", { tool_name: "list_task_types" }),
      ev("plan_emitted", { plan_id: "plan-1", step_count: 2 }),
    ];
    const turns = groupEventsIntoTurns(events, "Plan something");
    expect(turns).toHaveLength(1);
    expect(turns[0].result?.type).toBe("plan_emitted");
    expect(turns[0].inProgress).toBe(false);
    expect(turns[0].steps.some((s) => s.type === "tool_call_started")).toBe(true);
  });

  it("demotes an earlier plan_emitted to a step when loop_completed follows", () => {
    const events: ApiSessionEvent[] = [
      ev("loop_started", { user_input: "Run it" }),
      ev("plan_emitted", { plan_id: "plan-1", step_count: 1 }),
      ev("tool_call_started", { tool_name: "execute_run" }),
      ev("loop_completed", { text: "done" }),
    ];
    const turns = groupEventsIntoTurns(events, "Run it");
    expect(turns).toHaveLength(1);
    expect(turns[0].result?.type).toBe("loop_completed");
    expect(turns[0].steps.some((s) => s.type === "plan_emitted")).toBe(true);
  });

  it("closes the turn on loop_suspended so it is not left inProgress", () => {
    const events: ApiSessionEvent[] = [
      ev("loop_started", { user_input: "Need review" }),
      ev("tool_call_started", { tool_name: "propose" }),
      ev("loop_suspended", { reason: "awaiting approval", leaf_id: "leaf-1" }),
    ];
    const turns = groupEventsIntoTurns(events, "Need review");
    expect(turns).toHaveLength(1);
    expect(turns[0].result?.type).toBe("loop_suspended");
    expect(turns[0].inProgress).toBe(false);
  });

  it("opens a live turn when loop_started follows a bare clarification_required", () => {
    // Create-plan used to write only clarification_required; the scope reply
    // then wrote the first loop_started, which must not stay absorbed into the
    // already-closed goal bubble (that froze the UI with no spinner).
    const events: ApiSessionEvent[] = [
      ev("clarification_required", {
        request_id: "context-turn-1",
        questions: "Choose project / experiment.",
        context_kind: "experiment",
        mode: "plan",
      }),
      ev("loop_started", { user_input: "randomwalk / rg", mode: "plan" }),
      ev("stage_started", { stage: "plan", message: "Drafting the task board…" }),
      ev("thinking_delta", { text: "placing build step" }),
    ];
    const turns = groupEventsIntoTurns(events, "Plan a PE Rg scan");
    expect(turns).toHaveLength(2);
    expect(turns[0].result?.type).toBe("clarification_required");
    expect(turns[0].inProgress).toBe(false);
    expect(turns[1]).toMatchObject({
      question: "randomwalk / rg",
      source: "user",
      inProgress: true,
    });
    expect(turns[1].steps.some((s) => s.type === "stage_started")).toBe(true);
    expect(turns[1].steps.some((s) => s.type === "thinking_delta")).toBe(true);
  });

  it("promotes clarification_required to the turn answer (not a side prompt)", () => {
    const events: ApiSessionEvent[] = [
      ev("clarification_required", {
        request_id: "context-turn-1",
        questions:
          "Which project and experiment should this plan belong to? Reply with `project / experiment`.",
        context_kind: "experiment",
        mode: "plan",
      }),
    ];
    const turns = groupEventsIntoTurns(events, "Plan a PE Rg scan");
    expect(turns).toHaveLength(1);
    expect(turns[0].result?.type).toBe("clarification_required");
    expect(turns[0].inProgress).toBe(false);
    expect(turns[0].steps).toEqual([]);
    expect((turns[0].result?.payload as { questions?: string }).questions).toContain(
      "project and experiment",
    );
  });
});

describe("derivePlanRef", () => {
  it("returns null for a chat session with no plan locator", () => {
    expect(derivePlanRef([ev("loop_completed", { text: "done" })])).toBeNull();
  });

  it("lifts the plan locator off a terminal loop_completed", () => {
    const events: ApiSessionEvent[] = [
      ev("tool_call_completed", { tool_name: "Saved the experiment draft" }),
      ev("loop_completed", {
        text: "**Melt plan** — ready.",
        plan: {
          run_id: "run-1",
          project_id: "proj",
          experiment_id: "exp",
          title: "Melt plan",
          step_count: 2,
          has_workflow: true,
        },
      }),
    ];
    expect(derivePlanRef(events)).toEqual({
      runId: "run-1",
      projectId: "proj",
      experimentId: "exp",
      title: "Melt plan",
      stepCount: 2,
      hasWorkflow: true,
    });
  });

  it("returns null when the locator is missing an id", () => {
    const events: ApiSessionEvent[] = [
      ev("loop_completed", { plan: { run_id: "run-1", project_id: "", experiment_id: "exp" } }),
    ];
    expect(derivePlanRef(events)).toBeNull();
  });

  it("opens deliverables from plan_emitted at the review gate", () => {
    const events: ApiSessionEvent[] = [
      ev("plan_emitted", {
        plan_id: "run-42",
        step_count: 3,
        plan: {
          run_id: "run-42",
          project_id: "p",
          experiment_id: "e",
          title: "Rg",
          step_count: 3,
          has_workflow: false,
        },
      }),
    ];
    expect(derivePlanRef(events)).toEqual({
      runId: "run-42",
      projectId: "p",
      experimentId: "e",
      title: "Rg",
      stepCount: 3,
      hasWorkflow: false,
    });
  });

  it("falls back to session mount metadata when events lag", () => {
    expect(
      derivePlanRef([], {
        runId: "run-9",
        projectId: "proj",
        experimentId: "exp",
        title: "Goal",
      }),
    ).toEqual({
      runId: "run-9",
      projectId: "proj",
      experimentId: "exp",
      title: "Goal",
      stepCount: 0,
      hasWorkflow: false,
    });
  });
});

describe("collectArtifacts", () => {
  it("returns an empty array when no tool result carried artifacts", () => {
    expect(collectArtifacts([ev("loop_completed", { text: "done" })])).toEqual([]);
  });

  it("gathers artifacts from tool_call_completed payloads in stream order", () => {
    const events: ApiSessionEvent[] = [
      ev("tool_call_completed", {
        result: { artifacts: [{ kind: "plot", title: "A" }] },
      }),
      ev("tool_call_completed", {
        result: { artifacts: [{ kind: "table", title: "B" }] },
      }),
    ];
    const artifacts = collectArtifacts(events);
    expect(artifacts.map((a) => a.title)).toEqual(["A", "B"]);
  });
});

describe("completedStageKinds", () => {
  it("reads the produced artifact kind off each synthesized stage step", () => {
    const events: ApiSessionEvent[] = [
      ev("tool_call_completed", {
        tool_name: "Saved the draft",
        result: { artifact: "user_plan" },
      }),
      ev("tool_call_completed", {
        tool_name: "Drafted the spec",
        result: { artifact: "experiment_report" },
      }),
      ev("loop_completed", { text: "done" }),
    ];
    const kinds = completedStageKinds(events);
    expect([...kinds].sort()).toEqual(["experiment_report", "user_plan"]);
  });

  it("returns an empty set when no stage step carried a kind", () => {
    expect(completedStageKinds([ev("loop_completed", { text: "done" })]).size).toBe(0);
  });
});

describe("PLAN_STAGES", () => {
  it("declares two-phase plan artifacts + optional execute tail", () => {
    // Mirrors ui planStages + services materialize artifact kinds.
    expect(PLAN_STAGES.map((s) => s.kind)).toEqual([
      "experiment_plan",
      "review_pack",
      "analysis_result",
      "frozen_experiment_plan",
      "plan_report",
      "experiment_spec",
      "bound_workflow",
      "workflow_source",
      "test_source",
      "execution_result",
      "intervention_request",
      "final_report",
      "audit_report",
    ]);
  });

  it("marks ONLY the --execute tail stages", () => {
    const tail = PLAN_STAGES.filter((s) => s.executeTail).map((s) => s.kind);
    expect(tail).toEqual(["final_report", "audit_report"]);
  });

  it("defaults to the task board and maps kinds to deliverable views", () => {
    expect(DEFAULT_PLAN_STAGE).toBe("experiment_plan");
    expect(planStage("experiment_plan")?.view).toBe("board");
    expect(planStage("experiment_spec")?.view).toBe("spec");
    expect(planStage("workflow_source")?.view).toBe("script");
    expect(planStage("analysis_result")?.view).toBe("review");
    expect(planStage("final_report")?.view).toBe("final");
    expect(planStage("audit_report")?.view).toBe("audit");
    expect(planStage("nope")).toBeUndefined();
  });
});

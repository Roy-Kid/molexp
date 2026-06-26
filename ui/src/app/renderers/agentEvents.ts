import {
  Bot,
  CheckCircle2,
  ClipboardList,
  FileText,
  HelpCircle,
  Milestone,
  Minimize2,
  Play,
  ShieldAlert,
  ShieldCheck,
  Terminal,
  Wrench,
  XCircle,
} from "lucide-react";
import type { ComponentType } from "react";
import type { ApiSessionEvent } from "@/app/types";

export interface PendingUserRequest {
  requestId: string;
  prompt: string | null;
}

export interface EventMeta {
  icon: ComponentType<{ className?: string }>;
  label: string;
  colorClass: string;
}

/**
 * Icon + label dispatcher for the snake_case `AgentEvent` vocabulary.
 *
 * Keyed on the event `kind` (carried on the UI event's `type` field — the
 * server snapshot already sets `type = kind`, and live SSE frames are
 * normalized by {@link normalizeStreamFrame}). Only the 16 canonical kinds
 * appear here; `AgentViewer` falls back to a neutral row for any unknown
 * `type`, so mixed-log sessions (pre-rename PascalCase events on disk) still
 * render without crashing.
 */
export const EVENT_META: Record<string, EventMeta> = {
  // Colors come exclusively from the molcrafts semantic token set:
  // info = activity/accent, success/warning/destructive = state,
  // muted-foreground = lifecycle metadata. No decorative hues.
  loop_started: { icon: Play, label: "Started", colorClass: "text-info" },
  loop_completed: { icon: CheckCircle2, label: "Completed", colorClass: "text-success" },
  stage_started: { icon: Milestone, label: "Stage started", colorClass: "text-muted-foreground" },
  stage_completed: {
    icon: Milestone,
    label: "Stage completed",
    colorClass: "text-muted-foreground",
  },
  plan_emitted: { icon: ClipboardList, label: "Plan created", colorClass: "text-info" },
  approval_requested: {
    icon: ShieldAlert,
    label: "Approval needed",
    colorClass: "text-warning",
  },
  approval_decided: {
    icon: ShieldCheck,
    label: "Approval decided",
    colorClass: "text-info",
  },
  tool_call_started: { icon: Terminal, label: "Tool call", colorClass: "text-info" },
  tool_call_completed: { icon: CheckCircle2, label: "Tool result", colorClass: "text-success" },
  artifact_written: {
    icon: FileText,
    label: "Artifact written",
    colorClass: "text-info",
  },
  preflight_failed: { icon: XCircle, label: "Preflight failed", colorClass: "text-destructive" },
  repair_proposed: { icon: Wrench, label: "Repair proposed", colorClass: "text-warning" },
  clarification_required: { icon: HelpCircle, label: "Question", colorClass: "text-warning" },
  compaction_performed: {
    icon: Minimize2,
    label: "Compaction",
    colorClass: "text-muted-foreground",
  },
  error: { icon: XCircle, label: "Error", colorClass: "text-destructive" },
  thinking_delta: { icon: Bot, label: "Thinking", colorClass: "text-muted-foreground" },
  token_delta: { icon: Bot, label: "Response", colorClass: "text-muted-foreground" },
};

/**
 * Normalize one live SSE frame into the UI's `{type, ts, payload}` shape.
 *
 * The stream interleaves typed `AgentEvent` frames (`{kind, timestamp, …}`)
 * with control frames (`{type:"done"}`, `{type:"waiting"}`). An AgentEvent
 * frame becomes `{type: kind, ts: timestamp, payload: frame}` so it keys
 * identically to the session snapshot (where the server already sets
 * `type = kind`). Control frames return `null` — the caller closes the stream
 * on `done` and skips `waiting` rather than appending them.
 */
export const normalizeStreamFrame = (data: Record<string, unknown>): ApiSessionEvent | null => {
  const ctrl = data.type;
  if (ctrl === "done" || ctrl === "waiting") return null;
  if (typeof data.kind === "string") {
    return {
      type: data.kind,
      ts: typeof data.timestamp === "string" ? data.timestamp : "",
      payload: data,
    };
  }
  // Already a {type, ts, payload} envelope (e.g. a snapshot event echoed back).
  return {
    type: typeof data.type === "string" ? data.type : "",
    ts: typeof data.ts === "string" ? data.ts : "",
    payload: (data.payload as Record<string, unknown>) ?? {},
  };
};

/**
 * Walk an event log backwards to detect whether the agent is currently
 * waiting on the user. In the snake_case vocabulary the only such event is
 * `clarification_required` (a PlanMode prompt); it carries `questions` rather
 * than a `request_id`, so a `gate`/synthetic id is used. Resolved once a
 * later `loop_completed` / `approval_decided` supersedes it.
 *
 * Largely inactive until the clarification/approval path fires agent-side.
 */
export const derivePendingUserRequest = (events: ApiSessionEvent[]): PendingUserRequest | null => {
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    if (ev.type === "loop_completed" || ev.type === "approval_decided") return null;
    if (ev.type === "clarification_required") {
      const payload = (ev.payload ?? {}) as Record<string, unknown>;
      const rid =
        typeof payload.request_id === "string"
          ? payload.request_id
          : typeof payload.gate === "string"
            ? payload.gate
            : "clarification";
      return {
        requestId: rid,
        prompt: typeof payload.questions === "string" ? payload.questions : null,
      };
    }
  }
  return null;
};

export interface ConversationTurn {
  /** Stable key for React lists. */
  key: string;
  /** The question that opened this turn. The first turn uses the goal. */
  question: string;
  /** Whether this turn was opened by the original goal vs a follow-up message. */
  source: "goal" | "user";
  /** The agent's final answer for this turn, if any. */
  result: ApiSessionEvent | null;
  /** Intermediate events (plans, tool calls, …) — collapsed by default. */
  steps: ApiSessionEvent[];
  /** True when this turn is still streaming (no terminal result yet). */
  inProgress: boolean;
  /** ISO timestamp of the loop_started that opened this turn, if seen. */
  startedTs: string | null;
}

/** Wall-clock seconds from a turn's opening loop_started to its result. */
export const turnDurationSeconds = (turn: ConversationTurn): number | null => {
  if (!turn.startedTs || !turn.result?.ts) return null;
  const started = new Date(turn.startedTs).getTime();
  const finished = new Date(turn.result.ts).getTime();
  if (Number.isNaN(started) || Number.isNaN(finished) || finished < started) return null;
  return (finished - started) / 1000;
};

const isResultEvent = (event: ApiSessionEvent): boolean =>
  event.type === "loop_completed" ||
  // plan_emitted IS the agent's answer for a plan-mode turn — the user reviews
  // and approves it as the headline; a later loop_completed overrides it.
  event.type === "plan_emitted";

const eventKey = (event: ApiSessionEvent, fallback: number): string =>
  `${event.type}-${event.ts}-${fallback}`;

/**
 * Group events into conversational turns. Each turn begins with a
 * `loop_started` event (carrying the turn's `user_input`): the first one is
 * absorbed into the implicit goal turn, and every subsequent `loop_started`
 * opens a new turn. A turn closes on `loop_completed` / `plan_emitted`.
 *
 * Intermediate events (tool calls, …) are surfaced as `steps` so the UI can
 * collapse them; a `loop_started` boundary is not itself a step row.
 */
export const groupEventsIntoTurns = (
  events: ApiSessionEvent[],
  goal: string,
): ConversationTurn[] => {
  const turns: ConversationTurn[] = [];

  let current: ConversationTurn = {
    key: "turn-goal",
    question: goal,
    source: "goal",
    result: null,
    steps: [],
    inProgress: true,
    startedTs: null,
  };
  let sawFirstLoopStarted = false;

  events.forEach((event, idx) => {
    if (event.type === "loop_started") {
      if (!sawFirstLoopStarted) {
        // The first loop_started opens the implicit goal turn — absorb it.
        sawFirstLoopStarted = true;
        current.startedTs = event.ts;
        return;
      }
      const payload = (event.payload ?? {}) as Record<string, unknown>;
      const question = typeof payload.user_input === "string" ? payload.user_input : "";
      current.inProgress = false;
      turns.push(current);
      current = {
        key: eventKey(event, idx),
        question,
        source: "user",
        result: null,
        steps: [],
        inProgress: true,
        startedTs: event.ts,
      };
      return;
    }

    if (isResultEvent(event)) {
      if (current.result) {
        current.steps.push(current.result);
      }
      current.result = event;
      current.inProgress = false;
      return;
    }

    current.steps.push(event);
  });

  turns.push(current);
  return turns;
};

const _payload = (event: ApiSessionEvent): Record<string, unknown> =>
  (event.payload ?? {}) as Record<string, unknown>;

const _str = (value: unknown): string => (typeof value === "string" ? value : "");

/**
 * Locator for a structured PlanMode deliverable, lifted off the terminal
 * `loop_completed` event's open payload (`payload.plan`, written by the server's
 * plan-record synthesizer). Present only on PlanMode sessions; chat sessions
 * return null. Drives the Deliverables panel, which fetches the full plan via
 * `GET /projects/{projectId}/experiments/{experimentId}/plans/{runId}`.
 */
export interface PlanRef {
  runId: string;
  projectId: string;
  experimentId: string;
  title: string;
  stepCount: number;
  hasWorkflow: boolean;
}

/**
 * Walk events backward for the terminal `loop_completed` carrying a `plan`
 * locator. Returns null unless all three ids are present (a partial locator
 * can't address a plan, so the panel falls back to the chat-artifact view).
 */
export const derivePlanRef = (events: ApiSessionEvent[]): PlanRef | null => {
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    if (ev.type !== "loop_completed") continue;
    const plan = (_payload(ev).plan ?? null) as Record<string, unknown> | null;
    if (!plan || typeof plan !== "object") continue;
    const runId = _str(plan.run_id);
    const projectId = _str(plan.project_id);
    const experimentId = _str(plan.experiment_id);
    if (!runId || !projectId || !experimentId) return null;
    return {
      runId,
      projectId,
      experimentId,
      title: _str(plan.title),
      stepCount: typeof plan.step_count === "number" ? plan.step_count : 0,
      hasWorkflow: plan.has_workflow === true,
    };
  }
  return null;
};

/**
 * Gather every inline artifact (plot/table/text) a chat session emitted via
 * `tool_call_completed` payloads, in stream order. Used by the Deliverables
 * panel for non-plan sessions; plan sessions surface structured deliverables
 * through {@link derivePlanRef} instead.
 */
export const collectArtifacts = (events: ApiSessionEvent[]): Record<string, unknown>[] => {
  const out: Record<string, unknown>[] = [];
  for (const ev of events) {
    if (ev.type !== "tool_call_completed") continue;
    const p = _payload(ev);
    const result = (p.result as Record<string, unknown> | undefined) ?? p;
    const artifacts = Array.isArray(result.artifacts) ? result.artifacts : [];
    for (const a of artifacts) {
      if (a && typeof a === "object") out.push(a as Record<string, unknown>);
    }
  }
  return out;
};

/**
 * The set of PlanMode artifact kinds a session has completed, read off the
 * synthesized stage steps (each `tool_call_completed` carries the produced kind
 * at `payload.result.artifact`). Drives the vertical progress rail's per-stage
 * state. Empty for chat sessions.
 */
export const completedStageKinds = (events: ApiSessionEvent[]): Set<string> => {
  const kinds = new Set<string>();
  for (const ev of events) {
    if (ev.type !== "tool_call_completed") continue;
    const result = (_payload(ev).result ?? {}) as Record<string, unknown>;
    const kind = _str(result.artifact);
    if (kind) kinds.add(kind);
  }
  return kinds;
};

/** One tool call's live state, folded from its started/completed delta pair. */
export interface ToolCallState {
  id: string;
  toolName: string;
  argsSummary: string;
  status: "started" | "completed";
  ok: boolean | null;
  resultSummary: string | null;
  /** ISO timestamp of the started event ("" when unknown). */
  startedTs: string;
  /** ISO timestamp of the completed event; null while still running. */
  completedTs: string | null;
}

/** Wall-clock seconds between a call's started/completed pair, if both known. */
export const toolCallDurationSeconds = (call: ToolCallState): number | null => {
  if (!call.startedTs || !call.completedTs) return null;
  const started = new Date(call.startedTs).getTime();
  const completed = new Date(call.completedTs).getTime();
  if (Number.isNaN(started) || Number.isNaN(completed) || completed < started) return null;
  return (completed - started) / 1000;
};

/** Render-ready streamed state for one turn (token answer, reasoning, tools). */
export interface StreamedTurn {
  /** Concatenated `token_delta` text — the answer as it streams. */
  answer: string;
  /** Concatenated `thinking_delta` text — kept strictly separate from `answer`. */
  thinking: string;
  /** One entry per `tool_call_started`, upgraded in place on completion. */
  toolCalls: ToolCallState[];
}

/**
 * Fold a turn's events into render-ready streamed state.
 *
 * Pure (no React, no input mutation): consecutive `token_delta` text
 * concatenates into `answer`; `thinking_delta` text into a separate `thinking`
 * (never leaking into `answer`); each `tool_call_started` pushes a `started`
 * tool entry, upgraded in place to `completed` (with `ok`/`resultSummary`) by
 * the matching `tool_call_completed` — matched FIFO by `tool_name`, falling
 * back to the earliest still-`started` entry. When no `token_delta` streamed,
 * `answer` falls back to a trailing `loop_completed.text`.
 */
export const foldStreamedTurn = (events: ApiSessionEvent[]): StreamedTurn => {
  let answer = "";
  let thinking = "";
  let fallback = "";
  const toolCalls: ToolCallState[] = [];

  for (const event of events) {
    const p = _payload(event);
    switch (event.type) {
      case "token_delta":
        answer += _str(p.text);
        break;
      case "thinking_delta":
        thinking += _str(p.text);
        break;
      case "tool_call_started":
        toolCalls.push({
          id: `${_str(p.tool_name) || "tool"}-${toolCalls.length}`,
          toolName: _str(p.tool_name),
          argsSummary: _str(p.args_summary),
          status: "started",
          ok: null,
          resultSummary: null,
          startedTs: event.ts,
          completedTs: null,
        });
        break;
      case "tool_call_completed": {
        const name = _str(p.tool_name);
        let idx = toolCalls.findIndex((t) => t.status === "started" && t.toolName === name);
        if (idx < 0) idx = toolCalls.findIndex((t) => t.status === "started");
        if (idx >= 0) {
          toolCalls[idx] = {
            ...toolCalls[idx],
            status: "completed",
            ok: typeof p.ok === "boolean" ? p.ok : true,
            resultSummary: typeof p.result_summary === "string" ? p.result_summary : null,
            completedTs: event.ts,
          };
        }
        break;
      }
      case "loop_completed":
        fallback = _str(p.text);
        break;
      default:
        break;
    }
  }

  return { answer: answer || fallback, thinking, toolCalls };
};

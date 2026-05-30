import {
  Bot,
  CheckCircle2,
  CircleUser,
  FileText,
  HelpCircle,
  Minimize2,
  ShieldAlert,
  Sparkles,
  Terminal,
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
  mode_started: { icon: Sparkles, label: "Started", colorClass: "text-violet-400" },
  mode_completed: { icon: CheckCircle2, label: "Completed", colorClass: "text-emerald-500" },
  stage_started: { icon: CircleUser, label: "Stage started", colorClass: "text-muted-foreground" },
  stage_completed: {
    icon: CheckCircle2,
    label: "Stage completed",
    colorClass: "text-muted-foreground",
  },
  plan_emitted: { icon: Sparkles, label: "Plan created", colorClass: "text-violet-500" },
  approval_requested: {
    icon: ShieldAlert,
    label: "Approval needed",
    colorClass: "text-orange-500",
  },
  approval_decided: {
    icon: CheckCircle2,
    label: "Approval decided",
    colorClass: "text-violet-400",
  },
  tool_call_started: { icon: Terminal, label: "Tool call", colorClass: "text-blue-500" },
  tool_call_completed: { icon: CheckCircle2, label: "Tool result", colorClass: "text-green-600" },
  artifact_written: {
    icon: FileText,
    label: "Artifact written",
    colorClass: "text-muted-foreground",
  },
  preflight_failed: { icon: XCircle, label: "Preflight failed", colorClass: "text-red-500" },
  repair_proposed: { icon: Sparkles, label: "Repair proposed", colorClass: "text-amber-500" },
  clarification_required: { icon: HelpCircle, label: "Question", colorClass: "text-fuchsia-500" },
  compaction_performed: {
    icon: Minimize2,
    label: "Compaction",
    colorClass: "text-muted-foreground",
  },
  error: { icon: XCircle, label: "Error", colorClass: "text-red-500" },
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
 * later `mode_completed` / `approval_decided` supersedes it.
 *
 * Largely inactive until the clarification/approval path fires agent-side.
 */
export const derivePendingUserRequest = (events: ApiSessionEvent[]): PendingUserRequest | null => {
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    if (ev.type === "mode_completed" || ev.type === "approval_decided") return null;
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
}

const isResultEvent = (event: ApiSessionEvent): boolean =>
  event.type === "mode_completed" ||
  // plan_emitted IS the agent's answer for a plan-mode turn — the user reviews
  // and approves it as the headline; a later mode_completed overrides it.
  event.type === "plan_emitted";

const eventKey = (event: ApiSessionEvent, fallback: number): string =>
  `${event.type}-${event.ts}-${fallback}`;

/**
 * Group events into conversational turns. Each turn begins with a
 * `mode_started` event (carrying the turn's `user_input`): the first one is
 * absorbed into the implicit goal turn, and every subsequent `mode_started`
 * opens a new turn. A turn closes on `mode_completed` / `plan_emitted`.
 *
 * Intermediate events (tool calls, …) are surfaced as `steps` so the UI can
 * collapse them; a `mode_started` boundary is not itself a step row.
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
  };
  let sawFirstModeStarted = false;

  events.forEach((event, idx) => {
    if (event.type === "mode_started") {
      if (!sawFirstModeStarted) {
        // The first mode_started opens the implicit goal turn — absorb it.
        sawFirstModeStarted = true;
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

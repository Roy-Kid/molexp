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

export interface ExperimentCatalogEntry {
  project_id: string;
  experiment_id: string;
  label: string;
}

export interface PendingUserRequest {
  requestId: string;
  prompt: string | null;
  /** When set, the conversation bubble shows a structured form. */
  contextKind?: string | null;
  catalog?: ExperimentCatalogEntry[];
  allowCreate?: boolean;
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
  // Plan/gateway LLM call projected into the agent-task session (cache — pruned).
  llm_call: { icon: Bot, label: "LLM call", colorClass: "text-info" },
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
    // Match server snapshot wire shape (`_event_to_wire`): type=kind, ts=timestamp,
    // payload = model_dump without `timestamp`. Keeping the full frame (incl.
    // timestamp) as payload makes SSE rows fail equality checks against
    // getSession snapshots and doubles every event — most visibly two
    // loop_started rows → Task + You bubbles for the same prompt.
    const { timestamp, ...payload } = data;
    return {
      type: data.kind,
      ts: typeof timestamp === "string" ? timestamp : "",
      payload,
    };
  }
  // Already a {type, ts, payload} envelope (e.g. a snapshot event echoed back).
  return {
    type: typeof data.type === "string" ? data.type : "",
    ts: typeof data.ts === "string" ? data.ts : "",
    payload: (data.payload as Record<string, unknown>) ?? {},
  };
};

/** Structural equality for stream dedupe (order-stable JSON). */
export const sameSessionEvent = (a: ApiSessionEvent, b: ApiSessionEvent): boolean =>
  a.type === b.type &&
  a.ts === b.ts &&
  JSON.stringify(a.payload ?? {}) === JSON.stringify(b.payload ?? {});

const _DELTA_TYPES = new Set(["thinking_delta", "token_delta"]);

/**
 * Merge consecutive thinking/token deltas into one event each.
 *
 * The agent stream often emits one event per token (1000+ rows). Keeping
 * them raw freezes the UI (re-render + layout thrash) and restarts CSS
 * spinners on every poll/SSE tick. Display and append paths both use this
 * so the transcript stays O(tools + turns), not O(tokens).
 */
export const coalesceStreamEvents = (events: ApiSessionEvent[]): ApiSessionEvent[] => {
  if (events.length < 2) return events;
  const out: ApiSessionEvent[] = [];
  for (const ev of events) {
    if (!_DELTA_TYPES.has(ev.type)) {
      out.push(ev);
      continue;
    }
    const last = out[out.length - 1];
    if (last && last.type === ev.type) {
      const prev = (last.payload ?? {}) as Record<string, unknown>;
      const next = (ev.payload ?? {}) as Record<string, unknown>;
      const prevText = typeof prev.text === "string" ? prev.text : "";
      const nextText = typeof next.text === "string" ? next.text : "";
      out[out.length - 1] = {
        ...last,
        ts: ev.ts || last.ts,
        payload: { ...prev, ...next, text: prevText + nextText },
      };
      continue;
    }
    out.push(ev);
  }
  return out;
};

/** Append one live frame, coalescing into the previous delta when kinds match. */
export const appendCoalescedEvent = (
  prev: ApiSessionEvent[],
  event: ApiSessionEvent,
): ApiSessionEvent[] => {
  if (prev.some((e) => sameSessionEvent(e, event))) return prev;
  if (!_DELTA_TYPES.has(event.type) || prev.length === 0) return [...prev, event];
  const last = prev[prev.length - 1];
  if (!last || last.type !== event.type) return [...prev, event];
  const prevPayload = (last.payload ?? {}) as Record<string, unknown>;
  const nextPayload = (event.payload ?? {}) as Record<string, unknown>;
  const prevText = typeof prevPayload.text === "string" ? prevPayload.text : "";
  const nextText = typeof nextPayload.text === "string" ? nextPayload.text : "";
  const merged: ApiSessionEvent = {
    ...last,
    ts: event.ts || last.ts,
    payload: { ...prevPayload, ...nextPayload, text: prevText + nextText },
  };
  return [...prev.slice(0, -1), merged];
};

/**
 * Walk an event log backwards to detect whether the agent is currently
 * waiting on the user. In the snake_case vocabulary the only such event is
 * `clarification_required` (a PlanMode prompt); it carries `questions` rather
 * than a `request_id`, so a `gate`/synthetic id is used.
 *
 * The question itself is rendered as the turn answer in the transcript.
 * This helper only drives composer routing (reply vs stop). Cleared once a
 * later terminal or follow-up boundary supersedes it
 * (`loop_completed` / `loop_started` / `plan_emitted` / `approval_decided`).
 */
export const derivePendingUserRequest = (events: ApiSessionEvent[]): PendingUserRequest | null => {
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    if (
      ev.type === "loop_completed" ||
      ev.type === "loop_started" ||
      ev.type === "plan_emitted" ||
      ev.type === "approval_decided"
    ) {
      return null;
    }
    if (ev.type === "clarification_required") {
      const payload = (ev.payload ?? {}) as Record<string, unknown>;
      const rid =
        typeof payload.request_id === "string"
          ? payload.request_id
          : typeof payload.gate === "string"
            ? payload.gate
            : "clarification";
      const catalogRaw = Array.isArray(payload.catalog) ? payload.catalog : [];
      const catalog: ExperimentCatalogEntry[] = catalogRaw
        .filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object")
        .map((row) => ({
          project_id: String(row.project_id ?? ""),
          experiment_id: String(row.experiment_id ?? ""),
          label: String(row.label ?? `${row.project_id ?? ""} / ${row.experiment_id ?? ""}`),
        }));
      return {
        requestId: rid,
        prompt: typeof payload.questions === "string" ? payload.questions : null,
        contextKind: typeof payload.context_kind === "string" ? payload.context_kind : null,
        catalog,
        allowCreate: payload.allow_create !== false,
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
  // Terminal plan failure (message + detail) — treat as the turn answer so the
  // chat never ends with a green "plan ready" when status is failed.
  event.type === "error" ||
  // Durable park (approval / should_stop suspend) — ends the turn without a
  // completion; without this the turn stays inProgress forever and the UI
  // freezes in the "streaming" treatment.
  event.type === "loop_suspended" ||
  // plan_emitted IS the agent's answer for a plan-mode turn — the user reviews
  // and approves it as the headline; a later loop_completed overrides it.
  event.type === "plan_emitted" ||
  // Clarifying questions (e.g. which project/experiment) are the turn's
  // conversational answer — not a composer banner.
  event.type === "clarification_required";

/** Progress breadcrumbs that should surface in the open turn (not only in steps). */
export const isProgressEvent = (event: ApiSessionEvent): boolean =>
  event.type === "stage_started" || event.type === "stage_completed";

const eventKey = (event: ApiSessionEvent, fallback: number): string =>
  `${event.type}-${event.ts}-${fallback}`;

/**
 * Group events into conversational turns. Each turn begins with a
 * `loop_started` event (carrying the turn's `user_input`): the first one is
 * absorbed into the implicit goal turn, and every subsequent `loop_started`
 * opens a new turn. A turn closes on `loop_completed` / `loop_suspended` /
 * `plan_emitted` / `error`.
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
      const payload = (event.payload ?? {}) as Record<string, unknown>;
      const question = typeof payload.user_input === "string" ? payload.user_input : "";
      if (!sawFirstLoopStarted) {
        sawFirstLoopStarted = true;
        // Create-plan can emit clarification_required *before* any loop_started.
        // A later scope reply's loop_started must open a NEW live turn — if we
        // "absorb" into the already-closed goal bubble, stages/thinking vanish
        // under a finished clarification answer (UI looks frozen).
        if (current.result || !current.inProgress) {
          current.inProgress = false;
          turns.push(current);
          current = {
            key: eventKey(event, idx),
            question: question || current.question,
            source: "user",
            result: null,
            steps: [],
            inProgress: true,
            startedTs: event.ts,
          };
          return;
        }
        // Normal case: first loop_started opens the implicit goal turn.
        current.startedTs = event.ts;
        if (question && !current.question) current.question = question;
        return;
      }
      // Duplicate stream/snapshot loop_started for the *same* open turn (same
      // prompt, no terminal result yet) must not open a second You bubble.
      if (current.inProgress && !current.result && question && question === current.question) {
        if (!current.startedTs) current.startedTs = event.ts;
        return;
      }
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
 * Locator for a structured PlanMode deliverable. Prefer event payloads
 * (`loop_completed.plan` or `plan_emitted`); session metadata is a fallback so
 * the Deliverables panel opens at the review gate, not only after materialize.
 * Fetches via `GET /projects/{projectId}/experiments/{experimentId}/plans/{runId}`.
 */
export interface PlanRef {
  runId: string;
  projectId: string;
  experimentId: string;
  title: string;
  stepCount: number;
  hasWorkflow: boolean;
}

export type PlanRefFallback = {
  runId?: string | null;
  projectId?: string | null;
  experimentId?: string | null;
  title?: string | null;
};

const _planRefFromPayload = (
  payload: Record<string, unknown>,
  fallback?: PlanRefFallback | null,
): PlanRef | null => {
  const nested = (payload.plan ?? null) as Record<string, unknown> | null;
  const plan = nested && typeof nested === "object" ? nested : payload;
  const runId =
    _str(plan.run_id) ||
    _str(plan.plan_id) ||
    _str(payload.run_id) ||
    _str(payload.plan_id) ||
    _str(fallback?.runId);
  const projectId = _str(plan.project_id) || _str(payload.project_id) || _str(fallback?.projectId);
  const experimentId =
    _str(plan.experiment_id) || _str(payload.experiment_id) || _str(fallback?.experimentId);
  if (!runId || !projectId || !experimentId) return null;
  return {
    runId,
    projectId,
    experimentId,
    title: _str(plan.title) || _str(payload.title) || _str(fallback?.title),
    stepCount:
      typeof plan.step_count === "number"
        ? plan.step_count
        : typeof payload.step_count === "number"
          ? payload.step_count
          : 0,
    hasWorkflow: plan.has_workflow === true || payload.has_workflow === true,
  };
};

/**
 * Walk events backward for a plan locator. Sources (newest first):
 * 1. `loop_completed` / `plan_emitted` payloads
 * 2. optional session metadata fallback (runId / projectId / experimentId)
 */
export const derivePlanRef = (
  events: ApiSessionEvent[],
  fallback?: PlanRefFallback | null,
): PlanRef | null => {
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    if (ev.type !== "loop_completed" && ev.type !== "plan_emitted") continue;
    const ref = _planRefFromPayload(_payload(ev), fallback);
    if (ref) return ref;
  }
  if (fallback?.runId && fallback.projectId && fallback.experimentId) {
    return {
      runId: String(fallback.runId),
      projectId: String(fallback.projectId),
      experimentId: String(fallback.experimentId),
      title: _str(fallback.title),
      stepCount: 0,
      hasWorkflow: false,
    };
  }
  return null;
};

/**
 * Gather every inline artifact (plot/table/text) a task emitted via
 * `tool_call_completed` payloads, in stream order. Used by the Deliverables
 * panel when no Plan turn is selected; Plan turns surface structured deliverables
 * through {@link derivePlanRef} instead.
 */
export const collectArtifacts = (events: ApiSessionEvent[]): Record<string, unknown>[] => {
  const out: Record<string, unknown>[] = [];
  for (const ev of events) {
    if (ev.type !== "tool_call_completed") continue;
    const p = _payload(ev);
    const result = (p.result as Record<string, unknown> | undefined) ?? {};
    const raw = Array.isArray(p.artifacts)
      ? p.artifacts
      : Array.isArray(result.artifacts)
        ? result.artifacts
        : [];
    for (const a of raw) {
      if (a && typeof a === "object") out.push(a as Record<string, unknown>);
    }
  }
  return out;
};

/**
 * The set of PlanMode artifact kinds a task has completed, read off the
 * synthesized stage steps (each `tool_call_completed` carries the produced kind
 * at `payload.result.artifact`). Drives the vertical progress rail's per-stage
 * state. Empty when the task has no completed Plan turn.
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

/**
 * Which rows of an event list should render their timestamp. Backend
 * bookkeeping stamps every step of a batch with the same ledger time, so a
 * run of identical adjacent timestamps shows only its first — later rows
 * keep the value on their hover tooltip only.
 */
export const visibleTimestampFlags = (events: readonly Pick<ApiSessionEvent, "ts">[]): boolean[] =>
  events.map((event, idx) => idx === 0 || events[idx - 1]?.ts !== event.ts);

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

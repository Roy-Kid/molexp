import {
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleUser,
  ClipboardList,
  Loader2,
  Target,
  Wrench,
} from "lucide-react";
import { type JSX, useMemo, useState } from "react";
import { StatusBadge } from "@/app/components/entity";
import {
  type ConversationTurn,
  EVENT_META,
  foldStreamedTurn,
  turnDurationSeconds,
} from "@/app/renderers/agentEvents";
import type { ApiSessionEvent } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { MarkdownContent } from "@/components/ui/markdown";
import { ThinkingBlock } from "@/components/ui/thinking-block";
import { ToolCallRow } from "@/components/ui/tool-call-row";
import { formatDurationCompact } from "@/lib/format-time";
import { ToolResultArtifacts } from "./artifacts";

const formatTs = (ts: string): string => {
  try {
    return new Date(ts).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return ts;
  }
};

const formatTokens = (count: number): string => {
  if (count < 1_000) return String(count);
  if (count < 1_000_000) return `${(count / 1_000).toFixed(1)}k`;
  return `${(count / 1_000_000).toFixed(1)}M`;
};

// ---------------------------------------------------------------------------
// Event row (one raw event inside the "internal steps" disclosure)
// ---------------------------------------------------------------------------

const EventRow = ({ event }: { event: ApiSessionEvent }): JSX.Element => {
  const [expanded, setExpanded] = useState(false);
  const meta = EVENT_META[event.type] ?? {
    icon: Bot,
    label: event.type,
    colorClass: "text-muted-foreground",
  };
  const Icon = meta.icon;
  const payload: Record<string, unknown> = event.payload ?? {};
  const hasDetail = Object.keys(payload).length > 0;

  return (
    <div className="group flex gap-3 py-2">
      <div className={`mt-0.5 flex-none ${meta.colorClass}`}>
        <Icon className="h-4 w-4" />
      </div>
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{meta.label}</span>
          {(event.type === "tool_call_started" || event.type === "tool_call_completed") &&
            Boolean(payload.tool_name) && (
              <Badge variant="secondary" className="h-4 px-1 font-mono text-[10px]">
                {String(payload.tool_name)}
              </Badge>
            )}
          <span className="ml-auto text-[10px] tabular-nums text-muted-foreground">
            {formatTs(event.ts)}
          </span>
          {hasDetail && (
            <button
              type="button"
              className="text-muted-foreground transition-colors hover:text-foreground"
              onClick={() => setExpanded((v) => !v)}
              aria-label={expanded ? "Collapse event detail" : "Expand event detail"}
            >
              {expanded ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              )}
            </button>
          )}
        </div>

        {event.type === "clarification_required" && Boolean(payload.questions) && (
          <div className="rounded-md border border-warning/25 bg-warning-soft px-3 py-2">
            <p className="text-xs text-warning-foreground">{String(payload.questions)}</p>
          </div>
        )}

        {expanded && hasDetail && (
          <pre className="overflow-x-auto rounded-md bg-muted/60 px-3 py-2 font-mono text-[11px] text-muted-foreground">
            {JSON.stringify(payload, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Answer (the agent's reply for one turn)
// ---------------------------------------------------------------------------

const TurnAnswer = ({
  result,
  inProgress,
}: {
  result: ApiSessionEvent | null;
  inProgress: boolean;
}): JSX.Element => {
  if (!result) {
    if (inProgress) {
      return (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-info" />
          <span>Working…</span>
        </div>
      );
    }
    return (
      <p className="text-sm italic text-muted-foreground">
        No final answer recorded for this turn.
      </p>
    );
  }

  const payload = (result.payload ?? {}) as Record<string, unknown>;

  if (result.type === "loop_completed") {
    const summary = typeof payload.text === "string" ? payload.text : "";
    return summary ? (
      <MarkdownContent text={summary} />
    ) : (
      <p className="text-sm italic text-muted-foreground">Session ended without a summary.</p>
    );
  }

  if (result.type === "plan_emitted") {
    const planId = typeof payload.plan_id === "string" ? payload.plan_id : "plan";
    const stepCount = typeof payload.step_count === "number" ? payload.step_count : 0;
    return (
      <div className="flex items-center gap-2 rounded-md border border-info/25 bg-info-soft px-3 py-2 text-xs text-info-foreground">
        <ClipboardList className="h-3.5 w-3.5 flex-none" />
        <span className="font-mono font-medium">{planId}</span>
        <span>
          · {stepCount} {stepCount === 1 ? "step" : "steps"} — see the deliverables panel
        </span>
      </div>
    );
  }

  if (result.type === "tool_call_completed") {
    return <ToolResultArtifacts payload={payload} />;
  }

  return (
    <pre className="overflow-x-auto rounded-md bg-muted/40 px-3 py-2 text-xs">
      {JSON.stringify(payload, null, 2)}
    </pre>
  );
};

/** Dim provenance footer: outcome · duration · token usage (CLI parity). */
const TurnFooter = ({ turn }: { turn: ConversationTurn }): JSX.Element | null => {
  if (!turn.result) return null;
  const payload = (turn.result.payload ?? {}) as Record<string, unknown>;
  const resultDump = (payload.result as Record<string, unknown> | undefined) ?? {};
  const usage = (resultDump.usage as Record<string, unknown> | undefined) ?? {};
  const tokensIn = typeof usage.input_tokens === "number" ? usage.input_tokens : 0;
  const tokensOut = typeof usage.output_tokens === "number" ? usage.output_tokens : 0;
  const duration = formatDurationCompact(turnDurationSeconds(turn));
  const isPlan = turn.result.type === "plan_emitted";

  return (
    <div className="flex items-center gap-1.5 border-t border-border/50 pt-2 text-[11px] text-muted-foreground">
      <CheckCircle2 className={`h-3 w-3 ${isPlan ? "text-info" : "text-success"}`} />
      <span>{isPlan ? "plan ready" : "done"}</span>
      {duration && <span className="tabular-nums">· {duration}</span>}
      {(tokensIn > 0 || tokensOut > 0) && (
        <span className="tabular-nums">
          · ↑{formatTokens(tokensIn)} ↓{formatTokens(tokensOut)} tok
        </span>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Internal steps disclosure — reasoning's sibling: the agent's tool calls and
// lifecycle events, demoted and collapsed (auto-open while still streaming).
// ---------------------------------------------------------------------------

const InternalSteps = ({ turn }: { turn: ConversationTurn }): JSX.Element | null => {
  const streamed = useMemo(
    () => foldStreamedTurn(turn.result ? [...turn.steps, turn.result] : turn.steps),
    [turn.steps, turn.result],
  );
  // Raw step events worth showing in the detail view: skip the deltas (folded
  // into thinking/answer). A `tool_call_completed` is folded into a ToolCallRow
  // only when its `tool_call_started` is also in this turn — so we keep lone
  // completions (e.g. PlanMode's synthesized stage steps, which have no started
  // frame) as rows rather than dropping them.
  const hasStarted = turn.steps.some((e) => e.type === "tool_call_started");
  const detailSteps = turn.steps.filter(
    (e) =>
      e.type !== "token_delta" &&
      e.type !== "thinking_delta" &&
      e.type !== "tool_call_started" &&
      !(hasStarted && e.type === "tool_call_completed"),
  );
  const count = streamed.toolCalls.length + detailSteps.length;
  const [open, setOpen] = useState(false);
  if (count === 0) return null;
  // While the turn streams, show live progress; once done, collapse by default.
  const expanded = turn.inProgress || open;

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 rounded-sm px-1 py-0.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
        aria-expanded={expanded}
        disabled={turn.inProgress}
      >
        {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <Wrench className="h-3 w-3" />
        <span>
          {expanded ? "Internal steps" : `${count} internal step${count === 1 ? "" : "s"}`}
        </span>
      </button>
      {expanded && (
        <div className="mt-1 space-y-0.5 rounded-md border border-border/50 bg-muted/20 px-2 py-1">
          {streamed.toolCalls.map((call) => (
            <ToolCallRow key={call.id} call={call} />
          ))}
          {detailSteps.map((event, idx) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: the event log is append-only, so position is a stable identity even when two events share a timestamp
            <EventRow key={`${turn.key}-step-${idx}-${event.type}`} event={event} />
          ))}
        </div>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// One conversational turn — the user's prompt + the agent's reply, rendered as
// two distinct, role-separated blocks (prompt on the right, reply on the left).
// ---------------------------------------------------------------------------

export const ConversationTurnView = ({ turn }: { turn: ConversationTurn }): JSX.Element => {
  const streamed = useMemo(
    () => foldStreamedTurn(turn.result ? [...turn.steps, turn.result] : turn.steps),
    [turn.steps, turn.result],
  );
  const PromptIcon = turn.source === "goal" ? Target : CircleUser;

  return (
    <div className="space-y-2.5">
      {/* User / goal prompt — right-aligned accent bubble */}
      <div className="flex justify-end">
        <div className="max-w-[88%] rounded-2xl rounded-br-sm border border-primary/20 bg-primary/5 px-3.5 py-2">
          <div className="mb-0.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-primary/80">
            <PromptIcon className="h-3 w-3" />
            {turn.source === "goal" ? "Task" : "You"}
          </div>
          <p className="whitespace-pre-wrap text-sm leading-snug text-foreground [overflow-wrap:anywhere]">
            {turn.question || <span className="italic text-muted-foreground">(no prompt)</span>}
          </p>
        </div>
      </div>

      {/* Assistant reply — bot avatar + bubble with reasoning/steps demoted */}
      <div className="flex gap-2.5">
        <div className="mt-0.5 flex h-6 w-6 flex-none items-center justify-center rounded-full border border-border/60 bg-card">
          {turn.inProgress ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-info" />
          ) : (
            <Bot className="h-3.5 w-3.5 text-muted-foreground" />
          )}
        </div>
        <div className="min-w-0 flex-1 space-y-2 rounded-2xl rounded-tl-sm border border-border/70 bg-card px-3.5 py-2.5 shadow-xs">
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            <span>Agent</span>
            {turn.inProgress && <StatusBadge status="running" size="sm" dot />}
          </div>

          {streamed.thinking && (
            <ThinkingBlock thinking={streamed.thinking} streaming={turn.inProgress} />
          )}

          <InternalSteps turn={turn} />

          {turn.inProgress && !turn.result && streamed.answer ? (
            // Token-by-token streaming answer before a terminal result lands.
            <MarkdownContent text={streamed.answer} />
          ) : (
            <TurnAnswer result={turn.result} inProgress={turn.inProgress} />
          )}

          <TurnFooter turn={turn} />
        </div>
      </div>
    </div>
  );
};

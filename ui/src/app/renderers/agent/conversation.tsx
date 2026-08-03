import {
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleUser,
  ClipboardList,
  Target,
  Wrench,
} from "lucide-react";
import { type JSX, useMemo, useState } from "react";
import {
  type ConversationTurn,
  EVENT_META,
  type ExperimentCatalogEntry,
  foldStreamedTurn,
  isProgressEvent,
  turnDurationSeconds,
  visibleTimestampFlags,
} from "@/app/renderers/agentEvents";
import type { ApiSessionEvent } from "@/app/types";
import { MarkdownContent } from "@/components/ui/markdown";
import { ProgressSpinner } from "@/components/ui/progress-spinner";
import { ThinkingBlock } from "@/components/ui/thinking-block";
import { ToolCallRow } from "@/components/ui/tool-call-row";
import { WorkbenchTag } from "@/components/workbench";
import { ChatAnswerBody } from "@/lib/chat-answer";
import { linkifyEntityTokens } from "@/lib/entity-linkify";
import { formatDurationCompact } from "@/lib/format-time";
import { cn } from "@/lib/utils";
import { ToolResultArtifacts, TurnEmbedArtifacts } from "./artifacts";
import { ExperimentScopeForm } from "./ExperimentScopeForm";
import { LandDecisionBar, looksLikeLandOffer } from "./LandDecisionBar";
import { PlanDecisionBar } from "./PlanDecisionBar";
import { PlanDocumentCard } from "./PlanDocumentCard";

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

const EventRow = ({
  event,
  showTimestamp = true,
}: {
  event: ApiSessionEvent;
  /** Suppress the visible timestamp (hover tooltip keeps it) when it merely repeats the previous row's. */
  showTimestamp?: boolean;
}): JSX.Element => {
  const [expanded, setExpanded] = useState(false);
  const meta = EVENT_META[event.type] ?? {
    icon: Bot,
    label: event.type,
    colorClass: "text-muted-foreground",
  };
  const Icon = meta.icon;
  const payload: Record<string, unknown> = event.payload ?? {};
  const hasDetail = Object.keys(payload).length > 0;
  // Plan breadcrumbs often share the generic EVENT_META label ("Stage started");
  // surface the real message/stage so rows are distinguishable.
  const stageLabel = (() => {
    if (event.type !== "stage_started" && event.type !== "stage_completed") return null;
    if (typeof payload.message === "string" && payload.message.trim())
      return payload.message.trim();
    if (typeof payload.stage === "string" && payload.stage.trim()) {
      return event.type === "stage_completed"
        ? `Stage ${payload.stage} done`
        : `Stage ${payload.stage}`;
    }
    return null;
  })();
  const rowLabel = stageLabel ?? meta.label;

  return (
    <div className="group flex gap-3 py-2">
      <div className={`mt-1 flex-none ${meta.colorClass}`}>
        <Icon className="h-4 w-4" />
      </div>
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{rowLabel}</span>
          {(event.type === "tool_call_started" || event.type === "tool_call_completed") &&
            Boolean(payload.tool_name) && (
              <WorkbenchTag className="h-4 px-1 font-mono text-micro">
                {String(payload.tool_name)}
              </WorkbenchTag>
            )}
          <span className="ml-auto text-micro tabular-nums text-muted-foreground" title={event.ts}>
            {showTimestamp ? formatTs(event.ts) : null}
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

        {event.type === "llm_call" && (
          <div className="space-y-1 text-xs text-muted-foreground">
            {Boolean(payload.agent_name) && (
              <p>
                <span className="font-medium text-foreground">agent</span>{" "}
                <span className="font-mono">{String(payload.agent_name)}</span>
                {Boolean(payload.model) && (
                  <span className="text-muted-foreground"> · {String(payload.model)}</span>
                )}
              </p>
            )}
            {!expanded && Boolean(payload.prompt_preview) && (
              <p className="line-clamp-2 font-mono text-micro text-muted-foreground/90">
                {String(payload.prompt_preview).slice(0, 200)}
              </p>
            )}
          </div>
        )}

        {expanded && event.type === "llm_call" && (
          <div className="space-y-2">
            {Boolean(payload.prompt_preview) && (
              <div>
                <p className="mb-1 text-micro font-semibold uppercase tracking-wide text-muted-foreground">
                  Prompt
                  {typeof payload.prompt_chars === "number"
                    ? ` · ${payload.prompt_chars} chars`
                    : ""}
                </p>
                <pre className="max-h-48 overflow-auto rounded-md bg-muted/60 px-3 py-2 font-mono text-micro text-muted-foreground whitespace-pre-wrap">
                  {String(payload.prompt_preview)}
                </pre>
              </div>
            )}
            {Boolean(payload.raw_preview) && (
              <div>
                <p className="mb-1 text-micro font-semibold uppercase tracking-wide text-muted-foreground">
                  Response
                  {typeof payload.raw_chars === "number" ? ` · ${payload.raw_chars} chars` : ""}
                </p>
                <pre className="max-h-48 overflow-auto rounded-md bg-muted/60 px-3 py-2 font-mono text-micro text-muted-foreground whitespace-pre-wrap">
                  {String(payload.raw_preview)}
                </pre>
              </div>
            )}
            {payload.cache === true && (
              <p className="text-micro text-muted-foreground/80">
                Cached session projection (pruned by age/size); audit originals live on the run
                artifacts.
              </p>
            )}
          </div>
        )}

        {expanded && hasDetail && event.type !== "llm_call" && (
          <pre className="overflow-x-auto rounded-md bg-muted/60 px-3 py-2 font-mono text-micro text-muted-foreground">
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
  linkIndex,
  interactiveClarification,
  onScopeSubmit,
  showLandDecision,
  onLandDecide,
  steps = [],
}: {
  result: ApiSessionEvent | null;
  inProgress: boolean;
  linkIndex?: Map<string, string>;
  /** True when this clarification is still the pending user request. */
  interactiveClarification?: boolean;
  onScopeSubmit?: (scope: string) => void | Promise<void>;
  /** Offer Yes/No archive after a successful chat turn. */
  showLandDecision?: boolean;
  onLandDecide?: (message: string) => void | Promise<void>;
  steps?: ApiSessionEvent[];
}): JSX.Element => {
  if (!result) {
    if (inProgress) {
      // Text-only — the turn avatar is the single spinner (no stacked loaders).
      return <p className="text-sm text-muted-foreground">Working…</p>;
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
    const failed = payload.failed === true;
    if (!summary) {
      return (
        <p className="text-sm italic text-muted-foreground">Session ended without a summary.</p>
      );
    }
    if (failed) {
      return (
        <div className="space-y-2">
          <div className="border-y border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            Plan failed
            {typeof payload.stage === "string" && payload.stage ? ` · stage ${payload.stage}` : ""}
          </div>
          <ChatAnswerBody
            text={summary}
            linkify={linkIndex ? (s) => linkifyEntityTokens(s, linkIndex) : undefined}
          />
        </div>
      );
    }
    const hadSuccessfulCodeRun = steps.some((ev) => {
      if (ev.type !== "tool_call_completed") return false;
      const p = (ev.payload ?? {}) as Record<string, unknown>;
      const name = String(p.tool_name ?? "");
      return name === "code_run" && p.ok !== false;
    });
    // Explicit archive/land phrasing, or a successful code_run on this turn.
    const landOffer =
      Boolean(showLandDecision) &&
      Boolean(onLandDecide) &&
      !failed &&
      (looksLikeLandOffer(summary) || hadSuccessfulCodeRun);
    return (
      <div className="space-y-3">
        <ChatAnswerBody
          text={summary}
          linkify={linkIndex ? (s) => linkifyEntityTokens(s, linkIndex) : undefined}
        />
        {landOffer && onLandDecide ? <LandDecisionBar onDecide={onLandDecide} /> : null}
      </div>
    );
  }

  if (result.type === "error") {
    const message =
      typeof payload.message === "string"
        ? payload.message
        : typeof payload.error === "string"
          ? payload.error
          : "Unknown error";
    const detail = typeof payload.detail === "string" ? payload.detail : "";
    const stage = typeof payload.stage === "string" ? payload.stage : "";
    return (
      <div className="space-y-2 border-y border-destructive/30 bg-destructive/10 px-3 py-2">
        <p className="text-sm font-medium text-destructive">
          {stage ? `Error at ${stage}` : "Error"}
        </p>
        <p className="whitespace-pre-wrap text-sm text-destructive [overflow-wrap:anywhere]">
          {message}
        </p>
        {detail ? (
          <pre className="max-h-64 overflow-auto rounded bg-background/60 px-2 py-2 font-mono text-micro text-muted-foreground whitespace-pre-wrap">
            {detail.slice(0, 4000)}
          </pre>
        ) : null}
      </div>
    );
  }

  if (result.type === "plan_emitted") {
    // The experiment plan book IS the agent answer (chat form), not an approval card.
    const bodyMd = typeof payload.body_md === "string" ? payload.body_md.trim() : "";
    const title =
      (typeof payload.title === "string" && payload.title.trim()) ||
      (typeof (payload.plan as { title?: string } | undefined)?.title === "string"
        ? String((payload.plan as { title: string }).title)
        : "Experiment Plan");
    const projectId =
      (typeof payload.project_id === "string" && payload.project_id) ||
      (typeof (payload.plan as { project_id?: string } | undefined)?.project_id === "string"
        ? String((payload.plan as { project_id: string }).project_id)
        : "");
    const experimentId =
      (typeof payload.experiment_id === "string" && payload.experiment_id) ||
      (typeof (payload.plan as { experiment_id?: string } | undefined)?.experiment_id === "string"
        ? String((payload.plan as { experiment_id: string }).experiment_id)
        : "");
    const runId =
      (typeof payload.run_id === "string" && payload.run_id) ||
      (typeof payload.plan_id === "string" && payload.plan_id) ||
      (typeof (payload.plan as { run_id?: string } | undefined)?.run_id === "string"
        ? String((payload.plan as { run_id: string }).run_id)
        : "");

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <ClipboardList className="h-3.5 w-3.5 flex-none text-info" />
          <span className="font-medium text-foreground">{title}</span>
        </div>
        {bodyMd ? (
          <div className="max-h-[36rem] overflow-auto rounded-md border border-border/50 bg-muted/20 px-3 py-3 text-sm leading-relaxed">
            <MarkdownContent text={bodyMd} />
          </div>
        ) : projectId && experimentId && runId ? (
          <PlanDocumentCard projectId={projectId} experimentId={experimentId} runId={runId} />
        ) : (
          <p className="text-sm text-muted-foreground">Plan document unavailable.</p>
        )}
      </div>
    );
  }

  if (result.type === "loop_suspended") {
    const reason =
      typeof payload.reason === "string" && payload.reason.trim()
        ? payload.reason
        : "Waiting for your approval.";
    return (
      <div className="flex items-center gap-2 border-y border-warning/25 bg-warning-soft px-3 py-2 text-xs text-warning-foreground">
        <ClipboardList className="h-3.5 w-3.5 flex-none" />
        <span>{reason} Open the Approvals inbox to continue.</span>
      </div>
    );
  }

  if (result.type === "clarification_required") {
    const questions =
      typeof payload.questions === "string" && payload.questions.trim()
        ? payload.questions
        : "I need a bit more context before continuing.";
    const contextKind = typeof payload.context_kind === "string" ? payload.context_kind : null;
    const catalogRaw = Array.isArray(payload.catalog) ? payload.catalog : [];
    const catalog: ExperimentCatalogEntry[] = catalogRaw
      .filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object")
      .map((row) => ({
        project_id: String(row.project_id ?? ""),
        experiment_id: String(row.experiment_id ?? ""),
        label: String(row.label ?? `${row.project_id ?? ""} / ${row.experiment_id ?? ""}`),
      }));

    if (
      interactiveClarification &&
      onScopeSubmit &&
      (contextKind === "experiment" || catalog.length > 0 || payload.allow_create === true)
    ) {
      return (
        <ExperimentScopeForm
          intro={questions.split("\n")[0] || questions}
          catalog={catalog}
          allowCreate={payload.allow_create !== false}
          onSubmit={onScopeSubmit}
        />
      );
    }
    return (
      <ChatAnswerBody
        text={questions}
        linkify={linkIndex ? (s) => linkifyEntityTokens(s, linkIndex) : undefined}
      />
    );
  }

  if (result.type === "stage_started" || result.type === "stage_completed") {
    const message =
      typeof payload.message === "string" && payload.message.trim()
        ? payload.message
        : typeof payload.stage === "string"
          ? `Stage ${payload.stage}`
          : result.type;
    return <p className="text-sm text-muted-foreground">{message}</p>;
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
  const isSuspended = turn.result.type === "loop_suspended";
  const isClarification = turn.result.type === "clarification_required";

  if (isClarification) {
    return null;
  }

  return (
    <div className="flex items-center gap-2 border-t border-border/50 pt-2 text-micro text-muted-foreground">
      <CheckCircle2 className={`h-3 w-3 ${isPlan || isSuspended ? "text-info" : "text-success"}`} />
      <span>{isSuspended ? "waiting for approval" : isPlan ? "ready for review" : "done"}</span>
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
// lifecycle events, demoted and collapsed by default (same pattern as
// ThinkingBlock). The user can expand/collapse at any time, including while
// the turn is still streaming.
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
  // stage_* are plan progress breadcrumbs (shown as the live "Working…" line),
  // not internal tool steps — listing them here produced three identical
  // "Stage started" rows with no spinning state.
  const detailSteps = turn.steps.filter(
    (e) =>
      e.type !== "token_delta" &&
      e.type !== "thinking_delta" &&
      e.type !== "tool_call_started" &&
      e.type !== "stage_started" &&
      e.type !== "stage_completed" &&
      !(hasStarted && e.type === "tool_call_completed"),
  );
  // Backend batches often stamp adjacent steps with one ledger time; show
  // the timestamp only where it changes (the rest keep a hover tooltip).
  // Position is a stable identity — the event log is append-only — so the
  // row key may safely encode it.
  const timestampVisible = visibleTimestampFlags(detailSteps);
  const detailRows = detailSteps.map((event, idx) => ({
    key: `${turn.key}-step-${idx}-${event.type}`,
    event,
    showTimestamp: timestampVisible[idx] ?? true,
  }));
  const count = streamed.toolCalls.length + detailSteps.length;
  const [open, setOpen] = useState(false);
  if (count === 0) return null;

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-sm px-1 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
        aria-expanded={open}
        aria-label={open ? "Collapse internal steps" : "Expand internal steps"}
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <Wrench className="h-3 w-3" />
        <span>
          {open
            ? "Internal steps"
            : `${count} step${count === 1 ? "" : "s"}${turn.inProgress ? "…" : ""}`}
        </span>
      </button>
      {open && (
        <div className="mt-1 space-y-1 border-t border-border/50 pt-1">
          {streamed.toolCalls.map((call) => (
            <ToolCallRow key={call.id} call={call} />
          ))}
          {detailRows.map((row) => (
            <EventRow key={row.key} event={row.event} showTimestamp={row.showTimestamp} />
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

export const ConversationTurnView = ({
  turn,
  linkIndex,
  interactiveClarification,
  onScopeSubmit,
  showLandDecision,
  onLandDecide,
  planDecisionTaskId,
  planDecisionRunId,
  onPlanDecided,
}: {
  turn: ConversationTurn;
  linkIndex?: Map<string, string>;
  interactiveClarification?: boolean;
  onScopeSubmit?: (scope: string) => void | Promise<void>;
  showLandDecision?: boolean;
  onLandDecide?: (message: string) => void | Promise<void>;
  /** When set on a plan_emitted turn, show slim Approve/Reject under the answer. */
  planDecisionTaskId?: string | null;
  /** Plan run id for matching the approvals inbox when task ids diverge. */
  planDecisionRunId?: string | null;
  onPlanDecided?: () => void;
}): JSX.Element => {
  const streamed = useMemo(
    () => foldStreamedTurn(turn.result ? [...turn.steps, turn.result] : turn.steps),
    [turn.steps, turn.result],
  );
  const PromptIcon = turn.source === "goal" ? Target : CircleUser;
  const isPendingScopeForm =
    Boolean(interactiveClarification) &&
    turn.result?.type === "clarification_required" &&
    !turn.inProgress;

  return (
    <div className="space-y-3">
      {/* User / goal prompt — right-aligned accent bubble */}
      <div className="flex justify-end">
        <div className="max-w-[88%] rounded-[var(--radius-panel)] rounded-br-sm border border-primary/20 bg-primary/5 px-4 py-2">
          <div className="mb-1 flex items-center gap-2 text-micro font-semibold uppercase tracking-wide text-primary/80">
            <PromptIcon className="h-3 w-3" />
            {turn.source === "goal" ? "Task" : "You"}
          </div>
          <p className="whitespace-pre-wrap text-sm leading-snug text-foreground [overflow-wrap:anywhere]">
            {turn.question || <span className="italic text-muted-foreground">(no prompt)</span>}
          </p>
        </div>
      </div>

      {/* Assistant reply — bot avatar + bubble with reasoning/steps demoted.
          In-progress = spinning avatar only (same mechanism as StatusBadge running). */}
      <div className="flex gap-3">
        <div
          className={cn(
            "mt-1 flex h-6 w-6 flex-none items-center justify-center rounded-full border bg-card",
            turn.inProgress
              ? "border-status-running/40 text-status-running"
              : "border-border/60 text-muted-foreground",
          )}
          role="img"
          aria-label={turn.inProgress ? "Agent running" : "Agent"}
        >
          {turn.inProgress ? (
            <ProgressSpinner className="text-status-running" label="Agent running" />
          ) : (
            <Bot className="h-3.5 w-3.5" aria-hidden />
          )}
        </div>
        <div className="min-w-0 flex-1 space-y-2 rounded-[var(--radius-panel)] rounded-tl-sm border border-border/70 bg-card px-4 py-3">
          <div className="flex items-center gap-2 text-micro font-semibold uppercase tracking-wide text-muted-foreground">
            <span>{turn.inProgress ? "Agent working" : "Agent"}</span>
          </div>

          {/* One primary live signal only — avatar already spins. Avoid stacking
              Thinking… + N steps… + Working… (looks frozen / overloaded). */}
          {streamed.thinking && (
            <ThinkingBlock thinking={streamed.thinking} streaming={turn.inProgress} />
          )}

          <InternalSteps turn={turn} />

          {/* embed_plot / embed_structure artifacts from tool_call_completed */}
          <TurnEmbedArtifacts events={turn.steps} />

          {turn.inProgress && !turn.result && streamed.answer ? (
            <MarkdownContent
              text={linkIndex ? linkifyEntityTokens(streamed.answer, linkIndex) : streamed.answer}
            />
          ) : turn.inProgress && !turn.result ? (
            (() => {
              const progress = [...turn.steps].reverse().find(isProgressEvent);
              const activeTool = streamed.toolCalls.find((c) => c.status === "started");
              // Prefer stage breadcrumb, else active tool name, else nothing
              // when ThinkingBlock already conveys "busy".
              if (progress) {
                return <TurnAnswer result={progress} inProgress={false} linkIndex={linkIndex} />;
              }
              if (activeTool) {
                return (
                  <p className="text-sm text-muted-foreground">
                    Running{" "}
                    <span className="font-mono text-xs text-foreground">{activeTool.toolName}</span>
                    …
                  </p>
                );
              }
              if (streamed.thinking) {
                // ThinkingBlock is the sole progress line — no extra Working….
                return null;
              }
              return <TurnAnswer result={null} inProgress={true} linkIndex={linkIndex} />;
            })()
          ) : (
            <TurnAnswer
              result={turn.result}
              inProgress={turn.inProgress}
              linkIndex={linkIndex}
              interactiveClarification={isPendingScopeForm}
              onScopeSubmit={onScopeSubmit}
              showLandDecision={showLandDecision}
              onLandDecide={onLandDecide}
              steps={turn.steps}
            />
          )}

          {planDecisionTaskId && turn.result?.type === "plan_emitted" ? (
            <PlanDecisionBar
              taskId={planDecisionTaskId}
              runId={planDecisionRunId}
              onDecided={onPlanDecided}
            />
          ) : null}

          <TurnFooter turn={turn} />
        </div>
      </div>
    </div>
  );
};

import {
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleUser,
  ClipboardList,
  Cpu,
  HelpCircle,
  Loader2,
  Send,
  Settings,
  ShieldAlert,
  Target,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CommandPalette, useCommandPalette } from "@/app/components/CommandPalette";
import { EntityHeader, StatusBadge } from "@/app/components/entity";
import {
  AgentNotConfiguredError,
  type ApiAgentHealth,
  type ApiCommand,
  agentAdminApi,
  agentApi,
  commandsApi,
} from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ApiAgentSession, ApiSessionEvent, RendererProps } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MarkdownContent } from "@/components/ui/markdown";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { ThinkingBlock } from "@/components/ui/thinking-block";
import { ToolCallRow } from "@/components/ui/tool-call-row";
import { formatDurationCompact } from "@/lib/format-time";
import { MolplotRawChart } from "@/plugins/molplot";
import { AgentSettingsViewer } from "./AgentSettingsViewer";
import {
  type ConversationTurn,
  derivePendingUserRequest,
  EVENT_META,
  foldStreamedTurn,
  groupEventsIntoTurns,
  normalizeStreamFrame,
  turnDurationSeconds,
} from "./agentEvents";

// ---------------------------------------------------------------------------
// Shared visual recipe — one composer shell + one column width everywhere so
// the agent surface reads as a single instrument, not assembled parts.
// ---------------------------------------------------------------------------

const COLUMN = "mx-auto w-full max-w-3xl";

const COMPOSER_SHELL =
  "flex items-end gap-2 rounded-lg border border-border bg-card px-3 py-2 shadow-xs " +
  "transition-[border-color,box-shadow] focus-within:border-ring focus-within:ring-2 " +
  "focus-within:ring-ring/25";

const COMPOSER_BAR = "border-t border-border/60 bg-background px-4 pb-4 pt-3 md:px-8";

const TEXTAREA_CLASS =
  "max-h-48 min-h-[24px] flex-1 resize-none bg-transparent px-1 py-1 text-sm leading-6 " +
  "placeholder:text-muted-foreground focus:outline-none disabled:opacity-60";

// ---------------------------------------------------------------------------
// Event row
// ---------------------------------------------------------------------------

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

const getAgentTaskId = (session: ApiAgentSession): string => session.taskId ?? session.sessionId;

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

        {/* Inline content for common event types */}
        {event.type === "loop_completed" && Boolean(payload.text) && (
          <div className="rounded-md border border-success/25 bg-success-soft px-3 py-2">
            <p className="text-xs text-success-foreground">{String(payload.text)}</p>
          </div>
        )}

        {event.type === "tool_call_completed" && <ToolResultArtifacts payload={payload} />}

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
// Artifact body (inline plot/table/text)
// ---------------------------------------------------------------------------

const ArtifactBody = ({ payload }: { payload: Record<string, unknown> }): JSX.Element | null => {
  const kind = String(payload.kind ?? "");
  const title = typeof payload.title === "string" ? payload.title : "";
  const inner = (payload.payload as Record<string, unknown> | undefined) ?? payload;

  if (kind === "plot") {
    // Agent-emitted specs are untyped — we accept whatever they emit
    // and let plotly validate at runtime via MolplotRawChart.
    const data = Array.isArray(inner.data) ? (inner.data as unknown[]) : [];
    const layout = (inner.layout as Record<string, unknown> | undefined) ?? {};
    return (
      <div className="space-y-2 rounded-md border border-border/60 bg-card p-3">
        {title && <p className="text-xs font-medium text-foreground">{title}</p>}
        <MolplotRawChart
          spec={{
            data,
            layout: { autosize: true, margin: { l: 48, r: 16, t: 16, b: 40 }, ...layout },
            config: { displayModeBar: false, responsive: true },
          }}
          style={{ width: "100%", height: 300 }}
        />
      </div>
    );
  }

  if (kind === "table") {
    const columns = Array.isArray(inner.columns) ? (inner.columns as string[]) : [];
    const rows = Array.isArray(inner.rows) ? (inner.rows as unknown[][]) : [];
    if (columns.length === 0 || rows.length === 0) return null;
    return (
      <div className="overflow-x-auto rounded-md border border-border/60">
        {title && (
          <p className="border-b border-border/60 bg-muted/40 px-3 py-1 text-xs font-medium">
            {title}
          </p>
        )}
        <table className="w-full text-xs">
          <thead className="bg-muted/30">
            <tr>
              {columns.map((c) => (
                <th key={`col-${c}`} className="px-3 py-1.5 text-left font-medium">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 50).map((row) => {
              const rowKey = row.map((value) => String(value ?? "")).join("|");
              return (
                <tr key={`row-${rowKey}`} className="border-t border-border/40">
                  {columns.map((column, colIdx) => (
                    <td key={`cell-${column}`} className="px-3 py-1 tabular-nums">
                      {String(row[colIdx] ?? "")}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
        {rows.length > 50 && (
          <p className="border-t border-border/40 bg-muted/20 px-3 py-1 text-[10px] text-muted-foreground">
            Showing 50 of {rows.length} rows
          </p>
        )}
      </div>
    );
  }

  if (kind === "text" && typeof inner.body === "string") {
    return (
      <pre className="overflow-x-auto whitespace-pre-wrap rounded-md border border-border/60 bg-muted/40 px-3 py-2 text-[11px] text-foreground">
        {inner.body}
      </pre>
    );
  }

  return null;
};

/**
 * Renders artifacts folded inside a ToolCallCompleted payload (§6.5).
 *
 * Reads `result.artifacts` (canonical) or `payload.artifacts` (loose mock)
 * and dispatches each entry to ArtifactBody. Falls back silently when the
 * tool call carried no inline artifacts.
 */
const ToolResultArtifacts = ({
  payload,
}: {
  payload: Record<string, unknown>;
}): JSX.Element | null => {
  const result = (payload.result as Record<string, unknown> | undefined) ?? payload;
  const artifacts = Array.isArray(result.artifacts)
    ? (result.artifacts as Record<string, unknown>[])
    : [];
  if (artifacts.length === 0) return null;
  return (
    <div className="space-y-2">
      {artifacts.map((artifact) => {
        // Artifacts inside a single ToolCallCompleted are append-only —
        // identity is `kind:title`, falling back to a JSON fingerprint
        // so two identical-kind artifacts still get distinct keys.
        const title = typeof artifact.title === "string" && artifact.title ? artifact.title : "";
        const fingerprint = title || JSON.stringify(artifact.payload ?? artifact);
        const key = `${String(artifact.kind ?? "?")}:${fingerprint}`;
        return <ArtifactBody key={key} payload={artifact} />;
      })}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Turn card — one conversational round-trip (question → answer)
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
          · {stepCount} {stepCount === 1 ? "step" : "steps"} — review the plan to continue
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

const TurnCard = ({ turn }: { turn: ConversationTurn }): JSX.Element => {
  const [stepsOpen, setStepsOpen] = useState(false);
  const stepCount = turn.steps.length;

  // Streamed render state for this turn: token answer, reasoning, tool calls.
  const streamed = useMemo(
    () => foldStreamedTurn(turn.result ? [...turn.steps, turn.result] : turn.steps),
    [turn.steps, turn.result],
  );

  const QuestionIcon = turn.source === "goal" ? Target : CircleUser;

  return (
    <article className="overflow-hidden rounded-lg border border-border/70 bg-card shadow-xs">
      <header className="flex items-start gap-2.5 border-b border-border/60 bg-muted/30 px-4 py-2.5">
        <QuestionIcon
          className="mt-0.5 h-3.5 w-3.5 flex-none text-muted-foreground"
          aria-label={turn.source === "goal" ? "Task goal" : "Your message"}
        />
        <p
          className="min-w-0 flex-1 whitespace-pre-wrap text-sm font-medium leading-snug text-foreground [overflow-wrap:anywhere]"
          title={turn.question}
        >
          {turn.question || <span className="italic text-muted-foreground">(no question)</span>}
        </p>
        {turn.inProgress && <StatusBadge status="running" size="sm" dot />}
      </header>

      <div className="space-y-2.5 px-4 py-3">
        {streamed.thinking && (
          <ThinkingBlock thinking={streamed.thinking} streaming={turn.inProgress} />
        )}

        {streamed.toolCalls.length > 0 && (
          <div className="-mx-1.5">
            {streamed.toolCalls.map((call) => (
              <ToolCallRow key={call.id} call={call} />
            ))}
          </div>
        )}

        {turn.inProgress && !turn.result && streamed.answer ? (
          // Token-by-token streaming answer before a terminal result lands —
          // rendered as markdown so structure appears as it streams.
          <MarkdownContent text={streamed.answer} />
        ) : (
          <TurnAnswer result={turn.result} inProgress={turn.inProgress} />
        )}

        {stepCount > 0 && (
          <div>
            <button
              type="button"
              onClick={() => setStepsOpen((v) => !v)}
              className="flex items-center gap-1.5 rounded-md text-xs text-muted-foreground transition-colors hover:text-foreground"
              aria-expanded={stepsOpen}
            >
              {stepsOpen ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              )}
              <span>
                {stepsOpen ? "Hide" : "Show"} {stepCount} {stepCount === 1 ? "step" : "steps"}
              </span>
            </button>
            {stepsOpen && (
              <div className="mt-2 rounded-md border border-border/50 bg-muted/20 px-3 py-1">
                {turn.steps.map((event, stepIdx) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: the event log is append-only, so the position is a stable identity even when two events share a timestamp
                  <EventRow key={`${turn.key}-step-${stepIdx}-${event.type}`} event={event} />
                ))}
              </div>
            )}
          </div>
        )}

        <TurnFooter turn={turn} />
      </div>
    </article>
  );
};

// ---------------------------------------------------------------------------
// Chat box (mid-session messages)
// ---------------------------------------------------------------------------

const ChatBox = ({
  awaitingRequestId,
  awaitingPrompt,
  disabled,
  onSubmit,
}: {
  awaitingRequestId: string | null;
  awaitingPrompt: string | null;
  disabled: boolean;
  onSubmit: (content: string, requestId: string | null) => Promise<void>;
}): JSX.Element => {
  const [content, setContent] = useState("");
  const [sending, setSending] = useState(false);

  const handleSend = async (): Promise<void> => {
    const trimmed = content.trim();
    if (!trimmed || sending || disabled) return;
    setSending(true);
    try {
      await onSubmit(trimmed, awaitingRequestId);
      setContent("");
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent): void => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      void handleSend();
    }
  };

  const placeholder = awaitingRequestId
    ? "Reply to the agent's question…"
    : "Message the agent… (⌘+Enter to send)";

  return (
    <div className={COMPOSER_BAR}>
      {awaitingRequestId && (
        <div
          className={`${COLUMN} mb-2 flex items-start gap-2 rounded-md border border-warning/30 bg-warning-soft px-3 py-2 text-xs text-warning-foreground`}
        >
          <HelpCircle className="mt-0.5 h-3.5 w-3.5 flex-none" />
          <p className="flex-1">
            <span className="font-semibold">Agent is waiting</span>
            {awaitingPrompt ? `: ${awaitingPrompt}` : "."}
          </p>
        </div>
      )}
      <div className={`${COLUMN} ${COMPOSER_SHELL}`}>
        <textarea
          rows={1}
          className={TEXTAREA_CLASS}
          placeholder={placeholder}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <Button
          size="icon"
          onClick={() => {
            void handleSend();
          }}
          disabled={disabled || sending || !content.trim()}
          className="h-8 w-8 flex-none rounded-md"
          aria-label="Send message"
        >
          {sending ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Send className="h-3.5 w-3.5" />
          )}
        </Button>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Goal input form
// ---------------------------------------------------------------------------

/**
 * Discriminated intent emitted by :func:`GoalInput`. Encapsulates "what the
 * user wants to launch" so the parent can route to the right API call
 * (``createSession`` vs. ``launchSkill``) without re-parsing the text.
 */
export type LaunchIntent =
  | {
      kind: "goal";
      description: string;
      criteria: string[];
      planMode: boolean;
      instructionsOverride?: string;
    }
  | {
      kind: "skill";
      skillId: string;
      parameters: Record<string, string>;
      planMode: boolean;
    };

const HELP_TEXT_LINES = [
  "Available commands:",
  "  /plan      — toggle plan mode (read-only inspection) for the next launch",
  "  /clear     — clear the input",
  "  /model     — open Provider settings to change the active model",
  "  /help      — show this list",
  "Skills with a slash name appear here too. Type the name and press Tab to autocomplete.",
];

const GoalInput = ({
  onSubmit,
  disabled,
  onOpenSettings,
  placeholder = "Describe your goal — start with / for a saved command",
}: {
  onSubmit: (intent: LaunchIntent) => Promise<void> | void;
  disabled: boolean;
  onOpenSettings?: () => void;
  placeholder?: string;
}): JSX.Element => {
  const [description, setDescription] = useState("");
  const [overrideText, setOverrideText] = useState("");
  const [planMode, setPlanMode] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [info, setInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [providerLabel, setProviderLabel] = useState<string>("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const anchorRef = useRef<HTMLDivElement | null>(null);
  const palette = useCommandPalette();

  // Keep the palette in sync with the textarea content.
  useEffect(() => {
    palette.syncFromValue(description);
  }, [description, palette]);

  // Fetch the active provider/model once so the input can show it inline.
  // Soft-fail: missing provider means no badge.
  useEffect(() => {
    let cancelled = false;
    agentAdminApi
      .getProvider()
      .then((p) => {
        if (cancelled) return;
        setProviderLabel(p.model || p.provider);
      })
      .catch(() => {
        if (!cancelled) setProviderLabel("");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const dispatchIntent = useCallback(
    async (intent: LaunchIntent): Promise<void> => {
      setError(null);
      setInfo(null);
      try {
        await onSubmit(intent);
        setDescription("");
        setOverrideText("");
      } catch (err) {
        setError(String(err));
      }
    },
    [onSubmit],
  );

  const handleBuiltin = useCallback(
    (name: string): void => {
      switch (name) {
        case "plan":
          setPlanMode((v) => !v);
          setDescription("");
          setInfo("Plan mode toggled for the next launch.");
          return;
        case "clear":
          setDescription("");
          setOverrideText("");
          setPlanMode(false);
          setShowAdvanced(false);
          setInfo("Input cleared.");
          return;
        case "model":
          setDescription("");
          setInfo("Open Settings → Provider to change the model.");
          onOpenSettings?.();
          return;
        case "help":
          setDescription("");
          setInfo(HELP_TEXT_LINES.join("\n"));
          return;
        default:
          setError(`Unhandled builtin /${name}.`);
      }
    },
    [onOpenSettings],
  );

  const submitGoal = useCallback(async (): Promise<void> => {
    const trimmed = description.trim();
    if (!trimmed) return;
    const override = overrideText.trim() || undefined;
    await dispatchIntent({
      kind: "goal",
      description: trimmed,
      criteria: [],
      planMode,
      instructionsOverride: override,
    });
  }, [description, dispatchIntent, overrideText, planMode]);

  const submitSlash = useCallback(async (): Promise<void> => {
    const raw = description.trim();
    if (!raw.startsWith("/")) return;
    const parsed = await commandsApi.parse(raw);
    if (parsed.kind === "error") {
      setError(parsed.error || "Invalid command.");
      return;
    }
    if (parsed.kind === "builtin") {
      handleBuiltin(parsed.name);
      return;
    }
    // skill
    await dispatchIntent({
      kind: "skill",
      skillId: parsed.skillId,
      parameters: parsed.parameters,
      planMode: parsed.planMode || planMode,
    });
  }, [description, dispatchIntent, handleBuiltin, planMode]);

  const handleSendButton = useCallback((): void => {
    const trimmed = description.trim();
    if (!trimmed) return;
    if (trimmed.startsWith("/")) {
      void submitSlash();
    } else {
      void submitGoal();
    }
  }, [description, submitGoal, submitSlash]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
      // Palette keyboard nav takes precedence — Tab/Esc/arrows.
      if (palette.handleKeyDown(e)) {
        if (e.key === "Tab") {
          const replaced = palette.applyActive(description);
          if (replaced !== null) setDescription(replaced);
          e.preventDefault();
        } else {
          e.preventDefault();
        }
        return;
      }
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleSendButton();
      }
      if (e.key === "Enter" && palette.open) {
        // Enter while palette is open: apply active suggestion instead of submit.
        const replaced = palette.applyActive(description);
        if (replaced !== null) {
          setDescription(replaced);
          e.preventDefault();
        }
      }
    },
    [description, handleSendButton, palette],
  );

  const handlePaletteSelect = useCallback(
    (cmd: ApiCommand): void => {
      const replaced = `/${cmd.slashName} `;
      setDescription(replaced);
      palette.close();
      textareaRef.current?.focus();
    },
    [palette],
  );

  return (
    <div className={COMPOSER_BAR}>
      {info && (
        <div
          className={`${COLUMN} mb-2 rounded-md border border-border/60 bg-muted/40 px-3 py-1.5 text-xs`}
        >
          <pre className="whitespace-pre-wrap font-mono">{info}</pre>
        </div>
      )}
      {error && (
        <div
          className={`${COLUMN} mb-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-1.5 text-xs text-destructive`}
        >
          {error}
        </div>
      )}
      {showAdvanced && (
        <div className={`${COLUMN} mb-2 rounded-md border border-border/60 bg-card p-3`}>
          <label
            htmlFor="prompt-override"
            className="mb-1 block text-xs font-medium text-muted-foreground"
          >
            System prompt override (replaces workspace + skill layers)
          </label>
          <textarea
            id="prompt-override"
            rows={4}
            className="w-full resize-none rounded-md border border-input bg-background px-3 py-2 font-mono text-[11px] focus:outline-none focus:ring-2 focus:ring-ring"
            placeholder="Custom instructions for this single task..."
            value={overrideText}
            onChange={(e) => setOverrideText(e.target.value)}
            disabled={disabled}
          />
        </div>
      )}
      <div className={`${COLUMN} relative`}>
        <div ref={anchorRef} className={COMPOSER_SHELL}>
          <textarea
            ref={textareaRef}
            rows={1}
            className={TEXTAREA_CLASS}
            placeholder={placeholder}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
          />
          <Button
            size="icon"
            onClick={handleSendButton}
            disabled={disabled || !description.trim()}
            className="h-8 w-8 flex-none rounded-md"
            aria-label="Start agent task"
          >
            {disabled ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Send className="h-3.5 w-3.5" />
            )}
          </Button>
        </div>
        <CommandPalette state={palette} anchorRef={anchorRef} onPick={handlePaletteSelect} />
      </div>
      <div className={`${COLUMN} mt-1.5 flex items-center gap-2 text-[11px] text-muted-foreground`}>
        {providerLabel ? (
          <button
            type="button"
            onClick={onOpenSettings}
            className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 font-mono text-[10px] transition-colors hover:bg-muted hover:text-foreground"
            title="Active model — click to change in Settings"
          >
            <Cpu className="h-3 w-3 opacity-60" />
            <span>{providerLabel}</span>
          </button>
        ) : null}
        <button
          type="button"
          onClick={() => setPlanMode((v) => !v)}
          className={
            "rounded-md px-1.5 py-0.5 text-[10px] font-medium transition-colors " +
            (planMode
              ? "bg-primary/10 text-primary hover:bg-primary/15"
              : "hover:bg-muted hover:text-foreground")
          }
          title={
            planMode
              ? "Plan mode on: agent inspects and emits a plan; no writes"
              : "Toggle plan mode for the next launch"
          }
          aria-pressed={planMode}
        >
          Plan {planMode ? "on" : "off"}
        </button>
        <button
          type="button"
          onClick={() => setShowAdvanced((v) => !v)}
          className="inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 text-[10px] font-medium transition-colors hover:bg-muted hover:text-foreground"
          title="Show advanced options (system prompt override)"
          aria-expanded={showAdvanced}
        >
          {showAdvanced ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
          <span>Advanced</span>
        </button>
        <span className="ml-auto">⌘+Enter to send · / to browse commands</span>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Agent health banner (shown when no API key is configured)
// ---------------------------------------------------------------------------

const AgentHealthBanner = ({
  health,
  onOpenSettings,
}: {
  health: ApiAgentHealth;
  onOpenSettings: () => void;
}): JSX.Element => (
  <div className="flex items-start gap-3 rounded-md border border-warning/30 bg-warning-soft px-4 py-3 text-sm">
    <ShieldAlert className="mt-0.5 h-5 w-5 flex-none text-warning" />
    <div className="flex-1 text-warning-foreground">
      <p className="font-medium">Agent not ready</p>
      <p className="mt-0.5 text-xs opacity-90">{health.reason}</p>
    </div>
    <Button size="sm" variant="outline" onClick={onOpenSettings}>
      Configure provider
    </Button>
  </div>
);

// ---------------------------------------------------------------------------
// Session header
// ---------------------------------------------------------------------------

const HeaderSettingsAction = ({ onOpenSettings }: { onOpenSettings: () => void }): JSX.Element => (
  <Button
    variant="ghost"
    size="icon"
    className="h-7 w-7"
    onClick={onOpenSettings}
    title="Agent settings"
    aria-label="Agent settings"
  >
    <Settings className="h-4 w-4" />
  </Button>
);

const SessionHeader = ({ session }: { session: ApiAgentSession }): JSX.Element => (
  <EntityHeader icon={Bot} title={session.goal} status={session.status} />
);

const NewSessionHeader = ({ onOpenSettings }: { onOpenSettings: () => void }): JSX.Element => {
  return (
    <EntityHeader
      icon={Bot}
      title="New agent task"
      subtitle="Set the task goal"
      actions={<HeaderSettingsAction onOpenSettings={onOpenSettings} />}
    />
  );
};

// ---------------------------------------------------------------------------
// Loading skeleton — sketches two turn cards instead of a centered spinner.
// ---------------------------------------------------------------------------

const SessionSkeleton = (): JSX.Element => (
  <div className={`${COLUMN} space-y-4 px-4 pb-6 pt-4 md:px-8`}>
    {["first", "second"].map((slot) => (
      <div key={slot} className="overflow-hidden rounded-lg border border-border/70 bg-card">
        <div className="border-b border-border/60 bg-muted/30 px-4 py-2.5">
          <Skeleton className="h-4 w-2/5" />
        </div>
        <div className="space-y-2.5 px-4 py-3">
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-4/5" />
          <Skeleton className="h-3 w-3/5" />
        </div>
      </div>
    ))}
  </div>
);

// ---------------------------------------------------------------------------
// Main AgentViewer
// ---------------------------------------------------------------------------

export const AgentViewer = (props: RendererProps): JSX.Element | null => {
  if (props.selection.objectId === "settings") {
    return <AgentSettingsViewerWrapper {...props} />;
  }
  return <AgentSessionViewer {...props} />;
};

const AgentSettingsViewerWrapper = ({ snapshot }: RendererProps): JSX.Element => {
  const nav = useNavigationState(snapshot);
  return (
    <AgentSettingsViewer
      snapshot={snapshot}
      onLaunchSession={(sessionId) =>
        nav.setSelection({ objectType: "agent", objectId: sessionId })
      }
    />
  );
};

const AgentSessionViewer = ({
  selection,
  snapshot,
  onRefresh,
}: RendererProps): JSX.Element | null => {
  const sessionId = selection.objectId === "new" ? null : selection.objectId;
  const nav = useNavigationState(snapshot);
  const [session, setSession] = useState<ApiAgentSession | null>(null);
  const [events, setEvents] = useState<ApiSessionEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<ApiAgentHealth | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const esRef = useRef<EventSource | null>(null);

  // Fetch agent health up-front so the new-session view can warn the user
  // about a missing API key before they spend time typing a goal.
  useEffect(() => {
    let cancelled = false;
    agentApi
      .getHealth()
      .then((h) => {
        if (!cancelled) setHealth(h);
      })
      .catch(() => {
        // Health endpoint may legitimately be unavailable (older server,
        // network blip); leave health=null and don't render a banner.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const openSettings = useCallback(() => {
    nav.setSelection({ objectType: "agent", objectId: "settings" });
  }, [nav]);

  // Load session when sessionId changes
  useEffect(() => {
    if (!sessionId) {
      setSession(null);
      setEvents([]);
      return;
    }
    setLoading(true);
    setError(null);
    agentApi
      .getSession(sessionId)
      .then((s) => {
        setSession(s);
        setEvents(s.events ?? []);
        onRefresh();
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [sessionId, onRefresh]);

  // SSE stream for running sessions. Depend on sessionId+status only —
  // including the whole `session` object would resubscribe on every poll
  // tick (the polling effect mutates the reference) and the mock SSE
  // handler would re-deliver the same events, growing state forever.
  useEffect(() => {
    if (!sessionId) return;
    if (session?.status !== "running") return;
    const es = agentApi.streamEvents(sessionId);
    esRef.current = es;
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "done") {
          es.close();
          agentApi.getSession(sessionId).then((s) => {
            setSession((prev) => {
              if (!prev || prev.sessionId !== sessionId) return s;
              if (prev.status === s.status) return prev;
              return { ...prev, status: s.status, stats: s.stats };
            });
          });
          return;
        }
        // Normalize live AgentEvent frames ({kind, timestamp, …}) into the UI's
        // {type, ts, payload} shape; `waiting` (and any control frame) → null.
        const normalized = normalizeStreamFrame(data);
        if (normalized) {
          setEvents((prev) => [...prev, normalized]);
        }
      } catch {
        // ignore parse errors
      }
    };
    return () => {
      es.close();
    };
  }, [sessionId, session?.status]);

  // Auto-scroll: jump to the latest activity when a session loads, then
  // follow the stream only while the user is already reading the tail
  // (never yank the viewport away from someone scrolled up).
  const scrollViewport = useCallback((): HTMLElement | null => {
    const root = scrollRef.current;
    if (!root) return null;
    return root.querySelector<HTMLElement>("[data-radix-scroll-area-viewport]") ?? root;
  }, []);

  useEffect(() => {
    if (loading) return;
    const viewport = scrollViewport();
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
  }, [loading, scrollViewport]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: `events` is the stream tick this effect follows
  useEffect(() => {
    const viewport = scrollViewport();
    if (!viewport) return;
    const distanceFromBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
    if (distanceFromBottom < 160) {
      viewport.scrollTop = viewport.scrollHeight;
    }
  }, [events, scrollViewport]);

  // While running, poll session stats so token/usage counts stay live
  // even though they aren't part of the SSE event payloads. Depend on
  // sessionId+status only — including the session ref would re-create
  // the interval on every tick.
  useEffect(() => {
    if (!sessionId) return;
    if (session?.status !== "running") return;
    let cancelled = false;
    const tick = async (): Promise<void> => {
      try {
        const fresh = await agentApi.getSession(sessionId);
        if (cancelled) return;
        setSession((prev) => {
          if (!prev || prev.sessionId !== sessionId) return prev;
          if (
            prev.status === fresh.status &&
            JSON.stringify(prev.stats ?? null) === JSON.stringify(fresh.stats ?? null)
          ) {
            return prev;
          }
          return { ...prev, status: fresh.status, stats: fresh.stats };
        });
      } catch {
        // ignore transient polling errors
      }
    };
    const id = setInterval(tick, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [sessionId, session?.status]);

  const handleLaunchIntent = useCallback(
    async (intent: LaunchIntent) => {
      setSubmitting(true);
      setError(null);
      try {
        const created: ApiAgentSession =
          intent.kind === "goal"
            ? await agentApi.createSession(intent.description, intent.criteria, {
                planMode: intent.planMode || undefined,
                instructionsOverride: intent.instructionsOverride,
              })
            : await agentAdminApi.launchSkill(intent.skillId, intent.parameters, {
                planMode: intent.planMode,
              });
        setSession(created);
        setEvents(created.events ?? []);
        nav.setSelection({ objectType: "agent", objectId: getAgentTaskId(created) });
        onRefresh();
      } catch (err) {
        if (err instanceof AgentNotConfiguredError) {
          agentApi
            .getHealth()
            .then(setHealth)
            .catch(() => {});
          setError(err.message);
          return;
        }
        setError(String(err));
        throw err;
      } finally {
        setSubmitting(false);
      }
    },
    [nav, onRefresh],
  );

  const handleChatSubmit = useCallback(
    async (content: string, requestId: string | null) => {
      if (!session) return;
      try {
        await agentApi.postMessage(getAgentTaskId(session), content, requestId);
      } catch (err) {
        setError(String(err));
      }
    },
    [session],
  );

  // Detect whether the agent is currently waiting on the user's reply.
  const pendingUserRequest = useMemo(() => derivePendingUserRequest(events), [events]);

  // --- "new" state: quiet empty state with the composer at the bottom ---
  if (!sessionId || (!loading && !session)) {
    const notReady = health !== null && !health.ready;
    const recent = snapshot.agentSessions.slice(0, 5);
    return (
      <div className="flex h-full flex-col bg-background">
        <NewSessionHeader onOpenSettings={openSettings} />
        <div className="flex flex-1 flex-col overflow-auto">
          <div className={`${COLUMN} flex flex-1 flex-col gap-6 px-4 py-8 md:px-8`}>
            {notReady && health && (
              <AgentHealthBanner health={health} onOpenSettings={openSettings} />
            )}
            {error && (
              <div className="flex items-center justify-between gap-3 rounded-md border border-destructive/40 bg-destructive/10 px-4 py-2 text-sm text-destructive">
                <span className="flex-1">{error}</span>
                {notReady && (
                  <Button size="sm" variant="outline" onClick={openSettings}>
                    Open agent settings
                  </Button>
                )}
              </div>
            )}

            <div className="flex flex-col items-center gap-2 pt-10 text-center">
              <div className="flex h-10 w-10 items-center justify-center rounded-md border border-border/60 bg-card">
                <Bot className="h-5 w-5 text-muted-foreground" />
              </div>
              <h2 className="text-base font-semibold text-foreground">Start an agent task</h2>
              <p className="max-w-md text-sm text-muted-foreground">
                Describe a goal. The agent plans the steps, calls molexp tools, and reports results
                with artifacts.
              </p>
            </div>

            {recent.length > 0 && (
              <div className="space-y-1.5">
                <p className="px-1 text-xs font-medium text-muted-foreground">Recent tasks</p>
                {recent.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => nav.setSelection({ objectType: "agent", objectId: s.id })}
                    className="flex w-full items-center gap-3 rounded-md border border-border/60 bg-card px-3 py-2 text-left transition-colors hover:bg-muted/40"
                  >
                    <Bot className="h-4 w-4 flex-none text-muted-foreground" />
                    <p className="flex-1 truncate text-sm">{s.goal}</p>
                    <StatusBadge status={s.status} size="sm" />
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
        <GoalInput
          onSubmit={handleLaunchIntent}
          disabled={submitting || notReady}
          onOpenSettings={openSettings}
        />
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex h-full flex-col bg-background">
        <SessionSkeleton />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
        <XCircle className="h-10 w-10 text-destructive" />
        <p className="text-sm text-muted-foreground">{error}</p>
        <Button variant="outline" size="sm" onClick={() => setError(null)}>
          Dismiss error
        </Button>
      </div>
    );
  }

  if (!session) return null;

  const isRunning = session.status === "running";
  const turns = groupEventsIntoTurns(events, session.goal);

  return (
    <div className="flex h-full flex-col bg-background">
      <SessionHeader session={session} />

      <ScrollArea className="flex-1" ref={scrollRef as React.RefObject<HTMLDivElement>}>
        <div className={`${COLUMN} flex flex-col gap-4 px-4 pb-6 pt-4 md:px-8`}>
          {turns.map((turn) => (
            <TurnCard key={turn.key} turn={turn} />
          ))}
        </div>
      </ScrollArea>

      {isRunning ? (
        <ChatBox
          awaitingRequestId={pendingUserRequest?.requestId ?? null}
          awaitingPrompt={pendingUserRequest?.prompt ?? null}
          disabled={false}
          onSubmit={handleChatSubmit}
        />
      ) : (
        <GoalInput
          onSubmit={handleLaunchIntent}
          disabled={submitting}
          onOpenSettings={openSettings}
          placeholder="Send a follow-up message..."
        />
      )}
    </div>
  );
};

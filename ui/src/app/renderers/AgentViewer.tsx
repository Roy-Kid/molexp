import {
  Bot,
  ChevronDown,
  ChevronRight,
  CircleUser,
  HelpCircle,
  Loader2,
  Send,
  Settings,
  ShieldAlert,
  Sparkles,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CommandPalette, useCommandPalette } from "@/app/components/CommandPalette";
import { EntityHeader, StatusBadge } from "@/app/components/entity";
import { PlanCard } from "@/app/renderers/PlanCard";
import {
  AgentNotConfiguredError,
  type ApiAgentHealth,
  type ApiCommand,
  agentAdminApi,
  agentApi,
  commandsApi,
} from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type {
  ApiAgentSession,
  ApiSessionEvent,
  RendererProps,
  WorkspaceSnapshot,
} from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Plot } from "@/lib/plot";
import { AgentSettingsViewer } from "./AgentSettingsViewer";
import {
  type ConversationTurn,
  derivePendingUserRequest,
  EVENT_META,
  groupEventsIntoTurns,
} from "./agentEvents";

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

const getAgentTaskId = (session: ApiAgentSession): string => session.taskId ?? session.sessionId;

const EventRow = ({
  event,
  sessionId,
  onApprovalRespond,
  onPlanResolved,
}: {
  event: ApiSessionEvent;
  sessionId: string;
  onApprovalRespond: (requestId: string, approved: boolean) => void;
  onPlanResolved?: () => void;
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

  return (
    <div className="group flex gap-3 py-2">
      <div className={`mt-0.5 flex-none ${meta.colorClass}`}>
        <Icon className="h-4 w-4" />
      </div>
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{meta.label}</span>
          {event.type === "ToolCallRequested" && Boolean(payload.tool_name) && (
            <Badge variant="secondary" className="font-mono text-[10px] h-4 px-1">
              {String(payload.tool_name)}
            </Badge>
          )}
          {event.type === "ToolCallCompleted" && Boolean(payload.run_id) && (
            <Badge variant="secondary" className="font-mono text-[10px] h-4 px-1">
              {String(payload.run_id)}
            </Badge>
          )}
          <span className="ml-auto text-[10px] text-muted-foreground tabular-nums">
            {formatTs(event.ts)}
          </span>
          {hasDetail && (
            <button
              type="button"
              className="text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setExpanded((v) => !v)}
              aria-label={expanded ? "Collapse" : "Expand"}
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
        {event.type === "PlanCreated" && (
          <PlanCard sessionId={sessionId} event={event} onResolved={onPlanResolved} />
        )}

        {event.type === "SessionCompleted" && (
          <div className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 dark:border-emerald-800 dark:bg-emerald-950/30">
            {Boolean(payload.summary) && (
              <p className="text-xs text-emerald-700 dark:text-emerald-400">
                {String(payload.summary)}
              </p>
            )}
          </div>
        )}

        {event.type === "ToolApprovalRequested" && Boolean(payload.request_id) && (
          <div className="flex items-center gap-2 rounded-md border border-orange-200 bg-orange-50 px-3 py-2 dark:border-orange-800 dark:bg-orange-950/30">
            <p className="flex-1 text-xs text-orange-700 dark:text-orange-400">
              Approve{" "}
              <span className="font-mono font-semibold">
                {String(payload.tool_name ?? "action")}
              </span>
              ?
            </p>
            <Button
              size="sm"
              variant="outline"
              className="h-6 border-orange-400 text-orange-700 hover:bg-orange-100"
              onClick={() => onApprovalRespond(String(payload.request_id), false)}
            >
              Deny
            </Button>
            <Button
              size="sm"
              className="h-6 bg-orange-500 text-white hover:bg-orange-600"
              onClick={() => onApprovalRespond(String(payload.request_id), true)}
            >
              Approve
            </Button>
          </div>
        )}

        {event.type === "ToolCallCompleted" && <ToolResultArtifacts payload={payload} />}

        {event.type === "UserMessageRequested" && Boolean(payload.prompt) && (
          <div className="rounded-md border border-fuchsia-200 bg-fuchsia-50 px-3 py-2 dark:border-fuchsia-800 dark:bg-fuchsia-950/30">
            <p className="text-xs text-fuchsia-700 dark:text-fuchsia-300">
              {String(payload.prompt)}
            </p>
          </div>
        )}

        {event.type === "UserMessageReceived" && Boolean(payload.content) && (
          <p className="text-xs italic text-muted-foreground">“{String(payload.content)}”</p>
        )}

        {expanded && hasDetail && (
          <pre className="overflow-x-auto rounded-md bg-muted/60 px-3 py-2 text-[11px] font-mono text-muted-foreground">
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
    // Plotly's strict PlatData type is too narrow for agent-generated specs;
    // we accept whatever the agent emitted and let Plotly validate at runtime.
    const data = Array.isArray(inner.data) ? (inner.data as never) : ([] as never);
    const layout = (inner.layout as object | undefined) ?? {};
    return (
      <div className="space-y-2 rounded-md border border-indigo-200 bg-indigo-50/40 p-3 dark:border-indigo-900 dark:bg-indigo-950/20">
        {title && (
          <p className="text-xs font-semibold text-indigo-700 dark:text-indigo-300">{title}</p>
        )}
        <Plot
          data={data}
          layout={{ autosize: true, margin: { l: 48, r: 16, t: 16, b: 40 }, ...layout }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%", height: 300 }}
          useResizeHandler
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
          <p className="border-b border-border/60 bg-muted/40 px-3 py-1 text-xs font-semibold">
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
      <pre className="overflow-x-auto whitespace-pre-wrap rounded-md bg-muted/40 px-3 py-2 text-[11px] text-foreground">
        {inner.body}
      </pre>
    );
  }

  return null;
};

// ---------------------------------------------------------------------------
// Turn card — one conversational round-trip (question → answer)
// ---------------------------------------------------------------------------

const TurnAnswer = ({
  result,
  inProgress,
  sessionId,
  onPlanResolved,
}: {
  result: ApiSessionEvent | null;
  inProgress: boolean;
  sessionId: string;
  onPlanResolved?: () => void;
}): JSX.Element => {
  if (!result) {
    if (inProgress) {
      return (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
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

  if (result.type === "PlanCreated") {
    return <PlanCard sessionId={sessionId} event={result} onResolved={onPlanResolved} />;
  }

  const payload = (result.payload ?? {}) as Record<string, unknown>;

  if (result.type === "SessionCompleted") {
    const summary = typeof payload.summary === "string" ? payload.summary : "";
    return (
      <div className="space-y-2">
        {summary ? (
          <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-foreground">
            {summary}
          </p>
        ) : (
          <p className="text-sm italic text-muted-foreground">Session ended without a summary.</p>
        )}
      </div>
    );
  }

  if (result.type === "ToolCallCompleted") {
    return <ToolResultArtifacts payload={payload} />;
  }

  return (
    <pre className="overflow-x-auto rounded-md bg-muted/40 px-3 py-2 text-xs">
      {JSON.stringify(payload, null, 2)}
    </pre>
  );
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

const TurnCard = ({
  turn,
  index,
  total,
  sessionId,
  onApprovalRespond,
  onPlanResolved,
}: {
  turn: ConversationTurn;
  index: number;
  total: number;
  sessionId: string;
  onApprovalRespond: (requestId: string, approved: boolean) => void;
  onPlanResolved?: () => void;
}): JSX.Element => {
  const isLast = index === total - 1;
  const [stepsOpen, setStepsOpen] = useState(false);
  const stepCount = turn.steps.length;

  const QuestionIcon = turn.source === "goal" ? Sparkles : CircleUser;
  const questionLabel = turn.source === "goal" ? "Goal" : "You asked";

  // Surface approvals immediately even when the steps section is collapsed —
  // a hidden approval would block the agent indefinitely.
  const pendingApproval = turn.steps.find((s) => {
    if (s.type !== "ToolApprovalRequested") return false;
    const p = (s.payload ?? {}) as Record<string, unknown>;
    if (typeof p.request_id !== "string") return false;
    return !turn.steps.some(
      (other) =>
        other.type === "ApprovalDecisionEvent" &&
        ((other.payload ?? {}) as Record<string, unknown>).request_id === p.request_id,
    );
  });

  return (
    <article className="rounded-xl border border-border/70 bg-card shadow-sm">
      <header className="flex items-start gap-3 border-b border-border/60 px-4 py-3">
        <div
          className={
            turn.source === "goal"
              ? "mt-0.5 flex-none rounded-full bg-violet-500/10 p-1.5 text-violet-500"
              : "mt-0.5 flex-none rounded-full bg-blue-500/10 p-1.5 text-blue-500"
          }
        >
          <QuestionIcon className="h-3.5 w-3.5" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            {questionLabel}
          </p>
          <p className="mt-0.5 whitespace-pre-wrap text-sm font-medium text-foreground">
            {turn.question || <span className="italic text-muted-foreground">(no question)</span>}
          </p>
        </div>
      </header>

      <div className="space-y-3 px-4 py-3">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex-none rounded-full bg-emerald-500/10 p-1.5 text-emerald-500">
            <Bot className="h-3.5 w-3.5" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              {turn.result?.type === "SessionCompleted" && isLast ? "Final answer" : "Answer"}
            </p>
            <div className="mt-1">
              <TurnAnswer
                result={turn.result}
                inProgress={turn.inProgress}
                sessionId={sessionId}
                onPlanResolved={onPlanResolved}
              />
            </div>
          </div>
        </div>

        {pendingApproval && (
          <div className="ml-9">
            <EventRow
              event={pendingApproval}
              sessionId={sessionId}
              onApprovalRespond={onApprovalRespond}
            />
          </div>
        )}

        {stepCount > 0 && (
          <div className="ml-9">
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
              <div className="mt-2 rounded-md border border-border/50 bg-muted/20 px-3 py-2">
                {turn.steps.map((event) => (
                  <EventRow
                    key={`${turn.key}-step-${event.type}-${event.ts}`}
                    event={event}
                    sessionId={sessionId}
                    onApprovalRespond={onApprovalRespond}
                    onPlanResolved={onPlanResolved}
                  />
                ))}
              </div>
            )}
          </div>
        )}
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
    <div className="bg-gradient-to-t from-background via-background to-background/0 px-4 pb-4 pt-3 md:px-8 md:pb-6">
      {awaitingRequestId && (
        <div className="mx-auto mb-2 flex max-w-5xl items-start gap-2 rounded-md border border-fuchsia-200 bg-fuchsia-50 px-3 py-2 text-xs text-fuchsia-700 dark:border-fuchsia-800 dark:bg-fuchsia-950/30 dark:text-fuchsia-300">
          <HelpCircle className="mt-0.5 h-3.5 w-3.5 flex-none" />
          <p className="flex-1">
            <span className="font-semibold">Agent is waiting</span>
            {awaitingPrompt ? `: ${awaitingPrompt}` : "."}
          </p>
        </div>
      )}
      <div className="mx-auto flex max-w-5xl items-end gap-2 rounded-2xl border border-border bg-card px-3 py-2 shadow-md focus-within:border-primary/60 focus-within:ring-2 focus-within:ring-ring/30">
        <textarea
          rows={1}
          className="max-h-48 min-h-[24px] flex-1 resize-none bg-transparent px-1 py-1 text-sm leading-6 placeholder:text-muted-foreground focus:outline-none"
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
          className="h-8 w-8 flex-none rounded-full"
          aria-label="Send"
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

  // Fetch the active provider/model once so the input can show it inline,
  // Codex/Claude-style. Soft-fail: missing provider means no badge.
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
      const replaced = `/${cmd.slashName}${cmd.parameters.length > 0 ? " " : " "}`;
      setDescription(replaced);
      palette.close();
      textareaRef.current?.focus();
    },
    [palette],
  );

  return (
    <div className="bg-gradient-to-t from-background via-background to-background/0 px-4 pb-4 pt-3 md:px-8 md:pb-6">
      {info && (
        <div className="mx-auto mb-2 max-w-5xl rounded-md border border-border/60 bg-muted/40 px-3 py-1.5 text-xs">
          <pre className="whitespace-pre-wrap font-mono">{info}</pre>
        </div>
      )}
      {error && (
        <div className="mx-auto mb-2 max-w-5xl rounded-md border border-destructive/40 bg-destructive/10 px-3 py-1.5 text-xs text-destructive">
          {error}
        </div>
      )}
      {showAdvanced && (
        <div className="mx-auto mb-2 max-w-5xl rounded-xl border border-border/60 bg-card/60 p-3">
          <label
            htmlFor="prompt-override"
            className="mb-1 block text-[10px] font-semibold uppercase tracking-wide text-muted-foreground"
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
      <div className="relative mx-auto max-w-5xl">
        <div
          ref={anchorRef}
          className="flex items-end gap-2 rounded-2xl border border-border bg-card px-3 py-2 shadow-md focus-within:border-primary/60 focus-within:ring-2 focus-within:ring-ring/30"
        >
          <textarea
            ref={textareaRef}
            rows={1}
            className="max-h-48 min-h-[24px] flex-1 resize-none bg-transparent px-1 py-1 text-sm leading-6 placeholder:text-muted-foreground focus:outline-none"
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
            className="h-8 w-8 flex-none rounded-full"
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
      <div className="mx-auto mt-1.5 flex max-w-5xl items-center gap-2 text-[11px] text-muted-foreground">
        {providerLabel ? (
          <button
            type="button"
            onClick={onOpenSettings}
            className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 font-mono text-[10px] transition-colors hover:bg-muted hover:text-foreground"
            title="Active model — click to change in Settings"
          >
            <Sparkles className="h-3 w-3 opacity-60" />
            <span>{providerLabel}</span>
          </button>
        ) : null}
        <button
          type="button"
          onClick={() => setPlanMode((v) => !v)}
          className={
            "rounded-md px-1.5 py-0.5 text-[10px] font-medium transition-colors " +
            (planMode
              ? "bg-violet-500/10 text-violet-600 hover:bg-violet-500/20"
              : "hover:bg-muted hover:text-foreground")
          }
          title={
            planMode
              ? "Plan mode on: agent inspects and emits a plan; no writes"
              : "Toggle plan mode for the next launch"
          }
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
  <div className="flex items-start gap-3 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm dark:border-amber-700 dark:bg-amber-950/30">
    <ShieldAlert className="mt-0.5 h-5 w-5 flex-none text-amber-600 dark:text-amber-400" />
    <div className="flex-1 text-amber-900 dark:text-amber-100">
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

const LiveIndicator = (): JSX.Element => (
  <span className="inline-flex items-center gap-1 rounded-md bg-info-soft px-1.5 py-0.5 text-[10px] font-medium text-info-foreground">
    <Loader2 className="h-3 w-3 animate-spin" />
    Live
  </span>
);

const SessionHeader = ({
  session,
  snapshot,
}: {
  session: ApiAgentSession;
  snapshot: WorkspaceSnapshot;
}): JSX.Element => {
  const { breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);
  const isRunning = session.status === "running";
  return (
    <EntityHeader
      breadcrumbs={breadcrumbs}
      canNavigateUp={canNavigateUp}
      onNavigateUp={navigateUp}
      icon={Bot}
      title={session.goal}
      status={session.status}
      titleAccessory={isRunning ? <LiveIndicator /> : undefined}
    />
  );
};

const NewSessionHeader = ({
  snapshot,
  onOpenSettings,
}: {
  snapshot: WorkspaceSnapshot;
  onOpenSettings: () => void;
}): JSX.Element => {
  const { breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);
  return (
    <EntityHeader
      breadcrumbs={breadcrumbs}
      canNavigateUp={canNavigateUp}
      onNavigateUp={navigateUp}
      icon={Sparkles}
      title="New agent task"
      subtitle="Set the task goal"
      actions={<HeaderSettingsAction onOpenSettings={onOpenSettings} />}
    />
  );
};

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
        if (data.type !== "waiting") {
          setEvents((prev) => [...prev, data as ApiSessionEvent]);
        }
      } catch {
        // ignore parse errors
      }
    };
    return () => {
      es.close();
    };
  }, [sessionId, session?.status]);

  // Auto-scroll to bottom when events change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, []);

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

  const handleApprovalRespond = useCallback(
    async (requestId: string, approved: boolean) => {
      if (!session) return;
      try {
        await agentApi.respondApproval(getAgentTaskId(session), requestId, approved);
      } catch (err) {
        setError(String(err));
      }
    },
    [session],
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

  // --- "new" state: ChatGPT-style hero with composer at the bottom ---
  if (!sessionId || (!loading && !session)) {
    const notReady = health !== null && !health.ready;
    const recent = snapshot.agentSessions.slice(0, 5);
    return (
      <div className="flex h-full flex-col bg-background">
        <NewSessionHeader snapshot={snapshot} onOpenSettings={openSettings} />
        <div className="flex flex-1 flex-col overflow-auto">
          <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-6 px-4 py-8 md:px-8">
            {notReady && health && (
              <AgentHealthBanner health={health} onOpenSettings={openSettings} />
            )}
            {error && (
              <div className="flex items-center justify-between gap-3 rounded-md border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/30 dark:text-red-400">
                <span className="flex-1">{error}</span>
                {notReady && (
                  <Button size="sm" variant="outline" onClick={openSettings}>
                    Open Agent Settings
                  </Button>
                )}
              </div>
            )}

            <div className="flex flex-col items-center gap-3 pt-8 text-center">
              <div className="rounded-full bg-violet-500/10 p-3 text-violet-500">
                <Sparkles className="h-6 w-6" />
              </div>
              <h2 className="text-xl font-semibold">What can the agent help you with?</h2>
              <p className="max-w-md text-sm text-muted-foreground">Set a task goal to begin.</p>
            </div>

            {recent.length > 0 && (
              <div className="space-y-1.5">
                <p className="px-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Recent tasks
                </p>
                {recent.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => nav.setSelection({ objectType: "agent", objectId: s.id })}
                    className="flex w-full items-center gap-3 rounded-lg border border-border/60 bg-card px-3 py-2 text-left transition-colors hover:bg-muted/40"
                  >
                    <Bot className="h-4 w-4 flex-none text-violet-400" />
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
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
        <XCircle className="h-10 w-10 text-red-500" />
        <p className="text-sm text-muted-foreground">{error}</p>
        <Button variant="outline" size="sm" onClick={() => setError(null)}>
          Retry
        </Button>
      </div>
    );
  }

  if (!session) return null;

  const isRunning = session.status === "running";
  const turns = groupEventsIntoTurns(events, session.goal);

  // After a structured plan is approved/rejected, re-fetch the session
  // so the freshly-flipped planMode flag, status, and post-handoff
  // events land in state. The PlanCard fires this callback inline.
  const refreshAfterPlanDecision = (): void => {
    agentApi
      .getSession(getAgentTaskId(session))
      .then((s) => {
        setSession(s);
        setEvents(s.events ?? []);
      })
      .catch(() => {
        // Polling will pick up the change on the next tick.
      });
  };

  return (
    <div className="flex h-full flex-col bg-background">
      <SessionHeader session={session} snapshot={snapshot} />

      <ScrollArea className="flex-1" ref={scrollRef as React.RefObject<HTMLDivElement>}>
        <div className="mx-auto flex max-w-5xl flex-col gap-4 px-4 pb-6 pt-4 md:px-8">
          {turns.map((turn, turnIdx) => (
            <TurnCard
              key={turn.key}
              turn={turn}
              index={turnIdx}
              total={turns.length}
              sessionId={getAgentTaskId(session)}
              onApprovalRespond={handleApprovalRespond}
              onPlanResolved={refreshAfterPlanDecision}
            />
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

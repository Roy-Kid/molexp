import { Bot, Send, Settings, ShieldAlert, Square, XCircle } from "lucide-react";
import { type JSX, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CommandPalette, useCommandPalette } from "@/app/components/CommandPalette";
import { EntityHeader, StatusBadge } from "@/app/components/entity";
import {
  AgentNotConfiguredError,
  type ApiAgentHealth,
  type ApiCommand,
  agentAdminApi,
  agentApi,
  commandsApi,
  workspaceApi,
} from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ApiAgentSession, ApiSessionEvent, RendererProps } from "@/app/types";
import { Code as InlineCode } from "@/components/ui/code";
import { ProgressSpinner } from "@/components/ui/progress-spinner";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
  WorkbenchAction,
  WorkbenchDismissAction,
  WorkbenchIconAction,
} from "@/components/workbench";
import { agentTaskDisplayTitle } from "@/lib/agent-task-title";
import { buildEntityLinkIndex } from "@/lib/entity-linkify";
import { cn } from "@/lib/utils";
import { AgentSettingsViewer } from "./AgentSettingsViewer";
import { ApprovalsInbox } from "./agent/ApprovalsInbox";
import { type AgentMode, nextAgentMode } from "./agent/agentMode";
import { ConversationTurnView } from "./agent/conversation";
import { DeliverablesPanel, hasDeliverables } from "./agent/DeliverablesPanel";
import { PlanProgressRail } from "./agent/PlanProgressRail";
import { DEFAULT_PLAN_STAGE, PLAN_STAGES } from "./agent/planStages";
import {
  appendCoalescedEvent,
  coalesceStreamEvents,
  derivePendingUserRequest,
  derivePlanRef,
  groupEventsIntoTurns,
  normalizeStreamFrame,
  sameSessionEvent,
} from "./agentEvents";

// ---------------------------------------------------------------------------
// Shared visual recipe — one composer shell + one column width everywhere so
// the agent surface reads as a single instrument, not assembled parts.
// ---------------------------------------------------------------------------

const COLUMN = "mx-auto w-full max-w-3xl";

/** Mode-tinted composer shell — Chat neutral; Plan light blue border only (readable text). */
const composerShellClass = (mode: AgentMode): string =>
  cn(
    "flex items-end gap-2 rounded-panel border px-3 py-2 bg-card",
    "transition-[border-color,box-shadow,background-color] focus-within:ring-2",
    mode === "plan"
      ? "border-info/50 bg-info-soft/25 focus-within:border-info focus-within:ring-info/15"
      : "border-border focus-within:border-ring focus-within:ring-ring/25",
  );

const COMPOSER_BAR = "border-t border-border/60 bg-background px-4 pb-4 pt-3 md:px-8";

const TEXTAREA_CLASS =
  "max-h-48 min-h-6 flex-1 resize-none border-0 bg-transparent px-1 py-1 text-body-lg leading-6 " +
  "placeholder:text-muted-foreground focus-visible:ring-0 disabled:opacity-60";

const getAgentTaskId = (session: ApiAgentSession): string => session.taskId ?? session.sessionId;
const MESSAGE_HISTORY_KEY = "molexp.agent.messageHistory";
const MESSAGE_HISTORY_LIMIT = 80;

const readMessageHistory = (): string[] => {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(MESSAGE_HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
};

const writeMessageHistory = (items: string[]): void => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(
    MESSAGE_HISTORY_KEY,
    JSON.stringify(items.slice(0, MESSAGE_HISTORY_LIMIT)),
  );
};

const rememberMessage = (content: string): void => {
  const trimmed = content.trim();
  if (!trimmed) return;
  writeMessageHistory([trimmed, ...readMessageHistory().filter((item) => item !== trimmed)]);
};

const useMessageHistory = (
  value: string,
  setValue: (value: string) => void,
): ((e: React.KeyboardEvent<HTMLTextAreaElement>) => boolean) => {
  const cursorRef = useRef<number | null>(null);
  const draftRef = useRef("");

  return useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>): boolean => {
      if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return false;
      const target = e.currentTarget;
      const atEdge =
        e.key === "ArrowUp"
          ? target.selectionStart === 0 && target.selectionEnd === 0
          : target.selectionStart === value.length && target.selectionEnd === value.length;
      if (!atEdge || e.shiftKey || e.metaKey || e.ctrlKey || e.altKey) return false;
      const history = readMessageHistory();
      if (history.length === 0) return false;
      e.preventDefault();

      if (cursorRef.current === null) {
        draftRef.current = value;
        cursorRef.current = e.key === "ArrowUp" ? 0 : null;
      } else if (e.key === "ArrowUp") {
        cursorRef.current = Math.min(cursorRef.current + 1, history.length - 1);
      } else {
        cursorRef.current -= 1;
      }

      if (cursorRef.current === null || cursorRef.current < 0) {
        cursorRef.current = null;
        setValue(draftRef.current);
      } else {
        setValue(history[cursorRef.current] ?? draftRef.current);
      }
      return true;
    },
    [setValue, value],
  );
};

/** Single control: click (or Shift+Tab) cycles Chat ↔ Plan. */
const ModeToggle = ({
  mode,
  onChange,
}: {
  mode: AgentMode;
  onChange: (mode: AgentMode) => void;
}): JSX.Element => (
  <WorkbenchAction
    kind="ghost"
    size="content"
    type="button"
    onClick={() => onChange(nextAgentMode(mode))}
    className={cn(
      "rounded-control px-2.5 py-1 text-micro font-medium transition-colors",
      mode === "plan"
        ? "bg-info-soft/40 text-info-foreground hover:bg-info-soft/55"
        : "bg-muted/60 text-foreground hover:bg-muted",
    )}
    title={
      mode === "chat"
        ? "Chat mode — click to switch to Plan (Shift+Tab)"
        : "Plan mode — click to switch to Chat (Shift+Tab)"
    }
    aria-label={`Agent mode: ${mode}. Click to switch.`}
  >
    {mode === "chat" ? "Chat" : "Plan"}
  </WorkbenchAction>
);

// ---------------------------------------------------------------------------
// Chat box (mid-session messages + stop while running)
// ---------------------------------------------------------------------------

const ChatBox = ({
  awaitingRequestId,
  awaitingPrompt,
  scopeFormActive,
  disabled,
  isRunning,
  mode,
  onModeChange,
  onSubmit,
  onStop,
}: {
  awaitingRequestId: string | null;
  awaitingPrompt: string | null;
  /** Scope is filled via in-bubble form — hide free-text reply. */
  scopeFormActive?: boolean;
  disabled: boolean;
  /**
   * True while a turn is exclusive (running or waiting_approval) — primary
   * action becomes Stop so a refresh cannot leave the user without a way out.
   */
  isRunning: boolean;
  mode: AgentMode;
  onModeChange: (mode: AgentMode) => void;
  onSubmit: (content: string, requestId: string | null, mode: AgentMode) => Promise<void>;
  onStop: () => Promise<void>;
}): JSX.Element => {
  const [content, setContent] = useState("");
  const [sending, setSending] = useState(false);
  const [stopping, setStopping] = useState(false);

  // While a turn is running the composer is for stop (not a second message).
  // Exception: the agent asked a clarifying question — then send a reply.
  const showStop = isRunning && !awaitingRequestId;
  const handleHistoryKey = useMessageHistory(content, setContent);

  const handleSend = async (): Promise<void> => {
    const trimmed = content.trim();
    if (!trimmed || sending || disabled || showStop) return;
    setSending(true);
    try {
      await onSubmit(trimmed, awaitingRequestId, mode);
      rememberMessage(trimmed);
      setContent("");
    } finally {
      setSending(false);
    }
  };

  const handleStop = async (): Promise<void> => {
    if (stopping || !showStop) return;
    setStopping(true);
    try {
      await onStop();
    } finally {
      setStopping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent): void => {
    if (handleHistoryKey(e as React.KeyboardEvent<HTMLTextAreaElement>)) return;
    if (e.key === "Tab" && e.shiftKey) {
      e.preventDefault();
      onModeChange(nextAgentMode(mode));
      return;
    }
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      if (showStop) {
        void handleStop();
      } else {
        void handleSend();
      }
    }
  };

  // Clarifying questions render in the transcript as the agent answer;
  // the composer only changes placeholder so the user knows to reply.
  const placeholder = showStop
    ? "Turn in progress — stop to send another message (⌘+Enter)"
    : scopeFormActive
      ? "Use the form in the conversation above…"
      : awaitingRequestId
        ? "Reply in the conversation… (⌘+Enter to send)"
        : "Message the agent… (⌘+Enter to send)";

  return (
    <div className={COMPOSER_BAR}>
      <div className={`${COLUMN} ${composerShellClass(mode)}`}>
        <Textarea
          rows={1}
          className={TEXTAREA_CLASS}
          placeholder={placeholder}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled || showStop || Boolean(scopeFormActive)}
          aria-label={
            scopeFormActive
              ? "Scope form is above — fill project and experiment there"
              : awaitingRequestId
                ? awaitingPrompt
                  ? `Reply to agent: ${awaitingPrompt}`
                  : "Reply to the agent's question"
                : undefined
          }
        />
        {showStop ? (
          <WorkbenchIconAction
            label="Stop agent"
            kind="danger"
            onClick={() => {
              void handleStop();
            }}
            disabled={stopping}
            size="default"
            className="flex-none"
            title="Stop this turn"
          >
            {stopping ? (
              <ProgressSpinner label="Stopping" />
            ) : (
              <Square className="h-3 w-3 fill-current" />
            )}
          </WorkbenchIconAction>
        ) : (
          <WorkbenchIconAction
            label="Send message"
            kind="primary"
            onClick={() => {
              void handleSend();
            }}
            disabled={disabled || sending || !content.trim() || Boolean(scopeFormActive)}
            size="default"
            className="flex-none"
          >
            {sending ? <ProgressSpinner label="Sending" /> : <Send className="h-3.5 w-3.5" />}
          </WorkbenchIconAction>
        )}
      </div>
      <div className={`${COLUMN} mt-2 flex items-center gap-2 text-micro text-muted-foreground`}>
        <ModeToggle mode={mode} onChange={onModeChange} />
        <span className="ml-auto hidden sm:inline">
          {scopeFormActive ? "Fill the form above" : "Shift+Tab · ⌘+Enter"}
        </span>
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
      mode: AgentMode;
      instructionsOverride?: string;
    }
  | {
      kind: "skill";
      skillId: string;
      parameters: Record<string, string>;
      mode: AgentMode;
    };

const HELP_TEXT_LINES = [
  "Available commands:",
  "  /plan      — toggle Plan mode for the next turn",
  "  /clear     — clear the input",
  "  /model     — open Agent Settings (provider / models / API keys)",
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
  const [mode, setMode] = useState<AgentMode>("chat");
  const [info, setInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const anchorRef = useRef<HTMLDivElement | null>(null);
  const palette = useCommandPalette();
  const handleHistoryKey = useMessageHistory(description, setDescription);

  // Keep the palette in sync with the textarea content.
  useEffect(() => {
    palette.syncFromValue(description);
  }, [description, palette]);

  const dispatchIntent = useCallback(
    async (intent: LaunchIntent): Promise<void> => {
      setError(null);
      setInfo(null);
      try {
        await onSubmit(intent);
        if (intent.kind === "goal") rememberMessage(intent.description);
        setDescription("");
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
          setMode(nextAgentMode);
          setDescription("");
          setInfo("Plan mode toggled for the next launch.");
          return;
        case "clear":
          setDescription("");
          setMode("chat");
          setInfo("Input cleared.");
          return;
        case "model":
          setDescription("");
          if (onOpenSettings) {
            onOpenSettings();
            setInfo("Opened Agent Settings — choose the model under Provider.");
          } else {
            setInfo("Open Agent Settings → Provider to switch models.");
          }
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
    await dispatchIntent({
      kind: "goal",
      description: trimmed,
      criteria: [],
      mode,
    });
  }, [description, dispatchIntent, mode]);

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
      mode: parsed.planMode || mode === "plan" ? "plan" : "chat",
    });
  }, [description, dispatchIntent, handleBuiltin, mode]);

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
      if (e.key === "Tab" && e.shiftKey) {
        e.preventDefault();
        palette.close();
        setMode(nextAgentMode);
        return;
      }
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
      if (!palette.open && handleHistoryKey(e)) return;
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
    [description, handleHistoryKey, handleSendButton, palette],
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
        <div className={`${COLUMN} mb-2 border-l border-border/60 px-3 py-1 text-label`}>
          <pre className="whitespace-pre-wrap font-mono">{info}</pre>
        </div>
      )}
      {error && (
        <div
          className={`${COLUMN} mb-2 border-y border-destructive/40 bg-destructive/10 px-3 py-2 text-label text-destructive`}
        >
          {error}
        </div>
      )}
      <div className={`${COLUMN} relative`}>
        <div ref={anchorRef} className={composerShellClass(mode)}>
          <Textarea
            ref={textareaRef}
            rows={1}
            className={TEXTAREA_CLASS}
            placeholder={placeholder}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
          />
          <WorkbenchIconAction
            label="Start agent task"
            kind="primary"
            onClick={handleSendButton}
            disabled={disabled || !description.trim()}
            className="h-control w-control flex-none rounded-control"
            aria-label="Start agent task"
          >
            {disabled ? <ProgressSpinner label="Starting" /> : <Send className="h-3.5 w-3.5" />}
          </WorkbenchIconAction>
        </div>
        <CommandPalette state={palette} anchorRef={anchorRef} onPick={handlePaletteSelect} />
      </div>
      <div className={`${COLUMN} mt-2 flex items-center gap-2 text-micro text-muted-foreground`}>
        <ModeToggle mode={mode} onChange={setMode} />
        {onOpenSettings ? (
          <WorkbenchIconAction
            label="Agent settings"
            kind="ghost"
            size="compact"
            className="h-control-compact w-control-compact"
            onClick={onOpenSettings}
            title="Provider, models, API keys, MCP"
          >
            <Settings className="h-3.5 w-3.5" />
          </WorkbenchIconAction>
        ) : null}
        <span className="ml-auto hidden sm:inline">Shift+Tab · ⌘+Enter · / commands</span>
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
  <div className="flex items-start gap-3 border-y border-warning/30 bg-warning-soft px-4 py-3 text-body-lg">
    <ShieldAlert className="mt-1 h-5 w-5 flex-none text-warning" />
    <div className="flex-1 text-warning-foreground">
      <p className="font-medium">Agent not ready</p>
      <p className="mt-1 text-label opacity-90">{health.reason}</p>
    </div>
    <WorkbenchIconAction label="Configure provider" onClick={onOpenSettings}>
      <Settings className="size-4" />
    </WorkbenchIconAction>
  </div>
);

// ---------------------------------------------------------------------------
// Session header
// ---------------------------------------------------------------------------

const HeaderSettingsAction = ({ onOpenSettings }: { onOpenSettings: () => void }): JSX.Element => (
  <WorkbenchIconAction
    label="Agent settings"
    kind="ghost"
    className="h-control-compact w-control-compact"
    onClick={onOpenSettings}
    title="Agent settings"
    aria-label="Agent settings"
  >
    <Settings className="h-4 w-4" />
  </WorkbenchIconAction>
);

const SessionHeader = ({ session }: { session: ApiAgentSession }): JSX.Element => (
  // Markdown-stripped short title (plan tasks carry a curated report title);
  // the raw multi-line goal stays reachable via the hover tooltip.
  <EntityHeader
    icon={Bot}
    title={agentTaskDisplayTitle(session)}
    titleTooltip={session.goal}
    status={session.status}
  />
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
// Loading skeleton — sketches two conversation rows instead of a centered spinner.
// ---------------------------------------------------------------------------

const SessionSkeleton = (): JSX.Element => (
  <div className={`${COLUMN} divide-y divide-border/60 border-y border-border/60 px-4 md:px-8`}>
    {["first", "second"].map((slot) => (
      <div key={slot} className="py-4">
        <div className="pb-3">
          <Skeleton className="h-4 w-2/5" />
        </div>
        <div className="space-y-3">
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
  const mountScope =
    selection.objectType === "agent" && selection.objectId === "new" ? selection.scope : undefined;
  const nav = useNavigationState(snapshot);
  const [session, setSession] = useState<ApiAgentSession | null>(null);
  const [events, setEvents] = useState<ApiSessionEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<ApiAgentHealth | null>(null);
  const [composerMode, setComposerMode] = useState<AgentMode>("chat");
  // PlanMode progress rail selection → which document the right panel shows.
  const [selectedStage, setSelectedStage] = useState<string>(DEFAULT_PLAN_STAGE);
  // Bumped after approve / terminal events so Deliverables re-fetch GET /plans.
  const [planRefreshKey, setPlanRefreshKey] = useState(0);
  // Live artifact kinds from GET /plans — keeps the progress rail honest.
  const [planArtifactKinds, setPlanArtifactKinds] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const esRef = useRef<EventSource | null>(null);
  // Known-entity link index for conversation prose (vision-loop-10).
  const linkIndex = useMemo(() => buildEntityLinkIndex(snapshot), [snapshot]);

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

  // Load session when the selected task changes — not when onRefresh identity
  // changes (that would re-fetch and snap composerMode back to disk activeMode,
  // undoing a Chat↔Plan toggle the user just made).
  // biome-ignore lint/correctness/useExhaustiveDependencies: only re-load on task switch
  useEffect(() => {
    if (!sessionId) {
      setSession(null);
      setEvents([]);
      setComposerMode("chat");
      return;
    }
    setLoading(true);
    setError(null);
    agentApi
      .getSession(sessionId)
      .then((s) => {
        setSession(s);
        setEvents(coalesceStreamEvents(s.events ?? []));
        setComposerMode(s.activeMode ?? (s.planMode ? "plan" : "chat"));
        onRefresh();
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [sessionId]);

  // Live statuses that should keep an event stream open (plan also parks on
  // waiting_approval / awaiting_user — not only chat "running").
  const isLiveStatus =
    session?.status === "running" ||
    session?.status === "waiting_approval" ||
    session?.status === "awaiting_user";

  // SSE stream. Depend on sessionId+status only — including the whole
  // `session` object would resubscribe on every poll tick.
  useEffect(() => {
    if (!sessionId) return;
    if (!isLiveStatus) return;
    const es = agentApi.streamEvents(sessionId);
    esRef.current = es;
    const closeStream = (): void => {
      es.close();
      if (esRef.current === es) esRef.current = null;
    };
    const refreshSession = (): void => {
      agentApi
        .getSession(sessionId)
        .then((s) => {
          setSession((prev) => {
            if (!prev) return s;
            // Match by task id OR session id — plan tasks often use task-… as both,
            // chat uses distinct taskId / sessionId.
            const prevId = getAgentTaskId(prev);
            if (prevId !== sessionId && prev.sessionId !== sessionId) return s;
            return {
              ...prev,
              ...s,
              // Keep whichever id the route used so navigation stays stable.
              taskId: prev.taskId ?? s.taskId,
              sessionId: prev.sessionId || s.sessionId,
            };
          });
          setEvents((current) => {
            const next = coalesceStreamEvents(s.events ?? []);
            return next.length >= current.length || current.length === 0 ? next : current;
          });
        })
        .catch(() => {});
    };
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "done" || data.type === "error") {
          closeStream();
          refreshSession();
          return;
        }
        // Normalize live AgentEvent frames ({kind, timestamp, …}) into the UI's
        // {type, ts, payload} shape; `waiting` (and any control frame) → null.
        const normalized = normalizeStreamFrame(data);
        if (normalized) {
          // Merge consecutive thinking/token deltas — per-token rows thrash
          // React and restart CSS spinners (looks "frozen").
          setEvents((prev) => appendCoalescedEvent(prev, normalized));
        }
      } catch {
        // ignore parse errors
      }
    };
    es.onerror = () => {
      closeStream();
      refreshSession();
    };
    return () => {
      closeStream();
    };
  }, [sessionId, isLiveStatus]);

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

  // Poll while live so plan events written to disk (and status flips to
  // waiting_approval / completed) land even if SSE blips. Fire once immediately
  // — do not wait a full interval after "foo / bar" kicks off the plan.
  useEffect(() => {
    if (!sessionId) return;
    if (!isLiveStatus) return;
    let cancelled = false;
    const tick = async (): Promise<void> => {
      try {
        const fresh = await agentApi.getSession(sessionId);
        if (cancelled) return;
        setEvents((current) => {
          const next = coalesceStreamEvents(fresh.events ?? []);
          // Prefer coalesced server snapshot when it has at least as many
          // non-delta facts; length alone is wrong after coalesce (1 vs 1000).
          if (next.length === 0) return current;
          if (current.length === 0) return next;
          // Keep the previous array identity when content is unchanged so
          // React does not thrash the tree (and restart CSS spinners) every poll.
          if (
            current.length === next.length &&
            current.every((ev, i) => {
              const other = next[i];
              return other != null && sameSessionEvent(ev, other);
            })
          ) {
            return current;
          }
          return next;
        });
        setSession((prev) => {
          if (!prev) return fresh;
          const prevId = getAgentTaskId(prev);
          if (prevId !== sessionId && prev.sessionId !== sessionId) return prev;
          if (
            prev.status === fresh.status &&
            prev.activePlanTaskId === fresh.activePlanTaskId &&
            JSON.stringify(prev.stats ?? null) === JSON.stringify(fresh.stats ?? null)
          ) {
            return prev;
          }
          return {
            ...prev,
            status: fresh.status,
            stats: fresh.stats,
            activeMode: fresh.activeMode,
            activeTurnId: fresh.activeTurnId,
            activePlanTaskId: fresh.activePlanTaskId,
            projectId: fresh.projectId ?? prev.projectId,
            experimentId: fresh.experimentId ?? prev.experimentId,
            runId: fresh.runId ?? prev.runId,
            events: fresh.events,
          };
        });
      } catch {
        // ignore transient polling errors
      }
    };
    void tick();
    const id = setInterval(tick, 1500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [sessionId, isLiveStatus]);

  const handleLaunchIntent = useCallback(
    async (intent: LaunchIntent) => {
      setSubmitting(true);
      setError(null);
      try {
        const created: ApiAgentSession =
          intent.kind === "goal"
            ? await agentApi.createSession(intent.description, intent.criteria, {
                mode: intent.mode,
                instructionsOverride: intent.instructionsOverride,
                projectId: mountScope?.projectId,
                experimentId: mountScope?.experimentId,
                runId: mountScope?.runId,
              })
            : await agentAdminApi.launchSkill(intent.skillId, intent.parameters, {
                mode: intent.mode,
              });
        setSession(created);
        setEvents(coalesceStreamEvents(created.events ?? []));
        setComposerMode(
          created.activeMode ?? (created.planMode || intent.mode === "plan" ? "plan" : "chat"),
        );
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
    [nav, onRefresh, mountScope],
  );

  const handleChatSubmit = useCallback(
    async (content: string, requestId: string | null, mode: AgentMode) => {
      if (!session) return;
      const taskId = getAgentTaskId(session);
      try {
        // Status flip only — transcript events come from the single server
        // source (agent/_tasks events.json). Do not invent a local loop_started
        // (it duplicated server writes and split turns).
        setSession((prev) => (prev ? { ...prev, status: "running" } : prev));
        await agentApi.postMessage(taskId, content, requestId, mode);
        // Reload transcript so server-side errors (start_plan, etc.) show in-chat.
        try {
          const fresh = await agentApi.getSession(taskId);
          setSession(fresh);
          setEvents(coalesceStreamEvents(fresh.events ?? []));
          // Keep composer on the mode the server actually ran (plan vs chat).
          if (fresh.activeMode === "plan" || fresh.activeMode === "chat") {
            setComposerMode(fresh.activeMode);
          } else if (mode === "plan" || mode === "chat") {
            setComposerMode(mode);
          }
        } catch {
          onRefresh();
        }
      } catch (err) {
        setSession((prev) => (prev ? { ...prev, status: session.status } : prev));
        const raw = err instanceof Error ? err.message : String(err);
        const conflict = /already in flight|409|Conflict/i.test(raw) && !/stop/i.test(raw);
        setError(
          conflict
            ? `${raw}\n\nA turn is still active on this task. Press Stop, then send again.`
            : raw,
        );
        // Pull any error events the server recorded before raising.
        // Also re-sync status so Stop appears for running / waiting_approval.
        try {
          const fresh = await agentApi.getSession(taskId);
          setSession(fresh);
          setEvents(coalesceStreamEvents(fresh.events ?? []));
        } catch {
          /* keep optimistic transcript */
        }
      }
    },
    [session, onRefresh],
  );

  const handleStop = useCallback(async () => {
    if (!session) return;
    const taskId = getAgentTaskId(session);
    try {
      await agentApi.cancelSession(taskId);
      // Keep the accumulated transcript — getSession only returns the current
      // turn's events, so merging status/stats preserves the full conversation.
      setSession((prev) => {
        if (!prev) return prev;
        return { ...prev, status: "cancelled" };
      });
      onRefresh();
    } catch (err) {
      setError(String(err));
    }
  }, [session, onRefresh]);

  // Detect whether the agent is currently waiting on the user's reply.
  const pendingUserRequest = useMemo(() => derivePendingUserRequest(events), [events]);
  const scopeFormActive = pendingUserRequest?.contextKind === "experiment";

  const handleScopeSubmit = useCallback(
    async (scope: string) => {
      await handleChatSubmit(scope, pendingUserRequest?.requestId ?? null, "plan");
    },
    [handleChatSubmit, pendingUserRequest?.requestId],
  );

  // Plan products (board → workflow source) live in the right panel, not chat.
  // Computed before the loading/session early returns so the polling effect
  // below keeps a stable hook order (no hooks after conditional returns).
  const planRef = session
    ? derivePlanRef(events, {
        runId: session.runId,
        projectId: session.projectId,
        experimentId: session.experimentId,
        title: session.title ?? session.goal,
      })
    : null;

  // Poll plan artifacts while a plan run is live so the rail / deliverables
  // advance even when the event stream only carries stage_started breadcrumbs.
  // biome-ignore lint/correctness/useExhaustiveDependencies: selectedStage read intentionally not a dep
  useEffect(() => {
    if (loading || !planRef?.projectId || !planRef.experimentId || !planRef.runId) {
      setPlanArtifactKinds([]);
      return;
    }
    let cancelled = false;
    const pull = (): void => {
      void workspaceApi
        .getPlan(planRef.projectId, planRef.experimentId, planRef.runId)
        .then((detail) => {
          if (cancelled) return;
          const kinds = detail.artifactKinds ?? [];
          setPlanArtifactKinds(kinds);
          // Prefer the newest completed stage when the user hasn't clicked away
          // from the default, so Approve → workflow source is visible.
          if (kinds.length > 0) {
            const last = [...kinds]
              .reverse()
              .find((k) => PLAN_STAGES.some((s) => s.kind === k && !s.executeTail));
            if (last && selectedStage === DEFAULT_PLAN_STAGE) {
              setSelectedStage(last);
            }
          }
        })
        .catch(() => {
          /* plan may not be readable yet */
        });
    };
    pull();
    const live =
      session?.status === "running" || session?.status === "waiting_approval" || planRefreshKey > 0;
    if (!live) return;
    const id = window.setInterval(pull, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [
    loading,
    planRef?.projectId,
    planRef?.experimentId,
    planRef?.runId,
    session?.status,
    planRefreshKey,
  ]);

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
              <div className="flex items-center justify-between gap-3 border-y border-destructive/40 bg-destructive/10 px-4 py-2 text-body-lg text-destructive">
                <span className="flex-1">{error}</span>
                {notReady && (
                  <WorkbenchIconAction label="Open agent settings" onClick={openSettings}>
                    <Settings className="size-4" />
                  </WorkbenchIconAction>
                )}
              </div>
            )}

            <ApprovalsInbox
              variant="list"
              onOpenItem={(item) =>
                nav.setSelection({ objectType: "agent", objectId: item.taskId })
              }
              onDecided={onRefresh}
            />

            <div className="flex flex-col items-center gap-2 pt-4 text-center">
              <Bot className="h-5 w-5 text-muted-foreground" />
              <h2 className="text-base font-semibold text-foreground">Start an agent task</h2>
              <p className="max-w-md text-body-lg text-muted-foreground">
                <strong className="font-medium text-foreground">Chat</strong> explores and runs
                scratch scripts under{" "}
                <InlineCode className="text-label">agent/.scratch/</InlineCode> (no default
                project/run land). <strong className="font-medium text-foreground">Plan</strong>{" "}
                builds a reviewable workflow graph. Switch with Shift+Tab.
              </p>
              {mountScope && (
                <span className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-muted/40 px-3 py-1 text-label text-muted-foreground">
                  Mounted on{" "}
                  <span className="font-medium text-foreground">
                    {mountScope.runId
                      ? `run ${mountScope.runId}`
                      : mountScope.experimentId
                        ? `experiment ${mountScope.experimentId}`
                        : `project ${mountScope.projectId}`}
                  </span>
                </span>
              )}
            </div>

            {recent.length > 0 && (
              <div className="space-y-2">
                <p className="px-1 text-label font-medium text-muted-foreground">Recent tasks</p>
                <div className="divide-y divide-border/60 border-y border-border/60">
                  {recent.map((s) => (
                    <WorkbenchAction
                      kind="ghost"
                      size="content"
                      key={s.id}
                      type="button"
                      onClick={() => nav.setSelection({ objectType: "agent", objectId: s.id })}
                      className="flex w-full items-center gap-3 px-3 py-2 text-left transition-colors hover:bg-muted/40"
                    >
                      <Bot className="h-4 w-4 flex-none text-muted-foreground" />
                      <p className="flex-1 truncate text-body-lg">{s.goal}</p>
                      <StatusBadge status={s.status} size="sm" />
                    </WorkbenchAction>
                  ))}
                </div>
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

  if (!session) return null;

  if (loading) {
    return (
      <div className="flex h-full flex-col bg-background">
        <SessionSkeleton />
      </div>
    );
  }

  if (!session) return null;

  // Stop for any exclusive turn — including plan gates. Sending is blocked
  // server-side for these statuses; without Stop the UI looks dead after refresh.
  const isRunning = session.status === "running" || session.status === "waiting_approval";
  const turns = groupEventsIntoTurns(events, session.goal);
  const showSplit = hasDeliverables(events) || planRef !== null;

  const refreshAfterPlanDecision = (): void => {
    onRefresh();
    setPlanRefreshKey((k) => k + 1);
    setSelectedStage("bound_workflow");
    // Realization continues after approve — poll so workflow_source lands.
    const id = getAgentTaskId(session);
    let n = 0;
    const tick = (): void => {
      void agentApi.getSession(id).then((s) => {
        setSession(s);
        setEvents(coalesceStreamEvents(s.events ?? []));
        setPlanRefreshKey((k) => k + 1);
      });
      n += 1;
      if (n < 12) window.setTimeout(tick, 2000);
    };
    tick();
  };

  const errorBanner = error ? (
    <div
      className={`${COLUMN} mx-4 mt-3 flex items-start gap-2 border border-destructive/40 bg-destructive/10 px-3 py-2 text-body-lg text-destructive md:mx-6`}
      role="alert"
    >
      <XCircle className="mt-0.5 h-4 w-4 flex-none" />
      <p className="min-w-0 flex-1 whitespace-pre-wrap [overflow-wrap:anywhere]">{error}</p>
      <WorkbenchDismissAction onClick={() => setError(null)} />
    </div>
  ) : null;

  const conversation = (
    <div className="flex h-full min-h-0 flex-col">
      {errorBanner}
      <ScrollArea className="min-h-0 flex-1" ref={scrollRef as React.RefObject<HTMLDivElement>}>
        <div className={`${COLUMN} flex flex-col gap-6 px-4 pb-6 pt-4 md:px-6`}>
          {turns.map((turn, idx) => (
            <ConversationTurnView
              key={turn.key}
              turn={turn}
              linkIndex={linkIndex}
              interactiveClarification={Boolean(scopeFormActive) && idx === turns.length - 1}
              onScopeSubmit={scopeFormActive ? handleScopeSubmit : undefined}
              showLandDecision={
                !isRunning &&
                idx === turns.length - 1 &&
                !turn.inProgress &&
                session.activeMode !== "plan"
              }
              onLandDecide={!isRunning ? (msg) => handleChatSubmit(msg, null, "chat") : undefined}
              planDecisionTaskId={
                // Show decide strip on the plan answer whenever we are (or were)
                // at the review gate — bar hides itself if no pending item.
                idx === turns.length - 1 && turn.result?.type === "plan_emitted"
                  ? getAgentTaskId(session)
                  : null
              }
              planDecisionRunId={
                idx === turns.length - 1 && turn.result?.type === "plan_emitted"
                  ? (session.runId ?? planRef?.runId ?? null)
                  : null
              }
              onPlanDecided={refreshAfterPlanDecision}
            />
          ))}
        </div>
      </ScrollArea>

      {/* Always continue the same session — GoalInput (create) is only for /new. */}
      <ChatBox
        awaitingRequestId={pendingUserRequest?.requestId ?? null}
        awaitingPrompt={pendingUserRequest?.prompt ?? null}
        scopeFormActive={scopeFormActive}
        disabled={false}
        isRunning={isRunning}
        mode={composerMode}
        onModeChange={setComposerMode}
        onSubmit={handleChatSubmit}
        onStop={handleStop}
      />
    </div>
  );

  return (
    <div className="flex h-full flex-col bg-background">
      <SessionHeader session={session} />

      {showSplit ? (
        <div className="flex min-h-0 flex-1">
          {planRef ? (
            <PlanProgressRail
              events={events}
              status={session.status}
              selectedKind={selectedStage}
              onSelectStage={setSelectedStage}
              artifactKinds={planArtifactKinds}
            />
          ) : null}
          <ResizablePanelGroup
            direction="horizontal"
            autoSaveId="agent-session-split"
            className="min-h-0 flex-1"
          >
            <ResizablePanel defaultSize={58} minSize={38}>
              {conversation}
            </ResizablePanel>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={42} minSize={26}>
              <DeliverablesPanel
                events={events}
                activeStageKind={selectedStage}
                planFallback={planRef}
                refreshKey={planRefreshKey}
              />
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>
      ) : (
        <div className="min-h-0 flex-1">{conversation}</div>
      )}
    </div>
  );
};

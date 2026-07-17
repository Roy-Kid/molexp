import {
  Bot,
  ChevronDown,
  ChevronRight,
  Cpu,
  HelpCircle,
  Loader2,
  Send,
  Settings,
  ShieldAlert,
  Square,
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
import { Button } from "@/components/ui/button";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { agentTaskDisplayTitle } from "@/lib/agent-task-title";
import { buildEntityLinkIndex } from "@/lib/entity-linkify";
import { AgentSettingsViewer } from "./AgentSettingsViewer";
import { ApprovalsInbox } from "./agent/ApprovalsInbox";
import { type AgentMode, nextAgentMode } from "./agent/agentMode";
import { ConversationTurnView } from "./agent/conversation";
import { DeliverablesPanel, hasDeliverables } from "./agent/DeliverablesPanel";
import { PlanProgressRail } from "./agent/PlanProgressRail";
import { DEFAULT_PLAN_STAGE } from "./agent/planStages";
import {
  derivePendingUserRequest,
  derivePlanRef,
  groupEventsIntoTurns,
  normalizeStreamFrame,
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

const ModeToggle = ({
  mode,
  onChange,
}: {
  mode: AgentMode;
  onChange: (mode: AgentMode) => void;
}) => (
  <button
    type="button"
    onClick={() => onChange(nextAgentMode(mode))}
    className={
      "rounded-md px-1.5 py-0.5 text-[10px] font-medium transition-colors " +
      (mode === "plan"
        ? "bg-primary/10 text-primary hover:bg-primary/15"
        : "hover:bg-muted hover:text-foreground")
    }
    title="Switch agent mode (Shift+Tab)"
    aria-label={`Agent mode: ${mode}`}
  >
    {mode === "chat" ? "Chat" : "Plan"}
  </button>
);

// ---------------------------------------------------------------------------
// Chat box (mid-session messages + stop while running)
// ---------------------------------------------------------------------------

const ChatBox = ({
  awaitingRequestId,
  awaitingPrompt,
  disabled,
  isRunning,
  mode,
  onModeChange,
  onSubmit,
  onStop,
}: {
  awaitingRequestId: string | null;
  awaitingPrompt: string | null;
  disabled: boolean;
  /** True while a turn is in flight — primary action becomes Stop. */
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

  const placeholder = showStop
    ? "Agent is running… (⌘+Enter to stop)"
    : awaitingRequestId
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
          disabled={disabled || showStop}
        />
        {showStop ? (
          <Button
            size="icon"
            variant="destructive"
            onClick={() => {
              void handleStop();
            }}
            disabled={stopping}
            className="h-8 w-8 flex-none rounded-md"
            aria-label="Stop agent"
            title="Stop this turn"
          >
            {stopping ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Square className="h-3 w-3 fill-current" />
            )}
          </Button>
        ) : (
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
        )}
      </div>
      <div className={`${COLUMN} mt-1.5 flex items-center gap-2 text-[11px] text-muted-foreground`}>
        <ModeToggle mode={mode} onChange={onModeChange} />
        <span className="ml-auto">Shift+Tab to switch · ⌘+Enter to send</span>
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
  "  /plan      — toggle the auditable nine-step Plan agent for the next turn",
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
  const [mode, setMode] = useState<AgentMode>("chat");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [info, setInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [providerLabel, setProviderLabel] = useState<string>("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const anchorRef = useRef<HTMLDivElement | null>(null);
  const palette = useCommandPalette();
  const handleHistoryKey = useMessageHistory(description, setDescription);

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
        if (intent.kind === "goal") rememberMessage(intent.description);
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
          setMode(nextAgentMode);
          setDescription("");
          setInfo("Plan mode toggled for the next launch.");
          return;
        case "clear":
          setDescription("");
          setOverrideText("");
          setMode("chat");
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
      mode,
      instructionsOverride: override,
    });
  }, [description, dispatchIntent, overrideText, mode]);

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
        <ModeToggle mode={mode} onChange={setMode} />
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
        <span className="ml-auto">Shift+Tab to switch · ⌘+Enter to send · / for commands</span>
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
  // Which PlanMode stage the progress rail has selected; drives the right panel.
  const [selectedStage, setSelectedStage] = useState<string>(DEFAULT_PLAN_STAGE);
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
        setComposerMode(s.activeMode ?? "chat");
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
    const closeStream = (): void => {
      es.close();
      if (esRef.current === es) esRef.current = null;
    };
    const refreshSession = (): void => {
      agentApi
        .getSession(sessionId)
        .then((s) => {
          setSession((prev) => {
            if (!prev || getAgentTaskId(prev) !== sessionId) return s;
            return { ...prev, status: s.status, stats: s.stats };
          });
          setEvents(s.events ?? []);
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
          setEvents((prev) => {
            const duplicate = prev.some(
              (event) =>
                event.type === normalized.type &&
                event.ts === normalized.ts &&
                JSON.stringify(event.payload) === JSON.stringify(normalized.payload),
            );
            return duplicate ? prev : [...prev, normalized];
          });
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
        setEvents((current) =>
          (fresh.events?.length ?? 0) >= current.length ? (fresh.events ?? current) : current,
        );
        setSession((prev) => {
          if (!prev || prev.sessionId !== sessionId) return prev;
          if (
            prev.status === fresh.status &&
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
          };
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
                mode: intent.mode,
                instructionsOverride: intent.instructionsOverride,
                projectId: mountScope?.projectId,
                experimentId: mountScope?.experimentId,
                runId: mountScope?.runId,
              })
            : await agentAdminApi.launchSkill(intent.skillId, intent.parameters, {
                planMode: intent.mode === "plan",
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
    [nav, onRefresh, mountScope],
  );

  const handleChatSubmit = useCallback(
    async (content: string, requestId: string | null, mode: AgentMode) => {
      if (!session) return;
      const taskId = getAgentTaskId(session);
      try {
        // Optimistically show the user's message and flip status so the SSE
        // stream re-subscribes for the new turn (postMessage continues the
        // same session — it does NOT create a new task).
        const now = new Date().toISOString();
        setEvents((prev) => [
          ...prev,
          {
            type: "loop_started",
            ts: now,
            payload: { user_input: content, kind: "loop_started", mode },
          },
        ]);
        setSession((prev) => (prev ? { ...prev, status: "running" } : prev));
        await agentApi.postMessage(taskId, content, requestId, mode);
        onRefresh();
      } catch (err) {
        setSession((prev) => (prev ? { ...prev, status: session.status } : prev));
        setError(String(err));
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

            <ApprovalsInbox onDecided={onRefresh} />

            <div className="flex flex-col items-center gap-2 pt-4 text-center">
              <div className="flex h-10 w-10 items-center justify-center rounded-md border border-border/60 bg-card">
                <Bot className="h-5 w-5 text-muted-foreground" />
              </div>
              <h2 className="text-base font-semibold text-foreground">Start an agent task</h2>
              <p className="max-w-md text-sm text-muted-foreground">
                Describe a goal, then switch between Chat and the auditable nine-step Plan agent for
                each turn.
              </p>
              {mountScope && (
                <span className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-muted/40 px-3 py-1 text-xs text-muted-foreground">
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
  // Pull the agent's products (plan/spec/script, or chat artifacts) into a
  // dedicated panel; a session with no products stays a single conversation
  // column rather than being squeezed into a half-width view.
  const showSplit = hasDeliverables(events);
  // A PlanMode session also gets a left progress rail tracking its pipeline.
  const planRef = derivePlanRef(events);

  const conversation = (
    <div className="flex h-full min-h-0 flex-col">
      <ScrollArea className="min-h-0 flex-1" ref={scrollRef as React.RefObject<HTMLDivElement>}>
        <div className={`${COLUMN} flex flex-col gap-5 px-4 pb-6 pt-4 md:px-6`}>
          {turns.map((turn) => (
            <ConversationTurnView key={turn.key} turn={turn} linkIndex={linkIndex} />
          ))}
        </div>
      </ScrollArea>

      {session.status === "waiting_approval" && (
        <div className={`${COLUMN} px-4 pb-2 md:px-6`}>
          <ApprovalsInbox
            taskId={session.activePlanTaskId ?? getAgentTaskId(session)}
            onDecided={onRefresh}
          />
        </div>
      )}

      {/* Always continue the same session — GoalInput (create) is only for /new. */}
      <ChatBox
        awaitingRequestId={pendingUserRequest?.requestId ?? null}
        awaitingPrompt={pendingUserRequest?.prompt ?? null}
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
          {planRef && (
            <PlanProgressRail
              events={events}
              status={session.status}
              selectedKind={selectedStage}
              onSelectStage={setSelectedStage}
            />
          )}
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
              <DeliverablesPanel events={events} activeStageKind={selectedStage} />
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>
      ) : (
        <div className="min-h-0 flex-1">{conversation}</div>
      )}
    </div>
  );
};

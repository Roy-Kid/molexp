import {
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Loader2,
  Play,
  RotateCcw,
  Send,
  ShieldAlert,
  Sparkles,
  Terminal,
  Workflow,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { EmptyState, EntityHeader, EntityMetric, StatusBadge } from "@/app/components/entity";
import { agentApi } from "@/app/state/api";
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
import { Separator } from "@/components/ui/separator";

// ---------------------------------------------------------------------------
// Event row
// ---------------------------------------------------------------------------

const EVENT_META: Record<string, { icon: typeof Bot; label: string; colorClass: string }> = {
  PlanCreatedEvent: { icon: Sparkles, label: "Plan created", colorClass: "text-violet-500" },
  ToolCallEvent: { icon: Terminal, label: "Tool call", colorClass: "text-blue-500" },
  ToolResultEvent: { icon: CheckCircle2, label: "Tool result", colorClass: "text-green-600" },
  WorkflowStartedEvent: { icon: Workflow, label: "Workflow started", colorClass: "text-sky-500" },
  ObservationEvent: { icon: Bot, label: "Observation", colorClass: "text-muted-foreground" },
  ReplanEvent: { icon: RotateCcw, label: "Replanning", colorClass: "text-amber-500" },
  ApprovalRequestEvent: {
    icon: ShieldAlert,
    label: "Approval needed",
    colorClass: "text-orange-500",
  },
  SessionCompletedEvent: { icon: CheckCircle2, label: "Completed", colorClass: "text-emerald-500" },
};

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

const EventRow = ({
  event,
  sessionId: _sessionId,
  onApprovalRespond,
}: {
  event: ApiSessionEvent;
  sessionId: string;
  onApprovalRespond: (requestId: string, approved: boolean) => void;
}): JSX.Element => {
  const [expanded, setExpanded] = useState(false);
  const meta = EVENT_META[event.type] ?? {
    icon: Bot,
    label: event.type,
    colorClass: "text-muted-foreground",
  };
  const Icon = meta.icon;
  const hasDetail = Object.keys(event.payload).length > 0;

  return (
    <div className="group flex gap-3 py-2">
      <div className={`mt-0.5 flex-none ${meta.colorClass}`}>
        <Icon className="h-4 w-4" />
      </div>
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{meta.label}</span>
          {event.type === "ToolCallEvent" && event.payload.tool_name && (
            <Badge variant="secondary" className="font-mono text-[10px] h-4 px-1">
              {String(event.payload.tool_name)}
            </Badge>
          )}
          {event.type === "WorkflowStartedEvent" && event.payload.run_id && (
            <Badge variant="secondary" className="font-mono text-[10px] h-4 px-1">
              {String(event.payload.run_id)}
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
        {event.type === "PlanCreatedEvent" && Array.isArray(event.payload.plan_steps) && (
          <ol className="ml-2 space-y-0.5">
            {(event.payload.plan_steps as string[]).map((step, stepNum) => (
              <li
                key={`plan-step-${step.slice(0, 40)}`}
                className="flex gap-2 text-xs text-muted-foreground"
              >
                <span className="font-mono text-primary/60">{stepNum + 1}.</span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
        )}

        {event.type === "ObservationEvent" && event.payload.content && (
          <p className="text-xs text-muted-foreground">{String(event.payload.content)}</p>
        )}

        {event.type === "ReplanEvent" && (
          <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 dark:border-amber-800 dark:bg-amber-950/30">
            {event.payload.reason && (
              <p className="text-xs text-amber-700 dark:text-amber-400">
                {String(event.payload.reason)}
              </p>
            )}
            {Array.isArray(event.payload.new_plan) &&
              (event.payload.new_plan as string[]).map((step, stepNum) => (
                <p
                  key={`replan-step-${step.slice(0, 40)}`}
                  className="text-xs text-amber-700 dark:text-amber-400"
                >
                  {stepNum + 1}. {step}
                </p>
              ))}
          </div>
        )}

        {event.type === "SessionCompletedEvent" && (
          <div className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 dark:border-emerald-800 dark:bg-emerald-950/30">
            {event.payload.summary && (
              <p className="text-xs text-emerald-700 dark:text-emerald-400">
                {String(event.payload.summary)}
              </p>
            )}
          </div>
        )}

        {event.type === "ApprovalRequestEvent" && event.payload.request_id && (
          <div className="flex items-center gap-2 rounded-md border border-orange-200 bg-orange-50 px-3 py-2 dark:border-orange-800 dark:bg-orange-950/30">
            <p className="flex-1 text-xs text-orange-700 dark:text-orange-400">
              Approve{" "}
              <span className="font-mono font-semibold">
                {String(event.payload.tool_name ?? "action")}
              </span>
              ?
            </p>
            <Button
              size="sm"
              variant="outline"
              className="h-6 border-orange-400 text-orange-700 hover:bg-orange-100"
              onClick={() => onApprovalRespond(String(event.payload.request_id), false)}
            >
              Deny
            </Button>
            <Button
              size="sm"
              className="h-6 bg-orange-500 text-white hover:bg-orange-600"
              onClick={() => onApprovalRespond(String(event.payload.request_id), true)}
            >
              Approve
            </Button>
          </div>
        )}

        {expanded && hasDetail && (
          <pre className="overflow-x-auto rounded-md bg-muted/60 px-3 py-2 text-[11px] font-mono text-muted-foreground">
            {JSON.stringify(event.payload, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Goal input form
// ---------------------------------------------------------------------------

const GoalInput = ({
  onSubmit,
  disabled,
}: {
  onSubmit: (description: string, criteria: string[]) => void;
  disabled: boolean;
}): JSX.Element => {
  const [description, setDescription] = useState("");
  const [criteriaText, setCriteriaText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (): void => {
    const trimmed = description.trim();
    if (!trimmed) return;
    const criteria = criteriaText
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    onSubmit(trimmed, criteria);
    setDescription("");
    setCriteriaText("");
  };

  const handleKeyDown = (e: React.KeyboardEvent): void => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="space-y-3 rounded-xl border border-border bg-card p-4">
      <div className="flex items-center gap-2 text-sm font-semibold">
        <Sparkles className="h-4 w-4 text-violet-500" />
        New Goal
      </div>
      <textarea
        ref={textareaRef}
        className="w-full resize-none rounded-lg border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
        rows={3}
        placeholder="Describe your goal... (⌘+Enter to submit)"
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
      />
      <textarea
        className="w-full resize-none rounded-lg border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
        rows={2}
        placeholder="Success criteria (one per line, optional)"
        value={criteriaText}
        onChange={(e) => setCriteriaText(e.target.value)}
        disabled={disabled}
      />
      <div className="flex justify-end">
        <Button disabled={disabled || !description.trim()} onClick={handleSubmit} className="gap-2">
          {disabled ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          {disabled ? "Submitting..." : "Start Session"}
        </Button>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Session header
// ---------------------------------------------------------------------------

const SessionHeader = ({
  session,
  eventCount,
  snapshot,
}: {
  session: ApiAgentSession;
  eventCount: number;
  snapshot: WorkspaceSnapshot;
}): JSX.Element => {
  const toolCallCount = session.events.filter((event) => event.type === "ToolCallEvent").length;
  const { breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);
  return (
    <EntityHeader
      breadcrumbs={breadcrumbs}
      canNavigateUp={canNavigateUp}
      onNavigateUp={navigateUp}
      icon={Bot}
      title={session.goalDescription}
      status={session.status}
      metrics={
        <>
          <EntityMetric label="Events" value={eventCount} />
          <EntityMetric label="Tool Calls" value={toolCallCount} />
        </>
      }
    />
  );
};

// ---------------------------------------------------------------------------
// Main AgentViewer
// ---------------------------------------------------------------------------

export const AgentViewer = ({
  selection,
  snapshot,
  onRefresh,
}: RendererProps): JSX.Element | null => {
  const sessionId = selection.objectId === "new" ? null : selection.objectId;
  const [session, setSession] = useState<ApiAgentSession | null>(null);
  const [events, setEvents] = useState<ApiSessionEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const esRef = useRef<EventSource | null>(null);

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
        setEvents(s.events);
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [sessionId]);

  // SSE stream for running sessions
  useEffect(() => {
    if (!session || session.status !== "running") return;
    const es = agentApi.streamEvents(session.sessionId);
    esRef.current = es;
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "done") {
          es.close();
          agentApi.getSession(session.sessionId).then((s) => {
            setSession(s);
            setEvents(s.events);
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
  }, [session?.sessionId, session?.status, session]);

  // Auto-scroll to bottom when events change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, []);

  const handleGoalSubmit = useCallback(
    async (description: string, criteria: string[]) => {
      setSubmitting(true);
      setError(null);
      try {
        const created = await agentApi.createSession(description, criteria);
        setSession(created);
        setEvents(created.events);
        onRefresh();
      } catch (err) {
        setError(String(err));
      } finally {
        setSubmitting(false);
      }
    },
    [onRefresh],
  );

  const handleApprovalRespond = useCallback(
    async (requestId: string, approved: boolean) => {
      if (!session) return;
      try {
        await agentApi.respondApproval(session.sessionId, requestId, approved);
      } catch (err) {
        setError(String(err));
      }
    },
    [session],
  );

  // --- "new" state: show goal input ---
  if (!sessionId || (!loading && !session)) {
    return (
      <div className="flex h-full flex-col gap-4 overflow-auto p-4">
        <GoalInput onSubmit={handleGoalSubmit} disabled={submitting} />
        {error && (
          <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/30 dark:text-red-400">
            {error}
          </div>
        )}
        {snapshot.agentSessions.length > 0 && (
          <div className="space-y-1">
            <p className="px-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Recent sessions
            </p>
            {snapshot.agentSessions.slice(0, 5).map((s) => (
              <div
                key={s.id}
                className="flex items-center gap-3 rounded-lg border border-border/60 bg-background px-3 py-2 text-left hover:bg-muted/40"
              >
                <Bot className="h-4 w-4 flex-none text-violet-400" />
                <p className="flex-1 truncate text-sm">{s.goalDescription}</p>
                <StatusBadge status={s.status} size="sm" />
              </div>
            ))}
          </div>
        )}
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

  return (
    <div className="flex h-full flex-col bg-background">
      <SessionHeader session={session} eventCount={events.length} snapshot={snapshot} />

      <div className="flex items-center gap-2 border-b border-border/70 bg-muted/10 px-6 py-2 md:px-8">
        <p className="flex-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Event stream
        </p>
        {session.status === "running" && (
          <span className="flex items-center gap-1 text-xs text-blue-500">
            <Loader2 className="h-3 w-3 animate-spin" />
            Live
          </span>
        )}
      </div>

      <ScrollArea className="flex-1" ref={scrollRef as React.RefObject<HTMLDivElement>}>
        {events.length === 0 ? (
          <EmptyState title="Waiting for events…" icon={<Play className="h-8 w-8" />} />
        ) : (
          <div className="space-y-1 px-6 pb-4 pr-6 md:px-8">
            {events.map((event, eventIdx) => (
              <div key={`event-${event.type}-${event.ts}`}>
                <EventRow
                  event={event}
                  sessionId={session.sessionId}
                  onApprovalRespond={handleApprovalRespond}
                />
                {eventIdx < events.length - 1 && <Separator className="opacity-30" />}
              </div>
            ))}
          </div>
        )}
      </ScrollArea>

      {session.status === "completed" && (
        <div className="border-t border-border/70 p-4">
          <GoalInput onSubmit={handleGoalSubmit} disabled={submitting} />
        </div>
      )}
    </div>
  );
};

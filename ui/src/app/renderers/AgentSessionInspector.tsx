/**
 * AgentSessionInspector — right-panel details for an agent session.
 *
 * Hosts the token/usage statistics that previously lived in the center
 * panel as a stats strip, plus session metadata (status, goal, timing).
 * Polls the session while it is running so figures stay live without
 * cluttering the main workspace.
 */

import { Bot, ChevronRight, FileText, Lock, Slash } from "lucide-react";
import type { JSX } from "react";
import { useEffect, useMemo, useState } from "react";
import type { SessionStatsResponse } from "@/api/generated";
import { isLegacySession, legacyBadgeMeta } from "@/app/renderers/agent_session/inspectorHelpers";
import { type ApiAgentSystemPrompt, agentApi, planApi } from "@/app/state/api";
import type { ApiAgentSession, RendererProps } from "@/app/types";
import { Badge } from "@/components/ui/badge";

const COMPACT_NUMBER = new Intl.NumberFormat(undefined, {
  notation: "compact",
  maximumFractionDigits: 1,
});

const formatTokens = (n: number | null | undefined): string => {
  if (!n || n <= 0) return "0";
  return COMPACT_NUMBER.format(n);
};

const formatDuration = (seconds: number | null | undefined): string => {
  if (seconds == null || Number.isNaN(seconds) || seconds < 0) return "—";
  if (seconds < 1) return "<1s";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds - m * 60);
  if (m < 60) return s ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m - h * 60}m`;
};

interface DetailRow {
  label: string;
  value: string;
  hint?: string;
}

const buildStatRows = (
  stats: SessionStatsResponse,
  isRunning: boolean,
  liveDuration: number | null,
): DetailRow[] => {
  const inputTokens = stats.inputTokens ?? 0;
  const outputTokens = stats.outputTokens ?? 0;
  const cacheRead = stats.cacheReadTokens ?? 0;
  const cacheWrite = stats.cacheWriteTokens ?? 0;
  const totalTokens = stats.totalTokens ?? inputTokens + outputTokens;

  const rows: DetailRow[] = [
    {
      label: "Tokens (total)",
      value: formatTokens(totalTokens),
      hint: totalTokens.toLocaleString(),
    },
    { label: "Input", value: formatTokens(inputTokens), hint: inputTokens.toLocaleString() },
    { label: "Output", value: formatTokens(outputTokens), hint: outputTokens.toLocaleString() },
  ];

  if (cacheRead + cacheWrite > 0) {
    rows.push({
      label: "Cache",
      value: formatTokens(cacheRead + cacheWrite),
      hint: `read ${cacheRead.toLocaleString()} · write ${cacheWrite.toLocaleString()}`,
    });
  }

  rows.push(
    { label: "Requests", value: String(stats.requests ?? 0) },
    { label: "Tool calls", value: String(stats.toolCalls ?? 0) },
    { label: "Events", value: String(stats.events ?? 0) },
    {
      label: isRunning ? "Elapsed" : "Duration",
      value: formatDuration(liveDuration),
      hint: stats.startedAt ? `Started ${new Date(stats.startedAt).toLocaleString()}` : undefined,
    },
  );
  return rows;
};

const buildSessionRows = (session: ApiAgentSession): DetailRow[] => [
  { label: "Task ID", value: session.taskId ?? session.sessionId },
  { label: "Runtime Session", value: session.sessionId },
  { label: "Goal", value: session.goalDescription || "—" },
  { label: "Created", value: session.createdAt || "—" },
];

export const AgentSessionInspector = (props: RendererProps): JSX.Element => {
  const { selection } = props;
  const sessionId =
    selection.objectType === "agent" &&
    selection.objectId !== "new" &&
    selection.objectId !== "settings"
      ? selection.objectId
      : null;

  const [session, setSession] = useState<ApiAgentSession | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) {
      setSession(null);
      return;
    }
    let cancelled = false;
    const load = async (): Promise<void> => {
      try {
        const fresh = await agentApi.getSession(sessionId);
        if (!cancelled) setSession(fresh);
      } catch (err) {
        if (!cancelled) setError(String(err));
      }
    };
    void load();
    const id = setInterval(() => {
      if (cancelled) return;
      void load();
    }, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [sessionId]);

  const isRunning = session?.status === "running";
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!isRunning || !session?.stats?.startedAt) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [isRunning, session?.stats?.startedAt]);

  const liveDuration = useMemo(() => {
    const stats = session?.stats;
    if (!stats) return null;
    if (stats.durationSeconds != null && !isRunning) return stats.durationSeconds;
    if (!stats.startedAt) return stats.durationSeconds ?? null;
    const startedMs = new Date(stats.startedAt).getTime();
    if (Number.isNaN(startedMs)) return stats.durationSeconds ?? null;
    return Math.max(0, (now - startedMs) / 1000);
  }, [session?.stats, isRunning, now]);

  const stats = session?.stats;
  const statsRows: DetailRow[] = stats ? buildStatRows(stats, isRunning, liveDuration) : [];
  const sessionRows: DetailRow[] = session ? buildSessionRows(session) : [];

  const legacy = isLegacySession(session);
  const legacyMeta = legacy ? legacyBadgeMeta() : null;

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex items-center justify-between border-b border-border/70 bg-muted/20 px-3 py-1.5">
        <h2 className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          <Bot className="h-3.5 w-3.5" /> Task details
        </h2>
        {legacyMeta ? (
          <Badge
            variant="outline"
            className="h-5 gap-1 px-1.5 text-[10px] uppercase tracking-wide"
            title={legacyMeta.tooltip}
          >
            <Lock className="h-3 w-3" /> {legacyMeta.label}
          </Badge>
        ) : session?.status ? (
          <Badge variant="secondary" className="h-5 px-1.5 text-[10px] uppercase tracking-wide">
            {session.status}
          </Badge>
        ) : null}
      </div>

      <div className="flex-1 overflow-auto">
        {!sessionId && <p className="px-3 py-2 text-xs text-muted-foreground">No task selected.</p>}
        {error && <p className="px-3 py-2 text-xs text-destructive">{error}</p>}

        {session && <ModeSection session={session} />}
        {sessionRows.length > 0 && <Section title="Task" rows={sessionRows} />}
        {statsRows.length > 0 && <Section title="Usage" rows={statsRows} />}
        {session && <SystemPromptSection sessionId={session.sessionId} />}
        {session && <CommandsHistorySection session={session} />}
      </div>
    </div>
  );
};

const ModeSection = ({ session }: { session: ApiAgentSession }): JSX.Element | null => {
  if (!session.planMode && !session.skillId) return null;
  return (
    <div className="border-b border-border/40">
      <p className="px-3 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        Mode
      </p>
      <div className="flex flex-wrap items-center gap-1.5 px-3 pb-2">
        {session.planMode ? (
          <Badge variant="outline" className="text-[10px] gap-1">
            <FileText className="h-3 w-3" /> plan mode
          </Badge>
        ) : null}
        {session.skillId ? (
          <Badge variant="secondary" className="text-[10px]">
            from skill
          </Badge>
        ) : null}
      </div>
    </div>
  );
};

const SystemPromptSection = ({ sessionId }: { sessionId: string }): JSX.Element => {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<ApiAgentSystemPrompt | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || data) return;
    let cancelled = false;
    planApi
      .getSystemPrompt(sessionId)
      .then((v) => {
        if (!cancelled) setData(v);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [open, data, sessionId]);

  return (
    <div className="border-b border-border/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 pb-1 pt-2 text-left text-[10px] font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground"
      >
        <ChevronRight className={`h-3 w-3 transition-transform ${open ? "rotate-90" : ""}`} />
        System prompt
      </button>
      {open && (
        <div className="px-3 pb-2">
          {error && <p className="text-[11px] text-destructive">{error}</p>}
          {data ? (
            <>
              <div className="mb-1 flex flex-wrap gap-1 text-[10px]">
                {data.workspaceInstructions ? <Badge variant="secondary">workspace</Badge> : null}
                {data.skillInstructions ? <Badge variant="secondary">skill</Badge> : null}
                {data.sessionOverride !== null ? <Badge variant="default">override</Badge> : null}
                {data.planMode ? <Badge variant="outline">plan addendum</Badge> : null}
              </div>
              <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded border border-border/50 bg-muted/40 p-2 font-mono text-[10px] leading-snug">
                {data.effective}
              </pre>
            </>
          ) : !error ? (
            <p className="text-[11px] text-muted-foreground">Loading…</p>
          ) : null}
        </div>
      )}
    </div>
  );
};

const SLASH_LINE_RE = /^\s*\/([a-z0-9-]+)/i;

const CommandsHistorySection = ({ session }: { session: ApiAgentSession }): JSX.Element | null => {
  const events = session.events ?? [];
  const commands = useMemo(() => {
    const rows: { ts: string; slashName: string }[] = [];
    for (const event of events) {
      if (event.type !== "UserMessageReceived") continue;
      const content = event.payload?.content;
      if (typeof content !== "string") continue;
      const match = SLASH_LINE_RE.exec(content);
      if (!match) continue;
      rows.push({ ts: event.ts, slashName: match[1].toLowerCase() });
    }
    return rows;
  }, [events]);

  if (commands.length === 0) return null;

  return (
    <div className="border-b border-border/40 last:border-b-0">
      <p className="px-3 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        Commands invoked
      </p>
      <ul className="space-y-0.5 px-3 pb-2 text-[11px]">
        {commands.map((row) => (
          <li
            key={`${row.ts}-${row.slashName}`}
            className="flex items-center justify-between gap-2"
          >
            <span className="flex items-center gap-1.5 font-mono">
              <Slash className="h-3 w-3 text-muted-foreground" />
              {row.slashName}
            </span>
            <span className="text-muted-foreground tabular-nums">
              {new Date(row.ts).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
};

const Section = ({ title, rows }: { title: string; rows: DetailRow[] }): JSX.Element => (
  <div className="border-b border-border/40 last:border-b-0">
    <p className="px-3 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
      {title}
    </p>
    <dl className="divide-y divide-border/40">
      {rows.map((row) => (
        <div key={row.label} className="flex items-baseline justify-between gap-2 px-3 py-1.5">
          <dt className="text-[11px] font-medium text-muted-foreground">{row.label}</dt>
          <dd
            className="break-all text-right text-xs font-medium tabular-nums text-foreground"
            title={row.hint}
          >
            {row.value}
          </dd>
        </div>
      ))}
    </dl>
  </div>
);

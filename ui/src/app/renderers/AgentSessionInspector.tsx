/**
 * AgentSessionInspector — right-panel details for an agent session.
 *
 * Tabs:
 *   * **Details** — mode, task metadata, usage, system prompt, slash history
 *   * **Artifacts** — chat embed plots/tables/structures (not a forced center
 *     split; Plan deliverables still use the plan split when a plan locator
 *     exists)
 *
 * The main agent view owns live SSE/stat refresh; the inspector fetches the
 * selected task snapshot without starting a second poller for the same task.
 */

import { Bot, ChevronRight, FileText, Lock, Package, Slash } from "lucide-react";
import type { JSX } from "react";
import { useEffect, useMemo, useState } from "react";
import type { SessionStatsResponse } from "@/api/generated";
import { StatusBadge } from "@/app/components/entity";
import { ArtifactBody } from "@/app/renderers/agent/artifacts";
import { isLegacySession, legacyBadgeMeta } from "@/app/renderers/agent_session/inspectorHelpers";
import { collectArtifacts } from "@/app/renderers/agentEvents";
import { type ApiAgentSystemPrompt, agentApi, planApi } from "@/app/state/api";
import type { ApiAgentSession, RendererProps } from "@/app/types";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { WorkbenchTag } from "@/components/workbench";

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
  { label: "Goal", value: session.goal || "—" },
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
    return () => {
      cancelled = true;
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
  const artifacts = useMemo(() => collectArtifacts(session?.events ?? []), [session?.events]);

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex items-center justify-between border-b border-border/70 bg-muted/20 px-3 py-2">
        <h2 className="flex items-center gap-2 text-micro font-medium uppercase tracking-wide text-muted-foreground">
          <Bot className="h-3.5 w-3.5" /> Task details
        </h2>
        {legacyMeta ? (
          <WorkbenchTag
            meaning="metadata"
            className="h-5 gap-1 px-2 text-micro uppercase tracking-wide"
            title={legacyMeta.tooltip}
          >
            <Lock className="h-3 w-3" /> {legacyMeta.label}
          </WorkbenchTag>
        ) : session?.status ? (
          <StatusBadge status={session.status} size="sm" dot />
        ) : null}
      </div>

      {!sessionId && <p className="px-3 py-2 text-xs text-muted-foreground">No task selected.</p>}
      {error && <p className="px-3 py-2 text-xs text-destructive">{error}</p>}

      {session ? (
        <Tabs defaultValue="details" className="flex min-h-0 flex-1 flex-col gap-0">
          <TabsList
            variant="line"
            className="h-9 w-full justify-start rounded-none border-b border-border/60 bg-transparent px-2"
          >
            <TabsTrigger value="details" className="text-xs">
              Details
            </TabsTrigger>
            <TabsTrigger value="artifacts" className="gap-1.5 text-xs">
              Artifacts
              {artifacts.length > 0 ? (
                <WorkbenchTag className="h-4 min-w-4 px-1 text-micro tabular-nums">
                  {artifacts.length}
                </WorkbenchTag>
              ) : null}
            </TabsTrigger>
          </TabsList>
          <TabsContent value="details" className="min-h-0 flex-1 overflow-auto">
            <ModeSection session={session} />
            {sessionRows.length > 0 && <Section title="Task" rows={sessionRows} />}
            {statsRows.length > 0 && <Section title="Usage" rows={statsRows} />}
            <SystemPromptSection taskId={session.taskId ?? session.sessionId} session={session} />
            <CommandsHistorySection session={session} />
          </TabsContent>
          <TabsContent value="artifacts" className="min-h-0 flex-1 overflow-auto">
            <ArtifactsTab artifacts={artifacts} />
          </TabsContent>
        </Tabs>
      ) : null}
    </div>
  );
};

const ArtifactsTab = ({ artifacts }: { artifacts: Record<string, unknown>[] }): JSX.Element => {
  if (artifacts.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 px-4 py-8 text-center">
        <Package className="h-5 w-5 text-muted-foreground" />
        <p className="text-xs text-muted-foreground">
          No chat artifacts yet. Plots and structures from{" "}
          <code className="text-micro">embed_plot</code> /{" "}
          <code className="text-micro">embed_structure</code> appear here.
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-3 px-3 py-3">
      <p className="text-micro font-semibold uppercase tracking-wide text-muted-foreground">
        Conversation artifacts
      </p>
      {artifacts.map((artifact) => {
        const kind = String(artifact.kind ?? "item");
        const title = typeof artifact.title === "string" && artifact.title ? artifact.title : "";
        // Artifacts carry no stable id; identity is `kind:title` with a JSON
        // fingerprint fallback so two identical-kind items stay distinct.
        const fingerprint = title || JSON.stringify(artifact.payload ?? artifact).slice(0, 120);
        return (
          <div key={`${kind}:${fingerprint}`} className="space-y-1">
            <p className="text-micro text-muted-foreground">
              <span className="font-mono">{kind}</span>
              {title ? ` · ${title}` : ""}
            </p>
            <ArtifactBody payload={artifact} />
          </div>
        );
      })}
    </div>
  );
};

const ModeSection = ({ session }: { session: ApiAgentSession }): JSX.Element | null => {
  if (session.activeMode !== "plan" && !session.skillId) return null;
  return (
    <div className="border-b border-border/40">
      <p className="px-3 pb-1 pt-2 text-micro font-semibold uppercase tracking-wide text-muted-foreground">
        Mode
      </p>
      <div className="flex flex-wrap items-center gap-2 px-3 pb-2">
        {session.activeMode === "plan" ? (
          <WorkbenchTag meaning="metadata" className="text-micro gap-1">
            <FileText className="h-3 w-3" /> planning agent
          </WorkbenchTag>
        ) : null}
        {session.skillId ? <WorkbenchTag className="text-micro">from skill</WorkbenchTag> : null}
      </div>
    </div>
  );
};

const SystemPromptSection = ({
  taskId,
  session,
}: {
  taskId: string;
  session: ApiAgentSession;
}): JSX.Element => {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<ApiAgentSystemPrompt | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || data) return;
    let cancelled = false;
    // Prefer task id (agent-tasks live surface); fall back to session id.
    planApi
      .getSystemPrompt(taskId || session.sessionId)
      .then((v) => {
        if (!cancelled) setData(v);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [open, data, taskId, session.sessionId]);

  return (
    <div className="border-b border-border/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 pb-1 pt-2 text-left text-micro font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground"
      >
        <ChevronRight className={`h-3 w-3 transition-transform ${open ? "rotate-90" : ""}`} />
        System prompt
      </button>
      {open && (
        <div className="px-3 pb-2">
          {error && <p className="text-micro text-destructive">{error}</p>}
          {data ? (
            <>
              <div className="mb-1 flex flex-wrap gap-1 text-micro">
                {data.workspaceInstructions ? <WorkbenchTag>workspace</WorkbenchTag> : null}
                {data.skillInstructions ? <WorkbenchTag>skill</WorkbenchTag> : null}
                {data.sessionOverride !== null ? (
                  <WorkbenchTag meaning="selection">override</WorkbenchTag>
                ) : null}
                {data.planMode ? (
                  <WorkbenchTag meaning="metadata">plan addendum</WorkbenchTag>
                ) : null}
              </div>
              <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded border border-border/50 bg-muted/40 p-2 font-mono text-micro leading-snug">
                {data.effective}
              </pre>
            </>
          ) : !error ? (
            <p className="text-micro text-muted-foreground">Loading…</p>
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
      // Each turn opens with loop_started carrying the user's raw input —
      // that is where slash invocations live in the snake_case vocabulary.
      if (event.type !== "loop_started") continue;
      const content = event.payload?.user_input;
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
      <p className="px-3 pb-1 pt-2 text-micro font-semibold uppercase tracking-wide text-muted-foreground">
        Commands invoked
      </p>
      <ul className="space-y-1 px-3 pb-2 text-micro">
        {commands.map((row) => (
          <li
            key={`${row.ts}-${row.slashName}`}
            className="flex items-center justify-between gap-2"
          >
            <span className="flex items-center gap-2 font-mono">
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
    <p className="px-3 pb-1 pt-2 text-micro font-semibold uppercase tracking-wide text-muted-foreground">
      {title}
    </p>
    <dl className="divide-y divide-border/40">
      {rows.map((row) => (
        <div key={row.label} className="flex items-baseline justify-between gap-2 px-3 py-2">
          <dt className="text-micro font-medium text-muted-foreground">{row.label}</dt>
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

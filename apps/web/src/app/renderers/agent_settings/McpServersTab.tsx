/**
 * McpServersTab — full CRUD for MCP servers + secrets, VSCode multi-scope.
 *
 * Two layers (User / Workspace) with Workspace-overrides-User merge,
 * mirroring VSCode's settings model. Each entry shows its scope, transport,
 * resolved-secret status, and supports Test / Edit / Delete.
 *
 * Secrets reference syntax: ``${SECRET:KEY}`` in env values (stdio) or
 * header values (http/sse). Missing keys are detected
 * client-side and surfaced inline so the user can set them without
 * leaving the dialog.
 */

import {
  AlertCircle,
  CheckCircle2,
  ChevronRight,
  Edit3,
  KeyRound,
  Lock,
  Plus,
  Save,
  Server,
  Trash2,
  Unplug,
  Wrench,
  Zap,
} from "lucide-react";
import type { JSX } from "react";
import { useCallback, useEffect, useId, useMemo, useState } from "react";
import { AgentUnavailableError, resetAgentProbes } from "@/app/state/agentProbe";
import {
  type ApiAgentTool,
  type ApiAgentToolList,
  type ApiMcpOAuthStatus,
  type ApiMcpScope,
  type ApiMcpSecretList,
  type ApiMcpServer,
  type ApiMcpServerList,
  type ApiMcpServerTestResult,
  type ApiMcpToolGroup,
  agentAdminApi,
  type McpOAuth2AuthInput,
  type McpServerSpecInput,
  type McpServerUpsertInput,
  mcpSource,
} from "@/app/state/api";
import { emitMcpConfigChanged } from "@/app/state/mcpEvents";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Code as InlineCode } from "@/components/ui/code";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { WorkbenchAction, WorkbenchIconAction, WorkbenchTag } from "@/components/workbench";
import { cn } from "@/lib/utils";
import { KnowledgeSourcesPanel } from "./KnowledgeSourcesPanel";
import { UnavailableCapability } from "./UnavailableCapability";

const SECRET_REF_RE = /\$\{SECRET:([A-Za-z_]\w*)\}/g;
const NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

type ScopeFilter = "all" | ApiMcpScope;
type TransportType = McpServerSpecInput["type"];

const TRANSPORTS: TransportType[] = ["stdio", "http", "sse"];

const TRANSPORT_LABEL: Record<TransportType, string> = {
  stdio: "stdio",
  http: "http (streamable)",
  sse: "sse (legacy)",
};

const SCOPE_LABEL: Record<ApiMcpScope, string> = {
  user: "User",
  workspace: "Workspace",
};

const collectRefs = (values: string[]): string[] => {
  const refs = new Set<string>();
  for (const v of values) {
    for (const m of v.matchAll(SECRET_REF_RE)) refs.add(m[1]);
  }
  return Array.from(refs).sort();
};

// ── Tiny presentation helpers ──────────────────────────────────────────────
// Encapsulate the patterns used in 3+ places so the markup below stays scannable.

const Code = ({ children }: { children: React.ReactNode }): JSX.Element => (
  <InlineCode className="rounded-control bg-muted px-1 text-label">{children}</InlineCode>
);

const ErrorBanner = ({ children }: { children: React.ReactNode }): JSX.Element => (
  <div className="bg-destructive/10 px-2 py-1 text-label text-destructive">{children}</div>
);

interface StatusBadgeProps {
  tone: "success" | "destructive" | "muted";
  children: React.ReactNode;
  title?: string;
}

const StatusBadge = ({ tone, children, title }: StatusBadgeProps): JSX.Element => {
  return (
    <WorkbenchTag
      meaning={tone === "success" ? "completed" : tone === "destructive" ? "failed" : "category"}
      className="text-label"
      title={title}
    >
      {children}
    </WorkbenchTag>
  );
};

interface IconButtonProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  tone?: "default" | "destructive";
}

const IconButton = ({
  icon: Icon,
  label,
  onClick,
  disabled,
  tone = "default",
}: IconButtonProps): JSX.Element => (
  <Tooltip>
    <TooltipTrigger asChild>
      <WorkbenchIconAction
        label={label}
        disabled={disabled}
        onClick={onClick}
        className={tone === "destructive" ? "text-destructive hover:text-destructive" : ""}
      >
        <Icon className="size-4" />
      </WorkbenchIconAction>
    </TooltipTrigger>
    <TooltipContent>{label}</TooltipContent>
  </Tooltip>
);

// ───────────────────────────────────────────────────────────────────────────
//  Top-level tab
// ───────────────────────────────────────────────────────────────────────────

export const McpServersTab = (): JSX.Element => {
  const [data, setData] = useState<ApiMcpServerList | null>(null);
  const [toolData, setToolData] = useState<ApiAgentToolList>({ tools: [], mcpGroups: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [scopeFilter, setScopeFilter] = useState<ScopeFilter>("all");
  const [editing, setEditing] = useState<EditState | null>(null);
  const [pendingDelete, setPendingDelete] = useState<ApiMcpServer | null>(null);
  const [testResults, setTestResults] = useState<Record<string, ApiMcpServerTestResult | null>>({});
  const [busy, setBusy] = useState<Record<string, boolean>>({});

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    setUnavailable(false);
    try {
      // Servers are required; tools are best-effort so a tools 503 never bricks MCP settings.
      const servers = await agentAdminApi.listMcpServers();
      setData(servers);
      try {
        setToolData(await agentAdminApi.listToolsAndGroups());
      } catch {
        setToolData({ tools: [], mcpGroups: [] });
      }
    } catch (err) {
      if (err instanceof AgentUnavailableError) setUnavailable(true);
      else setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const filtered = useMemo(() => {
    if (!data) return [] as ApiMcpServer[];
    if (scopeFilter === "all") return data.servers;
    return data.servers.filter((s) => s.scope === scopeFilter);
  }, [data, scopeFilter]);

  const targetScope: ApiMcpScope = scopeFilter === "user" ? "user" : "workspace";

  const confirmDelete = useCallback(async () => {
    if (!pendingDelete) return;
    const server = pendingDelete;
    const key = `${server.scope}:${server.name}`;
    setPendingDelete(null);
    setBusy((b) => ({ ...b, [key]: true }));
    try {
      await agentAdminApi.deleteMcpServer(server.name, server.scope);
      emitMcpConfigChanged();
      await refresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy((b) => ({ ...b, [key]: false }));
    }
  }, [pendingDelete, refresh]);

  const handleTest = useCallback(async (server: ApiMcpServer) => {
    const key = `${server.scope}:${server.name}`;
    setBusy((b) => ({ ...b, [key]: true }));
    setTestResults((r) => ({ ...r, [key]: null }));
    try {
      const result = await agentAdminApi.testMcpServer(server.name, server.scope);
      setTestResults((r) => ({ ...r, [key]: result }));
      if (result.ok) setToolData(await agentAdminApi.listToolsAndGroups());
    } catch (err) {
      setTestResults((r) => ({
        ...r,
        [key]: {
          ok: false,
          name: server.name,
          scope: server.scope,
          transport: server.transport,
          latencyMs: 0,
          toolCount: 0,
          error: String(err),
        },
      }));
    } finally {
      setBusy((b) => ({ ...b, [key]: false }));
    }
  }, []);

  return (
    <div className="mx-auto flex h-full w-full max-w-5xl flex-col gap-4 px-4 py-5 sm:px-6 sm:py-6">
      <p className="text-label text-muted-foreground">
        MCP servers expose external tools to the agent. Configuration is layered:{" "}
        <strong>Workspace</strong> entries override <strong>User</strong> entries with the same name
        (VSCode-style). Secrets live in a separate keyring; reference them with{" "}
        <Code>$&#123;SECRET:NAME&#125;</Code> in env or header values.
      </p>

      <KnowledgeSourcesPanel />

      <div className="flex items-center justify-between gap-3">
        <Tabs value={scopeFilter} onValueChange={(v) => setScopeFilter(v as ScopeFilter)}>
          <TabsList>
            <TabsTrigger value="all">All</TabsTrigger>
            <TabsTrigger value="user">User</TabsTrigger>
            <TabsTrigger value="workspace">Workspace</TabsTrigger>
          </TabsList>
        </Tabs>
        <WorkbenchIconAction
          label="Add MCP server"
          disabled={unavailable}
          onClick={() =>
            setEditing({
              mode: "create",
              scope: targetScope,
              name: "",
              spec: { type: "stdio", command: "", args: [], env: {} },
            })
          }
        >
          <Plus className="size-3.5" />
        </WorkbenchIconAction>
      </div>

      {data && (
        <p className="text-label text-muted-foreground">
          <span>Workspace: </span>
          <Code>{data.workspacePath}</Code>
          <span className="mx-1">·</span>
          <span>User: </span>
          <Code>{data.userPath}</Code>
        </p>
      )}
      {error && <ErrorBanner>{error}</ErrorBanner>}
      {unavailable && (
        <UnavailableCapability
          title="MCP configuration unavailable"
          description="This server does not expose MCP administration. Existing server-side MCP configuration is unaffected; install the optional agent backend to manage it here."
          onRetry={() => {
            resetAgentProbes();
            void refresh();
          }}
        />
      )}

      <ScrollArea className="flex-1">
        <div className="flex flex-col gap-2 pr-2">
          {loading && (
            <>
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
            </>
          )}
          {!loading && !unavailable && filtered.length === 0 && (
            <div className="flex flex-col items-center gap-2 py-6 text-center">
              <Server className="size-8 text-muted-foreground" />
              <p className="text-body-lg text-muted-foreground">No servers at this scope.</p>
              <WorkbenchIconAction
                label="Add MCP server"
                onClick={() =>
                  setEditing({
                    mode: "create",
                    scope: targetScope,
                    name: "",
                    spec: { type: "stdio", command: "", args: [], env: {} },
                  })
                }
              >
                <Plus className="size-3.5" />
              </WorkbenchIconAction>
            </div>
          )}
          {!loading &&
            !unavailable &&
            filtered.map((server) => {
              const key = `${server.scope}:${server.name}`;
              const source = mcpSource(server.name);
              return (
                <ServerCard
                  key={key}
                  server={server}
                  test={testResults[key] ?? null}
                  busy={busy[key] ?? false}
                  tools={toolData.tools.filter((tool) => tool.source === source)}
                  toolGroup={
                    toolData.mcpGroups.find((group) => group.server === server.name) ?? null
                  }
                  onEdit={() =>
                    setEditing({
                      mode: "edit",
                      scope: server.scope,
                      name: server.name,
                      spec: serverToSpec(server),
                    })
                  }
                  onDelete={() => setPendingDelete(server)}
                  onTest={() => void handleTest(server)}
                />
              );
            })}
        </div>
      </ScrollArea>

      {editing && (
        <ServerEditor
          state={editing}
          existingNames={(data?.servers ?? [])
            .filter((s) => s.scope === editing.scope)
            .map((s) => s.name)}
          onCancel={() => setEditing(null)}
          onSaved={async () => {
            setEditing(null);
            await refresh();
          }}
        />
      )}

      <AlertDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => !open && setPendingDelete(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete MCP server?</AlertDialogTitle>
            <AlertDialogDescription>
              This removes <Code>{pendingDelete?.name}</Code> from{" "}
              {pendingDelete ? SCOPE_LABEL[pendingDelete.scope] : ""} scope. The server entry is
              gone, but referenced secrets remain in the keyring.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => void confirmDelete()}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

// ───────────────────────────────────────────────────────────────────────────
//  Server card
// ───────────────────────────────────────────────────────────────────────────

interface ServerCardProps {
  server: ApiMcpServer;
  test: ApiMcpServerTestResult | null;
  busy: boolean;
  tools: ApiAgentTool[];
  toolGroup: ApiMcpToolGroup | null;
  onEdit: () => void;
  onDelete: () => void;
  onTest: () => void;
}

const ServerCard = ({
  server,
  test,
  busy,
  tools,
  toolGroup,
  onEdit,
  onDelete,
  onTest,
}: ServerCardProps): JSX.Element => {
  const hasUnresolved = server.unresolvedSecrets.length > 0;
  const [expanded, setExpanded] = useState(false);
  const reportedToolCount = toolGroup?.toolCount ?? test?.toolCount ?? tools.length;
  return (
    <section className={cn("bg-surface/60", server.shadowed ? "bg-muted/30 opacity-80" : "")}>
      <header className="pb-2 px-3 pt-3">
        <div className="flex flex-wrap items-center gap-2">
          <WorkbenchIconAction
            label={`${expanded ? "Collapse" : "Expand"} ${server.name}`}
            className="size-7"
            aria-expanded={expanded}
            onClick={() => setExpanded((value) => !value)}
          >
            <ChevronRight
              className={`size-4 transition-transform ${expanded ? "rotate-90" : ""}`}
            />
          </WorkbenchIconAction>
          <Server className="size-4 text-muted-foreground" />
          <h3 className="font-mono text-body-lg font-medium text-foreground">{server.name}</h3>
          <WorkbenchTag meaning="metadata" className="text-label">
            {server.transport || "?"}
          </WorkbenchTag>
          <WorkbenchTag className="text-label">{SCOPE_LABEL[server.scope]}</WorkbenchTag>
          {server.knowledgeSources && server.knowledgeSources.length > 0 && (
            <WorkbenchTag
              className="text-label"
              title={`MOLMCP_SOURCES=${server.knowledgeSources.join(",")}`}
            >
              sources: {server.knowledgeSources.join(", ")}
            </WorkbenchTag>
          )}
          {server.shadowed && (
            <WorkbenchTag
              meaning="metadata"
              className="text-label"
              title="A Workspace entry with the same name overrides this one."
            >
              shadowed
            </WorkbenchTag>
          )}
          {server.auth?.type === "oauth2" && (
            <StatusBadge
              tone={server.auth.connected ? "success" : "destructive"}
              title={
                server.auth.connected
                  ? "OAuth tokens are stored. The runtime will use them."
                  : "OAuth required but no tokens stored — open the editor and click Connect."
              }
            >
              {server.auth.connected ? "OAuth: connected" : "OAuth: not connected"}
            </StatusBadge>
          )}
          {!server.valid && (
            <WorkbenchTag meaning="failed" className="text-label">
              invalid
            </WorkbenchTag>
          )}
          <div className="ml-auto flex gap-1">
            <IconButton
              icon={Zap}
              label="Test connection"
              disabled={busy || server.shadowed}
              onClick={onTest}
            />
            <IconButton icon={Edit3} label="Edit" disabled={busy} onClick={onEdit} />
            <IconButton
              icon={Trash2}
              label="Delete"
              tone="destructive"
              disabled={busy}
              onClick={onDelete}
            />
          </div>
        </div>
      </header>
      {expanded && (
        <div className="px-3 pb-3 space-y-3 pt-0 text-label">
          {!server.valid && server.invalidReason && (
            <p className="text-destructive">{server.invalidReason}</p>
          )}
          <dl className="grid grid-cols-(--definition-grid-columns) gap-x-2 gap-y-1">
            {server.transport === "stdio" ? (
              <>
                <dt className="text-muted-foreground">command</dt>
                <dd>
                  <InlineCode className="break-all">{server.command ?? "—"}</InlineCode>
                </dd>
                {server.args.length > 0 && (
                  <>
                    <dt className="text-muted-foreground">args</dt>
                    <dd className="break-all">
                      <InlineCode>{server.args.join(" ")}</InlineCode>
                    </dd>
                  </>
                )}
                {server.envKeys.length > 0 && (
                  <>
                    <dt className="text-muted-foreground">env</dt>
                    <dd>
                      <InlineCode>{server.envKeys.join(", ")}</InlineCode>
                    </dd>
                  </>
                )}
              </>
            ) : (
              <>
                <dt className="text-muted-foreground">url</dt>
                <dd className="break-all">
                  <InlineCode>{server.url ?? "—"}</InlineCode>
                </dd>
                {server.headerKeys.length > 0 && (
                  <>
                    <dt className="text-muted-foreground">headers</dt>
                    <dd>
                      <InlineCode>{server.headerKeys.join(", ")}</InlineCode>
                    </dd>
                  </>
                )}
              </>
            )}
          </dl>
          {server.secretRefs.length > 0 && (
            <div className="flex flex-wrap items-center gap-1 pt-1">
              <KeyRound className="size-3 text-muted-foreground" />
              <span className="text-muted-foreground">secrets:</span>
              {server.secretRefs.map((key) => {
                const missing = server.unresolvedSecrets.includes(key);
                return (
                  <StatusBadge
                    key={key}
                    tone={missing ? "destructive" : "success"}
                    title={missing ? "Set this secret in the editor or User scope." : "Resolved"}
                  >
                    {key}
                  </StatusBadge>
                );
              })}
            </div>
          )}
          {hasUnresolved && (
            <p className="flex items-center gap-1 text-label text-destructive">
              <AlertCircle className="size-3" />
              Runtime will skip this server until missing secrets are set.
            </p>
          )}
          {test && (
            <div
              className={
                "mt-1 border-y px-2 py-1 text-label " +
                (test.ok
                  ? "border-success/40 bg-success-soft text-success-foreground"
                  : "border-destructive/40 bg-destructive/10 text-destructive")
              }
            >
              {test.ok ? (
                <span className="flex items-center gap-1">
                  <CheckCircle2 className="size-3" />
                  connected · {test.toolCount} tools · {test.latencyMs} ms
                </span>
              ) : (
                <span className="flex items-center gap-1">
                  <AlertCircle className="size-3" />
                  {test.error ?? "Test failed"}
                </span>
              )}
            </div>
          )}
          <div className="space-y-2 border-t border-border pt-3">
            <div className="flex flex-wrap items-center gap-2">
              <h4 className="text-label font-semibold uppercase tracking-wide text-muted-foreground">
                Capabilities
              </h4>
              <WorkbenchTag meaning="metadata" className="gap-1 text-label">
                <Wrench className="size-3" /> Tools · {reportedToolCount}
              </WorkbenchTag>
              {server.auth?.type === "oauth2" && (
                <WorkbenchTag meaning="metadata" className="text-label">
                  OAuth 2.0
                </WorkbenchTag>
              )}
            </div>
            {toolGroup && !toolGroup.ok && (
              <ErrorBanner>{toolGroup.error ?? "Tool discovery failed."}</ErrorBanner>
            )}
          </div>
          <div className="space-y-2">
            <h4 className="text-label font-semibold uppercase tracking-wide text-muted-foreground">
              Tools
            </h4>
            {tools.length === 0 ? (
              <p className="border-y border-dashed border-border/60 py-2 text-muted-foreground">
                {test?.ok
                  ? "This server reported no tools."
                  : "No tools discovered. Test the connection to refresh this server's capabilities."}
              </p>
            ) : (
              <div className="divide-y divide-border/60 border-y border-border/60">
                {tools.map((tool) => (
                  <Collapsible key={tool.name} className="py-2">
                    <CollapsibleTrigger className="flex w-full justify-start gap-2 text-left">
                      <InlineCode className="font-mono text-label">{tool.name}</InlineCode>
                      {tool.parameters.length > 0 && (
                        <span className="text-micro text-muted-foreground">
                          {tool.parameters.length} parameter
                          {tool.parameters.length === 1 ? "" : "s"}
                        </span>
                      )}
                      {tool.requiresApproval && (
                        <WorkbenchTag meaning="failed" className="ml-auto text-micro">
                          approval
                        </WorkbenchTag>
                      )}
                    </CollapsibleTrigger>
                    <CollapsibleContent className="space-y-2 pt-2 text-muted-foreground">
                      {tool.description && (
                        <p className="whitespace-pre-line">{tool.description}</p>
                      )}
                      {tool.parameters.length > 0 && (
                        <dl className="grid grid-cols-(--definition-grid-columns) gap-x-2 gap-y-1 border-t border-border/60 pt-2">
                          {tool.parameters.map((parameter) => (
                            <div key={parameter.name} className="contents">
                              <dt className="font-mono text-foreground">{parameter.name}</dt>
                              <dd>
                                {parameter.annotation}
                                {!parameter.required && " · optional"}
                              </dd>
                            </div>
                          ))}
                        </dl>
                      )}
                    </CollapsibleContent>
                  </Collapsible>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
};

// ───────────────────────────────────────────────────────────────────────────
//  Editor dialog
// ───────────────────────────────────────────────────────────────────────────

interface EditState {
  mode: "create" | "edit";
  scope: ApiMcpScope;
  name: string;
  spec: McpServerSpecInput;
}

interface KvRow {
  uid: string;
  key: string;
  value: string;
}

let _kvUidCounter = 0;
const _kvUid = (): string => `kv-${++_kvUidCounter}`;

const specToEnvRows = (spec: McpServerSpecInput): KvRow[] => {
  if (spec.type !== "stdio") return [];
  return Object.entries(spec.env).map(([key, value]) => ({ uid: _kvUid(), key, value }));
};

const specToHeaderRows = (spec: McpServerSpecInput): KvRow[] => {
  if (spec.type === "stdio") return [];
  return Object.entries(spec.headers).map(([key, value]) => ({ uid: _kvUid(), key, value }));
};

const initialAuthFromSpec = (spec: McpServerSpecInput): HttpAuthState => {
  if (spec.type === "stdio") return DEFAULT_AUTH;
  if (spec.auth?.type === "oauth2") {
    return {
      ...DEFAULT_AUTH,
      mode: "oauth2",
      oauthScopesText: spec.auth.scopes.join("\n"),
      oauthClientId: spec.auth.clientId ?? "",
    };
  }
  return { ...DEFAULT_AUTH, mode: detectAuthMode(Object.keys(spec.headers)) };
};

const rowsToMap = (rows: KvRow[]): Record<string, string> => {
  const out: Record<string, string> = {};
  for (const r of rows) {
    if (r.key.trim()) out[r.key.trim()] = r.value;
  }
  return out;
};

const serverToSpec = (server: ApiMcpServer): McpServerSpecInput => {
  if (server.transport === "stdio") {
    // Prefer public env literals from the API (e.g. MOLMCP_SOURCES); secret keys
    // keep ${SECRET:…} placeholders so we never invent values.
    const env: Record<string, string> = { ...(server.env ?? {}) };
    for (const k of server.envKeys) {
      if (!(k in env)) {
        env[k] = `\${SECRET:${k}}`;
      }
    }
    return {
      type: "stdio",
      command: server.command ?? "",
      args: [...server.args],
      env,
    };
  }
  // For http variants we don't see header values from the API (only keys);
  // rebuild best-effort placeholders so the editor shows the keys. The user
  // can re-paste real values when editing.
  return {
    type: (server.transport as "http" | "sse") || "http",
    url: server.url ?? "",
    headers: Object.fromEntries(server.headerKeys.map((k) => [k, ""])),
  };
};

interface ServerEditorProps {
  state: EditState;
  existingNames: string[];
  onCancel: () => void;
  onSaved: () => Promise<void>;
}

const ServerEditor = ({
  state,
  existingNames,
  onCancel,
  onSaved,
}: ServerEditorProps): JSX.Element => {
  const fieldId = useId();
  const id = (name: string): string => `${fieldId}-${name}`;
  const [name, setName] = useState(state.name);
  const [scope, setScope] = useState<ApiMcpScope>(state.scope);
  const [spec, setSpec] = useState<McpServerSpecInput>(state.spec);
  const [argsText, setArgsText] = useState(
    state.spec.type === "stdio" ? state.spec.args.join("\n") : "",
  );
  const [envRows, setEnvRows] = useState<KvRow[]>(specToEnvRows(state.spec));
  const [auth, setAuth] = useState<HttpAuthState>(initialAuthFromSpec(state.spec));
  const [customHeaderRows, setCustomHeaderRows] = useState<KvRow[]>(() =>
    stripAuthOwnedHeaders(specToHeaderRows(state.spec), initialAuthFromSpec(state.spec)),
  );
  const [secrets, setSecrets] = useState<ApiMcpSecretList | null>(null);
  const [secretDrafts, setSecretDrafts] = useState<Record<string, string>>({});
  const [oauthStatus, setOauthStatus] = useState<ApiMcpOAuthStatus | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // OAuth status load (only meaningful for saved OAuth-mode entries; the
  // route returns 400 for non-OAuth servers, which we silently swallow).
  const refreshOauthStatus = useCallback(() => {
    if (state.mode !== "edit" || auth.mode !== "oauth2") {
      setOauthStatus(null);
      return;
    }
    void agentAdminApi
      .getMcpOauthStatus(state.name, scope)
      .then(setOauthStatus)
      .catch(() => setOauthStatus(null));
  }, [state.mode, state.name, scope, auth.mode]);

  useEffect(() => {
    refreshOauthStatus();
  }, [refreshOauthStatus]);

  // Load secrets at the chosen scope so we can show set/unset for refs
  useEffect(() => {
    let cancelled = false;
    agentAdminApi
      .listMcpSecrets(scope)
      .then((s) => {
        if (!cancelled) setSecrets(s);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [scope]);

  // Headers contributed by structured auth modes — visible in the
  // referenced-secrets summary even before submit.
  const compiledAuthHeaderValues = useMemo(() => {
    if (spec.type === "stdio") return [] as string[];
    return Object.values(compileAuth(auth, name).authHeaders);
  }, [spec.type, auth, name]);

  const referencedKeys = useMemo(() => {
    if (spec.type === "stdio") return collectRefs(envRows.map((r) => r.value));
    return collectRefs([...customHeaderRows.map((r) => r.value), ...compiledAuthHeaderValues]);
  }, [spec.type, envRows, customHeaderRows, compiledAuthHeaderValues]);

  const setKeys = useMemo(
    () => new Set((secrets?.secrets ?? []).filter((s) => s.isSet).map((s) => s.key)),
    [secrets],
  );

  const handleTransportChange = useCallback((next: TransportType) => {
    if (next === "stdio") {
      setSpec({ type: "stdio", command: "", args: [], env: {} });
      setArgsText("");
      setEnvRows([]);
    } else {
      setSpec({ type: next, url: "", headers: {} });
      setCustomHeaderRows([]);
      setAuth(DEFAULT_AUTH);
    }
  }, []);

  const buildSpec = useCallback((): McpServerSpecInput | string => {
    if (spec.type === "stdio") {
      const command = spec.command.trim();
      if (!command) return "Command is required.";
      const args = argsText
        .split("\n")
        .map((s) => s)
        .filter((s) => s.length > 0);
      return { type: "stdio", command, args, env: rowsToMap(envRows) };
    }
    const url = spec.url.trim();
    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      return "URL must start with http:// or https://";
    }
    // Structured auth headers come first; user-typed extras can override
    // (e.g., custom Authorization for an exotic scheme) but normally don't.
    const compiled = compileAuth(auth, name);
    const headers = { ...compiled.authHeaders, ...rowsToMap(customHeaderRows) };
    const oauthAuth: McpOAuth2AuthInput | null =
      auth.mode === "oauth2"
        ? {
            type: "oauth2",
            scopes: auth.oauthScopesText
              .split(/\s+/)
              .map((s) => s.trim())
              .filter((s) => s.length > 0),
            clientId: auth.oauthClientId.trim() === "" ? null : auth.oauthClientId.trim(),
          }
        : null;
    return { type: spec.type, url, headers, auth: oauthAuth };
  }, [spec, argsText, envRows, customHeaderRows, auth, name]);

  const handleSecretSave = useCallback(
    async (key: string) => {
      const value = secretDrafts[key];
      if (value === undefined || value === "") return;
      try {
        await agentAdminApi.setMcpSecret(key, value, scope);
        setSecretDrafts((d) => ({ ...d, [key]: "" }));
        const next = await agentAdminApi.listMcpSecrets(scope);
        setSecrets(next);
      } catch (err) {
        setError(String(err));
      }
    },
    [scope, secretDrafts],
  );

  const handleSubmit = useCallback(async () => {
    setError(null);
    if (!NAME_RE.test(name)) {
      setError("Invalid name. Use lowercase letters, digits, underscore, hyphen; max 64 chars.");
      return;
    }
    if (state.mode === "create" && existingNames.includes(name)) {
      setError(`A server named '${name}' already exists at ${SCOPE_LABEL[scope]} scope.`);
      return;
    }
    const built = buildSpec();
    if (typeof built === "string") {
      setError(built);
      return;
    }
    // Auth-driven secrets (Bearer/Basic/API Key) take priority over
    // any same-key manual draft below — the structured field is the
    // single source of truth for that key while in that auth mode.
    const compiled = built.type !== "stdio" ? compileAuth(auth, name) : { secretDrafts: {} };
    const allDrafts: Record<string, string> = { ...secretDrafts, ...compiled.secretDrafts };
    const payload: McpServerUpsertInput = { name, scope, spec: built };
    setSaving(true);
    try {
      const pending = Object.entries(allDrafts).filter(([, v]) => v !== "");
      await Promise.all(pending.map(([k, v]) => agentAdminApi.setMcpSecret(k, v, scope)));
      if (state.mode === "create") {
        await agentAdminApi.createMcpServer(payload);
      } else {
        await agentAdminApi.replaceMcpServer(state.name, payload);
      }
      emitMcpConfigChanged();
      await onSaved();
    } catch (err) {
      setError(String(err));
    } finally {
      setSaving(false);
    }
  }, [name, scope, state, existingNames, buildSpec, auth, secretDrafts, onSaved]);

  const isStdio = spec.type === "stdio";

  return (
    <Dialog open onOpenChange={(open) => !open && !saving && onCancel()}>
      <DialogContent className="max-h-dialog-viewport-tall max-w-2xl overflow-hidden">
        <DialogHeader>
          <DialogTitle>
            {state.mode === "create" ? (
              "Add MCP server"
            ) : (
              <span>
                Edit <InlineCode className="font-mono">{state.name}</InlineCode>
              </span>
            )}
          </DialogTitle>
        </DialogHeader>
        <ScrollArea className="max-h-dialog-scroll-compact pr-2">
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label htmlFor={id("name")} className="text-label">
                  Name
                </Label>
                <Input
                  id={id("name")}
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="github-search"
                  disabled={state.mode === "edit"}
                  className="font-mono"
                />
                <p className="mt-1 text-label text-muted-foreground">
                  Lowercase letters, digits, underscore, hyphen.
                </p>
              </div>
              <div>
                <Label htmlFor={id("scope")} className="text-label">
                  Scope
                </Label>
                <Select
                  value={scope}
                  onValueChange={(v) => setScope(v as ApiMcpScope)}
                  disabled={state.mode === "edit"}
                >
                  <SelectTrigger id={id("scope")}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="workspace">Workspace (this workspace only)</SelectItem>
                    <SelectItem value="user">User (all workspaces)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div>
              <Label htmlFor={id("transport")} className="text-label">
                Transport
              </Label>
              <Select
                value={spec.type}
                onValueChange={(v) => handleTransportChange(v as TransportType)}
              >
                <SelectTrigger id={id("transport")}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TRANSPORTS.map((t) => (
                    <SelectItem key={t} value={t}>
                      {TRANSPORT_LABEL[t]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {isStdio ? (
              <>
                <div>
                  <Label htmlFor={id("command")} className="text-label">
                    Command
                  </Label>
                  <Input
                    id={id("command")}
                    value={spec.command}
                    onChange={(e) => setSpec({ ...spec, command: e.target.value })}
                    placeholder="npx"
                    className="font-mono"
                  />
                </div>
                <div>
                  <Label htmlFor={id("args")} className="text-label">
                    Args (one per line)
                  </Label>
                  <Textarea
                    id={id("args")}
                    rows={4}
                    value={argsText}
                    onChange={(e) => setArgsText(e.target.value)}
                    placeholder={"-y\n@modelcontextprotocol/server-github"}
                    className="font-mono text-label"
                  />
                  <p className="mt-1 text-label text-muted-foreground">
                    <Code>$&#123;workspaceRoot&#125;</Code> is expanded at runtime.
                  </p>
                </div>
                {(name === "molmcp" || name.includes("molmcp")) && (
                  <div className="rounded-control border border-info/30 bg-info-soft/20 px-3 py-2">
                    <Label className="text-label font-medium">MOLMCP_SOURCES (package pin)</Label>
                    <p className="mb-1.5 text-micro text-muted-foreground">
                      Comma-separated packages molmcp may expose. Leave empty for all. Prefer the
                      Knowledge sources panel above for a default pin applied to every plan.
                    </p>
                    <Input
                      className="font-mono text-label"
                      placeholder="molpy,molvis,molplot"
                      value={envRows.find((r) => r.key === "MOLMCP_SOURCES")?.value ?? ""}
                      onChange={(e) => {
                        const v = e.target.value;
                        setEnvRows((rows) => {
                          const rest = rows.filter((r) => r.key !== "MOLMCP_SOURCES");
                          if (!v.trim()) return rest;
                          const existing = rows.find((r) => r.key === "MOLMCP_SOURCES");
                          return [
                            ...rest,
                            {
                              uid: existing?.uid ?? _kvUid(),
                              key: "MOLMCP_SOURCES",
                              value: v,
                            },
                          ];
                        });
                      }}
                    />
                  </div>
                )}
                <KvEditor
                  label="Environment variables"
                  rows={envRows}
                  setRows={setEnvRows}
                  valuePlaceholder="literal value or ${SECRET:KEY}"
                />
              </>
            ) : (
              <>
                <div>
                  <Label htmlFor={id("url")} className="text-label">
                    URL
                  </Label>
                  <Input
                    id={id("url")}
                    value={(spec as { url: string }).url}
                    onChange={(e) =>
                      setSpec({
                        ...(spec as {
                          type: "http" | "sse";
                          headers: Record<string, string>;
                        }),
                        url: e.target.value,
                      })
                    }
                    placeholder="https://api.example.com/mcp"
                    className="font-mono"
                  />
                </div>
                <HttpAuthSection
                  serverName={name}
                  scope={scope}
                  auth={auth}
                  setAuth={setAuth}
                  customRows={customHeaderRows}
                  setCustomRows={setCustomHeaderRows}
                  serverPersisted={state.mode === "edit"}
                  oauthConnected={oauthStatus?.hasTokens ?? false}
                  onOauthChanged={refreshOauthStatus}
                />
              </>
            )}

            {referencedKeys.length > 0 && (
              <section className="border-t border-border/60 pt-3">
                <div className="mb-2 flex items-center gap-2 text-label font-medium">
                  <KeyRound className="size-3" /> Referenced secrets
                </div>
                <p className="mb-2 text-label text-muted-foreground">
                  Plaintext values stay on this machine, written to <Code>.mcp_secrets.json</Code>{" "}
                  at the chosen scope.
                </p>
                <div className="space-y-2">
                  {referencedKeys.map((key) => {
                    const isSet = setKeys.has(key);
                    return (
                      <div key={key} className="flex flex-col gap-1">
                        <div className="flex items-center gap-2 text-label">
                          <InlineCode>{key}</InlineCode>
                          <StatusBadge tone={isSet ? "success" : "destructive"}>
                            {isSet ? "set" : "missing"}
                          </StatusBadge>
                        </div>
                        <div className="flex items-center gap-2">
                          <Input
                            type="password"
                            aria-label={`Secret value for ${key}`}
                            placeholder={isSet ? "Type to replace…" : "Paste secret value"}
                            value={secretDrafts[key] ?? ""}
                            onChange={(e) =>
                              setSecretDrafts((d) => ({ ...d, [key]: e.target.value }))
                            }
                            autoComplete="off"
                          />
                          <WorkbenchIconAction
                            label={`Save secret ${key}`}
                            type="button"
                            disabled={!secretDrafts[key]}
                            onClick={() => void handleSecretSave(key)}
                          >
                            <Save className="size-4" />
                          </WorkbenchIconAction>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            )}

            {error && <ErrorBanner>{error}</ErrorBanner>}
          </div>
        </ScrollArea>
        <DialogFooter>
          <WorkbenchAction kind="ghost" size="default" onClick={onCancel} disabled={saving}>
            Cancel
          </WorkbenchAction>
          <WorkbenchAction
            kind="primary"
            size="default"
            onClick={() => void handleSubmit()}
            disabled={saving}
          >
            {saving ? "Saving…" : state.mode === "create" ? "Create" : "Save"}
          </WorkbenchAction>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

// ───────────────────────────────────────────────────────────────────────────
//  HTTP Authentication helpers + section
// ───────────────────────────────────────────────────────────────────────────
//
//  Five static auth modes are surfaced as first-class UI affordances; each
//  compiles to one or more headers + (optionally) a secret draft to persist.
//  Custom keeps the legacy free-form KvEditor for power users / oddball
//  schemes. OAuth 2.0 is intentionally absent — it requires server-side
//  plumbing (PKCE + redirect/callback handlers + TokenStorage) that the
//  current backend doesn't ship; a banner notes the gap.

type AuthMode = "none" | "bearer" | "basic" | "apikey" | "oauth2" | "custom";

interface HttpAuthState {
  mode: AuthMode;
  bearerToken: string;
  basicUser: string;
  basicPass: string;
  apiKeyHeader: string;
  apiKeyValue: string;
  // OAuth 2.0 — only the *intent* lives in spec; tokens are persisted
  // server-side via /oauth/start + /oauth/callback. ``oauthScopesText`` is a
  // newline-separated edit string (one scope per line); empty means "let the
  // IdP pick". ``oauthClientId`` is set only when the IdP doesn't support
  // Dynamic Client Registration.
  oauthScopesText: string;
  oauthClientId: string;
}

const DEFAULT_AUTH: HttpAuthState = {
  mode: "none",
  bearerToken: "",
  basicUser: "",
  basicPass: "",
  apiKeyHeader: "X-API-Key",
  apiKeyValue: "",
  oauthScopesText: "",
  oauthClientId: "",
};

const AUTH_MODE_LABEL: Record<AuthMode, string> = {
  none: "None",
  bearer: "Bearer Token",
  basic: "Basic Auth",
  apikey: "API Key",
  oauth2: "OAuth 2.0",
  custom: "Custom Headers",
};

// Slug a server name into a SCREAMING_SNAKE secret-key prefix. Falls back
// to "SERVER" when the name field is still empty during creation so the
// preview UI stays meaningful before the user types a name.
const secretSlug = (serverName: string): string => {
  const cleaned = serverName.toUpperCase().replace(/[^A-Z0-9_]/g, "_");
  return cleaned.length > 0 ? cleaned : "SERVER";
};

// UTF-8-safe base64 for Basic Auth credentials. ``btoa`` chokes on any
// codepoint above 0xFF, so encode through ``TextEncoder`` first.
const base64UTF8 = (s: string): string => {
  const bytes = new TextEncoder().encode(s);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary);
};

// Headers + secret drafts contributed by the auth state. Custom/None
// produce nothing — those modes leave the user's KvEditor rows untouched.
interface CompiledAuth {
  authHeaders: Record<string, string>;
  secretDrafts: Record<string, string>;
}

const compileAuth = (auth: HttpAuthState, serverName: string): CompiledAuth => {
  const slug = secretSlug(serverName);
  switch (auth.mode) {
    case "none":
    case "custom":
    case "oauth2":
      // OAuth doesn't contribute headers — the runtime injects them via
      // OAuthClientProvider after a successful Connect. Spec.auth carries
      // the intent (scopes/client_id) and is wired in handleSubmit.
      return { authHeaders: {}, secretDrafts: {} };
    case "bearer": {
      const key = `${slug}_TOKEN`;
      return {
        authHeaders: { Authorization: `Bearer \${SECRET:${key}}` },
        secretDrafts: auth.bearerToken ? { [key]: auth.bearerToken } : {},
      };
    }
    case "basic": {
      const key = `${slug}_BASIC`;
      const hasInput = auth.basicUser !== "" || auth.basicPass !== "";
      const encoded = hasInput ? base64UTF8(`${auth.basicUser}:${auth.basicPass}`) : "";
      return {
        authHeaders: { Authorization: `Basic \${SECRET:${key}}` },
        secretDrafts: hasInput ? { [key]: encoded } : {},
      };
    }
    case "apikey": {
      const headerName = auth.apiKeyHeader.trim() || "X-API-Key";
      const key = `${slug}_API_KEY`;
      return {
        authHeaders: { [headerName]: `\${SECRET:${key}}` },
        secretDrafts: auth.apiKeyValue ? { [key]: auth.apiKeyValue } : {},
      };
    }
  }
};

// Best-effort: header *values* are redacted by the API, so any single
// Authorization header is treated as Bearer (the dominant case) and any
// single X-API-* header as apikey. Anything else falls back to custom so
// the user's existing config doesn't get silently rewritten.
const detectAuthMode = (headerKeys: readonly string[]): AuthMode => {
  if (headerKeys.length === 0) return "none";
  const lower = headerKeys.map((k) => k.toLowerCase());
  const hasAuth = lower.includes("authorization");
  const nonAuth = lower.filter((k) => k !== "authorization");
  if (hasAuth && nonAuth.length === 0) return "bearer";
  if (!hasAuth && nonAuth.length === 1) {
    const onlyKey = nonAuth[0];
    if (onlyKey.includes("api") && (onlyKey.includes("key") || onlyKey.includes("token"))) {
      return "apikey";
    }
  }
  return "custom";
};

// Strip header rows the auth section owns so they don't double-up in the
// custom KvEditor. Auth modes own these specific rows:
//   bearer / basic       → "Authorization"
//   apikey               → user-chosen header name (case-insensitive)
//   none / custom        → owns nothing
const stripAuthOwnedHeaders = (rows: KvRow[], auth: HttpAuthState): KvRow[] => {
  if (auth.mode === "none" || auth.mode === "custom") return rows;
  const owned = new Set<string>();
  if (auth.mode === "bearer" || auth.mode === "basic") {
    owned.add("authorization");
  } else if (auth.mode === "apikey") {
    owned.add((auth.apiKeyHeader || "X-API-Key").trim().toLowerCase());
  }
  return rows.filter((r) => !owned.has(r.key.trim().toLowerCase()));
};

interface HttpAuthSectionProps {
  serverName: string;
  scope: ApiMcpScope;
  auth: HttpAuthState;
  setAuth: (next: HttpAuthState) => void;
  customRows: KvRow[];
  setCustomRows: (next: KvRow[]) => void;
  // True iff the server already exists on disk (i.e. saved). OAuth Connect
  // requires a saved server because the backend needs an entry to look up.
  serverPersisted: boolean;
  // Whether the current entry has stored OAuth tokens. Used to render the
  // status badge ("connected"/"disconnected") and decide button label.
  oauthConnected: boolean;
  // Notify parent the connection state changed so it can refresh.
  onOauthChanged: () => void;
}

const HttpAuthSection = ({
  serverName,
  scope,
  auth,
  setAuth,
  customRows,
  setCustomRows,
  serverPersisted,
  oauthConnected,
  onOauthChanged,
}: HttpAuthSectionProps): JSX.Element => {
  const fieldId = useId();
  const id = (name: string): string => `${fieldId}-${name}`;
  const slug = secretSlug(serverName);
  const onModeChange = (next: AuthMode) => {
    // Switching mode preserves typed credentials so users don't lose work
    // when toggling. Custom rows are scrubbed of headers the new mode owns.
    setAuth({ ...auth, mode: next });
    setCustomRows(stripAuthOwnedHeaders(customRows, { ...auth, mode: next }));
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Lock className="size-3.5 text-muted-foreground" />
        <Label htmlFor={id("mode")} className="text-label">
          Authentication
        </Label>
      </div>
      <Select value={auth.mode} onValueChange={(v) => onModeChange(v as AuthMode)}>
        <SelectTrigger id={id("mode")}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {(Object.keys(AUTH_MODE_LABEL) as AuthMode[]).map((m) => (
            <SelectItem key={m} value={m}>
              {AUTH_MODE_LABEL[m]}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {auth.mode === "bearer" && (
        <div className="space-y-1 border-t border-border/60 pt-3">
          <Label htmlFor={id("bearer-token")} className="text-label">
            Token
          </Label>
          <Input
            id={id("bearer-token")}
            type="password"
            placeholder="Paste access token"
            value={auth.bearerToken}
            onChange={(e) => setAuth({ ...auth, bearerToken: e.target.value })}
            autoComplete="off"
            className="font-mono"
          />
          <p className="text-label text-muted-foreground">
            Stored as <Code>{slug}_TOKEN</Code> in the secret keyring; sent as{" "}
            <Code>Authorization: Bearer …</Code>.
          </p>
        </div>
      )}

      {auth.mode === "basic" && (
        <div className="space-y-2 border-t border-border/60 pt-3">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <Label htmlFor={id("username")} className="text-label">
                Username
              </Label>
              <Input
                id={id("username")}
                value={auth.basicUser}
                onChange={(e) => setAuth({ ...auth, basicUser: e.target.value })}
                autoComplete="off"
                className="font-mono"
              />
            </div>
            <div>
              <Label htmlFor={id("password")} className="text-label">
                Password
              </Label>
              <Input
                id={id("password")}
                type="password"
                value={auth.basicPass}
                onChange={(e) => setAuth({ ...auth, basicPass: e.target.value })}
                autoComplete="off"
                className="font-mono"
              />
            </div>
          </div>
          <p className="text-label text-muted-foreground">
            Encoded as <Code>{slug}_BASIC</Code> = base64(user:pass); sent as{" "}
            <Code>Authorization: Basic …</Code>.
          </p>
        </div>
      )}

      {auth.mode === "apikey" && (
        <div className="space-y-2 border-t border-border/60 pt-3">
          <div>
            <Label htmlFor={id("api-key-header")} className="text-label">
              Header name
            </Label>
            <Input
              id={id("api-key-header")}
              value={auth.apiKeyHeader}
              onChange={(e) => setAuth({ ...auth, apiKeyHeader: e.target.value })}
              placeholder="X-API-Key"
              className="font-mono"
            />
          </div>
          <div>
            <Label htmlFor={id("api-key-value")} className="text-label">
              Value
            </Label>
            <Input
              id={id("api-key-value")}
              type="password"
              placeholder="Paste API key"
              value={auth.apiKeyValue}
              onChange={(e) => setAuth({ ...auth, apiKeyValue: e.target.value })}
              autoComplete="off"
              className="font-mono"
            />
          </div>
          <p className="text-label text-muted-foreground">
            Stored as <Code>{slug}_API_KEY</Code>; sent in the named header.
          </p>
        </div>
      )}

      {auth.mode === "oauth2" && (
        <OAuthConnectPanel
          serverName={serverName}
          scope={scope}
          oauthScopesText={auth.oauthScopesText}
          setOauthScopesText={(v) => setAuth({ ...auth, oauthScopesText: v })}
          oauthClientId={auth.oauthClientId}
          setOauthClientId={(v) => setAuth({ ...auth, oauthClientId: v })}
          serverPersisted={serverPersisted}
          connected={oauthConnected}
          onChanged={onOauthChanged}
        />
      )}

      <HeaderRows mode={auth.mode} rows={customRows} setRows={setCustomRows} />
    </div>
  );
};

// One render path for the custom-headers KvEditor across all auth modes:
//   - custom        — primary editor, always visible
//   - bearer/basic/apikey/oauth2 — collapsed under <details> (extras alongside)
//   - none          — visible only when the user has typed rows already
const HeaderRows = ({
  mode,
  rows,
  setRows,
}: {
  mode: AuthMode;
  rows: KvRow[];
  setRows: (next: KvRow[]) => void;
}): JSX.Element | null => {
  if (mode === "custom") {
    return (
      <KvEditor
        label="Headers"
        rows={rows}
        setRows={setRows}
        valuePlaceholder="Bearer ${SECRET:KEY}"
      />
    );
  }
  if (mode === "none") {
    return rows.length > 0 ? (
      <KvEditor
        label="Headers"
        rows={rows}
        setRows={setRows}
        valuePlaceholder="literal value or ${SECRET:KEY}"
      />
    ) : null;
  }
  return (
    <Collapsible className="border-y border-border/60">
      <CollapsibleTrigger className="py-2 text-label text-muted-foreground">
        Additional headers ({rows.length})
      </CollapsibleTrigger>
      <CollapsibleContent className="border-t border-border/60 py-2">
        <KvEditor
          label="Extra headers"
          rows={rows}
          setRows={setRows}
          valuePlaceholder="literal value or ${SECRET:KEY}"
        />
      </CollapsibleContent>
    </Collapsible>
  );
};

// ───────────────────────────────────────────────────────────────────────────
//  OAuth 2.0 Connect panel
// ───────────────────────────────────────────────────────────────────────────
//
//  Drives the interactive flow: collect scopes → kick /oauth/start → open
//  the authorize URL in a popup → wait for the SPA's /oauth-callback route
//  to POST the code back via window.postMessage. Token persistence happens
//  server-side; this UI only shows status + lets the user disconnect.
//
//  The /oauth-callback SPA route is expected to emit a window.postMessage
//  payload of shape ``{type: "molexp:oauth-callback", code, state}`` to its
//  opener. If the SPA isn't yet wired (e.g. running pre-callback-route
//  build), the user can paste the redirected URL into a text field instead.

interface OAuthConnectPanelProps {
  serverName: string;
  scope: ApiMcpScope;
  oauthScopesText: string;
  setOauthScopesText: (v: string) => void;
  oauthClientId: string;
  setOauthClientId: (v: string) => void;
  serverPersisted: boolean;
  connected: boolean;
  onChanged: () => void;
}

const OAUTH_POPUP_FEATURES = "width=720,height=820,menubar=no,toolbar=no";

const OAuthConnectPanel = ({
  serverName,
  scope,
  oauthScopesText,
  setOauthScopesText,
  oauthClientId,
  setOauthClientId,
  serverPersisted,
  connected,
  onChanged,
}: OAuthConnectPanelProps): JSX.Element => {
  const fieldId = useId();
  const scopesId = `${fieldId}-scopes`;
  const clientId = `${fieldId}-client-id`;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string | null>(null);

  const connect = useCallback(async () => {
    setError(null);
    setProgress("Requesting authorize URL…");
    setBusy(true);
    try {
      const start = await agentAdminApi.startMcpOauth(serverName, scope);
      const popup = window.open(start.authorizeUrl, "molexp-oauth", OAUTH_POPUP_FEATURES);
      if (!popup) {
        throw new Error(
          "Popup blocked. Allow popups for this site, or copy the authorize URL into a new tab.",
        );
      }
      setProgress("Waiting for browser authorization…");
      const callback = await waitForOAuthCallback(popup);
      setProgress("Exchanging tokens…");
      await agentAdminApi.callbackMcpOauth(serverName, scope, callback.code, callback.state);
      setProgress(null);
      emitMcpConfigChanged();
      onChanged();
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
      setProgress(null);
    } finally {
      setBusy(false);
    }
  }, [serverName, scope, onChanged]);

  const disconnect = useCallback(async () => {
    setError(null);
    setBusy(true);
    try {
      await agentAdminApi.disconnectMcpOauth(serverName, scope);
      emitMcpConfigChanged();
      onChanged();
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
    } finally {
      setBusy(false);
    }
  }, [serverName, scope, onChanged]);

  return (
    <div className="space-y-2 border-t border-border/60 pt-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-label font-medium">Status:</span>
        <StatusBadge tone={connected ? "success" : "muted"}>
          {connected ? "connected" : "not connected"}
        </StatusBadge>
        {!serverPersisted && (
          <span className="text-label text-muted-foreground">
            Save the server first, then click Connect.
          </span>
        )}
      </div>
      <div>
        <Label htmlFor={scopesId} className="text-label">
          Scopes (one per line, optional)
        </Label>
        <Textarea
          id={scopesId}
          rows={3}
          value={oauthScopesText}
          onChange={(e) => setOauthScopesText(e.target.value)}
          placeholder={"openid\nemail\noffline_access"}
          className="font-mono text-label"
        />
        <p className="mt-1 text-label text-muted-foreground">
          Leave empty to let the IdP pick. <Code>offline_access</Code> is recommended for long-lived
          sessions (refresh tokens).
        </p>
      </div>
      <div>
        <Label htmlFor={clientId} className="text-label">
          Client ID (optional)
        </Label>
        <Input
          id={clientId}
          value={oauthClientId}
          onChange={(e) => setOauthClientId(e.target.value)}
          placeholder="leave empty for Dynamic Client Registration"
          className="font-mono"
        />
      </div>
      <div className="flex items-center gap-2">
        <WorkbenchAction
          kind="primary"
          size="compact"
          type="button"
          disabled={busy || !serverPersisted}
          onClick={() => void connect()}
        >
          {connected ? "Reconnect" : "Connect"}
        </WorkbenchAction>
        {connected && (
          <WorkbenchIconAction
            label="Disconnect MCP server"
            className="text-destructive hover:bg-destructive/10 hover:text-destructive"
            type="button"
            disabled={busy}
            onClick={() => void disconnect()}
          >
            <Unplug className="size-4" />
          </WorkbenchIconAction>
        )}
        {progress && <span className="text-label text-muted-foreground">{progress}</span>}
      </div>
      {error && <ErrorBanner>{error}</ErrorBanner>}
    </div>
  );
};

interface OAuthCallbackPayload {
  code: string;
  state: string | null;
}

// Listen for the SPA's /oauth-callback page to postMessage the auth code,
// or fall back to a 5-minute timeout / popup-closed detection.
const waitForOAuthCallback = (popup: Window): Promise<OAuthCallbackPayload> => {
  return new Promise((resolve, reject) => {
    const FIVE_MIN_MS = 5 * 60 * 1000;
    let pollHandle: number | null = null;
    let timeoutHandle: number | null = null;

    const cleanup = () => {
      window.removeEventListener("message", onMessage);
      if (pollHandle !== null) window.clearInterval(pollHandle);
      if (timeoutHandle !== null) window.clearTimeout(timeoutHandle);
    };

    const onMessage = (event: MessageEvent) => {
      // Same-origin only — the SPA route lives on this origin.
      if (event.origin !== window.location.origin) return;
      const data = event.data;
      if (
        data &&
        typeof data === "object" &&
        data.type === "molexp:oauth-callback" &&
        typeof data.code === "string"
      ) {
        cleanup();
        try {
          popup.close();
        } catch {
          /* ignore */
        }
        resolve({ code: data.code, state: typeof data.state === "string" ? data.state : null });
      }
    };

    window.addEventListener("message", onMessage);
    pollHandle = window.setInterval(() => {
      if (popup.closed) {
        cleanup();
        reject(new Error("Authorization window closed before completion."));
      }
    }, 500);
    timeoutHandle = window.setTimeout(() => {
      cleanup();
      try {
        popup.close();
      } catch {
        /* ignore */
      }
      reject(new Error("Authorization timed out after 5 minutes."));
    }, FIVE_MIN_MS);
  });
};

// ───────────────────────────────────────────────────────────────────────────
//  Reusable key-value editor
// ───────────────────────────────────────────────────────────────────────────

interface KvEditorProps {
  label: string;
  rows: KvRow[];
  setRows: (next: KvRow[]) => void;
  valuePlaceholder: string;
}

const KvEditor = ({ label, rows, setRows, valuePlaceholder }: KvEditorProps): JSX.Element => {
  const update = (uid: string, patch: Partial<KvRow>) => {
    setRows(rows.map((r) => (r.uid === uid ? { ...r, ...patch } : r)));
  };
  const remove = (uid: string) => setRows(rows.filter((r) => r.uid !== uid));
  const add = () => setRows([...rows, { uid: _kvUid(), key: "", value: "" }]);

  return (
    <fieldset>
      <div className="mb-1 flex items-center justify-between">
        <legend className="text-label font-medium">{label}</legend>
        <WorkbenchIconAction label={`Add ${label} entry`} onClick={add}>
          <Plus className="size-3" />
        </WorkbenchIconAction>
      </div>
      {rows.length === 0 ? (
        <p className="text-label text-muted-foreground">None.</p>
      ) : (
        <div className="space-y-1">
          {rows.map((row) => (
            <div key={row.uid} className="flex items-center gap-2">
              <Input
                value={row.key}
                aria-label={`${label} key`}
                onChange={(e) => update(row.uid, { key: e.target.value })}
                placeholder="KEY"
                className="font-mono"
              />
              <Input
                value={row.value}
                aria-label={`${label} value for ${row.key || "new key"}`}
                onChange={(e) => update(row.uid, { value: e.target.value })}
                placeholder={valuePlaceholder}
                className="font-mono"
              />
              <IconButton
                icon={Trash2}
                label="Remove"
                tone="destructive"
                onClick={() => remove(row.uid)}
              />
            </div>
          ))}
        </div>
      )}
    </fieldset>
  );
};

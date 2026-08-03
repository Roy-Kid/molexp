/**
 * AgentSettingsViewer — read/write management for the agent runtime.
 *
 * Surfaces: model (Chat/Plan overview + providers), instructions, skills,
 * MCP. Chat vs Plan is switched in the composer; this page does not duplicate
 * that control. Network-backed tabs mount independently so a missing
 * agent-admin service yields one quiet state instead of repeated 503s.
 *
 * Tab descriptors live in `agent_settings/tabs.ts` for unit tests without
 * the full component graph.
 */

import {
  AlertCircle,
  Bot,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Database,
  Eye,
  EyeOff,
  FileText,
  PlayCircle,
  Plus,
  Settings,
  Slash,
  Trash2,
  Zap,
} from "lucide-react";
import type { JSX, ReactNode } from "react";
import { useCallback, useEffect, useId, useMemo, useState } from "react";
import { EmptyState, EntityPage } from "@/app/components/entity";
import { McpServersTab } from "@/app/renderers/agent_settings/McpServersTab";
import {
  baseUrlPlaceholder,
  DEFAULT_PROVIDER_REGISTRY,
  findRegistryEntry,
  type ProviderRegistryResponse,
  supportsBaseUrl,
} from "@/app/renderers/agent_settings/providerRegistry";
import { AGENT_SETTINGS_TABS, type AgentSettingsTabDef } from "@/app/renderers/agent_settings/tabs";
import { UnavailableCapability } from "@/app/renderers/agent_settings/UnavailableCapability";
import { AgentUnavailableError, resetAgentProbes } from "@/app/state/agentProbe";
import {
  type ApiAgentProvider,
  type ApiAgentProviderTestResult,
  type ApiModelTier,
  type ApiProviderConfiguration,
  type ApiProviderName,
  type ApiSkill,
  type ApiTierModels,
  agentAdminApi,
  type ProviderUpdateInput,
  RESERVED_SLASH_NAMES,
  type SkillUpsertInput,
  SLASH_NAME_PATTERN,
} from "@/app/state/api";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { WorkbenchAction, WorkbenchIconAction, WorkbenchTag } from "@/components/workbench";

interface SkillFormState {
  name: string;
  description: string;
  goalTemplate: string;
  slashName: string;
  instructions: string;
  defaultPlanMode: boolean;
  constraints: string;
  successCriteria: string;
  tags: string;
}

const EMPTY_SKILL_FORM: SkillFormState = {
  name: "",
  description: "",
  goalTemplate: "",
  slashName: "",
  instructions: "",
  defaultPlanMode: false,
  constraints: "",
  successCriteria: "",
  tags: "",
};

const lines = (text: string): string[] =>
  text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

const formToInput = (form: SkillFormState): SkillUpsertInput => ({
  name: form.name.trim(),
  goalTemplate: form.goalTemplate.trim(),
  description: form.description.trim(),
  slashName: form.slashName.trim(),
  instructions: form.instructions.trim(),
  defaultPlanMode: form.defaultPlanMode,
  constraints: lines(form.constraints),
  successCriteria: lines(form.successCriteria),
  tags: form.tags
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean),
});

/**
 * Validate the user's slash_name input client-side. Mirrors the regex /
 * reserved-name policy enforced by ``SkillStore`` so the user gets
 * immediate feedback without a server round-trip.
 */
const validateSlashName = (name: string): string | null => {
  const trimmed = name.trim();
  if (!trimmed) return null; // empty is allowed (launcher-only)
  if (!SLASH_NAME_PATTERN.test(trimmed)) {
    return "Use lowercase letters, digits, and hyphens. Max 32 chars.";
  }
  if ((RESERVED_SLASH_NAMES as readonly string[]).includes(trimmed)) {
    return `'/${trimmed}' is reserved by the chat input.`;
  }
  return null;
};

interface AgentSettingsViewerProps {
  onLaunchSession?: (sessionId: string) => void;
}

const TAB_ICON: Record<AgentSettingsTabDef["value"], typeof Settings> = {
  model: Cpu,
  instructions: FileText,
  skills: Slash,
  mcp: Database,
};

const renderTabContent = (
  contentKey: AgentSettingsTabDef["contentKey"],
  onLaunchSession?: (sessionId: string) => void,
): JSX.Element => {
  switch (contentKey) {
    case "providers-form":
      return <ProviderTab />;
    case "instructions-form":
      return (
        <SettingsScroll>
          <InstructionsTab />
        </SettingsScroll>
      );
    case "skills-list":
      return (
        <SettingsScroll wide>
          <SkillsTab onLaunchSession={onLaunchSession} />
        </SettingsScroll>
      );
    case "mcp-servers":
      return <McpServersTab />;
  }
};

export const AgentSettingsViewer = ({ onLaunchSession }: AgentSettingsViewerProps): JSX.Element => {
  const tabs = AGENT_SETTINGS_TABS.map((def) => {
    const Icon = TAB_ICON[def.value];
    return {
      value: def.value,
      label: (
        <span className="flex items-center">
          <Icon className="mr-2 h-4 w-4" /> {def.label}
        </span>
      ),
      content: renderTabContent(def.contentKey, onLaunchSession),
    };
  });
  return (
    <EntityPage
      icon={Settings}
      title="Agent settings"
      subtitle="Model, instructions, skills, MCP tools, and knowledge package scope"
      tabs={tabs}
    />
  );
};

const SettingsScroll = ({
  children,
  wide = false,
}: {
  children: React.ReactNode;
  wide?: boolean;
}) => (
  <ScrollArea className="flex-1">
    <div className={`mx-auto w-full ${wide ? "max-w-5xl" : "max-w-3xl"} px-4 pb-8 pt-4`}>
      {children}
    </div>
  </ScrollArea>
);

// ─── Provider tab ──────────────────────────────────────────────────────────
//
// Layout (user-facing):
//   1. Providers — API keys / base URLs (vendors)
//   2. Agents — which model Chat / Plan call (maps to router tiers)
//
// Wire mapping (unchanged backend):
//   Chat + Plan review  → models.default (+ legacy agent.model)
//   Plan authoring      → models.heavy
//   Light routing       → models.cheap
//
// Field schema / labels live in `providerRegistry.ts`.

const providerLabel = (registry: ProviderRegistryResponse, name: string): string =>
  findRegistryEntry(registry, name)?.label ?? name;

const providerModelHint = (registry: ProviderRegistryResponse, name: string): string =>
  findRegistryEntry(registry, name)?.modelHint ?? "";

/** UI rows for agent model assignment (not the internal tier jargon). */
const AGENT_MODEL_ROWS: readonly {
  tier: ApiModelTier;
  agent: "Chat" | "Plan";
  label: string;
}[] = [
  { tier: "default", agent: "Chat", label: "Model" },
  { tier: "heavy", agent: "Plan", label: "Authoring" },
  { tier: "cheap", agent: "Plan", label: "Light" },
];

const emptyTierModels = (): ApiTierModels => ({ cheap: "", default: "", heavy: "" });

/** Split ``provider:model``; bare ids use ``fallbackProvider``. */
const parseQualifiedModel = (
  value: string,
  fallbackProvider: ApiProviderName | "",
): { provider: ApiProviderName | ""; modelId: string } => {
  const text = value.trim();
  if (text.includes(":")) {
    const [p, ...rest] = text.split(":");
    return { provider: p as ApiProviderName, modelId: rest.join(":") };
  }
  return { provider: fallbackProvider, modelId: text };
};

const qualifyModel = (provider: string, modelId: string): string => {
  const id = modelId.trim();
  if (!id) return "";
  if (id.includes(":")) return id;
  return provider ? `${provider}:${id}` : id;
};

const ProviderTab = (): JSX.Element => {
  const registry = DEFAULT_PROVIDER_REGISTRY;
  const [config, setConfig] = useState<ApiAgentProvider | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    setUnavailable(false);
    try {
      setConfig(await agentAdminApi.getProvider());
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

  if (loading) {
    return <div className="px-4 py-3 text-sm text-muted-foreground">Loading model config…</div>;
  }
  if (unavailable) {
    return (
      <SettingsScroll>
        <UnavailableCapability
          title="Model configuration unavailable"
          description="This server does not expose agent model administration."
          onRetry={() => {
            resetAgentProbes();
            void refresh();
          }}
        />
      </SettingsScroll>
    );
  }

  const supported =
    config?.supportedProviders ??
    (registry.providers.map((entry) => entry.name) as ApiProviderName[]);
  const configurations = config?.configurations ?? [];
  const globalModels = config?.models ?? emptyTierModels();

  const fallbackProvider = (config?.provider as ApiProviderName) || supported[0] || "deepseek";

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-8 px-4 pb-8 pt-3">
        {error && <p className="text-xs text-destructive">{error}</p>}

        {/* 1. Vendors first */}
        <section className="space-y-3">
          <h2 className="text-base font-semibold">Providers</h2>
          <div className="space-y-2">
            {supported.map((provider) => {
              const stored = configurations.find((entry) => entry.provider === provider);
              const usedBy = AGENT_MODEL_ROWS.filter(({ tier }) =>
                (globalModels[tier] || "").startsWith(`${provider}:`),
              ).map(({ agent, label }) => `${agent} · ${label}`);
              return (
                <CredentialCard
                  key={provider}
                  provider={provider}
                  usedByTiers={usedBy}
                  initial={
                    stored ?? {
                      provider,
                      models: emptyTierModels(),
                      baseUrl: "",
                      apiKeyPreview: "",
                      apiKeySet: false,
                    }
                  }
                  registry={registry}
                  onChanged={setConfig}
                />
              );
            })}
          </div>
        </section>

        {/* 2. Per-agent model assignment */}
        <AgentModelTable
          supported={supported}
          initial={globalModels}
          fallbackProvider={fallbackProvider}
          registry={registry}
          onChanged={setConfig}
        />
      </div>
    </ScrollArea>
  );
};

/** Per-agent model picks — Chat / Plan rows, not abstract tier names. */
const AgentModelTable = ({
  supported,
  initial,
  fallbackProvider,
  registry,
  onChanged,
}: {
  supported: ApiProviderName[];
  initial: ApiTierModels;
  fallbackProvider: ApiProviderName;
  registry: ProviderRegistryResponse;
  onChanged: (config: ApiAgentProvider) => void;
}): JSX.Element => {
  const [rows, setRows] = useState(
    () =>
      Object.fromEntries(
        AGENT_MODEL_ROWS.map(({ tier }) => {
          const parsed = parseQualifiedModel(initial[tier] || "", fallbackProvider);
          return [tier, parsed];
        }),
      ) as Record<ApiModelTier, { provider: ApiProviderName | ""; modelId: string }>,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [testResult, setTestResult] = useState<ApiAgentProviderTestResult | null>(null);

  useEffect(() => {
    setRows(
      Object.fromEntries(
        AGENT_MODEL_ROWS.map(({ tier }) => {
          const parsed = parseQualifiedModel(initial[tier] || "", fallbackProvider);
          return [tier, parsed];
        }),
      ) as Record<ApiModelTier, { provider: ApiProviderName | ""; modelId: string }>,
    );
  }, [initial, fallbackProvider]);

  const models: ApiTierModels = {
    cheap: qualifyModel(rows.cheap.provider, rows.cheap.modelId),
    default: qualifyModel(rows.default.provider, rows.default.modelId),
    heavy: qualifyModel(rows.heavy.provider, rows.heavy.modelId),
  };
  const complete = AGENT_MODEL_ROWS.every(({ tier }) => models[tier].includes(":"));

  const submit = async (mode: "save" | "test"): Promise<void> => {
    setBusy(true);
    setError(null);
    setSaved(false);
    setTestResult(null);
    try {
      // Primary (default) is also the legacy agent.model used by chat sessions.
      const input: ProviderUpdateInput = { models, model: models.default };
      if (mode === "test") {
        setTestResult(await agentAdminApi.testProvider(input));
      } else {
        onChanged(await agentAdminApi.updateProvider(input));
        setSaved(true);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const chatRows = AGENT_MODEL_ROWS.filter((r) => r.agent === "Chat");
  const planRows = AGENT_MODEL_ROWS.filter((r) => r.agent === "Plan");

  const renderRow = (row: (typeof AGENT_MODEL_ROWS)[number]): JSX.Element => (
    <div
      key={row.tier}
      className="grid items-center gap-2 py-2 sm:grid-cols-[5.5rem_minmax(7rem,9rem)_minmax(0,1fr)] sm:gap-3"
    >
      <p className="text-xs font-medium text-foreground">{row.label}</p>
      <select
        className="h-9 w-full rounded-md bg-muted/50 px-2 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
        value={rows[row.tier].provider}
        aria-label={`${row.agent} ${row.label} provider`}
        onChange={(event) =>
          setRows((value) => ({
            ...value,
            [row.tier]: {
              ...value[row.tier],
              provider: event.target.value as ApiProviderName,
            },
          }))
        }
      >
        {supported.map((name) => (
          <option key={name} value={name}>
            {providerLabel(registry, name)}
          </option>
        ))}
      </select>
      <Input
        value={rows[row.tier].modelId}
        onChange={(event) =>
          setRows((value) => ({
            ...value,
            [row.tier]: { ...value[row.tier], modelId: event.target.value },
          }))
        }
        placeholder={providerModelHint(registry, rows[row.tier].provider || fallbackProvider)}
        className="border-0 bg-muted/50 font-mono text-xs shadow-none focus-visible:ring-2 focus-visible:ring-ring/40"
        aria-label={`${row.agent} ${row.label} model id`}
      />
    </div>
  );

  return (
    <section className="space-y-3">
      <h2 className="text-base font-semibold">Agents</h2>

      <div className="space-y-0.5 rounded-lg bg-muted/30 px-3 py-2">
        <div className="flex items-center gap-2 py-1.5">
          <Bot className="size-3.5 text-muted-foreground" aria-hidden />
          <h3 className="text-sm font-medium">Chat</h3>
        </div>
        {chatRows.map(renderRow)}
      </div>

      <div className="space-y-0.5 rounded-lg bg-info-soft/20 px-3 py-2">
        <div className="flex items-center gap-2 py-1.5">
          <BrainCircuit className="size-3.5 text-info" aria-hidden />
          <h3 className="text-sm font-medium">Plan</h3>
        </div>
        {planRows.map(renderRow)}
      </div>

      {error && <p className="text-xs text-destructive">{error}</p>}
      {saved && (
        <p className="flex items-center gap-1 text-xs text-success-foreground">
          <CheckCircle2 className="size-3.5" /> Saved.
        </p>
      )}
      {testResult && <ProviderTestResult result={testResult} />}
      <div className="flex justify-end gap-2">
        <WorkbenchAction
          kind="secondary"
          size="compact"
          disabled={busy || !complete}
          onClick={() => void submit("test")}
        >
          <Zap className="mr-1 size-4" /> Test
        </WorkbenchAction>
        <WorkbenchAction
          kind="primary"
          size="compact"
          disabled={busy || !complete}
          onClick={() => void submit("save")}
        >
          {busy ? "Saving…" : "Save"}
        </WorkbenchAction>
      </div>
    </section>
  );
};

/** Per-provider API key / base URL only. */
const CredentialCard = ({
  provider,
  usedByTiers,
  initial,
  registry,
  onChanged,
}: {
  provider: ApiProviderName;
  usedByTiers: string[];
  initial: ApiProviderConfiguration;
  registry: ProviderRegistryResponse;
  onChanged: (config: ApiAgentProvider) => void;
}): JSX.Element => {
  const fieldId = useId();
  const baseUrlId = `${fieldId}-base-url`;
  const apiKeyId = `${fieldId}-api-key`;
  const [expanded, setExpanded] = useState(usedByTiers.length > 0 || initial.apiKeySet);
  const [baseUrl, setBaseUrl] = useState(initial.baseUrl);
  const [apiKey, setApiKey] = useState("");
  const [revealKey, setRevealKey] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [confirmClearKey, setConfirmClearKey] = useState(false);
  const showBaseUrl = supportsBaseUrl(registry, provider);

  useEffect(() => {
    setBaseUrl(initial.baseUrl);
  }, [initial]);

  const submit = async (mode: "save" | "clear"): Promise<void> => {
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      const input: ProviderUpdateInput = {
        provider,
        baseUrl: baseUrl.trim(),
      };
      if (mode === "clear") input.apiKey = "";
      else if (apiKey !== "") input.apiKey = apiKey;
      onChanged(await agentAdminApi.updateProvider(input));
      setApiKey("");
      setRevealKey(false);
      setSaved(true);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="rounded-lg bg-muted/35">
      <header className="px-3 py-2.5">
        <button
          type="button"
          className="flex w-full items-center gap-3 text-left"
          aria-expanded={expanded}
          onClick={() => setExpanded((value) => !value)}
        >
          <Cpu className="size-4 text-muted-foreground" />
          <div className="min-w-0 flex-1">
            <h3 className="text-sm font-medium text-foreground">
              {providerLabel(registry, provider)}
            </h3>
            <p className="mt-0.5 truncate text-xs text-muted-foreground">
              {initial.apiKeySet ? `Key ${initial.apiKeyPreview}` : "No stored key"}
              {usedByTiers.length > 0 ? ` · ${usedByTiers.join(" · ")}` : ""}
            </p>
          </div>
          {usedByTiers.length > 0 && <WorkbenchTag className="text-micro">In use</WorkbenchTag>}
          <ChevronRight className={`size-4 transition-transform ${expanded ? "rotate-90" : ""}`} />
        </button>
      </header>
      {expanded && (
        <div className="space-y-4 px-3 pb-3 pt-1">
          {showBaseUrl && (
            <div>
              <Label htmlFor={baseUrlId} className="text-xs">
                Base URL
              </Label>
              <Input
                id={baseUrlId}
                value={baseUrl}
                onChange={(event) => setBaseUrl(event.target.value)}
                placeholder={baseUrlPlaceholder(registry, provider)}
              />
            </div>
          )}
          <div>
            <Label htmlFor={apiKeyId} className="text-xs">
              API key
            </Label>
            <div className="flex gap-2">
              <Input
                id={apiKeyId}
                type={revealKey ? "text" : "password"}
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                placeholder={
                  initial.apiKeySet
                    ? `Stored: ${initial.apiKeyPreview} — type to replace`
                    : "Paste API key"
                }
                autoComplete="off"
              />
              <WorkbenchIconAction
                label={revealKey ? "Hide API key" : "Show API key"}
                kind="ghost"
                type="button"
                onClick={() => setRevealKey((value) => !value)}
              >
                {revealKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
              </WorkbenchIconAction>
            </div>
          </div>
          {error && <p className="text-xs text-destructive">{error}</p>}
          {saved && (
            <p className="flex items-center gap-1 text-xs text-success-foreground">
              <CheckCircle2 className="size-3.5" /> Credentials saved.
            </p>
          )}
          <ConfirmDialog
            open={confirmClearKey}
            onOpenChange={setConfirmClearKey}
            title={`Clear ${providerLabel(registry, provider)} API key?`}
            description="This provider will fall back to its environment variable."
            confirmLabel="Clear key"
            destructive
            onConfirm={() => void submit("clear")}
          />
          <div className="flex flex-wrap justify-between gap-2">
            <WorkbenchAction
              kind="ghost"
              size="compact"
              disabled={busy || !initial.apiKeySet}
              onClick={() => setConfirmClearKey(true)}
            >
              Clear key
            </WorkbenchAction>
            <WorkbenchAction
              kind="primary"
              size="compact"
              disabled={busy}
              onClick={() => void submit("save")}
            >
              {busy ? "Saving…" : "Save credentials"}
            </WorkbenchAction>
          </div>
        </div>
      )}
    </section>
  );
};

const ProviderTestResult = ({ result }: { result: ApiAgentProviderTestResult }): JSX.Element => {
  const ok = result.ok;
  return (
    <div
      className={
        "border-y px-3 py-2 text-xs " +
        (ok
          ? "border-success/30 bg-success-soft text-success-foreground"
          : "border-destructive/40 bg-destructive/10 text-destructive")
      }
    >
      <div className="flex items-center gap-2 font-medium">
        {ok ? <CheckCircle2 className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
        {ok ? "Connection OK" : "Connection failed"}
        <span className="ml-auto font-mono text-micro opacity-80">
          {result.provider}:{result.model} · {result.latencyMs} ms
        </span>
      </div>
      {ok && result.reply && (
        <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-micro opacity-80">
          {result.reply}
        </pre>
      )}
      {!ok && result.error && (
        <p className="mt-1 break-words font-mono text-micro">{result.error}</p>
      )}
    </div>
  );
};

// ─── Instructions tab ──────────────────────────────────────────────────────

/**
 * Workspace-default system prompt addendum. Saved into
 * ``.agent_provider.json`` alongside the credentials and threaded into
 * every new session via the layered prompt composer.
 */
const InstructionsTab = (): JSX.Element => {
  const [config, setConfig] = useState<ApiAgentProvider | null>(null);
  const [draft, setDraft] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    agentAdminApi
      .getProvider()
      .then((next) => {
        if (cancelled) return;
        setConfig(next);
        setDraft(next.instructions);
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof AgentUnavailableError) setUnavailable(true);
        else setError(String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setError(null);
    try {
      const updated = await agentAdminApi.updateProvider({ instructions: draft });
      setConfig(updated);
      setDraft(updated.instructions);
      setSavedAt(Date.now());
    } catch (err) {
      setError(String(err));
    } finally {
      setSaving(false);
    }
  }, [draft]);

  const [confirmClear, setConfirmClear] = useState(false);

  const handleClear = useCallback(async () => {
    setSaving(true);
    setError(null);
    try {
      const updated = await agentAdminApi.updateProvider({ instructions: "" });
      setConfig(updated);
      setDraft("");
      setSavedAt(Date.now());
    } catch (err) {
      setError(String(err));
    } finally {
      setSaving(false);
    }
  }, []);

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading instructions…</p>;
  }

  if (unavailable) {
    return (
      <UnavailableCapability
        title="Workspace instructions unavailable"
        description="This server does not expose editable agent instructions. Add instructions through the server configuration, or install the agent administration dependencies."
        onRetry={() => {
          resetAgentProbes();
          setUnavailable(false);
          setLoading(true);
          agentAdminApi
            .getProvider()
            .then((next) => {
              setConfig(next);
              setDraft(next.instructions);
            })
            .catch((err) => {
              if (err instanceof AgentUnavailableError) setUnavailable(true);
              else setError(String(err));
            })
            .finally(() => setLoading(false));
        }}
      />
    );
  }

  const dirty = (config?.instructions ?? "") !== draft;

  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-muted-foreground">
        Workspace-default system prompt addendum. Appended to the molexp built-in preamble for every
        new session. Skills can layer additional instructions on top, and individual sessions may
        override the whole stack from the chat input.
      </p>

      <section className="space-y-3 border-t border-border/60 pt-3">
        <header>
          <h3 className="text-sm font-medium text-foreground">Workspace instructions</h3>
        </header>
        <div className="space-y-3">
          <Textarea
            rows={10}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={
              "Always cite source data with project/experiment/run ids.\n" +
              "Prefer existing workflow templates before writing new code."
            }
            className="font-mono text-xs"
          />
          <p className="text-micro text-muted-foreground">
            Saved alongside the provider credentials; never sent to the model directly — only
            attached as the agent's system prompt.
          </p>
          {error && <p className="text-xs text-destructive">{error}</p>}
          {savedAt && !error && (
            <p className="flex items-center gap-1 text-xs text-success-foreground">
              <CheckCircle2 className="h-3.5 w-3.5 text-success" />
              Saved. New sessions will use these instructions.
            </p>
          )}
          <ConfirmDialog
            open={confirmClear}
            onOpenChange={setConfirmClear}
            title="Clear the workspace instructions?"
            description="New sessions will start from the molexp built-in preamble only."
            confirmLabel="Clear instructions"
            destructive
            onConfirm={() => void handleClear()}
          />
          <div className="flex justify-between gap-2 pt-1">
            <WorkbenchAction
              kind="ghost"
              size="compact"
              disabled={saving || (config?.instructions ?? "") === ""}
              onClick={() => setConfirmClear(true)}
            >
              Clear
            </WorkbenchAction>
            <WorkbenchAction
              kind="primary"
              size="compact"
              disabled={saving || !dirty}
              onClick={() => void handleSave()}
            >
              {saving ? "Saving…" : "Save"}
            </WorkbenchAction>
          </div>
        </div>
      </section>
    </div>
  );
};

// ─── Skills ────────────────────────────────────────────────────────────────

/** Unique `{{placeholder}}` names in a goal template, in first-seen order. */
const templatePlaceholders = (goalTemplate: string): string[] =>
  Array.from(goalTemplate.matchAll(/\{\{\s*([A-Za-z_]\w*)\s*\}\}/g))
    .map((m) => m[1])
    .filter((v, i, a) => a.indexOf(v) === i);

/**
 * Parameter form shown before launching a skill whose goal template
 * carries `{{placeholders}}` — replaces the old chain of window.prompt
 * calls with one in-app dialog.
 */
const SkillLaunchDialog = ({
  skill,
  onOpenChange,
  onLaunch,
}: {
  skill: ApiSkill;
  onOpenChange: (open: boolean) => void;
  onLaunch: (params: Record<string, string>) => Promise<void>;
}): JSX.Element => {
  const placeholders = useMemo(() => templatePlaceholders(skill.goalTemplate), [skill]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [launching, setLaunching] = useState(false);

  const handleLaunch = async (): Promise<void> => {
    setLaunching(true);
    try {
      await onLaunch(values);
      onOpenChange(false);
    } finally {
      setLaunching(false);
    }
  };

  return (
    <Dialog open onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Launch {skill.name}</DialogTitle>
          <DialogDescription className="font-mono text-xs">{skill.goalTemplate}</DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          {placeholders.map((key) => (
            <div key={key}>
              <Label htmlFor={`param-${key}`} className="font-mono text-xs">
                {`{{${key}}}`}
              </Label>
              <Input
                id={`param-${key}`}
                value={values[key] ?? ""}
                onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
                autoFocus={key === placeholders[0]}
              />
            </div>
          ))}
        </div>
        <DialogFooter>
          <WorkbenchAction kind="ghost" size="compact" onClick={() => onOpenChange(false)}>
            Cancel
          </WorkbenchAction>
          <WorkbenchAction
            kind="primary"
            size="compact"
            disabled={launching}
            onClick={() => void handleLaunch()}
          >
            <PlayCircle className="mr-1 h-4 w-4" />
            {launching ? "Launching…" : "Launch session"}
          </WorkbenchAction>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

const CapabilityListHeader = ({
  title,
  description,
  count,
  actions,
}: {
  title: string;
  description: ReactNode;
  count: string;
  actions: ReactNode;
}): JSX.Element => (
  <div className="mb-3 flex flex-wrap items-start justify-between gap-3 border-b border-border pb-3">
    <div className="min-w-0 space-y-1">
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-medium text-foreground">{title}</h3>
        <WorkbenchTag className="text-micro font-normal">{count}</WorkbenchTag>
      </div>
      <div className="max-w-2xl text-sm text-muted-foreground">{description}</div>
    </div>
    <div className="flex shrink-0 items-center gap-2">{actions}</div>
  </div>
);

const SkillsTab = ({
  onLaunchSession,
}: {
  onLaunchSession?: (sessionId: string) => void;
}): JSX.Element => {
  const [skills, setSkills] = useState<ApiSkill[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<ApiSkill | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [deleting, setDeleting] = useState<ApiSkill | null>(null);
  const [launchingSkill, setLaunchingSkill] = useState<ApiSkill | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSkills(await agentAdminApi.listSkills());
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const handleDelete = useCallback(
    async (skill: ApiSkill) => {
      try {
        await agentAdminApi.deleteSkill(skill.id);
        await refresh();
      } catch (err) {
        setError(String(err));
      }
    },
    [refresh],
  );

  const launchWithParams = useCallback(
    async (skill: ApiSkill, params: Record<string, string>) => {
      try {
        const session = await agentAdminApi.launchSkill(skill.id, params);
        onLaunchSession?.(session.sessionId);
      } catch (err) {
        setError(String(err));
      }
    },
    [onLaunchSession],
  );

  const handleLaunch = useCallback(
    (skill: ApiSkill): void => {
      if (templatePlaceholders(skill.goalTemplate).length === 0) {
        void launchWithParams(skill, {});
        return;
      }
      setLaunchingSkill(skill);
    },
    [launchWithParams],
  );

  return (
    <div className="flex flex-col">
      <CapabilityListHeader
        title="Skills"
        description={
          <>
            Reusable workflows and domain instructions. Use
            <code className="mx-1 rounded bg-muted px-1">{"{{name}}"}</code>
            placeholders; a slash name makes the skill invokable from chat.
          </>
        }
        count={`${skills.length} configured`}
        actions={
          <WorkbenchAction
            kind="primary"
            size="compact"
            onClick={() => {
              setEditing(null);
              setShowForm(true);
            }}
          >
            <Plus className="mr-1 h-4 w-4" /> New skill
          </WorkbenchAction>
        }
      />
      {error && <p className="mb-2 text-xs text-destructive">{error}</p>}
      {showForm && (
        <SkillForm
          initial={editing}
          onCancel={() => {
            setShowForm(false);
            setEditing(null);
          }}
          onSaved={async () => {
            setShowForm(false);
            setEditing(null);
            await refresh();
          }}
        />
      )}
      <ConfirmDialog
        open={deleting !== null}
        onOpenChange={(open) => {
          if (!open) setDeleting(null);
        }}
        title={`Delete "${deleting?.name ?? ""}"?`}
        description="The reusable workflow and its slash name are removed. Existing tasks are not affected."
        confirmLabel="Delete skill"
        destructive
        onConfirm={() => {
          if (deleting) void handleDelete(deleting);
          setDeleting(null);
        }}
      />
      {launchingSkill && (
        <SkillLaunchDialog
          skill={launchingSkill}
          onOpenChange={(open) => {
            if (!open) setLaunchingSkill(null);
          }}
          onLaunch={(params) => launchWithParams(launchingSkill, params)}
        />
      )}
      <div className="flex flex-col gap-2">
        {loading && <p className="text-sm text-muted-foreground">Loading…</p>}
        {!loading && skills.length === 0 && (
          <EmptyState
            density="compact"
            icon={<Slash className="h-5 w-5" />}
            title="No skills yet"
            description="Create a reusable workflow to launch it here or invoke it as /name from chat."
          />
        )}
        {skills.map((skill) => (
          <section
            key={skill.id}
            className="rounded-[var(--radius-panel)] border border-border bg-surface border-border"
          >
            <header className="pb-2 px-3 pt-3">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  {skill.slashName ? (
                    <WorkbenchTag
                      meaning="metadata"
                      className="font-mono text-micro"
                      title="Type this in chat to invoke"
                    >
                      /{skill.slashName}
                    </WorkbenchTag>
                  ) : (
                    <WorkbenchTag className="text-micro">launcher only</WorkbenchTag>
                  )}
                  <h3 className="truncate text-sm font-medium text-foreground">{skill.name}</h3>
                  {skill.defaultPlanMode && (
                    <WorkbenchTag meaning="metadata" className="text-micro">
                      plan
                    </WorkbenchTag>
                  )}
                </div>
                <div className="flex gap-1">
                  <WorkbenchAction
                    kind="ghost"
                    size="compact"
                    onClick={() => handleLaunch(skill)}
                    title="Launch a task from this skill"
                    aria-label={`Launch ${skill.name}`}
                  >
                    <PlayCircle className="h-4 w-4" />
                  </WorkbenchAction>
                  <WorkbenchAction
                    kind="ghost"
                    size="compact"
                    onClick={() => {
                      setEditing(skill);
                      setShowForm(true);
                    }}
                    title="Edit skill"
                  >
                    Edit
                  </WorkbenchAction>
                  <WorkbenchAction
                    kind="ghost"
                    size="compact"
                    onClick={() => setDeleting(skill)}
                    title="Delete skill"
                    aria-label={`Delete ${skill.name}`}
                  >
                    <Trash2 className="h-4 w-4" />
                  </WorkbenchAction>
                </div>
              </div>
            </header>
            <div className="px-3 pb-3 pt-0">
              {skill.description && (
                <p className="mb-2 text-xs text-muted-foreground">{skill.description}</p>
              )}
              <pre className="mb-2 whitespace-pre-wrap rounded bg-muted px-2 py-1 text-xs">
                {skill.goalTemplate}
              </pre>
              {skill.instructions && (
                <p className="mb-2 text-micro italic text-muted-foreground">
                  +{skill.instructions.length} chars of additional instructions
                </p>
              )}
              <div className="flex flex-wrap gap-1">
                {skill.tags.map((tag) => (
                  <WorkbenchTag key={tag} className="text-micro">
                    {tag}
                  </WorkbenchTag>
                ))}
              </div>
            </div>
          </section>
        ))}
      </div>
    </div>
  );
};

const SkillForm = ({
  initial,
  onCancel,
  onSaved,
}: {
  initial: ApiSkill | null;
  onCancel: () => void;
  onSaved: () => Promise<void>;
}): JSX.Element => {
  const fieldId = useId();
  const id = (name: string): string => `${fieldId}-${name}`;
  const [form, setForm] = useState<SkillFormState>(() =>
    initial
      ? {
          name: initial.name,
          description: initial.description,
          goalTemplate: initial.goalTemplate,
          slashName: initial.slashName,
          instructions: initial.instructions,
          defaultPlanMode: initial.defaultPlanMode,
          constraints: initial.constraints.join("\n"),
          successCriteria: initial.successCriteria.join("\n"),
          tags: initial.tags.join(", "),
        }
      : EMPTY_SKILL_FORM,
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const slashError = validateSlashName(form.slashName);

  const handleSubmit = useCallback(async () => {
    if (!form.name.trim() || !form.goalTemplate.trim()) {
      setError("Name and goal template are required");
      return;
    }
    if (slashError) {
      setError(slashError);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const input = formToInput(form);
      if (initial) {
        await agentAdminApi.updateSkill(initial.id, input);
      } else {
        await agentAdminApi.createSkill(input);
      }
      await onSaved();
    } catch (err) {
      setError(String(err));
    } finally {
      setSaving(false);
    }
  }, [form, initial, onSaved, slashError]);

  return (
    <section className="mb-2 border-y border-primary/40 py-3">
      <header className="pb-2">
        <h3 className="text-sm font-medium text-foreground">
          {initial ? "Edit skill" : "New skill"}
        </h3>
      </header>
      <div className="space-y-2">
        <div>
          <Label htmlFor={id("name")} className="text-xs">
            Name
          </Label>
          <Input
            id={id("name")}
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="Plot energy vs temperature"
          />
        </div>
        <div>
          <Label htmlFor={id("slash-name")} className="text-xs">
            Slash name (optional) — invokes as{" "}
            <code className="rounded bg-muted px-1">/&lt;name&gt;</code>
          </Label>
          <div className="flex items-center gap-2">
            <span className="select-none font-mono text-sm text-muted-foreground">/</span>
            <Input
              id={id("slash-name")}
              value={form.slashName}
              onChange={(e) => setForm({ ...form, slashName: e.target.value })}
              placeholder="plot-energy"
              className="font-mono"
            />
          </div>
          {slashError ? (
            <p className="mt-1 text-micro text-destructive">{slashError}</p>
          ) : (
            <p className="mt-1 text-micro text-muted-foreground">
              Reserved: {RESERVED_SLASH_NAMES.join(", ")}. Leave empty to keep this as
              launcher-only.
            </p>
          )}
        </div>
        <div>
          <Label htmlFor={id("description")} className="text-xs">
            Description
          </Label>
          <Input
            id={id("description")}
            value={form.description}
            onChange={(e) => setForm({ ...form, description: e.target.value })}
            placeholder="Optional summary"
          />
        </div>
        <div>
          <Label htmlFor={id("goal-template")} className="text-xs">
            Goal template — use {"{{param}}"} for placeholders
          </Label>
          <Textarea
            id={id("goal-template")}
            rows={3}
            value={form.goalTemplate}
            onChange={(e) => setForm({ ...form, goalTemplate: e.target.value })}
            placeholder="Plot total_energy vs temperature in project {{project}}."
          />
        </div>
        <div>
          <Label htmlFor={id("instructions")} className="text-xs">
            Additional instructions (optional) — appended to the system prompt
          </Label>
          <Textarea
            id={id("instructions")}
            rows={3}
            value={form.instructions}
            onChange={(e) => setForm({ ...form, instructions: e.target.value })}
            placeholder="When plotting, prefer Plotly scatter and label units explicitly."
            className="font-mono text-xs"
          />
        </div>
        <div className="flex items-center gap-2">
          <input
            id={id("default-plan-mode")}
            type="checkbox"
            checked={form.defaultPlanMode}
            onChange={(e) => setForm({ ...form, defaultPlanMode: e.target.checked })}
            className="h-3.5 w-3.5 accent-primary"
          />
          <Label htmlFor={id("default-plan-mode")} className="text-xs">
            Launch with the auditable nine-stage Plan agent by default
          </Label>
        </div>
        <div>
          <Label htmlFor={id("constraints")} className="text-xs">
            Constraints (one per line)
          </Label>
          <Textarea
            id={id("constraints")}
            rows={2}
            value={form.constraints}
            onChange={(e) => setForm({ ...form, constraints: e.target.value })}
            placeholder="scope=project"
          />
        </div>
        <div>
          <Label htmlFor={id("success-criteria")} className="text-xs">
            Success criteria (one per line)
          </Label>
          <Textarea
            id={id("success-criteria")}
            rows={2}
            value={form.successCriteria}
            onChange={(e) => setForm({ ...form, successCriteria: e.target.value })}
            placeholder="A scatter plot is produced"
          />
        </div>
        <div>
          <Label htmlFor={id("tags")} className="text-xs">
            Tags (comma-separated)
          </Label>
          <Input
            id={id("tags")}
            value={form.tags}
            onChange={(e) => setForm({ ...form, tags: e.target.value })}
            placeholder="plot, sweep"
          />
        </div>
        {error && <p className="text-xs text-destructive">{error}</p>}
        <div className="flex justify-end gap-2 pt-1">
          <WorkbenchAction kind="ghost" size="compact" onClick={onCancel} disabled={saving}>
            Cancel
          </WorkbenchAction>
          <WorkbenchAction
            kind="primary"
            size="compact"
            onClick={() => void handleSubmit()}
            disabled={saving}
          >
            {saving ? "Saving…" : "Save"}
          </WorkbenchAction>
        </div>
      </div>
    </section>
  );
};

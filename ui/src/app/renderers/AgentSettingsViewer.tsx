/**
 * AgentSettingsViewer — read/write management for the agent runtime.
 *
 * Claude Code-style capability surfaces: agents, model, instructions,
 * skills, tools, and MCP. Each network-backed surface mounts independently,
 * so an unavailable optional agent-admin service produces one quiet state
 * instead of a wall of repeated 503 errors.
 *
 * The tab descriptors live in `agent_settings/tabs.ts` so they can be
 * unit-tested without pulling in the full component graph.
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
import { useCallback, useEffect, useMemo, useState } from "react";
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
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
  agents: Bot,
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
    case "agents-overview":
      return <AgentsOverview />;
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
      subtitle="Agents, instructions, skills, and MCP-provided tools"
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
    <div className={`mx-auto w-full ${wide ? "max-w-5xl" : "max-w-3xl"} px-4 pb-10 pt-4`}>
      {children}
    </div>
  </ScrollArea>
);

const AgentsOverview = (): JSX.Element => (
  <SettingsScroll wide>
    <div className="space-y-6">
      <div>
        <h2 className="text-base font-semibold">Built-in agents</h2>
        <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
          Every task is one conversation. Switch the agent for each turn from the composer with
          <kbd className="mx-1 rounded border bg-muted px-1.5 py-0.5 font-mono text-[11px]">
            Shift+Tab
          </kbd>
          or click the mode indicator.
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="border-border/80">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <Bot className="size-4" />
                </span>
                <div>
                  <CardTitle className="text-sm">Chat</CardTitle>
                  <p className="text-xs text-muted-foreground">Interactive workspace agent</p>
                </div>
              </div>
              <Badge variant="secondary">Built in</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p className="text-muted-foreground">
              Explores the workspace, calls tools, edits files, and performs explicit lifecycle
              actions in the current conversation.
            </p>
            <div className="flex flex-wrap gap-1.5">
              <Badge variant="outline">workspace context</Badge>
              <Badge variant="outline">tools</Badge>
              <Badge variant="outline">MCP</Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/80">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <span className="flex size-9 items-center justify-center rounded-lg bg-violet-500/10 text-violet-500">
                  <BrainCircuit className="size-4" />
                </span>
                <div>
                  <CardTitle className="text-sm">Plan</CardTitle>
                  <p className="text-xs text-muted-foreground">Auditable experiment planner</p>
                </div>
              </div>
              <Badge variant="secondary">Built in</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p className="text-muted-foreground">
              Runs the fixed nine-stage PlanMode pipeline with review gates, versioned revisions,
              and inspectable artifacts. It never executes automatically.
            </p>
            <div className="flex flex-wrap gap-1.5">
              <Badge variant="outline">9 stages</Badge>
              <Badge variant="outline">approvals</Badge>
              <Badge variant="outline">provenance</Badge>
            </div>
          </CardContent>
        </Card>
      </div>
      <div className="rounded-lg border border-border/70 bg-muted/20 px-4 py-3">
        <p className="text-sm font-medium">Configuration is shared</p>
        <p className="mt-1 text-xs text-muted-foreground">
          Model and workspace instructions apply to both agents. Skills provide reusable workflows;
          Tools and MCP control the capabilities available at runtime.
        </p>
      </div>
    </div>
  </SettingsScroll>
);

// ─── Provider tab ──────────────────────────────────────────────────────────
//
// Field schema, labels, and per-provider hints are owned by
// `providerRegistry.ts` (registry-driven per spec §7.1 / ac-005). This
// file holds no provider-name literals as switching keys; new providers
// are introduced by shipping a model plugin and updating the registry,
// not by editing this component.

const providerLabel = (registry: ProviderRegistryResponse, name: string): string =>
  findRegistryEntry(registry, name)?.label ?? name;

const providerModelHint = (registry: ProviderRegistryResponse, name: string): string =>
  findRegistryEntry(registry, name)?.modelHint ?? "";

const MODEL_TIERS: readonly {
  tier: ApiModelTier;
  label: string;
  description: string;
}[] = [
  { tier: "cheap", label: "Cheap", description: "Parsing, routing, and lightweight tasks" },
  { tier: "default", label: "Default", description: "Normal chat and general agent work" },
  { tier: "heavy", label: "Heavy", description: "Complex planning and deep reasoning" },
];

const emptyTierModels = (): ApiTierModels => ({ cheap: "", default: "", heavy: "" });

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

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-4 px-4 pb-10 pt-3">
        <div>
          <h2 className="text-base font-semibold">Model providers</h2>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            Configure each provider independently. The router selects Cheap, Default, or Heavy
            according to the task; saving a provider also makes it the active provider.
          </p>
        </div>
        {error && <p className="text-xs text-destructive">{error}</p>}
        <div className="space-y-3">
          {supported.map((provider) => {
            const stored = configurations.find((entry) => entry.provider === provider);
            const legacyModels =
              config?.provider === provider && config.model
                ? { cheap: config.model, default: config.model, heavy: config.model }
                : emptyTierModels();
            return (
              <ProviderCard
                key={provider}
                provider={provider}
                active={config?.provider === provider}
                initial={
                  stored ?? {
                    provider,
                    models: legacyModels,
                    baseUrl: config?.provider === provider ? config.baseUrl : "",
                    apiKeyPreview: config?.provider === provider ? config.apiKeyPreview : "",
                    apiKeySet: config?.provider === provider ? config.apiKeySet : false,
                  }
                }
                registry={registry}
                onChanged={(next) => setConfig(next)}
              />
            );
          })}
        </div>
      </div>
    </ScrollArea>
  );
};

const ProviderCard = ({
  provider,
  active,
  initial,
  registry,
  onChanged,
}: {
  provider: ApiProviderName;
  active: boolean;
  initial: ApiProviderConfiguration;
  registry: ProviderRegistryResponse;
  onChanged: (config: ApiAgentProvider) => void;
}): JSX.Element => {
  const [expanded, setExpanded] = useState(active);
  const [models, setModels] = useState<ApiTierModels>(initial.models);
  const [baseUrl, setBaseUrl] = useState(initial.baseUrl);
  const [apiKey, setApiKey] = useState("");
  const [revealKey, setRevealKey] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [testResult, setTestResult] = useState<ApiAgentProviderTestResult | null>(null);
  const [confirmClearKey, setConfirmClearKey] = useState(false);
  const showBaseUrl = supportsBaseUrl(registry, provider);

  useEffect(() => {
    setModels(initial.models);
    setBaseUrl(initial.baseUrl);
  }, [initial]);

  const submit = async (mode: "save" | "test" | "clear"): Promise<void> => {
    setBusy(true);
    setError(null);
    setSaved(false);
    setTestResult(null);
    try {
      const input: ProviderUpdateInput = {
        provider,
        models,
        baseUrl: baseUrl.trim(),
      };
      if (mode === "clear") input.apiKey = "";
      else if (apiKey !== "") input.apiKey = apiKey;
      if (mode === "test") {
        setTestResult(await agentAdminApi.testProvider(input));
      } else {
        const next = await agentAdminApi.updateProvider(input);
        setApiKey("");
        setRevealKey(false);
        setSaved(true);
        onChanged(next);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const complete = MODEL_TIERS.every(({ tier }) => models[tier].trim() !== "");

  return (
    <Card className={active ? "border-primary/50" : "border-border"}>
      <CardHeader className="pb-3">
        <button
          type="button"
          className="flex w-full items-center gap-3 text-left"
          aria-expanded={expanded}
          onClick={() => setExpanded((value) => !value)}
        >
          <Cpu className="size-4 text-muted-foreground" />
          <div className="min-w-0 flex-1">
            <CardTitle className="text-sm">{providerLabel(registry, provider)}</CardTitle>
            <p className="mt-0.5 truncate text-xs text-muted-foreground">
              {initial.apiKeySet ? `Key ${initial.apiKeyPreview}` : "No stored key"} ·{" "}
              {models.default || "No default model"}
            </p>
          </div>
          {active && <Badge variant="default">Active</Badge>}
          <ChevronRight className={`size-4 transition-transform ${expanded ? "rotate-90" : ""}`} />
        </button>
      </CardHeader>
      {expanded && (
        <CardContent className="space-y-4 border-t pt-4">
          <div className="space-y-2">
            <div className="grid grid-cols-[100px_minmax(0,1fr)] gap-3 px-1 text-[10px] uppercase tracking-wide text-muted-foreground">
              <span>Tier</span>
              <span>Model ID</span>
            </div>
            {MODEL_TIERS.map(({ tier, label, description }) => (
              <div
                key={tier}
                className="grid items-center gap-3 rounded-md border p-3 md:grid-cols-[100px_minmax(0,1fr)]"
              >
                <div>
                  <Label htmlFor={`${provider}-${tier}`} className="text-xs font-medium">
                    {label}
                  </Label>
                  <p className="text-[10px] text-muted-foreground">{description}</p>
                </div>
                <Input
                  id={`${provider}-${tier}`}
                  value={models[tier]}
                  onChange={(event) =>
                    setModels((value) => ({ ...value, [tier]: event.target.value }))
                  }
                  placeholder={providerModelHint(registry, provider)}
                  className="font-mono text-xs"
                />
              </div>
            ))}
          </div>
          {showBaseUrl && (
            <div>
              <Label className="text-xs">Base URL</Label>
              <Input
                value={baseUrl}
                onChange={(event) => setBaseUrl(event.target.value)}
                placeholder={baseUrlPlaceholder(registry, provider)}
              />
            </div>
          )}
          <div>
            <Label className="text-xs">API key</Label>
            <div className="flex gap-2">
              <Input
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
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => setRevealKey((value) => !value)}
              >
                {revealKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
              </Button>
            </div>
          </div>
          {error && <p className="text-xs text-destructive">{error}</p>}
          {saved && (
            <p className="flex items-center gap-1 text-xs text-success-foreground">
              <CheckCircle2 className="size-3.5" /> Saved and set active.
            </p>
          )}
          {testResult && <ProviderTestResult result={testResult} />}
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
            <Button
              variant="ghost"
              size="sm"
              disabled={busy || !initial.apiKeySet}
              onClick={() => setConfirmClearKey(true)}
            >
              Clear key
            </Button>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={busy || !complete}
                onClick={() => void submit("test")}
              >
                <Zap className="mr-1 size-4" /> Test default
              </Button>
              <Button size="sm" disabled={busy || !complete} onClick={() => void submit("save")}>
                {busy ? "Saving…" : "Save & use"}
              </Button>
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  );
};

const ProviderTestResult = ({ result }: { result: ApiAgentProviderTestResult }): JSX.Element => {
  const ok = result.ok;
  return (
    <div
      className={
        "rounded-md border px-3 py-2 text-xs " +
        (ok
          ? "border-success/30 bg-success-soft text-success-foreground"
          : "border-destructive/40 bg-destructive/10 text-destructive")
      }
    >
      <div className="flex items-center gap-2 font-medium">
        {ok ? <CheckCircle2 className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
        {ok ? "Connection OK" : "Connection failed"}
        <span className="ml-auto font-mono text-[10px] opacity-80">
          {result.provider}:{result.model} · {result.latencyMs} ms
        </span>
      </div>
      {ok && result.reply && (
        <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-[11px] opacity-80">
          {result.reply}
        </pre>
      )}
      {!ok && result.error && (
        <p className="mt-1 break-words font-mono text-[11px]">{result.error}</p>
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

      <Card className="border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Workspace instructions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 pt-0">
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
          <p className="text-[10px] text-muted-foreground">
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
            <Button
              variant="ghost"
              size="sm"
              disabled={saving || (config?.instructions ?? "") === ""}
              onClick={() => setConfirmClear(true)}
            >
              Clear
            </Button>
            <Button size="sm" disabled={saving || !dirty} onClick={() => void handleSave()}>
              {saving ? "Saving…" : "Save"}
            </Button>
          </div>
        </CardContent>
      </Card>
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
          <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button size="sm" disabled={launching} onClick={() => void handleLaunch()}>
            <PlayCircle className="mr-1 h-4 w-4" />
            {launching ? "Launching…" : "Launch session"}
          </Button>
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
        <Badge variant="secondary" className="text-[10px] font-normal">
          {count}
        </Badge>
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
          <Button
            size="sm"
            onClick={() => {
              setEditing(null);
              setShowForm(true);
            }}
          >
            <Plus className="mr-1 h-4 w-4" /> New skill
          </Button>
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
          <Card key={skill.id} className="border-border">
            <CardHeader className="pb-2">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  {skill.slashName ? (
                    <Badge
                      variant="outline"
                      className="font-mono text-[11px]"
                      title="Type this in chat to invoke"
                    >
                      /{skill.slashName}
                    </Badge>
                  ) : (
                    <Badge variant="secondary" className="text-[10px]">
                      launcher only
                    </Badge>
                  )}
                  <CardTitle className="truncate text-sm">{skill.name}</CardTitle>
                  {skill.defaultPlanMode && (
                    <Badge variant="outline" className="text-[10px]">
                      plan
                    </Badge>
                  )}
                </div>
                <div className="flex gap-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => handleLaunch(skill)}
                    title="Launch a task from this skill"
                    aria-label={`Launch ${skill.name}`}
                  >
                    <PlayCircle className="h-4 w-4" />
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setEditing(skill);
                      setShowForm(true);
                    }}
                    title="Edit skill"
                  >
                    Edit
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setDeleting(skill)}
                    title="Delete skill"
                    aria-label={`Delete ${skill.name}`}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              {skill.description && (
                <p className="mb-2 text-xs text-muted-foreground">{skill.description}</p>
              )}
              <pre className="mb-2 whitespace-pre-wrap rounded bg-muted px-2 py-1 text-xs">
                {skill.goalTemplate}
              </pre>
              {skill.instructions && (
                <p className="mb-2 text-[11px] italic text-muted-foreground">
                  +{skill.instructions.length} chars of additional instructions
                </p>
              )}
              <div className="flex flex-wrap gap-1">
                {skill.tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="text-[10px]">
                    {tag}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
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
    <Card className="mb-2 border-primary/40">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">{initial ? "Edit skill" : "New skill"}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div>
          <Label className="text-xs">Name</Label>
          <Input
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="Plot energy vs temperature"
          />
        </div>
        <div>
          <Label className="text-xs">
            Slash name (optional) — invokes as{" "}
            <code className="rounded bg-muted px-1">/&lt;name&gt;</code>
          </Label>
          <div className="flex items-center gap-2">
            <span className="select-none font-mono text-sm text-muted-foreground">/</span>
            <Input
              value={form.slashName}
              onChange={(e) => setForm({ ...form, slashName: e.target.value })}
              placeholder="plot-energy"
              className="font-mono"
            />
          </div>
          {slashError ? (
            <p className="mt-1 text-[10px] text-destructive">{slashError}</p>
          ) : (
            <p className="mt-1 text-[10px] text-muted-foreground">
              Reserved: {RESERVED_SLASH_NAMES.join(", ")}. Leave empty to keep this as
              launcher-only.
            </p>
          )}
        </div>
        <div>
          <Label className="text-xs">Description</Label>
          <Input
            value={form.description}
            onChange={(e) => setForm({ ...form, description: e.target.value })}
            placeholder="Optional summary"
          />
        </div>
        <div>
          <Label className="text-xs">Goal template — use {"{{param}}"} for placeholders</Label>
          <Textarea
            rows={3}
            value={form.goalTemplate}
            onChange={(e) => setForm({ ...form, goalTemplate: e.target.value })}
            placeholder="Plot total_energy vs temperature in project {{project}}."
          />
        </div>
        <div>
          <Label className="text-xs">
            Additional instructions (optional) — appended to the system prompt
          </Label>
          <Textarea
            rows={3}
            value={form.instructions}
            onChange={(e) => setForm({ ...form, instructions: e.target.value })}
            placeholder="When plotting, prefer Plotly scatter and label units explicitly."
            className="font-mono text-xs"
          />
        </div>
        <div className="flex items-center gap-2">
          <input
            id="defaultPlanMode"
            type="checkbox"
            checked={form.defaultPlanMode}
            onChange={(e) => setForm({ ...form, defaultPlanMode: e.target.checked })}
            className="h-3.5 w-3.5 accent-primary"
          />
          <Label htmlFor="defaultPlanMode" className="text-xs">
            Launch with the auditable nine-stage Plan agent by default
          </Label>
        </div>
        <div>
          <Label className="text-xs">Constraints (one per line)</Label>
          <Textarea
            rows={2}
            value={form.constraints}
            onChange={(e) => setForm({ ...form, constraints: e.target.value })}
            placeholder="scope=project"
          />
        </div>
        <div>
          <Label className="text-xs">Success criteria (one per line)</Label>
          <Textarea
            rows={2}
            value={form.successCriteria}
            onChange={(e) => setForm({ ...form, successCriteria: e.target.value })}
            placeholder="A scatter plot is produced"
          />
        </div>
        <div>
          <Label className="text-xs">Tags (comma-separated)</Label>
          <Input
            value={form.tags}
            onChange={(e) => setForm({ ...form, tags: e.target.value })}
            placeholder="plot, sweep"
          />
        </div>
        {error && <p className="text-xs text-destructive">{error}</p>}
        <div className="flex justify-end gap-2 pt-1">
          <Button variant="ghost" size="sm" onClick={onCancel} disabled={saving}>
            Cancel
          </Button>
          <Button size="sm" onClick={() => void handleSubmit()} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

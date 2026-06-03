/**
 * AgentSettingsViewer — read/write management for the agent runtime.
 *
 * Three top-level tabs (per agent-harness UI lockstep spec §8):
 *
 *   - Agent          — agent-core configuration: instructions, slash
 *                      commands, native tools (stacked sections).
 *   - Model providers — LLM provider/model + API key (registry-driven).
 *   - Tool sources    — pluggable tool sources (today: MCP servers).
 *
 * The tab descriptors live in `agent_settings/tabs.ts` so they can be
 * unit-tested without pulling in the full component graph.
 */

import {
  AlertCircle,
  CheckCircle2,
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
  Wrench,
  Zap,
} from "lucide-react";
import type { JSX } from "react";
import { useCallback, useEffect, useState } from "react";
import { EntityPage } from "@/app/components/entity";
import { McpServersTab } from "@/app/renderers/agent_settings/McpServersTab";
import {
  baseUrlPlaceholder,
  DEFAULT_PROVIDER_REGISTRY,
  findRegistryEntry,
  type ProviderRegistryResponse,
  supportsBaseUrl,
} from "@/app/renderers/agent_settings/providerRegistry";
import { AGENT_SETTINGS_TABS, type AgentSettingsTabDef } from "@/app/renderers/agent_settings/tabs";
import {
  type ApiAgentProvider,
  type ApiAgentProviderTestResult,
  type ApiAgentTool,
  type ApiAgentToolList,
  type ApiMcpToolGroup,
  type ApiProviderName,
  type ApiSkill,
  agentAdminApi,
  isMcpSource,
  mcpSource,
  NATIVE_SOURCE,
  type ProviderUpdateInput,
  RESERVED_SLASH_NAMES,
  type SkillUpsertInput,
  SLASH_NAME_PATTERN,
} from "@/app/state/api";
import { onMcpConfigChanged } from "@/app/state/mcpEvents";
import type { WorkspaceSnapshot } from "@/app/types";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
  /** Workspace snapshot — required for shared header navigation. */
  snapshot: WorkspaceSnapshot;
  onLaunchSession?: (sessionId: string) => void;
}

const TAB_ICON: Record<AgentSettingsTabDef["value"], typeof Settings> = {
  agent: Settings,
  providers: Cpu,
  "tool-sources": Database,
};

const renderTabContent = (
  contentKey: AgentSettingsTabDef["contentKey"],
  onLaunchSession?: (sessionId: string) => void,
): JSX.Element => {
  switch (contentKey) {
    case "agent-core":
      return <AgentCoreTab onLaunchSession={onLaunchSession} />;
    case "providers-form":
      return <ProviderTab />;
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
      subtitle="Agent core, model providers, tool sources"
      tabs={tabs}
    />
  );
};

// ─── Agent core tab (instructions + commands + native tools, stacked) ──────

interface AgentCoreTabProps {
  onLaunchSession?: (sessionId: string) => void;
}

const AgentCoreTab = ({ onLaunchSession }: AgentCoreTabProps): JSX.Element => {
  return (
    <div className="space-y-6">
      <section aria-labelledby="agent-core-instructions">
        <div className="mb-2 flex items-center gap-2">
          <FileText className="h-4 w-4 text-muted-foreground" />
          <h2 id="agent-core-instructions" className="text-base font-semibold">
            Instructions
          </h2>
        </div>
        <InstructionsTab />
      </section>

      <section aria-labelledby="agent-core-commands">
        <div className="mb-2 flex items-center gap-2">
          <Slash className="h-4 w-4 text-muted-foreground" />
          <h2 id="agent-core-commands" className="text-base font-semibold">
            Commands
          </h2>
        </div>
        <CommandsTab onLaunchSession={onLaunchSession} />
      </section>

      <section aria-labelledby="agent-core-tools">
        <div className="mb-2 flex items-center gap-2">
          <Wrench className="h-4 w-4 text-muted-foreground" />
          <h2 id="agent-core-tools" className="text-base font-semibold">
            Native tools
          </h2>
        </div>
        <ToolsTab />
      </section>
    </div>
  );
};

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

const ProviderTab = (): JSX.Element => {
  // Until backend Phase 3 ships `/api/agent/admin/providers`, the UI
  // bootstraps from the bundled defaults. Once the route is live and
  // wired through `agentAdminApi`, this becomes the fallback while the
  // network response is in flight.
  const registry = DEFAULT_PROVIDER_REGISTRY;
  const [config, setConfig] = useState<ApiAgentProvider | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState<{
    provider: ApiProviderName;
    model: string;
    baseUrl: string;
    apiKey: string;
    revealKey: boolean;
  }>({
    // Initial provider comes from the registry, not a literal — adding a
    // new provider in the registry shifts the default automatically.
    provider: DEFAULT_PROVIDER_REGISTRY.providers[0].name as ApiProviderName,
    model: "",
    baseUrl: "",
    apiKey: "",
    revealKey: false,
  });
  const [saving, setSaving] = useState(false);
  const [saveOk, setSaveOk] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<ApiAgentProviderTestResult | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await agentAdminApi.getProvider();
      setConfig(next);
      setDraft((d) => ({
        ...d,
        provider: next.provider,
        model: next.model,
        baseUrl: next.baseUrl,
        // Preserve any pending key the user typed; reset only when initial load.
      }));
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setError(null);
    setSaveOk(false);
    setTestResult(null);
    try {
      const patch: ProviderUpdateInput = {
        provider: draft.provider,
        model: draft.model.trim(),
        baseUrl: draft.baseUrl.trim(),
      };
      // Only send apiKey if user actually typed one. Empty string clears.
      if (draft.apiKey !== "") {
        patch.apiKey = draft.apiKey;
      }
      const updated = await agentAdminApi.updateProvider(patch);
      setConfig(updated);
      setDraft((d) => ({
        ...d,
        apiKey: "", // clear after save so the field doesn't linger
        revealKey: false,
        provider: updated.provider,
        model: updated.model,
        baseUrl: updated.baseUrl,
      }));
      setSaveOk(true);
    } catch (err) {
      setError(String(err));
    } finally {
      setSaving(false);
    }
  }, [draft]);

  const handleClearKey = useCallback(async () => {
    if (!window.confirm("Clear the stored API key? Sessions will fall back to env vars.")) return;
    setSaving(true);
    setError(null);
    setSaveOk(false);
    setTestResult(null);
    try {
      const updated = await agentAdminApi.updateProvider({ apiKey: "" });
      setConfig(updated);
      setDraft((d) => ({ ...d, apiKey: "", revealKey: false }));
      setSaveOk(true);
    } catch (err) {
      setError(String(err));
    } finally {
      setSaving(false);
    }
  }, []);

  const handleTest = useCallback(async () => {
    setTesting(true);
    setTestResult(null);
    setError(null);
    try {
      const patch: ProviderUpdateInput = {
        provider: draft.provider,
        model: draft.model.trim(),
        baseUrl: draft.baseUrl.trim(),
      };
      if (draft.apiKey !== "") {
        patch.apiKey = draft.apiKey;
      }
      setTestResult(await agentAdminApi.testProvider(patch));
    } catch (err) {
      setError(String(err));
    } finally {
      setTesting(false);
    }
  }, [draft]);

  if (loading) {
    return (
      <div className="px-4 pb-4 pt-2 text-sm text-muted-foreground">Loading provider config…</div>
    );
  }

  const supported =
    config?.supportedProviders ?? (registry.providers.map((p) => p.name) as ApiProviderName[]);
  // The registry's field schema decides whether a provider exposes a
  // base URL field (e.g. proxy / mirror / self-hosted gateway).
  const showBaseUrl = supportsBaseUrl(registry, draft.provider);

  return (
    <ScrollArea className="h-full">
      <div className="flex flex-col gap-4 px-4 pb-6 pt-2">
        <p className="text-sm text-muted-foreground">
          Choose which LLM the agent should call and supply your API key. The key is stored at{" "}
          <code className="rounded bg-muted px-1">.agent_provider.json</code> in the workspace root
          and never leaves this machine.
        </p>

        <SavedProviderList config={config} supported={supported} />

        <Card className="border-border">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm">Active configuration</CardTitle>
              {config?.apiKeySet ? (
                <Badge variant="default" className="gap-1 text-[10px]">
                  <CheckCircle2 className="h-3 w-3" /> Key configured
                </Badge>
              ) : (
                <Badge variant="secondary" className="text-[10px]">
                  No key — falls back to env vars
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-3 pt-0">
            <div>
              <Label className="text-xs">Provider</Label>
              <Select
                value={draft.provider}
                onValueChange={(value) =>
                  setDraft((d) => ({ ...d, provider: value as ApiProviderName }))
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {supported.map((p) => (
                    <SelectItem key={p} value={p}>
                      {providerLabel(registry, p)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label className="text-xs">Model</Label>
              <Input
                value={draft.model}
                onChange={(e) => setDraft({ ...draft, model: e.target.value })}
                placeholder={providerModelHint(registry, draft.provider)}
              />
              <p className="mt-1 text-[10px] text-muted-foreground">
                {providerModelHint(registry, draft.provider)}
              </p>
            </div>

            {showBaseUrl && (
              <div>
                <Label className="text-xs">Base URL (optional)</Label>
                <Input
                  value={draft.baseUrl}
                  onChange={(e) => setDraft({ ...draft, baseUrl: e.target.value })}
                  placeholder={baseUrlPlaceholder(registry, draft.provider)}
                />
              </div>
            )}

            <div>
              <Label className="text-xs">API key</Label>
              <div className="flex gap-2">
                <Input
                  type={draft.revealKey ? "text" : "password"}
                  value={draft.apiKey}
                  onChange={(e) => setDraft({ ...draft, apiKey: e.target.value })}
                  placeholder={
                    config?.apiKeySet
                      ? `Stored: ${config.apiKeyPreview} — type to replace`
                      : "Paste your API key"
                  }
                  autoComplete="off"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={() => setDraft((d) => ({ ...d, revealKey: !d.revealKey }))}
                  title={draft.revealKey ? "Hide" : "Reveal"}
                >
                  {draft.revealKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </Button>
              </div>
              <p className="mt-1 text-[10px] text-muted-foreground">
                Leaving the field blank keeps the existing key.
              </p>
            </div>

            {error && <p className="text-xs text-destructive">{error}</p>}
            {saveOk && !error && (
              <p className="text-xs text-emerald-600">Saved. New sessions will use this config.</p>
            )}

            {testResult && <ProviderTestResult result={testResult} />}

            <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
              <Button
                variant="ghost"
                size="sm"
                disabled={saving || !config?.apiKeySet}
                onClick={() => void handleClearKey()}
              >
                Clear stored key
              </Button>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={testing || saving || (!config?.apiKeySet && draft.apiKey === "")}
                  onClick={() => void handleTest()}
                  title="Send a minimal request to verify the key and model"
                >
                  <Zap className="mr-1 h-4 w-4" />
                  {testing ? "Testing…" : "Test connection"}
                </Button>
                <Button size="sm" disabled={saving} onClick={() => void handleSave()}>
                  {saving ? "Saving…" : "Save"}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  );
};

const SavedProviderList = ({
  config,
  supported,
}: {
  config: ApiAgentProvider | null;
  supported: ApiProviderName[];
}): JSX.Element => {
  return (
    <Card className="border-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Saved keys</CardTitle>
      </CardHeader>
      <CardContent className="space-y-1.5 pt-0">
        {supported.map((provider) => {
          const isActive = config?.provider === provider;
          const hasKey = isActive && Boolean(config?.apiKeySet);
          return (
            <div
              key={provider}
              className={
                "flex items-center gap-3 rounded-md border px-3 py-2 text-sm " +
                (hasKey
                  ? "border-emerald-500/40 bg-emerald-500/5"
                  : isActive
                    ? "border-primary/40 bg-primary/5"
                    : "border-border/60 bg-card")
              }
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">
                    {providerLabel(DEFAULT_PROVIDER_REGISTRY, provider)}
                  </span>
                  {isActive && (
                    <Badge variant="outline" className="h-4 text-[10px]">
                      Active
                    </Badge>
                  )}
                </div>
                {hasKey && config ? (
                  <p className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
                    {config.model} · {config.apiKeyPreview}
                  </p>
                ) : (
                  <p className="mt-0.5 text-[11px] text-muted-foreground">
                    {isActive ? "No key saved — using env vars" : "Not configured"}
                  </p>
                )}
              </div>
              {hasKey ? (
                <Badge variant="default" className="gap-1 text-[10px]">
                  <CheckCircle2 className="h-3 w-3" /> Saved
                </Badge>
              ) : (
                <Badge variant="secondary" className="text-[10px]">
                  Empty
                </Badge>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
};

const ProviderTestResult = ({ result }: { result: ApiAgentProviderTestResult }): JSX.Element => {
  const ok = result.ok;
  return (
    <div
      className={
        "rounded border px-3 py-2 text-xs " +
        (ok
          ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700"
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
        if (!cancelled) setError(String(err));
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

  const handleClear = useCallback(async () => {
    if (!window.confirm("Clear the workspace-default instructions?")) return;
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
    return <div className="px-4 pt-2 text-sm text-muted-foreground">Loading instructions…</div>;
  }

  const dirty = (config?.instructions ?? "") !== draft;

  return (
    <ScrollArea className="h-full">
      <div className="flex flex-col gap-3 px-4 pb-6 pt-2">
        <p className="text-sm text-muted-foreground">
          Workspace-default system prompt addendum. Appended to the molexp built-in preamble for
          every new session. Skills can layer additional instructions on top, and individual
          sessions may override the whole stack via the chat hero.
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
              <p className="text-xs text-emerald-600">
                Saved. New sessions will use these instructions.
              </p>
            )}
            <div className="flex justify-between gap-2 pt-1">
              <Button
                variant="ghost"
                size="sm"
                disabled={saving || (config?.instructions ?? "") === ""}
                onClick={() => void handleClear()}
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
    </ScrollArea>
  );
};

// ─── Commands tab (formerly Skills) ────────────────────────────────────────

const CommandsTab = ({
  onLaunchSession,
}: {
  onLaunchSession?: (sessionId: string) => void;
}): JSX.Element => {
  const [skills, setSkills] = useState<ApiSkill[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<ApiSkill | null>(null);
  const [showForm, setShowForm] = useState(false);

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
      if (!window.confirm(`Delete skill "${skill.name}"?`)) return;
      try {
        await agentAdminApi.deleteSkill(skill.id);
        await refresh();
      } catch (err) {
        setError(String(err));
      }
    },
    [refresh],
  );

  const handleLaunch = useCallback(
    async (skill: ApiSkill) => {
      const params: Record<string, string> = {};
      const placeholders = Array.from(skill.goalTemplate.matchAll(/\{\{\s*([A-Za-z_]\w*)\s*\}\}/g))
        .map((m) => m[1])
        .filter((v, i, a) => a.indexOf(v) === i);
      for (const key of placeholders) {
        const value = window.prompt(`Value for {{${key}}}:`);
        if (value === null) return;
        params[key] = value;
      }
      try {
        const session = await agentAdminApi.launchSkill(skill.id, params);
        onLaunchSession?.(session.sessionId);
      } catch (err) {
        setError(String(err));
      }
    },
    [onLaunchSession],
  );

  return (
    <div className="flex h-full flex-col px-4 pb-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground">
          Saved goal templates. Use <code className="rounded bg-muted px-1">{"{{name}}"}</code>{" "}
          placeholders for parameters; commands with a slash name are also invokable from the chat
          input as <code className="rounded bg-muted px-1">/&lt;name&gt;</code>.
        </p>
        <Button
          size="sm"
          onClick={() => {
            setEditing(null);
            setShowForm(true);
          }}
        >
          <Plus className="mr-1 h-4 w-4" /> New command
        </Button>
      </div>
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
      <ScrollArea className="mt-2 flex-1">
        <div className="flex flex-col gap-2 pr-2">
          {loading && <p className="text-sm text-muted-foreground">Loading…</p>}
          {!loading && skills.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No skills yet. Create one to save common goals for one-click launches.
            </p>
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
                      onClick={() => void handleLaunch(skill)}
                      title="Launch session from this skill"
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
                      title="Edit"
                    >
                      Edit
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => void handleDelete(skill)}
                      title="Delete"
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
      </ScrollArea>
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
        <CardTitle className="text-sm">{initial ? "Edit command" : "New command"}</CardTitle>
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
            className="h-3.5 w-3.5"
          />
          <Label htmlFor="defaultPlanMode" className="text-xs">
            Launch in plan mode by default (read-only inspection, agent emits a plan)
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

// ─── Tools tab ─────────────────────────────────────────────────────────────

// Groups native tools under "native"; each MCP server's tools under
// "mcp:<name>" so the UI renders one collapsible section per source.
// ``mcpGroups`` carries per-server status so a broken MCP still shows its
// heading + the error rather than silently disappearing.
type SortMode = "server" | "name";

interface ToolGroup {
  /** Display key — "native" or "mcp:<server>". */
  source: string;
  /** Friendly label rendered in the section header. */
  label: string;
  /** True for MCP servers; native tools never carry server status. */
  isMcp: boolean;
  ok: boolean;
  error: string | null;
  tools: ApiAgentTool[];
}

const buildGroups = (data: ApiAgentToolList): Map<string, ToolGroup> => {
  const groups = new Map<string, ToolGroup>();
  groups.set(NATIVE_SOURCE, {
    source: NATIVE_SOURCE,
    label: "Native tools",
    isMcp: false,
    ok: true,
    error: null,
    tools: [],
  });
  // Pre-create a group per MCP server so we render even when toolCount=0
  // (e.g. unreachable). Tools then drop into their group below.
  for (const grp of data.mcpGroups) {
    const key = mcpSource(grp.server);
    groups.set(key, {
      source: key,
      label: grp.server,
      isMcp: true,
      ok: grp.ok,
      error: grp.error,
      tools: [],
    });
  }
  for (const tool of data.tools) {
    const key = isMcpSource(tool.source) ? tool.source : NATIVE_SOURCE;
    const grp = groups.get(key);
    if (grp) grp.tools.push(tool);
  }
  return groups;
};

const sortGroups = (groups: Map<string, ToolGroup>, mode: SortMode): ToolGroup[] => {
  const arr = Array.from(groups.values());
  // Native always pinned first; MCP groups sort alphabetically by name.
  arr.sort((a, b) => {
    if (a.source === NATIVE_SOURCE && b.source !== NATIVE_SOURCE) return -1;
    if (b.source === NATIVE_SOURCE && a.source !== NATIVE_SOURCE) return 1;
    if (mode === "server") return a.label.localeCompare(b.label);
    return 0;
  });
  if (mode === "name") {
    for (const g of arr) g.tools.sort((a, b) => a.name.localeCompare(b.name));
  }
  return arr;
};

// One group's heading + collapsible tool rows. Headings stay static
// (always visible) so the user can see all servers at a glance; only the
// per-tool details collapse, keeping the list scannable in the common case
// where most rows are uninteresting metadata.
const ToolGroupSection = ({ group }: { group: ToolGroup }): JSX.Element => {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {group.label}
        </h3>
        {group.isMcp && (
          <Badge variant={group.ok ? "outline" : "destructive"} className="text-[10px]">
            {group.ok ? `${group.tools.length} tools` : "unreachable"}
          </Badge>
        )}
      </div>
      {group.isMcp && group.error && (
        <p className="rounded border border-destructive/40 bg-destructive/10 px-2 py-1 text-xs text-destructive">
          {group.error}
        </p>
      )}
      {group.tools.length === 0 && !group.isMcp && (
        <p className="text-xs text-muted-foreground">No tools.</p>
      )}
      {group.tools.length > 0 && (
        <Accordion type="multiple" className="rounded border bg-card">
          {group.tools.map((tool) => (
            <AccordionItem
              key={`${group.source}:${tool.name}`}
              value={`${group.source}:${tool.name}`}
              className="border-b last:border-b-0 px-3"
            >
              <AccordionTrigger className="py-1.5">
                <div className="flex flex-1 items-center gap-2 overflow-hidden">
                  <code className="truncate font-mono text-xs">{tool.name}</code>
                  {tool.parameters.length > 0 && (
                    <span className="shrink-0 text-[10px] text-muted-foreground">
                      ({tool.parameters.length} param{tool.parameters.length === 1 ? "" : "s"})
                    </span>
                  )}
                  {tool.description && (
                    <span className="truncate text-[11px] font-normal text-muted-foreground">
                      — {firstSentence(tool.description)}
                    </span>
                  )}
                  {tool.requiresApproval && (
                    <Badge variant="destructive" className="ml-auto shrink-0 text-[10px]">
                      approval
                    </Badge>
                  )}
                </div>
              </AccordionTrigger>
              <AccordionContent className="space-y-2 pb-2 text-xs">
                {tool.description && (
                  <p className="whitespace-pre-line text-muted-foreground">{tool.description}</p>
                )}
                {tool.parameters.length > 0 && (
                  <div className="flex flex-col gap-0.5 rounded bg-muted/30 p-2">
                    {tool.parameters.map((p) => (
                      <div key={p.name} className="flex items-center gap-2">
                        <code className="text-foreground">{p.name}</code>
                        <span className="text-muted-foreground">{p.annotation}</span>
                        {!p.required && (
                          <span className="text-[10px] text-muted-foreground">optional</span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      )}
    </div>
  );
};

// First sentence (or first 80 chars) of a tool's description — keeps the
// collapsed row useful without bloating it.
const firstSentence = (s: string): string => {
  const trimmed = s.trim().split(/\n/)[0] ?? "";
  const dot = trimmed.indexOf(". ");
  if (dot > 0 && dot < 120) return trimmed.slice(0, dot + 1);
  return trimmed.length > 80 ? `${trimmed.slice(0, 78)}…` : trimmed;
};

const ToolsTab = (): JSX.Element => {
  const [data, setData] = useState<ApiAgentToolList>({ tools: [], mcpGroups: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sort, setSort] = useState<SortMode>("server");
  const [reloadTick, setReloadTick] = useState(0);

  // Bus listener: any MCP-config mutation in this window forces a re-fetch
  // so users don't have to flip tabs to see new tools after editing a server.
  useEffect(() => onMcpConfigChanged(() => setReloadTick((t) => t + 1)), []);

  // reloadTick is a deliberate re-fetch trigger (incremented by
  // onMcpConfigChanged); the effect body doesn't read it but its
  // identity change drives the re-run.
  // biome-ignore lint/correctness/useExhaustiveDependencies: deliberate trigger
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    agentAdminApi
      .listToolsAndGroups()
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [reloadTick]);

  const groups = sortGroups(buildGroups(data), sort);
  const totalTools = data.tools.length;
  const totalMcp = data.mcpGroups.length;
  const failingMcp = data.mcpGroups.filter((g: ApiMcpToolGroup) => !g.ok).length;

  return (
    <div className="flex h-full flex-col px-4 pb-4">
      <div className="mb-2 flex items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground">
          {totalTools} tool{totalTools === 1 ? "" : "s"} across {totalMcp + 1} source
          {totalMcp + 1 === 1 ? "" : "s"}
          {failingMcp > 0 && (
            <span className="ml-2 text-destructive">· {failingMcp} server failed</span>
          )}
        </p>
        <div className="flex items-center gap-2">
          <Label className="text-xs text-muted-foreground">Sort by</Label>
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as SortMode)}
            className="rounded border bg-background px-2 py-1 text-xs"
          >
            <option value="server">Server / domain</option>
            <option value="name">Tool name</option>
          </select>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            disabled={loading}
            onClick={() => setReloadTick((t) => t + 1)}
          >
            Refresh
          </Button>
        </div>
      </div>
      {error && <p className="mb-2 text-xs text-destructive">{error}</p>}
      <ScrollArea className="flex-1">
        <div className="flex flex-col gap-3 pr-2">
          {loading && <p className="text-sm text-muted-foreground">Loading…</p>}
          {!loading && groups.map((group) => <ToolGroupSection key={group.source} group={group} />)}
        </div>
      </ScrollArea>
    </div>
  );
};

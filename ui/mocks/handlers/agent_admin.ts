/**
 * MSW mocks for /api/agent admin endpoints (mcp, tools, skills).
 *
 * Skills are persisted in an in-memory list so create/update/delete
 * round-trip during dev:mock sessions.
 */

import { http, HttpResponse } from "msw";

interface MockSkill {
  id: string;
  name: string;
  description: string;
  goalTemplate: string;
  slashName: string;
  instructions: string;
  defaultPlanMode: boolean;
  constraints: string[];
  successCriteria: string[];
  tags: string[];
  createdAt: string;
  updatedAt: string;
}

interface SkillBody {
  name?: string;
  goal_template?: string;
  description?: string;
  slash_name?: string;
  instructions?: string;
  default_plan_mode?: boolean;
  constraints?: string[];
  success_criteria?: string[];
  tags?: string[];
}

const RESERVED = new Set(["plan", "clear", "model", "help"]);
const SLASH_RE = /^[a-z0-9][a-z0-9-]{0,31}$/;

const _validateSlashName = (
  next: string,
  excludeId: string | null,
): string | null => {
  if (!next) return null;
  if (!SLASH_RE.test(next)) {
    return `Invalid slash_name '${next}'. Must match [a-z0-9][a-z0-9-]{0,31}.`;
  }
  if (RESERVED.has(next)) {
    return `slash_name '${next}' is reserved by the chat input.`;
  }
  for (const skill of _skills) {
    if (excludeId !== null && skill.id === excludeId) continue;
    if (skill.slashName === next) {
      return `slash_name '${next}' is already used by skill '${skill.id}'.`;
    }
  }
  return null;
};

const _requiredParams = (template: string): string[] => {
  const seen: string[] = [];
  for (const match of template.matchAll(/\{\{\s*([A-Za-z_]\w*)\s*\}\}/g)) {
    const key = match[1];
    if (!seen.includes(key)) seen.push(key);
  }
  return seen;
};

const _now = () => new Date().toISOString();

const _skills: MockSkill[] = [
  {
    id: "skill-energy-vs-temp",
    name: "Plot energy vs temperature",
    description: "Standard sweep visualization for ablation studies.",
    goalTemplate: "Plot total_energy vs temperature in project {{project}}.",
    slashName: "plot-energy",
    instructions: "Prefer Plotly scatter; label axes with units (eV, K).",
    defaultPlanMode: false,
    constraints: ["scope=project"],
    successCriteria: ["A scatter plot is produced"],
    tags: ["plot", "sweep"],
    createdAt: "2026-04-01T00:00:00Z",
    updatedAt: "2026-04-01T00:00:00Z",
  },
];

// ── MCP servers (multi-scope, mirrors the McpStore on the backend) ─────

type McpScope = "user" | "workspace";

interface McpStdioSpec {
  type: "stdio";
  command: string;
  args: string[];
  env: Record<string, string>;
}

interface McpOAuth2Auth {
  type: "oauth2";
  scopes: string[];
  clientId: string | null;
}

interface McpHttpSpec {
  type: "http" | "sse";
  url: string;
  headers: Record<string, string>;
  auth?: McpOAuth2Auth | null;
}

type McpSpec = McpStdioSpec | McpHttpSpec;

interface McpStoredEntry {
  name: string;
  scope: McpScope;
  spec: McpSpec;
}

interface McpUpsertBody {
  name: string;
  scope: McpScope;
  spec: McpSpec;
}

interface McpSecretSetBody {
  value: string;
  scope: McpScope;
}

const SECRET_REF_RE = /\$\{SECRET:([A-Za-z_]\w*)\}/g;

const _mcpServers: McpStoredEntry[] = [
  {
    name: "molexp-data",
    scope: "workspace",
    spec: {
      type: "stdio",
      command: "molexp",
      args: ["mcp-serve", "${workspaceRoot}"],
      env: {},
    },
  },
  {
    name: "python-sandbox",
    scope: "workspace",
    spec: {
      type: "stdio",
      command: "deno",
      args: ["run", "-N", "jsr:@pydantic/mcp-run-python", "stdio"],
      env: {},
    },
  },
];

const _mcpSecrets: Record<McpScope, Record<string, string>> = {
  user: {},
  workspace: {},
};

// Tracks which (scope, name) entries currently have OAuth tokens stored.
// In the real backend this lives on disk under .mcp_oauth/<server>.json.
const _mcpOauthTokens = new Set<string>();

const _findMcp = (scope: McpScope, name: string): number =>
  _mcpServers.findIndex((s) => s.scope === scope && s.name === name);

const _collectSecretRefs = (entry: McpStoredEntry): string[] => {
  const values: string[] =
    entry.spec.type === "stdio"
      ? Object.values(entry.spec.env)
      : Object.values(entry.spec.headers);
  const refs = new Set<string>();
  for (const v of values) {
    for (const m of v.matchAll(SECRET_REF_RE)) refs.add(m[1]);
  }
  return Array.from(refs).sort();
};

const _resolveSecret = (key: string): boolean =>
  key in _mcpSecrets.workspace || key in _mcpSecrets.user;

const _entryToResponse = (entry: McpStoredEntry, shadowed: boolean) => {
  const refs = _collectSecretRefs(entry);
  const unresolved = refs.filter((k) => !_resolveSecret(k));
  const stdio = entry.spec.type === "stdio" ? entry.spec : null;
  const http = entry.spec.type !== "stdio" ? entry.spec : null;
  const auth =
    http?.auth?.type === "oauth2"
      ? {
          type: "oauth2" as const,
          scopes: http.auth.scopes,
          clientId: http.auth.clientId,
          connected: _mcpOauthTokens.has(`${entry.scope}:${entry.name}`),
        }
      : null;
  return {
    name: entry.name,
    scope: entry.scope,
    transport: entry.spec.type,
    command: stdio?.command ?? null,
    args: stdio?.args ?? [],
    url: http?.url ?? null,
    envKeys: stdio ? Object.keys(stdio.env).sort() : [],
    headerKeys: http ? Object.keys(http.headers).sort() : [],
    secretRefs: refs,
    unresolvedSecrets: unresolved,
    shadowed,
    valid: true,
    invalidReason: "",
    auth,
  };
};

const _listMcpServers = () => {
  const wsNames = new Set(
    _mcpServers.filter((s) => s.scope === "workspace").map((s) => s.name),
  );
  return _mcpServers.map((entry) =>
    _entryToResponse(entry, entry.scope === "user" && wsNames.has(entry.name)),
  );
};

interface MockProvider {
  provider: string;
  model: string;
  baseUrl: string;
  apiKeyPreview: string;
  apiKeySet: boolean;
  instructions: string;
  supportedProviders: string[];
}

const _SUPPORTED_PROVIDERS = [
  "anthropic",
  "openai",
  "google",
  "deepseek",
  "openai-compatible",
];
const _DEFAULT_MODELS: Record<string, string> = {
  anthropic: "claude-sonnet-4-6",
  openai: "gpt-4o",
  google: "gemini-2.0-flash",
  deepseek: "deepseek-chat",
  "openai-compatible": "gpt-4o",
};

let _provider: MockProvider = {
  provider: "anthropic",
  model: "claude-sonnet-4-6",
  baseUrl: "",
  apiKeyPreview: "",
  apiKeySet: false,
  instructions: "",
  supportedProviders: _SUPPORTED_PROVIDERS,
};

const _maskKey = (k: string): string => {
  if (!k) return "";
  if (k.length < 8) return "*".repeat(k.length);
  return `${k.slice(0, 3)}...${k.slice(-4)}`;
};

// Track the *real* (mocked) key so subsequent updates that omit api_key
// preserve apiKeySet correctly — without ever returning the raw value.
let _storedKey = "";

const _tools = [
  {
    name: "submit_run",
    description: "Materialize and launch a run for an experiment. Mutates state.",
    parameters: [
      { name: "project_id", annotation: "str", required: true },
      { name: "experiment_id", annotation: "str", required: true },
      { name: "parameters", annotation: "dict", required: false },
    ],
    requiresApproval: true,
    source: "native",
  },
  {
    name: "get_run_status",
    description: "Read the current status of a run from the workspace catalog.",
    parameters: [{ name: "run_id", annotation: "str", required: true }],
    requiresApproval: false,
    source: "native",
  },
  {
    name: "wait_for_run",
    description: "Poll the catalog until a run reaches a terminal status.",
    parameters: [
      { name: "run_id", annotation: "str", required: true },
      { name: "timeout_seconds", annotation: "int", required: false },
    ],
    requiresApproval: false,
    source: "native",
  },
  {
    name: "retry_run",
    description: "Re-launch a previously failed run. Mutates state.",
    parameters: [{ name: "run_id", annotation: "str", required: true }],
    requiresApproval: true,
    source: "native",
  },
  {
    name: "ask_user",
    description: "Pause and prompt the user for free-form input via the chat box.",
    parameters: [{ name: "prompt", annotation: "str", required: true }],
    requiresApproval: false,
    source: "native",
  },
];

export const agentAdminHandlers = [
  http.get("/api/agent/health", () => {
    if (_provider.apiKeySet) {
      return HttpResponse.json({
        ready: true,
        provider: _provider.provider,
        model: _provider.model,
        source: "stored",
        reason: "",
        envVar: "ANTHROPIC_API_KEY",
      });
    }
    return HttpResponse.json({
      ready: false,
      provider: _provider.provider,
      model: _provider.model,
      source: "none",
      reason: `No API key configured for provider '${_provider.provider}'. Save one in Agent Settings → Provider.`,
      envVar: "ANTHROPIC_API_KEY",
    });
  }),

  http.get("/api/agent/provider", () => HttpResponse.json(_provider)),

  // Test connection mock — the real backend probes the actual provider, but
  // here we cannot reach the network. Always return ok=false with a clear
  // "[MOCK]" prefix so users running `dev:mock` aren't tricked into thinking
  // a fake key worked. Switch to `npm run dev` against a real backend to
  // actually validate credentials.
  http.post("/api/agent/provider/test", async ({ request }) => {
    const body = (await request.json()) as {
      provider?: string;
      model?: string;
      api_key?: string;
      base_url?: string;
    };
    const provider = body.provider ?? _provider.provider;
    const model = body.model ?? _provider.model;
    const effectiveKey = body.api_key && body.api_key !== "" ? body.api_key : _storedKey;
    if (!effectiveKey) {
      return HttpResponse.json({
        ok: false,
        provider,
        model,
        latencyMs: 0,
        reply: "",
        error: "No API key configured. Save one before testing.",
      });
    }
    return HttpResponse.json({
      ok: false,
      provider,
      model,
      latencyMs: 0,
      reply: "",
      error:
        "[MOCK] dev:mock cannot validate credentials. Run `npm run dev` against a real backend to actually test the API key.",
    });
  }),

  http.put("/api/agent/provider", async ({ request }) => {
    const body = (await request.json()) as {
      provider?: string;
      model?: string;
      api_key?: string;
      base_url?: string;
      instructions?: string;
    };
    if (body.provider !== undefined && !_SUPPORTED_PROVIDERS.includes(body.provider)) {
      return HttpResponse.json(
        { detail: `Unsupported provider '${body.provider}'` },
        { status: 400 },
      );
    }
    const switchingProvider = body.provider !== undefined && body.provider !== _provider.provider;
    const nextProvider = body.provider ?? _provider.provider;
    const nextModel =
      body.model ?? (switchingProvider ? _DEFAULT_MODELS[nextProvider] : _provider.model);
    if (body.api_key !== undefined) {
      _storedKey = body.api_key;
    }
    _provider = {
      ..._provider,
      provider: nextProvider,
      model: nextModel,
      baseUrl: body.base_url ?? _provider.baseUrl,
      instructions: body.instructions ?? _provider.instructions,
      apiKeyPreview: _maskKey(_storedKey),
      apiKeySet: Boolean(_storedKey),
    };
    return HttpResponse.json(_provider);
  }),

  http.get("/api/agent/mcp/servers", () =>
    HttpResponse.json({
      workspacePath: "/mock/workspace/.mcp.json",
      userPath: "/mock/home/.molexp/mcp.json",
      servers: _listMcpServers(),
    }),
  ),

  http.post("/api/agent/mcp/servers", async ({ request }) => {
    const body = (await request.json()) as McpUpsertBody;
    if (_findMcp(body.scope, body.name) !== -1) {
      return HttpResponse.json(
        { detail: `MCP server '${body.name}' already exists at scope '${body.scope}'.` },
        { status: 409 },
      );
    }
    const entry: McpStoredEntry = { name: body.name, scope: body.scope, spec: body.spec };
    _mcpServers.push(entry);
    return HttpResponse.json(_entryToResponse(entry, false), { status: 201 });
  }),

  http.put("/api/agent/mcp/servers/:name", async ({ params, request }) => {
    const body = (await request.json()) as McpUpsertBody;
    if (body.name !== params.name) {
      return HttpResponse.json(
        { detail: "Path name does not match body name." },
        { status: 400 },
      );
    }
    const idx = _findMcp(body.scope, body.name);
    const entry: McpStoredEntry = { name: body.name, scope: body.scope, spec: body.spec };
    if (idx === -1) _mcpServers.push(entry);
    else _mcpServers[idx] = entry;
    return HttpResponse.json(_entryToResponse(entry, false));
  }),

  http.delete("/api/agent/mcp/servers/:name", ({ params, request }) => {
    const url = new URL(request.url);
    const scope = (url.searchParams.get("scope") as McpScope) ?? "workspace";
    const idx = _findMcp(scope, String(params.name));
    if (idx === -1) {
      return HttpResponse.json(
        { detail: `MCP server '${params.name}' not found at scope '${scope}'.` },
        { status: 404 },
      );
    }
    _mcpServers.splice(idx, 1);
    return HttpResponse.json({
      message: `MCP server '${params.name}' deleted from ${scope}.`,
    });
  }),

  http.post("/api/agent/mcp/servers/:name/test", ({ params, request }) => {
    const url = new URL(request.url);
    const scope = (url.searchParams.get("scope") as McpScope) ?? "workspace";
    const idx = _findMcp(scope, String(params.name));
    if (idx === -1) {
      return HttpResponse.json(
        { detail: `MCP server '${params.name}' not found at scope '${scope}'.` },
        { status: 404 },
      );
    }
    // Mock assumes the user's config is real — synthetic success; the
    // real backend actually probes the server.
    return HttpResponse.json({
      ok: true,
      name: params.name,
      scope,
      transport: _mcpServers[idx].spec.type,
      latencyMs: 42,
      toolCount: 3,
      error: null,
    });
  }),

  http.post("/api/agent/mcp/servers/:name/oauth/start", ({ params, request }) => {
    const url = new URL(request.url);
    const scope = (url.searchParams.get("scope") as McpScope) ?? "workspace";
    const idx = _findMcp(scope, String(params.name));
    if (idx === -1) {
      return HttpResponse.json(
        { detail: `MCP server '${params.name}' not found at scope '${scope}'.` },
        { status: 404 },
      );
    }
    const entry = _mcpServers[idx];
    if (entry.spec.type === "stdio" || entry.spec.auth?.type !== "oauth2") {
      return HttpResponse.json(
        { detail: "Server is not configured for OAuth." },
        { status: 400 },
      );
    }
    // Synthetic authorize URL — clicking it in mock mode 404s, which is
    // fine: the mock only exists to exercise UI state transitions.
    return HttpResponse.json({
      name: params.name,
      scope,
      authorizeUrl: `https://mock-idp.local/oauth2/authorize?mock=1&server=${params.name}`,
    });
  }),

  http.post(
    "/api/agent/mcp/servers/:name/oauth/callback",
    async ({ params, request }) => {
      const url = new URL(request.url);
      const scope = (url.searchParams.get("scope") as McpScope) ?? "workspace";
      const idx = _findMcp(scope, String(params.name));
      if (idx === -1) {
        return HttpResponse.json(
          { detail: `MCP server '${params.name}' not found at scope '${scope}'.` },
          { status: 404 },
        );
      }
      const body = (await request.json()) as { code?: string };
      if (!body?.code) {
        return HttpResponse.json({ detail: "Missing code." }, { status: 400 });
      }
      _mcpOauthTokens.add(`${scope}:${params.name}`);
      const entry = _mcpServers[idx];
      const scopes = entry.spec.type !== "stdio" ? entry.spec.auth?.scopes ?? [] : [];
      return HttpResponse.json({
        name: params.name,
        scope,
        hasTokens: true,
        scopes,
      });
    },
  ),

  http.get("/api/agent/mcp/servers/:name/oauth", ({ params, request }) => {
    const url = new URL(request.url);
    const scope = (url.searchParams.get("scope") as McpScope) ?? "workspace";
    const idx = _findMcp(scope, String(params.name));
    if (idx === -1) {
      return HttpResponse.json(
        { detail: `MCP server '${params.name}' not found at scope '${scope}'.` },
        { status: 404 },
      );
    }
    const entry = _mcpServers[idx];
    if (entry.spec.type === "stdio" || entry.spec.auth?.type !== "oauth2") {
      return HttpResponse.json(
        { detail: "Server is not configured for OAuth." },
        { status: 400 },
      );
    }
    return HttpResponse.json({
      name: params.name,
      scope,
      hasTokens: _mcpOauthTokens.has(`${scope}:${params.name}`),
      scopes: entry.spec.auth.scopes,
    });
  }),

  http.delete("/api/agent/mcp/servers/:name/oauth", ({ params, request }) => {
    const url = new URL(request.url);
    const scope = (url.searchParams.get("scope") as McpScope) ?? "workspace";
    const had = _mcpOauthTokens.delete(`${scope}:${params.name}`);
    return HttpResponse.json({
      message: had
        ? `OAuth tokens cleared for '${params.name}'.`
        : `No OAuth tokens were stored for '${params.name}'.`,
    });
  }),

  http.get("/api/agent/mcp/secrets", ({ request }) => {
    const url = new URL(request.url);
    const scope = (url.searchParams.get("scope") as McpScope) ?? "workspace";
    const setKeys = new Set(Object.keys(_mcpSecrets[scope]));
    const refs = new Map<string, string[]>();
    for (const entry of _mcpServers) {
      if (entry.scope !== scope) continue;
      for (const k of _collectSecretRefs(entry)) {
        if (!refs.has(k)) refs.set(k, []);
        refs.get(k)!.push(entry.name);
      }
    }
    const allKeys = Array.from(new Set([...setKeys, ...refs.keys()])).sort();
    return HttpResponse.json({
      scope,
      path: scope === "workspace" ? "/mock/workspace/.mcp_secrets.json" : "/mock/home/.molexp/mcp_secrets.json",
      secrets: allKeys.map((key) => ({
        key,
        isSet: setKeys.has(key),
        referencedBy: (refs.get(key) ?? []).sort(),
      })),
    });
  }),

  http.put("/api/agent/mcp/secrets/:key", async ({ params, request }) => {
    const body = (await request.json()) as McpSecretSetBody;
    const key = String(params.key);
    if (body.value === "") {
      delete _mcpSecrets[body.scope][key];
      return HttpResponse.json({
        message: `Secret '${key}' cleared at ${body.scope}.`,
      });
    }
    _mcpSecrets[body.scope][key] = body.value;
    return HttpResponse.json({
      message: `Secret '${key}' saved at ${body.scope}.`,
    });
  }),

  http.get("/api/agent/tools", () => {
    // Synthesize one mock MCP group per stored server with type=http/sse so
    // the UI groups-by-server rendering has something to show in dev:mock.
    const mcpGroups = _mcpServers
      .filter((s) => s.spec.type !== "stdio")
      .map((s) => ({
        server: s.name,
        scope: s.scope,
        ok: true,
        toolCount: 2,
        error: null as string | null,
      }));
    const mcpTools = mcpGroups.flatMap((g) => [
      {
        name: `${g.server}_search`,
        description: `Search via ${g.server} (mock).`,
        parameters: [],
        requiresApproval: false,
        source: `mcp:${g.server}`,
      },
      {
        name: `${g.server}_get`,
        description: `Fetch from ${g.server} (mock).`,
        parameters: [],
        requiresApproval: false,
        source: `mcp:${g.server}`,
      },
    ]);
    return HttpResponse.json({
      tools: [..._tools, ...mcpTools],
      mcpGroups,
    });
  }),

  http.get("/api/agent/skills", () => HttpResponse.json({ skills: _skills })),

  http.post("/api/agent/skills", async ({ request }) => {
    const body = (await request.json()) as SkillBody;
    if (!body.name || !body.goal_template) {
      return HttpResponse.json({ detail: "name + goal_template required" }, { status: 400 });
    }
    const slashName = (body.slash_name ?? "").trim();
    const validation = _validateSlashName(slashName, null);
    if (validation) {
      return HttpResponse.json({ detail: validation }, { status: 400 });
    }
    const skill: MockSkill = {
      id: `skill-${Math.random().toString(36).slice(2, 10)}`,
      name: body.name,
      description: body.description ?? "",
      goalTemplate: body.goal_template,
      slashName,
      instructions: body.instructions ?? "",
      defaultPlanMode: body.default_plan_mode ?? false,
      constraints: body.constraints ?? [],
      successCriteria: body.success_criteria ?? [],
      tags: body.tags ?? [],
      createdAt: _now(),
      updatedAt: _now(),
    };
    _skills.push(skill);
    return HttpResponse.json(skill, { status: 201 });
  }),

  http.get("/api/agent/skills/:id", ({ params }) => {
    const skill = _skills.find((s) => s.id === params.id);
    if (!skill) return HttpResponse.json({ detail: "not found" }, { status: 404 });
    return HttpResponse.json(skill);
  }),

  http.patch("/api/agent/skills/:id", async ({ params, request }) => {
    const idx = _skills.findIndex((s) => s.id === params.id);
    if (idx === -1) return HttpResponse.json({ detail: "not found" }, { status: 404 });
    const body = (await request.json()) as SkillBody;
    const current = _skills[idx];
    if (body.slash_name !== undefined) {
      const validation = _validateSlashName(body.slash_name, current.id);
      if (validation) {
        return HttpResponse.json({ detail: validation }, { status: 400 });
      }
    }
    const updated: MockSkill = {
      ...current,
      ...(body.name !== undefined ? { name: body.name } : {}),
      ...(body.goal_template !== undefined ? { goalTemplate: body.goal_template } : {}),
      ...(body.description !== undefined ? { description: body.description } : {}),
      ...(body.slash_name !== undefined ? { slashName: body.slash_name } : {}),
      ...(body.instructions !== undefined ? { instructions: body.instructions } : {}),
      ...(body.default_plan_mode !== undefined
        ? { defaultPlanMode: body.default_plan_mode }
        : {}),
      ...(body.constraints !== undefined ? { constraints: body.constraints } : {}),
      ...(body.success_criteria !== undefined
        ? { successCriteria: body.success_criteria }
        : {}),
      ...(body.tags !== undefined ? { tags: body.tags } : {}),
      updatedAt: _now(),
    };
    _skills[idx] = updated;
    return HttpResponse.json(updated);
  }),

  http.delete("/api/agent/skills/:id", ({ params }) => {
    const idx = _skills.findIndex((s) => s.id === params.id);
    if (idx === -1) return HttpResponse.json({ detail: "not found" }, { status: 404 });
    _skills.splice(idx, 1);
    return HttpResponse.json({ message: "deleted" });
  }),

  http.post("/api/agent/skills/:id/launch", async ({ params, request }) => {
    const skill = _skills.find((s) => s.id === params.id);
    if (!skill) return HttpResponse.json({ detail: "not found" }, { status: 404 });
    const body = (await request
      .json()
      .catch(() => ({}))) as { plan_mode?: boolean; parameters?: Record<string, unknown> };
    const planMode = body.plan_mode ?? skill.defaultPlanMode;
    return HttpResponse.json({
      sessionId: `session-${Math.random().toString(36).slice(2, 10)}`,
      status: "running",
      goalDescription: skill.goalTemplate,
      createdAt: _now(),
      events: [],
      stats: {
        inputTokens: 0,
        outputTokens: 0,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        totalTokens: 0,
        requests: 0,
        toolCalls: 0,
        events: 0,
        startedAt: _now(),
        completedAt: null,
        durationSeconds: 0,
      },
      planMode,
      skillId: skill.id,
    });
  }),

  // ── Slash commands ────────────────────────────────────────────────────

  http.get("/api/agent/commands", () => {
    const builtins = [
      {
        slashName: "plan",
        name: "Plan mode",
        description:
          "Toggle plan mode for the next message (read-only inspection only).",
        parameters: [],
        defaultPlanMode: true,
        isBuiltin: true,
        skillId: null,
      },
      {
        slashName: "clear",
        name: "Clear conversation",
        description: "Discard the current chat transcript and start fresh.",
        parameters: [],
        defaultPlanMode: false,
        isBuiltin: true,
        skillId: null,
      },
      {
        slashName: "model",
        name: "Change model",
        description: "Show or change the active model.",
        parameters: [],
        defaultPlanMode: false,
        isBuiltin: true,
        skillId: null,
      },
      {
        slashName: "help",
        name: "Help",
        description: "Show available commands and a short usage reminder.",
        parameters: [],
        defaultPlanMode: false,
        isBuiltin: true,
        skillId: null,
      },
    ];
    const skillCommands = _skills
      .filter((s) => s.slashName)
      .map((s) => ({
        slashName: s.slashName,
        name: s.name,
        description: s.description,
        parameters: _requiredParams(s.goalTemplate).map((name) => ({
          name,
          required: true,
        })),
        defaultPlanMode: s.defaultPlanMode,
        isBuiltin: false,
        skillId: s.id,
      }));
    return HttpResponse.json({ commands: [...builtins, ...skillCommands] });
  }),

  http.post("/api/agent/commands/parse", async ({ request }) => {
    const body = (await request.json()) as { raw: string };
    const raw = (body.raw || "").trim();
    if (!raw.startsWith("/")) {
      return HttpResponse.json({
        kind: "error",
        name: "",
        skillId: "",
        parameters: {},
        planMode: false,
        error: "Slash commands must start with '/'.",
      });
    }
    const tokens = raw.slice(1).trim().split(/\s+/);
    const head = (tokens.shift() ?? "").toLowerCase();
    if (!head) {
      return HttpResponse.json({
        kind: "error",
        name: "",
        skillId: "",
        parameters: {},
        planMode: false,
        error: "Empty slash command.",
      });
    }
    const params: Record<string, string> = {};
    let parseError: string | null = null;
    for (const token of tokens) {
      if (!token.includes("=")) {
        parseError = `Argument '${token}' is missing a value. Use the form key=value.`;
        break;
      }
      const eq = token.indexOf("=");
      params[token.slice(0, eq)] = token.slice(eq + 1).replace(/^"|"$/g, "");
    }
    if (parseError) {
      return HttpResponse.json({
        kind: "error",
        name: head,
        skillId: "",
        parameters: {},
        planMode: false,
        error: parseError,
      });
    }
    if (RESERVED.has(head)) {
      return HttpResponse.json({
        kind: "builtin",
        name: head,
        skillId: "",
        parameters: params,
        planMode: head === "plan",
        error: "",
      });
    }
    const skill = _skills.find((s) => s.slashName === head);
    if (!skill) {
      return HttpResponse.json({
        kind: "error",
        name: head,
        skillId: "",
        parameters: {},
        planMode: false,
        error: `Unknown command '/${head}'. Define a skill with this slash name first.`,
      });
    }
    const required = _requiredParams(skill.goalTemplate);
    const missing = required.filter((k) => !(k in params));
    if (missing.length > 0) {
      return HttpResponse.json({
        kind: "error",
        name: head,
        skillId: skill.id,
        parameters: {},
        planMode: false,
        error: `Missing required parameter(s) for /${head}: ${missing.join(", ")}`,
      });
    }
    return HttpResponse.json({
      kind: "skill",
      name: head,
      skillId: skill.id,
      parameters: params,
      planMode: skill.defaultPlanMode,
      error: "",
    });
  }),
];

# Agent Harness Architecture: Model-Agnostic Core and Provider Plugins

**Status**: Draft · **Author**: @RoyKid · **Date**: 2026-05-03 · **Revised**: 2026-05-03 (decisions pass: all 7 OQs resolved — see §13 Design Decisions; SessionManager folded into AgentService per Decision O1)

## 1. Motivation

The current agent implementation grew around the first concrete backend:
PydanticAI. That was useful for getting a working end-to-end loop, but it
blurred three separate concerns:

| Concern | What it should own | What must not leak into it |
|---------|--------------------|-----------------------------|
| Agent harness | Context, tools, turn orchestration, session state, memory, events, policy, recovery | Provider-specific SDKs |
| Model adapter | A normalized way to ask one model for the next response | Agent session lifecycle or tool execution |
| Tool adapters | Native tools, MCP tools, future external capability surfaces | Model/provider APIs |

The architectural target is **harness engineering**: everything outside
the model is a first-class, model-agnostic layer. PydanticAI, OpenAI,
Anthropic, local models, or any future model integration should all plug
into the same harness contract.

**Goal**: introduce `molexp.agent` as the dependency-light agent logic
layer, and reduce PydanticAI to a provider plugin that implements the
model boundary.

**Non-goals**:

- No backward compatibility for `molexp.plugins.agent_pydanticai` imports,
  session files, private modules, or server internals.
- No attempt to keep PydanticAI as the orchestrator. The harness owns the
  loop; model plugins return normalized model outputs.
- No implementation of sophisticated memory, evaluation, or recovery in
  the first pass. These layers still get concrete module boundaries,
  protocols, and no-op implementations now.
- No external dependencies in the `molexp.agent` core. Use stdlib
  dataclasses, protocols, JSON-compatible dictionaries, and internal
  `molexp` APIs only.

## 2. Design Principles

1. **Agent is core logic, not a plugin.** The package surface is
   `molexp.agent`. Plugins may contribute model clients or tool sources,
   but the agent lifecycle lives in core.
2. **The model boundary is narrow.** A model plugin accepts a
   `ModelRequest` and returns model text, tool calls, usage, and streaming
   events in normalized molexp types.
3. **Tools are executed by molexp.** Model providers may suggest tool
   calls, but `ToolDispatcher` owns approval, validation, execution,
   result recording, and retries.
4. **Deferred does not mean absent.** Context compression, memory,
   evaluation, and recovery may start as no-ops, but their modules and
   contracts exist from the first refactor.
5. **Layer boundaries are import-enforced.** `molexp.agent` never imports
   `pydantic_ai`, provider SDKs, HTTP clients, MCP SDKs, or optional
   dependencies.
6. **State is inspectable.** Sessions, events, tool calls, plans, memory
   records, and failures are persisted as JSON/JSONL under the workspace.
7. **Policy wraps side effects.** Mutating tools, budgets, workspace
   boundaries, and recovery decisions are handled outside the model
   plugin.

## 3. Target Package Layout

```text
src/molexp/agent/
  __init__.py
  service.py                 # AgentService public entry point
  model.py                   # ModelClient protocol + request/response types
  types.py                   # Goal, Message, SessionEvent, IDs, status enums

  context/
    __init__.py
    manager.py               # ContextManager protocol + default implementation
    prompt.py                # PromptLayer, PromptComposer
    packet.py                # ContextPacket, budget metadata
    compression.py           # no-op now; contract for summarizers later

  tools/
    __init__.py
    spec.py                  # ToolSpec, ToolContext, ToolResult
    registry.py              # ToolRegistry, native_tool decorator
    dispatcher.py            # ToolDispatcher, validation, audit events
    policy.py                # ApprovalPolicy, skill allow/deny policy
    native/
      __init__.py
      workspace.py           # built-in workspace/project/run tools
      workflow.py            # built-in workflow/plan/run tools
      chat.py                # ask_user and interaction tools

  orchestration/
    __init__.py
    session.py               # AgentSession implementation
    manager.py               # live session registry and background task owner
    runner.py                # turn loop: context -> model -> tools -> state
    plan.py                  # plan-mode state machine
    approvals.py             # approval wait/resume primitives
    events.py                # EventBus + serializers

  state/
    __init__.py
    store.py                 # AgentStateStore aggregate
    sessions.py              # SessionStore + session metadata/history
    skills.py                # SkillStore and slash command materialization
    memory.py                # MemoryStore protocol + no-op/jsonl store
    config.py                # model/tool/agent settings, no provider SDKs

  observability/
    __init__.py
    trace.py                 # TraceSink protocol + JSONL sink
    usage.py                 # normalized usage counters
    artifacts.py             # inline artifact normalization
    evals.py                 # Evaluator protocol + no-op evaluator

  recovery/
    __init__.py
    constraints.py           # ConstraintSet, budgets, workspace limits
    errors.py                # typed failure taxonomy
    retry.py                 # RecoveryPolicy + no-op/simple retry

src/molexp/plugins/model_pydanticai/
  __init__.py
  client.py                  # PydanticAIModelClient implements ModelClient
  provider.py                # workspace provider config -> concrete model

src/molexp/plugins/tool_mcp/
  __init__.py
  source.py                  # MCP ToolSource implements ToolRegistry extension
  store.py                   # MCP server/secrets config
  oauth.py                   # OAuth flow; optional deps stay here
```

The names above are the target structure. If an early PR needs smaller
steps, it can land empty/no-op modules first, but the final import graph
must match this split.

## 4. Layer Model

The harness is organized from inner to outer as:

```text
ModelClient boundary
  -> Context management
  -> Tool system
  -> Execution orchestration
  -> State and memory
  -> Evaluation and observability
  -> Constraints and recovery
```

Imports should not literally follow that direction everywhere; protocols
are used to avoid cycles. The rule is behavioral:

- Inner layers do not know concrete outer policies.
- Outer layers may observe, constrain, or replace inner behavior through
  injected protocols.
- Provider plugins only implement `ModelClient`.
- Tool plugins only register `ToolSpec` plus callables.

## 5. Core Data Contracts

All core contracts are stdlib-only. JSON schema is represented as plain
`dict[str, object]`, not as Pydantic models.

### 5.1 Goals and messages

```python
@dataclass(frozen=True)
class Goal:
    description: str
    constraints: dict[str, Any] = field(default_factory=dict)
    success_criteria: list[str] = field(default_factory=list)
    mode: AgentMode = AgentMode.CHAT
    instructions_override: str | None = None
    # Skill is identified, not inlined: ContextManager re-resolves the
    # addendum from SkillStore on every turn so workspace/user/builtin
    # precedence stays in one place.
    skill_id: str | None = None

class AgentMode(str, Enum):
    CHAT = "chat"
    PLAN = "plan"
    REVIEW = "review"  # placeholder; semantics defined when needed.

@dataclass(frozen=True)
class Message:
    """Harness-level semantic message — the source of truth for context
    selection, event logs, and harness-side replay.

    ``content`` is a flat string view sufficient for prompt assembly.
    Provider-native shapes (Anthropic content blocks, OpenAI ``tool_calls``
    arrays, PydanticAI ``ModelMessage`` blobs) are *not* representable
    here. Per Decision M1, the model plugin records them in the parallel
    ``model_io.jsonl`` + ``provider_blobs/`` layer (see §6.4); the
    harness never parses that layer.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 5.2 Model boundary

```python
class ModelClient(Protocol):
    name: str

    async def complete(self, request: ModelRequest) -> ModelResponse:
        ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        ...

@dataclass(frozen=True)
class ModelRequest:
    session_id: str
    turn_id: str
    system: str
    messages: list[Message]
    tools: list[ToolSchema]
    response_format: dict[str, Any] | None = None
    budget: ModelBudget | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ModelResponse:
    text: str = ""
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    raw: Any = None
```

Provider plugins must not execute tools. If a provider SDK has its own
agent/tool abstraction, the plugin may use only the lower-level model
calling surface, or adapt the SDK while disabling its tool execution.

### 5.3 Tool boundary

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    source: str = "native"
    category: str = "workspace"
    mutates: bool = False
    requires_approval: bool = False
    tags: tuple[str, ...] = ()

@dataclass
class ToolContext:
    workspace: Any
    session_id: str
    turn_id: str
    run: Any | None = None
    memory: MemoryStore | None = None

@dataclass(frozen=True)
class ToolResult:
    ok: bool
    value: Any = None
    error: AgentFailure | None = None
    artifacts: list[ArtifactRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

Tool names are globally unique inside one session. Suggested source
prefixes:

- `native:<name>` for package-shipped tools.
- `mcp:<server>.<tool>` for MCP tools.
- `user:<name>` for user-supplied in-process tools.

The model-facing schema may omit the source prefix if the provider has
strict naming rules, but the harness must retain the canonical name in
state and events.

## 6. Layer Specifications

### 6.1 Context Management

**Owns**:

- System prompt layering: base + workspace instructions + skill addendum
  + session override.
- Conversation history selection.
- Workspace context packet selection.
- Context budget accounting.
- Deferred compression/retrieval hooks.

**Does not own**:

- Tool execution.
- Provider selection.
- Durable session writes.

Key contracts:

```python
class ContextManager(Protocol):
    async def build(self, request: ContextBuildRequest) -> ContextPacket:
        ...

@dataclass(frozen=True)
class ContextPacket:
    system: str
    messages: list[Message]
    included_refs: list[ContextRef]
    budget: ContextBudget
    diagnostics: list[str] = field(default_factory=list)
```

Initial implementation:

- `PromptComposer` concatenates layers with deterministic section
  headers.
- `ContextCompressor` is a no-op.
- History selection is simple tail selection by approximate character
  budget.
- Workspace context injection is explicit only; no automatic filesystem
  crawl in the first pass.

### 6.2 Tool System

**Owns**:

- Native tool registry.
- Tool schema generation.
- Skill allow/deny filtering.
- Approval policy checks.
- Tool call validation.
- Tool call execution and result normalization.
- Tool events and audit records.

**Does not own**:

- Model call orchestration.
- Long-term memory writes except through injected stores.

Key contracts:

```python
class ToolRegistry:
    def register(self, spec: ToolSpec, fn: ToolCallable) -> None: ...
    def list(self, policy: ToolPolicy | None = None) -> list[ToolSpec]: ...
    def get(self, name: str) -> RegisteredTool | None: ...

class ToolDispatcher:
    async def dispatch(
        self,
        call: ModelToolCall,
        ctx: ToolContext,
        policy: ToolPolicy,
    ) -> ToolResult: ...
```

Native tools move from the PydanticAI private module into
`molexp.agent.tools.native`. They may depend on internal workspace and
workflow APIs, but not on any model provider.

MCP becomes a tool-source plugin. It is not part of the PydanticAI model
plugin.

### 6.3 Execution Orchestration

**Owns**:

- Session lifecycle.
- One-turn and multi-turn loops.
- Model request/response sequencing.
- Tool-call loop.
- Approval waits.
- `ask_user` waits.
- Plan-mode handoff.
- Cancellation.
- Event fanout to server/UI.

**Does not own**:

- Provider-specific model construction.
- Tool implementation internals.
- Persistence format details beyond store interfaces.

**Component relationships** (resolved by Decision O1):

```text
AgentService              # workspace-scoped facade; owns the live
  │                       # session registry and the asyncio.Tasks
  │                       # that run each session. The only entry
  │                       # point server routes import.
  └─ AgentSession         # per-session handle: streams events,
       │                  # accepts user messages, and dispatches
       │                  # approval/plan decisions to the runner.
       └─ AgentRunner     # executes one turn end-to-end. Holds the
                          # context → model → tools → state pipeline.
```

`AgentService` is the import surface; `AgentSession` and `AgentRunner`
are internal. Server routes call `AgentService` to spawn / resume /
list sessions; they never instantiate `AgentRunner` or `AgentSession`
directly. The earlier-considered `SessionManager` was folded into
`AgentService` because the registry duties did not justify a separate
class.

Turn loop:

```text
1. Receive user message or launched skill goal.
2. Build ContextPacket.
3. Resolve visible ToolSpec list.
4. Call ModelClient.complete().
5. If response has text, append assistant message and emit events.
6. If response has tool calls:
   a. validate against ToolRegistry
   b. request approval if needed
   c. dispatch tool through ToolDispatcher
   d. append tool result message
   e. loop back to step 2 until final response, budget stop, or failure
7. Persist turn checkpoint and terminal state.
```

Plan mode is a state machine, not a prompt convention:

```text
CHAT -> PLAN_REQUESTED -> PLAN_EMITTED -> USER_DECISION
  approved: PLAN_APPROVED -> CHAT continuation with approved plan
  rejected: PLAN_REJECTED -> PLAN_REQUESTED with feedback
```

The model may still be instructed to produce a plan, but the harness
decides when the session is parked and how approval resumes.

**Reject-feedback delivery (Decision O2): synthetic user message.**
On `PLAN_REJECTED` the runner injects a user-role turn whose content is

```text
Plan rejected. Feedback: <text>. Revise the plan and emit it again.
```

Plan mode stays pure prose; the model has no special "plan tool"
obligation, and the message log replays identically with no plan-mode-
specific reader logic. Tool-result delivery via a built-in
`exit_plan_mode` was rejected — it would couple plan mode to the tool
surface and require every model plugin to handle a synthetic tool
result correctly.

### 6.4 State and Memory

**Owns**:

- On-disk session metadata.
- Event logs.
- Message history.
- Plan decisions.
- Skills and slash command materialization.
- Memory interfaces and initial no-op/jsonl implementations.

Workspace layout:

```text
<workspace>/.molexp-agent/
  config.json
  skills.json
  sessions/
    <session_id>/
      session.json
      messages.jsonl       # harness Message records — semantic source
      model_io.jsonl       # raw model request/response records (plugin)
      provider_blobs/      # large or binary provider-native payloads
        <blob_id>.bin      # referenced from model_io.jsonl entries
      events.jsonl         # append-only event log
      checkpoints/
        <turn_id>.json     # per-turn recovery checkpoints
      artifacts/
  memory/
    episodic.jsonl
    semantic.jsonl
```

`session.json` stores the latest summary. `events.jsonl` is append-only.
`messages.jsonl` is the replay source for model context. Checkpoints
store turn-level recovery data and may be omitted for completed turns
whose messages/events are sufficient to replay.

**Message-line schema (Decision M1): two-layer, semantic-only on top.**
`messages.jsonl` stores harness `Message` records and is the single
source for context selection, event-log replay, and any plugin-agnostic
tooling. Raw provider-native shapes go to a parallel `model_io.jsonl`
(one `ModelRequest` / `ModelResponse` pair per line, plus
`provider_blob` references for content too large or binary to fit
inline) and the `provider_blobs/` directory. The model plugin owns
`model_io.jsonl` and `provider_blobs/` exclusively; the harness never
reads them. Plugin-side replay reads `model_io.jsonl` to reconstruct a
provider-native session; harness-side replay reads `messages.jsonl`
only.

**Cross-process session resume (Decision O3): Phase 1 drops, Phase 5
rehydrates.** Phases 1–4 mark non-terminal sessions as `interrupted` on
server restart; the UI surfaces "session ended unexpectedly" and the
user starts a new one. Phase 5 adds rehydration: `AgentService` startup
scans `sessions/`, loads message history, and revives any non-terminal
session as `resumable`; the next inbound user message restarts the turn
loop from the persisted history.

The on-disk format from Phase 1b onward is already turn-complete —
every flushed turn (`messages.jsonl`, `events.jsonl`, `model_io.jsonl`)
is a self-contained unit, never half-flushed. Phase 5 turns rehydration
on without rewriting any session writer.

Initial memory behavior:

- `NoopMemoryStore` returns no memories and ignores writes.
- `JsonlMemoryStore` may land with only append/list operations.
- No embeddings, vector indexes, or external stores in this spec.

### 6.5 Evaluation and Observability

**Owns**:

- Normalized usage accounting.
- Structured traces/spans.
- Artifact capture.
- Evaluation hooks.
- Replay/debug records.

**Does not own**:

- UI presentation.
- Provider-specific telemetry SDKs.

Event categories:

| Event | Emitted by | Purpose |
|-------|------------|---------|
| `SessionStarted` | orchestration | session row creation |
| `TurnStarted` | orchestration | user turn boundary |
| `ContextBuilt` | context | budget and included refs |
| `ModelRequested` | runner | request metadata, no secrets |
| `ModelResponded` | model adapter | usage and finish reason |
| `ToolCallRequested` | runner | model-requested call |
| `ToolApprovalRequested` | tools/policy | human gate |
| `ToolCallCompleted` | dispatcher | result summary |
| `PlanCreated` | plan state machine | structured plan handoff |
| `PlanDecided` | plan state machine | approve/reject/edit |
| `FailureRecorded` | recovery | typed failure |
| `SessionCompleted` | orchestration | terminal summary |

Migration mapping from the current event surface (`molexp.plugins.agent_pydanticai.types`):

| Current event | New equivalent | Notes |
|---|---|---|
| `PlanCreatedEvent` | `PlanCreated` | Same payload; `WorkflowPreview` survives unchanged. |
| `ApprovalRequestEvent` | `ToolApprovalRequested` | Renamed; same shape. |
| `ToolCallEvent` | `ToolCallRequested` | |
| `ToolResultEvent` | `ToolCallCompleted` | Inline result moves into `ToolResult.value`. |
| `ResultArtifactEvent` | (folded into) `ToolCallCompleted` | Inline artifacts come via `ToolResult.artifacts: list[ArtifactRef]`; UI reads the `artifacts` field, not a separate event. |
| `WorkflowStartedEvent` | (folded into) `ToolCallCompleted` | Workflow launch is a tool call; `run_id`/`workflow_id` appear in `ToolResult.metadata`. |
| `ObservationEvent` | (none) | Was a prompt-convention text event, not a model event. Drop. |
| `ReplanEvent` | (none) | Replaced by the plan-mode state-machine transitions in §6.3. |
| `UserMessageRequestEvent` | `UserMessageRequested` | New canonical name; the orchestration layer emits when an `ask_user` tool runs. |
| `UserMessageEvent` | `UserMessageReceived` | Inbound user reply (whether to a request or unsolicited follow-up). |
| `SessionCompletedEvent` | `SessionCompleted` | Same shape. |

Initial implementation:

- JSONL trace sink.
- In-memory event bus for live server streaming.
- No-op evaluator.
- Usage counters copied from model plugin response into normalized
  `Usage`.

### 6.6 Constraints and Recovery

**Owns**:

- Model/tool budget limits.
- Side-effect constraints.
- Workspace boundary constraints.
- Retry and backoff decisions.
- Failure taxonomy.
- Resume decisions from checkpoints.

Failure taxonomy:

```python
class FailureKind(str, Enum):
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    TOOL_NOT_FOUND = "tool_not_found"
    POLICY_DENIED = "policy_denied"
    APPROVAL_DENIED = "approval_denied"
    CONTEXT_OVERFLOW = "context_overflow"
    INVALID_PLAN = "invalid_plan"
    USER_CANCELLED = "user_cancelled"
    WORKSPACE_CONFLICT = "workspace_conflict"
    INTERNAL_ERROR = "internal_error"
```

Initial recovery behavior:

- Model transient failures: one retry with simple delay.
- Tool failures: no automatic retry unless tool marks itself idempotent.
- Approval denied: feed denial back as a tool result and let the model
  continue or stop.
- Invalid plan: reject the plan internally with validation feedback.
- Context overflow: rebuild with smaller history; if still overflowing,
  fail with `CONTEXT_OVERFLOW`.

## 7. Plugin Boundaries

### 7.1 Model plugins

Model plugins register a `ModelClientFactory`.

```python
class ModelClientFactory(Protocol):
    provider_name: str

    def create(self, config: ModelConfig) -> ModelClient:
        ...
```

`model_pydanticai` is allowed to import PydanticAI. No other package is.
It must not expose PydanticAI classes in the core API.

**Provider configuration is shared core state.**
`agent.state.config.ProviderConfig` carries the generic shape
(`provider_name`, `api_key`, `base_url`, `model`, `instructions`) so the
UI and admin routes can render or edit settings without a model plugin
loaded. Each model plugin registers two contracts:

- `ProviderConfigValidator` — per-provider field rules (e.g. required
  `base_url` for `openai-compatible`, default base URL for `deepseek`).
- `ModelClientFactory.create(config) -> ModelClient` — concrete
  construction.

Adding a new provider means adding a plugin, not editing core. This
resolves the original OQ on provider-config ownership.

The PydanticAI plugin should avoid `pydantic_ai.Agent` as the main
runtime. If the SDK cannot be used as a raw model client, the adapter may
be transitional, but all tool execution and session orchestration must
still be disabled or bypassed.

### 7.2 Tool plugins

Tool plugins register a `ToolSource`.

```python
class ToolSource(Protocol):
    source_name: str

    async def list_tools(self, workspace: Any) -> list[ToolSpec]:
        ...

    async def call(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ...
```

MCP belongs here. OAuth, HTTP clients, MCP SDKs, server probing, and
secret stores live in `molexp.plugins.tool_mcp`, not in any model
plugin.

**v1 scope (Decision S2): native + MCP only.** The `ToolSource`
contract supports user-home or workspace Python tool packs in the
abstract, but v1 forbids any auto-discovered Python imports — both
user-home (`~/.molexp/tools/`) and workspace-implicit. Custom Python
tool packs require explicit declaration in the workspace's
`.molexp-agent/` configuration and are deferred to v1.1+. Rationale:
user-home auto-import would silently inject capabilities into every
workspace (reproducibility hazard) and run arbitrary code in the agent
process (trust escalation).

## 8. Server and UI Impact

Server routes should depend on the core service:

```python
from molexp.agent import AgentService
```

Route modules must not own global `_sessions` dictionaries. They
receive or construct a workspace-scoped `AgentService` (which owns the
live session registry — see Decision O1), then translate between
FastAPI schemas and core dataclasses.

Admin routes split into:

- Agent config: core settings, skills, native tools.
- Model providers: list/test/update provider configs through model plugin
  registry.
- Tool sources: MCP server/secrets/OAuth through tool plugin registry.

The UI can keep the same conceptual panels, but the backend labels should
change:

- "Provider" means model provider.
- "Tools" is model-agnostic and includes native + tool plugins.
- "MCP servers" belongs to tool sources, not PydanticAI.
- Session inspector reads `.molexp-agent/sessions/<id>`.

## 9. Removal and Migration Policy

This spec deliberately does **not** preserve old agent internals.

Remove:

- `src/molexp/plugins/agent_pydanticai/service.py`
- `src/molexp/plugins/agent_pydanticai/runtime.py`
- `src/molexp/plugins/agent_pydanticai/types.py`
- `src/molexp/plugins/agent_pydanticai/tools.py`
- `src/molexp/plugins/agent_pydanticai/tool_registry.py`
- `src/molexp/plugins/agent_pydanticai/_pydantic_ai/`

Move or rewrite:

- Provider config code -> `plugins/model_pydanticai/provider.py` or a
  generic `agent.state.config` plus provider-specific factory.
- MCP store/OAuth/probe -> `plugins/tool_mcp/`.
- Skills -> `molexp.agent.state.skills`.
- Native tools -> `molexp.agent.tools.native`.
- Session stores -> `molexp.agent.state.sessions`.
- Server route globals -> `molexp.agent.orchestration.manager`.

Old session files do not need to be read. Old imports should fail loudly
or be removed; no re-export layer is required.

## 10. Implementation Phases

**UI track runs alongside, not after.** Whenever a phase changes server
route schemas, generated TypeScript clients, MSW mocks, and UI renderers
must land in the same PR (or paired PRs merged together behind the same
flag). Phase 6 is reserved for documentation and naming polish only —
not a holding pen for UI work that should have shipped earlier.

### Phase 0 - Skeleton and import guard

- Create `src/molexp/agent/` package with all target subpackages.
- Add protocols/dataclasses/no-op implementations.
- Add an import-guard test that fails if `molexp.agent` imports
  `pydantic_ai`, HTTP clients, MCP SDKs, or provider SDKs.
- No behavior migration yet.

### Phase 1a - Types, registry, FakeModelClient

- Port `Goal`, `Message`, `SessionEvent`, `Usage`, `ToolSpec`,
  `ToolRegistry` into core. No behavior, no orchestration.
- Implement `FakeModelClient`: deterministic, scriptable
  (pre-seeded responses + tool-call replays). Becomes the test rig for
  every later phase.
- Land the `AgentService` API surface with stub `start_session` /
  `list_sessions` returning typed empty values.
- Tests: types round-trip; `FakeModelClient` is deterministic; the
  Phase 0 import-guard test still passes.

### Phase 1b - Text-only turn loop

- Implement `AgentRunner` for the no-tool path: context build →
  `ModelClient.complete` → append assistant message → emit events →
  persist `messages.jsonl`, `model_io.jsonl`, and `events.jsonl`.
- `AgentService` owns the asyncio.Task and the per-session event bus
  (Decision O1).
- `AgentSession.stream_events()` reads from the bus.
- Persist turn-completely: every flushed turn is a self-contained unit
  so Phase 5 rehydration (Decision O3) lights up without rewriting any
  session writer.
- Tests: full text-only session via `FakeModelClient`; events arrive
  in order; both `messages.jsonl` and `model_io.jsonl` round-trip
  cleanly.

### Phase 1c - Tool-call loop

- Add tool dispatch into `AgentRunner`: validate against `ToolRegistry`,
  apply `ToolPolicy`, execute, append tool message, loop.
- Implement `ToolDispatcher` end-to-end (no MCP yet — `tool_mcp` is
  Phase 4).
- Tests: model requests a native tool, dispatcher executes it, result
  feeds back into next model turn; unknown-tool returns
  `TOOL_NOT_FOUND` typed failure.

### Phase 2 - Native tools and plan mode

- Move native workspace/workflow/chat tools into
  `molexp.agent.tools.native`. Per Decision T1, tools register on an
  `AgentService`-owned `ToolRegistry` instance — `@native_tool` only
  tags candidates, no module-level singleton.
- Implement `ToolDispatcher` approval flow with HITL gates.
- Implement the plan-mode state machine. Reject feedback flows back as
  a synthetic user message (Decision O2); no `exit_plan_mode` tool.
- Persist plan decisions in `events.jsonl` (`PlanCreated` /
  `PlanDecided`) plus a per-turn checkpoint.
- Update server `/agent/sessions`, `/events`, `/messages`,
  `/plan-decision`, and `/approve` routes to use core.
- UI: regenerate TS client; update MSW handlers; map current event
  types to the new ones per §6.5; `npm test` and `npm run dev:mock`
  must both pass before merge.

### Phase 3 - PydanticAI as model plugin

- Create `molexp.plugins.model_pydanticai`.
- Implement `PydanticAIModelClient` behind the `ModelClient` protocol.
  Per Decision M1 the plugin owns `model_io.jsonl` writes and
  `provider_blobs/` storage exclusively; the harness never reads
  either. See §14 R1 for the `Agent.run()` rebuild risk and §14 R2 for
  the raw-message serialization risk.
- Register `ProviderConfigValidator` and `ModelClientFactory` per §7.1.
- Update provider settings routes to use the model plugin registry.
- Delete old `agent_pydanticai` runtime/session/catalog code.
- UI: provider settings panel reads from the new plugin registry
  endpoint; remove "PydanticAI" labels; regenerate client.

### Phase 4 - MCP as tool plugin

- Create `molexp.plugins.tool_mcp`.
- Move MCP store, secret store, OAuth, probe, and tool listing into the
  tool plugin.
- Expose MCP tools through `ToolSource`, not model-specific toolsets.
- UI: MCP settings panel moves to the tool-source admin route; OAuth
  flow triggers point at `tool_mcp` endpoints; MSW handlers updated.

### Phase 5 - State, observability, recovery hardening

- Switch session persistence to `.molexp-agent/`.
- Add JSONL trace sink and replay helpers.
- Add failure taxonomy to model and tool errors.
- Add simple recovery policy and budget constraints.
- Add no-op evaluator hooks to terminal session flow.
- Migrate legacy `sessions/` folders: write tombstone `session.json`
  records under `.molexp-agent/` (see §14 R4); UI lists legacy sessions
  read-only.
- Light up cross-process session rehydration per Decision O3:
  `AgentService` startup scans `sessions/`, marks non-terminal
  sessions `resumable`, and revives the turn loop on the next inbound
  user message. The Phase 1b on-disk format is already turn-complete,
  so no session writer changes.
- Land `.molexp-agent/` catalog indexing per Decision S1: index every
  session as a *system* asset (default-hidden in the file view) so
  audit/replay tooling can query it without cluttering the user's
  workspace browser.

### Phase 6 - UI and docs cleanup

- Update generated API models.
- Update agent settings text to reflect model/tool split.
- Remove references to PydanticAI as "the agent".
- Rewrite getting-started agent docs around the harness architecture.

## 11. Test Plan

Core unit tests:

1. `molexp.agent` imports without optional dependencies installed.
2. `ContextManager` composes prompt layers deterministically.
3. `ToolRegistry` rejects duplicate names.
4. `ToolPolicy` applies allow/deny and mutating-tool approval.
5. `ToolDispatcher` validates unknown tools and returns typed failures.
6. `AgentRunner` completes a text-only fake-model session.
7. `AgentRunner` executes a fake model tool call and feeds the result
   back into the model.
8. Plan mode parks on `PlanCreated`, resumes on approval, and loops on
   rejection.
9. `SessionStore` writes `session.json`, `messages.jsonl`, and
   `events.jsonl`.
10. Recovery policy handles model failure, tool failure, and invalid plan
    without provider-specific exceptions.

Integration tests:

1. Server can create/list/get/stream sessions using `FakeModelClient`.
2. Server plan decision route resumes a parked session.
3. Native workspace tools can create a project/experiment/run through
   the core dispatcher.
4. PydanticAI model plugin satisfies the `ModelClient` contract.
5. MCP tool plugin satisfies the `ToolSource` contract with a fake MCP
   server/probe.

Import boundary tests:

1. No `pydantic_ai` import outside `plugins/model_pydanticai`.
2. No MCP SDK import outside `plugins/tool_mcp`.
3. No model provider SDK import inside `molexp.agent`.

## 12. Acceptance Criteria

- `molexp.agent` is the only Python package that defines agent session
  semantics.
- PydanticAI is not mentioned in core agent APIs, event types, route
  schemas, or native tool code.
- The harness can run a complete fake-model session with no optional
  dependencies installed.
- Native tools are registered and executed through the core dispatcher.
- Plan mode is implemented as an orchestration state machine.
- Session state is persisted under `.molexp-agent/` as JSON/JSONL.
- Server routes use `AgentService` (which owns the live session
  registry per Decision O1); route-local session globals are removed.
- Deferred memory/eval/recovery modules exist with no-op implementations
  and tests proving they are wired into the session flow.

## 13. Design Decisions

All seven OQs from the previous draft are resolved. Each entry below
names the decision, the rationale, and the alternatives that were
rejected. Tags: **M** = message/state schema, **O** = orchestration,
**T** = tools, **S** = state/storage. New OQs (if any arise) should be
appended below as `OQ-<tag><n>` entries pending decisions.

### M1 — On-disk message schema (Resolved 2026-05-03)

**Decision**: Two-layer, semantic-only on top. `messages.jsonl` stores
harness `Message` records and is the single source of truth for
context selection, event-log replay, and any plugin-agnostic tooling.
Raw provider-native shapes go to a parallel `model_io.jsonl` (one
`ModelRequest` / `ModelResponse` per line) plus a `provider_blobs/`
directory for content too large or binary to fit inline. The model
plugin owns the raw layer exclusively; the harness never parses it.

**Rationale**: The harness needs a stable, plugin-agnostic shape so
context-selection and UI code work without loading any model plugin. A
unified `MessageRecord { harness, raw }` line was rejected because the
`raw` field would couple the on-disk schema to whichever plugin wrote
it. Two parallel journals keep the layers independent and auditable;
plugin-side replay still works from `model_io.jsonl` alone.

### O1 — Component decomposition (Resolved 2026-05-03)

**Decision**: Three classes. `AgentService` (workspace facade + live
session registry + asyncio.Task owner — i.e. the former Service and
Manager merged), `AgentSession` (per-session handle), `AgentRunner`
(single-turn pipeline). `SessionManager` is folded into `AgentService`.

**Rationale**: The registry duties of the proposed `SessionManager`
did not justify a separate class — the workspace-scoped facade
naturally owns the live registry. Collapsing `AgentSession` or
`AgentRunner` into the facade was rejected: per-session state and the
single-turn pipeline are independent concerns and benefit from
isolated tests.

### O2 — Plan-rejection feedback channel (Resolved 2026-05-03)

**Decision**: Synthetic user message. On `PLAN_REJECTED` the runner
injects a user-role turn with content
`"Plan rejected. Feedback: <text>. Revise the plan and emit it
again."`. No built-in `exit_plan_mode` tool.

**Rationale**: User-role turns are the most natural input shape for
every provider; no model has special semantics for tool-result turns
that don't follow real tool calls. Replay is simpler — the message log
contains nothing plan-mode-specific, so a recorded session replays
identically to a fresh one. Tool-result delivery would couple plan
mode to the tool surface and demand consistent handling from every
model plugin.

### O3 — Cross-process session resume (Resolved 2026-05-03)

**Decision**: Phase 1 drops, Phase 5 rehydrates. Phases 1–4 mark
non-terminal sessions `interrupted` on server restart. Phase 5 adds
rehydration via an `AgentService` startup scan that loads message
history and marks non-terminal sessions `resumable`. The on-disk
format from Phase 1b onward is already turn-complete (every flushed
turn is self-contained), so Phase 5 lights up rehydration without
rewriting any session writer.

**Rationale**: Phase 1 cannot absorb the design surface of full
rehydration on top of a turn-loop rewrite. Deferring to Phase 5 keeps
Phase 1 focused; making the on-disk format rehydration-ready from day
one prevents a second storage rewrite later.

### T1 — Tool registry scope (Resolved 2026-05-03)

**Decision**: Explicit registration on an `AgentService`-owned
registry. Each `AgentService` instance creates its own `ToolRegistry`;
native tools are imported and registered at construction time. The
`@native_tool` decorator only tags a callable as a candidate — it
does not auto-register on a module-level global.

**Rationale**: Per-workspace tool isolation is the deciding factor.
Module-level singletons cause tests to contaminate each other and
break concurrent `AgentService` instances in the same process.
Decorator-tag plus service-owned registration gives both ergonomics
and isolation.

### S1 — `.molexp-agent/` catalog integration (Resolved 2026-05-03)

**Decision**: Phase 5+ indexes `.molexp-agent/` in the workspace
catalog, but each entry is tagged as a *system* asset and hidden by
default in the file view. Audit and replay tooling queries the
catalog; ordinary user file browsing does not see the entries.

**Rationale**: Catalog indexing is needed for cross-session queries
("find every session that ran tool X"). Surfacing every session as a
regular asset would clutter the user's workspace view. The
system-asset tag already exists for similar concerns and slots in at
low cost.

### S2 — User-home Python tool packs (Resolved 2026-05-03)

**Decision**: v1 `ToolSource` implementations are limited to native
(built-in) plus MCP (Phase 4). Auto-discovered Python tool packs from
`~/.molexp/tools/` or any other user-home path are forbidden in v1.
Workspace-declared local Python tool packs (registered explicitly in
the workspace's `.molexp-agent/` configuration) are deferred to
v1.1+.

**Rationale**: User-home auto-import would silently inject
capabilities into every workspace the server opens, breaking
reproducibility across machines. It also escalates trust — anything
in user-home runs in the agent process. Workspace-declared packs are
a controlled extension that can land later when the declaration
format is settled.

## 14. Known Risks and Degradation

### R1 — Rebuilding `pydantic_ai.Agent.run()` is bigger than it looks

`Agent.run()` currently provides: tool-execution loop, retry on
transient errors, structured-output validation, message-history schema
(`ModelMessagesTypeAdapter`), streaming event dispatch with usage
accumulation, and MCP toolset integration. Phase 1b/1c rebuild most of
this in `AgentRunner`. Underestimating this cost is the single largest
schedule risk.

**Mitigation**:

- Phase 1a includes a *recorded baseline*: capture the current
  `Agent.run()` behavior on a fixture session (text-only, then
  text+tool) as JSON traces. Phase 1b/1c must reproduce these traces
  bit-for-bit modulo timestamps.
- Reserve a transitional-adapter escape hatch: if the rewrite stalls,
  `model_pydanticai` may temporarily wrap `Agent.run()` and translate
  events inside the plugin, violating §7.1 ("must not execute tools").
  Track as explicit tech debt with a Phase 5 exit ticket. The escape
  hatch is *not* a recommendation — it is a survival path so a stalled
  Phase 3 cannot block the rest of the cutover.

### R2 — Provider-native message serializer correctness

Per Decision M1, raw provider-native messages live in `model_io.jsonl`
+ `provider_blobs/`, owned exclusively by the model plugin. Each new
model plugin therefore writes its own raw-message serializer. Sloppy
serialization — e.g. dropping Anthropic content-block ordering, losing
OpenAI `tool_call_id` correlation, or eliding parts that are only
non-obvious on multi-step tool runs — silently breaks plugin-side
replay even though `messages.jsonl` looks fine.

**Mitigation**:

- Cover replay parity in tests: every Phase 3 test that reuses a
  recorded session must produce model output equal to the original
  recording for the same provider.
- The Phase 1a `FakeModelClient` is the reference implementation of a
  round-trip-correct `model_io.jsonl` writer; new plugins copy its
  serializer skeleton and add provider-specific marshalling.

### R3 — UI lockstep slips

`/agent/*` route schema changes in Phases 2, 3, 4 each invalidate the
generated TypeScript client and MSW handlers. If UI work is deferred to
Phase 6, the UI will be unrunnable for weeks.

**Mitigation**:

- Each schema-changing phase has an explicit UI sub-bullet (§10).
- A phase does not merge until `npm test` and `npm run dev:mock` both
  pass against the updated routes.

### R4 — Existing workspace state during cutover

Existing workspaces have `sessions/<id>/{metadata,history}.json` from
the current implementation. `.molexp-agent/` is a new path. §9 is "no
backward compat for session files," but live workspaces will contain
both during the cutover and the UI must not crash on either.

**Mitigation**:

- Phase 5 ships a one-shot migration tool: scans `sessions/`, copies
  `goal` + final summary to `.molexp-agent/sessions/<id>/session.json`
  with `status="legacy"`. Drops messages (no replay possible across the
  schema change). UI shows legacy sessions read-only.
- If a workspace has zero `sessions/` entries, the migration is a
  no-op.

### R5 — `pydantic_ai` MCP integration disappears with the plugin split

Today MCP toolsets are loaded inside `_pydantic_ai/runtime.py` via the
PydanticAI MCP client classes (`MCPServerStdio`, `MCPServerSSE`,
`MCPServerStreamableHTTP`). Phase 4 moves MCP into `tool_mcp`. The
PydanticAI MCP classes assume a PydanticAI Agent owns the toolset; we
need an MCP client implementation that fits the harness `ToolSource`
contract (list + call) and is independent of any model plugin.

**Mitigation**:

- Phase 4 evaluates the official MCP Python SDK
  (`mcp` package) as the basis for `tool_mcp` rather than reusing
  PydanticAI's MCP wrappers. PydanticAI's wrappers may be referenced
  for behavior parity but should not be imported.
- If an SDK gap is found (e.g. OAuth handling), it is a v1.1
  follow-up — not a Phase 4 blocker. The workaround is to keep
  affected MCP servers offline until the gap is closed.


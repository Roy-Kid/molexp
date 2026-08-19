<!-- mol:note:topic:harness-plugin-kernel -->
# Harness plugin kernel

**Status:** live discipline (2026-08-18). Target architecture for the host; today's packages are the default plugins of one profile, not a different ontology.

> **Companion:** artifact / provenance / IR / approval internals stay in [`harness-goal.md`](harness-goal.md). Cross-layer coordination (WorkspaceContext → KnowledgeDelta) stays in [`integration.md`](integration.md). Those documents describe **bundles** mounted on this kernel. They do not define the kernel.

**Rule:** molexp is a plugin composition. The atom is one **model call** (`AgentGateway.call` → pydantic-ai primitive). Chat is one shot and does not loop. Tool-using work is one ReAct (`stream_agentic`); ReAct already stops when the model stops calling tools. **Plan is a molexp Workflow** whose nodes each run one call/ReAct and persist artifacts. Tools are Host plugins. Do not grow `ChatLoop` / `InteractiveLoop` / `run_plan` as kernels.

**Supersedes:** ChatLoop and InteractiveLoop as first-class loops; Plan as a privileged two-phase pipeline beside chat; the idea that plugins hang off InteractiveLoop.

---

## 0. The smallest harness

```text
plugin host (context + lifecycle + event bus)
  + llm adapter (pydantic-ai)
  + one model call
```

Two call shapes, both pydantic-ai, both entered through `AgentGateway.call`:

| Shape | pydantic-ai | When |
|---|---|---|
| **one-shot** | `Agent.run` / `complete_text` / `complete_structured` | Chat. Any node that only needs a schema or a reply. **Must not loop.** |
| **ReAct** | `Agent.iter` / `stream_agentic` | A node that may use tools. Stops when the model stops calling tools. That *is* the should-stop. |

`ChatLoop` and `InteractiveLoop` are deleted. Chat is one-shot. ReAct is `stream_agentic`. Plan is a workflow.

### 0.1 Names

| Name | Meaning |
|---|---|
| **AgentCall** | Harness envelope around one pydantic-ai call (one-shot or one ReAct). Persist raw + parsed artifacts. |
| **tool plugin** | Registers on `ctx.tools` (effect; unloads). ReAct sees the current set. |
| **Plan** | A `CompiledWorkflow`. Each task node performs one AgentCall and writes artifacts. Form/acceptance failure is a **workflow edge** (retry that node or fail the run), not a second agent loop. |
| **ChatMode** | One-shot face: one `call`, no session loop. |
| **REPL / server session** | A `Session` of many turns. Each user line is one ReAct that sees prior messages. The conversation is the session, not a molexp loop. |

### 0.2 What InteractiveLoop's outer `should_stop` becomes

ReAct already ends a call. The old outer guard ("board incomplete → another ReAct") is **workflow control flow**: after the ReAct node, a validator node; fail/loop-back is `wf.loop` / `wf.branch`, recorded on the execution. It is not a molexp-owned conversation loop.

---

## 1. Everything is a plugin

Copied as *discipline* from [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) / Cordis — not as a TypeScript runtime to vendor.

### 1.1 Five host ideas

1. **A plugin is a Service.** It declares `inject` (services it waits for) and `apply(ctx)` (what it publishes). Function or class; same contract.
2. **A context is a service map.** Other plugins find capabilities by key (`ctx.workspace`, `ctx.workflow`, `ctx.llm`, `ctx.tools`, `ctx.approval`, …), not by importing a concrete package at the composition boundary.
3. **Dependency is `inject`, not boot order.** A plugin that names required services does not activate until those services exist. Today's layer DAG is the **static inject graph** of the default profile.
4. **Typed events are the extension points.** Observe (`emit`), wrap (`waterfall`), fan-out (`parallel`), ordered (`serial`). Policy and adapters attach here. Changing the AgentCall driver to add a feature is a defect.
5. **Registrations are reversible effects.** Tools, prompt sections, providers, listeners, capability rows go through an effect that **unwinds on unload**. A plugin that cannot leave the process cleanly is not a plugin.

There is no privileged product core to patch. You extend molexp by mounting a plugin beside the others.

### 1.2 Capability seam (complete or it is not a seam)

A **seam** is one swappable capability with three roles. One role alone is not a seam.

| Role | Owns | molexp examples already close |
|---|---|---|
| **Service Definition** | `ctx.<key>` + vocabulary types | `AgentGateway`, `CapabilityRegistry`, `FileSystem`, `Executor`, `ApprovalStore` |
| **Service Provider** | one implementation of that definition | `RouterBackedAgentGateway` / `StubAgentGateway`; `LocalFileSystem` / `RemoteFileSystem` / `CachingFileSystem`; `LocalExecutor` / `DryRunExecutor`; `SQLiteApprovalStore` |
| **Consumer** | model-facing tool, Stage, or lifecycle verb that only talks to the definition | plan board tools, `InvokeCapability`, `Run.execute` via `set_run_executor`, harvest |

Swapping a provider must move every consumer of that seam. Filesystem + subprocess share one execution world: point them at a remote target and Bash-equivalent, preview, and job submit move together. That is why compute target, `FileSystem`, and molq must be one world, not three forks.

### 1.3 What is *not* the plugin host

`molexp.plugins.PluginRegistry` (today: optional-extra loaders for `gh` / tensorboard / molpy preview / molvis / submit_molq) is **one bundle of optional providers**, not the host. Entry-point groups `molexp.cli_plugins` / `molexp.ui_plugins` are consumers of the same host, not a second kernel.

The host belongs in the harness kernel (context + lifecycle + AgentCall dispatch). Do not grow a parallel registry in `workspace` or `agent`.

---

## 2. Default plugin graph

Every current layer is a plugin (or a small family of plugins). The import-guard DAG is how the default profile may `inject`. It is not a reason those packages sit outside the plugin system.

```text
                    ┌──────────── host ────────────┐
                    │  context · lifecycle · events │
                    │  AgentCall dispatch           │
                    └──────────────┬───────────────┘
                                   │ mounts
         knowledge                 │
         (concept-type registry)   │
                ▲                  │
                │ inject           │
           workspace ──────────────┤  ctx.workspace  (Folder, Run, assets, events)
                ▲                  │
        ┌───────┴────────┐         │
        │                │         │
     workflow          llm ────────┤  ctx.workflow / ctx.llm
        │                │         │
        └───────┬────────┘         │
                │                  │
             harness ──────────────┤  stages, stores, approval, registry
             (also the host)       │
                ▲                  │
            services               │  application drivers (plan_runtime, …)
                ▲                  │
         cli / server / ui         │  face plugins
                                   │
     molpy / molvis / molq / molmcp / gh / metrics
                                   │  science + adapter plugins
```

### 2.1 Service keys (`ctx.xxx` — DeepSeek names)

Access is attribute style (`ctx.tools`), same as Cordis. `Keys` is only the string catalog; composition code does not use `ctx.require(Keys.TOOLS)` as the primary API.

Shared spine (names match DeepSeek Harness):

| `ctx` key | Plugin publishes | Role |
|---|---|---|
| `ctx.llm` | llm adapter | pydantic-ai behind this seam only |
| `ctx.tools` | tools plugin | model-facing registry + `tools/*` pipeline |
| `ctx.fs` | workspace / fs provider | `FileSystem` (local / remote / cached) |
| `ctx.approval` | approval plugin | store-first + `approval/request` waterfall |
| `ctx.sessions` | session plugin | append-only agent session log |
| `ctx.systemPrompt` | prompt-section plugin | cache-stable sections; not the mutable board |
| `ctx.jobs` | molq plugin | submit / cancel / poll |
| `ctx.sandbox` | sandbox plugin (when mounted) | process confinement |
| `ctx.commands` | face plugins | human verbs that skip the model |
| `ctx.settings` | operator config | `~/.molexp/config.json` / `molexp.config` |
| `ctx.credentials` | credentials plugin | API keys; never in prompts |

Domain extras (molexp science — still `ctx.xxx`; DeepSeek has no equivalent):

| `ctx` key | Plugin publishes | Role |
|---|---|---|
| `ctx.knowledge` | knowledge plugin | concept-type registry |
| `ctx.workspace` | workspace plugin | `Folder` / `Run` / assets — plugin programmatic interface, not host-owned |
| `ctx.workflow` | workflow plugin | compile / execute / resume |
| `ctx.artifacts` | store plugin | content-addressed harness artifacts |
| `ctx.events` | store + workspace | append-only timelines (not folded into sessions) |
| `ctx.lineage` | store plugin | artifact edges |
| `ctx.capabilities` | catalog / molmcp | science API rows; not `ctx.tools` |

Do **not** add `ctx.agentLoop` (their loop plugin — our atom is one pydantic-ai call). Do **not** add `ctx.agents` / `ctx.planMode` until those plugins actually mount. Do not reserve unused keys.

A domain plugin `apply` publishes its service **and** may register model-facing functions on `ctx.tools`. Other plugins consume `ctx.workspace` / `ctx.workflow`; the model consumes only `ctx.tools`. Composition-boundary consumers do not import `molexp.workflow._engine` or a concrete molq client.

Events (DeepSeek names where the seam matches): `agent/pre-step`, `llm/stream`, `tools/pre-execute`, `tools/execute`, `tools/post-execute`, `approval/request`, `fs/*`. `agent/pre-call` is retired.

### 2.2 Workspace and workflow are plugins

**Workspace plugin.** Provides durable identity and bytes (`Folder` family, `ops/run.json`, `assets.json`, OKF concepts). Providers differ by `FileSystem` and compute target; the definition does not. Consumers: workflow execute, agent session folders, harvest, plan tools, git projection. On-disk layout and the verb law (`run` / `resume` / `rerun` / `cancel`) are **this plugin's public contract**, not host internals — other plugins must not invent a parallel run record.

**Workflow plugin.** Provides compile + values-on-edges execute + resume seed. It injects `ctx.workspace` (run dir, cache, atomic JSON). It must not inject `ctx.llm` as a required service. The `set_run_executor` inversion already is a seam: workspace consumes a provider registered by the workflow plugin.

**LLM plugin.** Publishes `ctx.llm`. The only `import pydantic_ai` site. Not a product Agent class. Chat = one structured call; a REPL line = one ReAct via `ctx.llm`.

**Harness plugins.** The host (`molexp.harness.host`) *plus* default store / approval / plan-board / realize plugins. `run_plan` is not the kernel — plan/solve are bundles composed on the host.

**Science plugins** (`molpy`, `molvis`, `molq`, molmcp, metrics ingest). Each is a seam: package public API is the definition, the molexp adapter is the provider, a `ToolCapability` or preview route is the consumer. Science methods are never host tools; the host discovers them through `ctx.capabilities` / molmcp.

---

## 3. Lifecycle

```text
compose          profile lists bundles, then patches (user / home / --patch)
  → load         topological by inject; missing required service fails loud
  → apply        plugin publishes ctx.<key>, registers effects
  → ready        AgentCall and other service methods may run
  → intercept    waterfalls: agent/pre-step, llm/stream, tools/pre-execute, fs/*, approval/request
  → unload       dispose effects in reverse; services leave the context
```

### 3.1 Compose (profiles and bundles)

A running molexp is a plugin tree composed at boot.

| Object | Meaning | Today's seed |
|---|---|---|
| **profile** | named composition | `chat` / `plan` / `run` / `curate` / `serve` (operator + CLI entry) |
| **bundle** | distributable plugin rows + the code they mount | `harness` stores+gateway; `workspace`; `workflow`; `agent`; `submit_molq`; UI |
| **patch** | replace or insert a row by id | `~/.molexp/config.json`, extra extras, MCP catalog |

Suggested default stacks (later rows see earlier services):

```text
chat     : host + llm                  → one structured call
plan     : chat + workspace + capabilities + approval + board + realize
run      : workspace + workflow + executor (+ jobs if molq mounted)
curate   : workspace + approval + change-proposal handlers
serve    : plan + run + curate + server/ui face plugins
```

`dsh --profile web --dump-config` is the DeepSeek inspection verb. molexp needs the same: dump the plugin tree the process actually booted. Until that exists, `molexp config` + extras + MCP catalog are the incomplete stand-in.

### 3.2 Load / apply / unload

- **Load waits on `inject`.** No manual "import workspace then workflow then harness" in new composition code. The existing import-guard tests stay as the static checker of the default profile.
- **Apply is the only publication point.** A plugin that mutates process globals or writes disk in `import` is violating lifecycle (workspace already forbids I/O in `__init__`; keep that as an apply-time rule).
- **Unload unwinds effects.** Capability rows, tools, MCP sessions (`keep_alive=False` already), approval listeners, executor registration (`set_run_executor`) must have disposers. A second profile must be mountable in one process without leaking the first profile's tools.

### 3.3 AgentCall lifecycle (the atom)

```text
agent/pre-step          waterfall — rewrite or reject the AgentCallSpec
  persist prompt/input artifacts        (model-visible ⟺ logged)
  ctx.llm request                       (structured = one request;
                                         agentic = request ↔ tools/* until stop)
    tools/pre-execute   waterfall — approval, sandbox, path policy
    tools/execute
    tools/post-execute
  persist output_artifact + raw_response_artifact
agent/post-call
```

`AgentCallSpec.call_mode` stays a closed `Literal["structured", "agentic"]`. Structured is the **minimum** form. Agentic is the same envelope with the tools seam enabled.

**Model-visible ⟺ logged.** Anything that reaches the model (system prompt sections, tool schemas, prior knowledge, claimed user text) must be reconstructable from artifacts + the event log. A new model-visible input requires a new event or artifact kind. `AssembleKnowledgeContext` is the pattern: prior knowledge becomes a `knowledge_context` artifact with lineage, never a prompt-side side-channel.

### 3.4 Plugin-owned lifecycles (not the host's)

These stay inside their plugin. The host does not re-encode them.

| Plugin | Owns | Host must not |
|---|---|---|
| workspace | Run status × verbs; `ops/run.json`; Folder CRUD; asset manifests | invent a second run fingerprint or status store |
| workflow | compile / execute / resume seed; cache identity | load `_engine` in the host process |
| agent | Session log; ChatLoop / InteractiveLoop | import `pydantic_ai` outside `_pydanticai/` |
| approval | pending / grant / reject; store-first replay | treat `ApprovalPendingError` as failure |
| molq | job submit / cancel / poll | embed scheduler types in workspace |

---

## 4. Where new behavior goes

| Goal | Mechanism |
|---|---|
| Add a model provider | register on `ctx.llm` |
| Add a model-facing tool | register on `ctx.tools` (schema joins prompt assembly) |
| Add storage / identity | workspace plugin (or a `ctx.fs` provider) |
| Add graph execute / resume | workflow plugin provider behind `ctx.workflow` |
| Add HPC / jobs | `ctx.jobs` provider (molq); consumers stay on the definition |
| Add science API | capability row + molmcp page; never a hand-maintained tool list |
| Intercept a call, tool, or write | `agent/*` / `tools/*` / `fs/*` waterfall |
| Add a human-facing command | face plugin (`cli` / `server` / UI), not an AgentCall |
| Give one session a different tool set | compose a preset / isolate realm (DeepSeek agent-scope) |
| Change plan / chat / curate shape | a different **bundle**, not a fork of the host |
| Change the AgentCall driver itself | update this file first |

---

## 5. Laws (host)

1. **Atom is AgentCall.** The smallest harness is one structured call. Bundles compose calls; they are not a second atom.
2. **Everything is a plugin.** Workspace, workflow, agent, knowledge, science adapters, faces, and orchestration mount beside each other. Today's layer DAG is the default inject graph.
3. **Seams are complete.** Definition + Provider + Consumer. One role is not a seam.
4. **Registrations are effects.** Unload unwinds. Import-time side effects are forbidden.
5. **Model-visible ⟺ logged.** New model-visible input ⇒ new artifact or event.
6. **Plugins, not driver changes.** New behavior attaches to a documented key or event. Editing the AgentCall driver or `run_plan` to add a product feature is the wrong layer.
7. **Agent proposes; plugins dispose.** The agent plugin returns structured proposals. Writes, executes, submits, and approvals go through the owning plugin's service, usually behind `ctx.approval`.
8. **Misconfiguration fails at load.** A missing inject, an unknown capability id in a required bundle, a broken extra — loud at compose/load, never a silent skip.
9. **Do not vendor Cordis.** The discipline is binding. The Python host is ours (or a future chosen runtime) and must stay inside the layer/inject DAG. A TypeScript kernel is not an import.

---

## 6. Reconcile with what already shipped

Keep; they become plugin-internal contracts or policy plugins:

- Artifact + append-only events + lineage (`harness-goal.md` §1.2–1.4)
- Workflow IR ≠ Bound Workflow
- Tests as artifacts
- Store-first approval
- Run verb law and on-disk layout (workspace plugin contract)
- `AgentCallSpec` / `AgentCallResult` (already the atom)
- `FileSystem` / `Executor` / `CapabilityRegistry` Protocols (already seam definitions)
- `set_run_executor` (already a provider registration)

Re-read as bundles, not as the kernel:

- `run_plan` two-phase pipeline
- `ChatMode` (the `chat` bundle on InteractiveLoop)
- `integration.md` coordination loop (WorkspaceContext → KnowledgeDelta)
- `harness.capabilities` catalogs (rows a plugin registers)

Replace over time:

- `molexp.plugins.PluginRegistry` as a one-off optional-dep cache → providers on the host
- Stages that import concrete siblings instead of injecting `ctx.*`
- Prompt assembly that is not registered as an effect
- MCP / tool surfaces that cannot unload

---

## 7. Implementation stance

The host is in `molexp.harness.host` (not on the frozen 22-symbol package root).

| Symbol | Role |
|---|---|
| `Context` / `Host` / `Keys` / `Plugin` | kernel |
| `compose_chat` / `compose_plan` / `compose_curate` / `compose_run` | profiles |
| `AgentCallPlugin` | atom + waterfall; injects `artifacts` and rebinds persist. Chat and plan nodes enter here. |
| `RunStoresPlugin` / `CapabilitiesPlugin` | default mounts |

Production composers: `run_plan.run`, `ChatMode.run` (unload in `finally`), `services.curate_runtime` `_build_ctx`.

`compose_run` / `ToolsPlugin` / `WorkflowPlugin` / `Host.dump_config` ship. Science adapters mount as `extra=` plugins on a composer. Do not add a second kernel. Do not grow `run_plan` to own interpret / harvest / multi-cycle research.

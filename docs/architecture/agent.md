# Agent Layer Architecture

The agent layer (`molexp.agent`) is a clean, user-facing wrapper around
[pydantic-ai](https://github.com/pydantic/pydantic-ai). The library is
an implementation detail hidden inside the private `_pydanticai/`
subpackage and **never appears in the public surface**.
(`pydantic-graph` belongs to the workflow layer and is never imported
under `agent/` — see the confinement rule below.)

## Public API

The agent layer exposes five user-visible names plus the two concrete
loops:

```python
from molexp.agent import (
    AgentLoop, AgentRunner, AgentRunResult, AgentRuntime, AgentSession,
)
from molexp.agent.loops import ChatLoop, InteractiveLoop
```

`ChatLoop` is one LLM round-trip; `InteractiveLoop` is the emergent
tool loop driving `Router.stream_agentic`. ("Loop" is the agent-layer
LLM-conversation concept; "Mode" is reserved for `molexp.harness.Mode`
orchestration — the former agent-side PlanMode / RunMode pipelines moved
to the harness layer.)

Construction is plain Python — no factory functions
(`create_agent(...)` / `build_agent(...)` / `Agent(provider=...)`).
`AgentRunner(*, loop, model=…, router=…)` lazily builds the underlying
pydantic-ai router on first `.run()`, so `import molexp.agent` is cheap.
A loop receives an `AgentRuntime` (`session` + `router` +
`execution_env`) and emits everything it sees through the injected
`AsyncIteratorEventSink`.

## Layer Boundaries

```text
harness ──uses──▶ agent ──uses──▶ workspace
                  (agent and workflow are SIBLINGS — no edge between them)
```

`molexp.agent` may import from `molexp.workspace.*` (the storage layer
below it). It must **not** import `molexp.workflow` or `molexp.harness`
— workflow is a sibling and harness sits above; harness reaches the
agent layer through `molexp.agent.router` (the SDK-free `Router`
Protocol module). It must also not import sibling application layers
(`molexp.plugins`, `molexp.server`, `molexp.cli`).

### pydantic-ai firewall

The only files under `src/molexp/agent/` allowed to `import pydantic_ai`
live inside `agent/_pydanticai/`:

| File | Purpose |
|---|---|
| `_pydanticai/router.py` | `PydanticAIRouter` — concrete `Router` implementation; one `Agent` instance per `(tier, schema \| None)` |
| `_pydanticai/mcp.py` | `build_mcp_server` helper used to attach MCP toolsets to a loop |
| `_pydanticai/messages_codec.py` | message-history (de)serialization between molexp sessions and pydantic-ai |
| `_pydanticai/errors.py` | `ProviderError` — the single SDK-failure boundary |

The firewall is enforced by `tests/test_agent/test_import_guard.py`.
`import molexp.agent` does not eagerly load `pydantic_ai` — every
construction site is hidden behind a lazy import that fires only on the
first `AgentRunner.run()` call.

### pydantic-graph confinement

`pydantic_graph` is exclusively imported under
`src/molexp/workflow/_pydantic_graph/`. **Nothing under
`src/molexp/agent/` may import it.** The agent never schedules a
workflow itself; the harness does, and even there the engine runs in an
executor subprocess rather than in-process.

## Don't Reinvent pydantic-ai

> Anything pydantic-ai provides natively in *model-side execution* (tool
> dispatch, MCP, retries, message history, structured output) MUST use
> pydantic-ai; do not build parallel implementations under
> `molexp.agent`.

Concrete consequences:

* Tools — pass `pydantic_ai.tools.Tool` instances or bare callables to
  `AgentRunner(tools=...)`; the router forwards them verbatim into
  `Agent(tools=...)`. No molexp middle layer.
* MCP servers — `Agent(toolsets=[MCPServerStdio(...)])`. molexp does not
  iterate over MCP needs by hand.
* Retries — `Agent(retries=N)`. The router's outer retry budget on top
  of pydantic-ai is a structured-path safety net, not a re-implementation.
* Message history — pydantic-ai's `RunResult.all_messages()` and
  `Agent(message_history=...)`.
* Structured output — `Agent(output_type=Schema)`.

molexp retains ownership of the **session / event / on-disk** layer
(`Session` persistence, the `AgentEvent` stream, the `Agent` /
`AgentSession` folders) because pydantic-ai does not cover those. The
multi-stage experiment pipeline is owned by `molexp.harness`, one layer
up.

## Pipeline orchestration lives in the harness

The agent layer used to host its own multi-step pipelines (PlanMode,
AuthorMode, RunMode, ReviewMode) built on `molexp.workflow`. That
orchestration moved out: `molexp.harness` now owns the `Mode` ledger
and the canonical `PlanMode` (9 stages) and `RunMode` (10 stages)
pipelines. The harness drives the LLM through the agent's
`Router` Protocol via `RouterBackedAgentGateway` — the single sanctioned
`harness → agent` edge. See [Plan Mode Architecture](plan-mode.md) for
the stage list and artifact layout.

## Sessions and events

A conversation is recorded by a `Session` (re-exported as
`AgentSession`): an append-only entry tree with a leaf pointer.
Persistence is pluggable behind the `SessionStorage` Protocol —
`JsonlSessionStorage` on disk, `InMemorySessionStorage` in tests.
On-disk, an `Agent` folder (`kind = "agent.agent"`) and `AgentSession`
folder (`kind = "agent.session"`) attach to any workspace `Folder`
through the generic `add_folder` CRUD, so the workspace layer never
learns their agent-specific shape.

Every observable step a loop takes is emitted as an `AgentEvent`
(a discriminated union) through an `AsyncIteratorEventSink`. The CLI's
`AgentEventRenderer` and the server's SSE stream consume the same event
contract, so terminal and web clients render identical conversations.

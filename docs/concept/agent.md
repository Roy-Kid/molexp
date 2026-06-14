# Agent

The agent layer turns natural-language research intent into LLM
conversations that can read a workspace and call tools. It is a
*library* — it does not own the application shell (no FastAPI routes,
no CLI command tree) and it does not orchestrate multi-stage
experiment pipelines. It is a thin facade over
[pydantic-ai](https://github.com/pydantic/pydantic-ai) plus the
session, event, and on-disk plumbing that pydantic-ai does not provide.

> **Loop vs Mode.** The agent-layer LLM-conversation concept is a
> **Loop**. **Mode** is reserved for the harness layer, which owns the
> multi-stage Plan / Run pipelines (see [Plan Mode](../architecture/plan-mode.md)).
> Pipeline orchestration used to live in the agent layer; it now lives
> in `molexp.harness`, which reaches the agent through the SDK-free
> `molexp.agent.router.Router` Protocol.

## Public surface

The agent exposes five user-visible names plus the two concrete loops:

```python
from molexp.agent import (
    AgentRunner,
    AgentLoop,
    AgentRunResult,
    AgentRuntime,
    AgentSession,
)
from molexp.agent.loops import ChatLoop, InteractiveLoop
```

- `AgentRunner` — the orchestration entry point. You construct it with
  a loop plus a model (or a per-tier `models` map, or a ready-made
  `router`); it lazily builds the underlying pydantic-ai router on the
  first `.run()`, then drives the loop. `run()` returns an
  `AgentRunResult`; `run_events()` yields the typed `AgentEvent` stream
  for live rendering.
- `AgentLoop` — the abstract base every loop implements:
  `async def run(*, runtime, sink, user_input) -> None`. The loop never
  builds its own router or session — it receives them in the
  `AgentRuntime` and emits everything it observes through the injected
  event sink.
- `AgentRuntime` — the frozen bundle a loop receives at run time:
  `session` + `router` + `execution_env`. It is what makes a loop
  testable in isolation.
- `AgentSession` — the on-disk handle for a conversation (a re-export
  of the runtime `Session`): an append-only entry tree plus a leaf
  pointer, persisted through a `SessionStorage`.
- `AgentRunResult` — the frozen value returned by `AgentRunner.run`,
  carrying the final text, token usage, and an optional structured
  failure.

Two loops ship today:

- `ChatLoop` — a single LLM round-trip. Send one message, get one
  completion event.
- `InteractiveLoop` — the emergent tool loop the CLI defaults to. It
  drives `Router.stream_agentic`, forwarding each chunk (text deltas,
  tool calls, tool results) as it arrives, and ships read-only
  `read_file` / `list_directory` / `search_code` tools.

```python
from molexp.agent import AgentRunner, AgentSession
from molexp.agent.loops import ChatLoop, ChatLoopConfig

runner = AgentRunner(
    loop=ChatLoop(config=ChatLoopConfig(system_prompt="…")),
    model="anthropic:claude-sonnet-4-5",
)
result = await runner.run(AgentSession(), "summarize this dataset")
print(result.text)
```

The interactive loop is what `molexp agent` exposes as a terminal REPL —
it streams the same events the web UI renders.

## Layer position

`agent` is a **sibling of `workflow`**, sitting one layer above
`workspace` and one layer below `harness`:

```text
harness ──uses──▶ agent ──uses──▶ workspace
                  (agent and workflow are siblings — no edge between them)
```

- **workspace** — `Workspace`, `Run`, `RunContext`, `AssetCatalog`,
  the `Agent` / `AgentSession` folders. Sessions persist as folders in
  the workspace through workspace's generic `add_folder` CRUD; the
  workspace layer stays unaware of their agent-specific shape.
- **harness** sits *above* the agent and reaches it through exactly one
  module — `molexp.agent.router` — so the agent never depends on the
  pipeline machinery that consumes it.

The agent must **not** import `molexp.workflow`, `molexp.harness`, or
any application layer (`plugins` / `server` / `cli`). It stays a
library.

## SDK isolation

Two import-boundary rules keep the layer light and honest:

- `pydantic_ai` is confined to `src/molexp/agent/_pydanticai/`.
  `import molexp.agent` does **not** load it; the router is built lazily
  on the first `AgentRunner.run()`, so call sites that never reach the
  LLM stay cold.
- `pydantic_graph` is **never** imported under `agent/` — that is the
  workflow layer's private concern.

Import-boundary tests (`tests/test_agent/test_import_guard.py`)
mechanically enforce both rules.

## Sessions and events

`AgentSession` is the durable record of a conversation. It is an
append-only entry tree with a leaf pointer, persisted through a
`SessionStorage` — `JsonlSessionStorage` on disk, `InMemorySessionStorage`
in tests. Everything a loop observes flows out as an `AgentEvent`
(a discriminated union) through an `AsyncIteratorEventSink`, so the CLI
renderer and the server's SSE stream consume one event contract.

## What's not here

- **Pipeline orchestration** — Plan / Run staging, artifact lineage,
  approval gates, and audit reports live in `molexp.harness`, not the
  agent. The harness drives the agent only through `agent.router.Router`.
- **The LLM provider matrix** — pydantic-ai handles provider routing
  inside `_pydanticai/router.py`. The agent's public API stays
  provider-neutral; a model string (or per-tier map) is the only handle.
- **Workflow execution** — owned by `molexp.workflow`. The agent is a
  *consumer* of that layer through the harness, never an alternate
  scheduler.

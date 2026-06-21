# Agent

The agent layer is molexp's **pydantic-ai facade**: it turns a model
configuration plus an *agent loop* into a driven LLM conversation that
streams typed events. It is a *library* — it owns no application shell
(no FastAPI routes, no CLI command tree) and it does **not** orchestrate
experiment pipelines. Multi-stage orchestration (`PlanMode` / `RunMode`)
lives one layer up in `molexp.harness`; see the bottom of this page.

## Public surface

`molexp.agent` exposes exactly five user-visible names
(`src/molexp/agent/__init__.py.__all__`):

```python
from molexp.agent import (
    AgentRunner,
    AgentLoop,
    AgentRunResult,
    AgentRuntime,
    AgentSession,
)
```

- `AgentRunner` — orchestration entry point. Constructed with a **loop**
  plus a model configuration; it lazily builds the private pydantic-ai
  router on the first `run`, assembles an `AgentRuntime`, injects it into
  the loop, drains the loop's event stream, and returns the terminal
  `AgentRunResult`.
- `AgentLoop` — abstract base for one LLM conversation. A loop is a plain
  `async def run(*, runtime, sink, user_input) -> None`; every
  `AgentEvent` it produces flows through the injected sink. User code
  never constructs the router or the runtime.
- `AgentRuntime` — the frozen bundle a loop receives at run time:
  `session` + `router` + `execution_env`.
- `AgentSession` / `Session` — per-conversation state (message history +
  session id), backed on disk by `JsonlSessionStorage` when a workspace
  is configured, otherwise `InMemorySessionStorage`.
- `AgentRunResult` — frozen pydantic result returned by `AgentRunner.run`
  (carries `text` and the accumulated `events` stream).

Two concrete loops ship under `molexp.agent.loops`:

- `ChatLoop` — a single LLM round-trip.
- `InteractiveLoop` — the emergent tool loop, driving
  `Router.stream_agentic` until the model stops requesting tools.

```python
from molexp.agent import AgentRunner
from molexp.agent.loops.chat import ChatLoop

runner = AgentRunner(loop=ChatLoop(), model="openai:gpt-5.2")
session = runner.session("demo")
result = await runner.run(session, "summarize this dataset")
print(result.text)
```

### Model configuration

The model is given exactly one of three mutually-exclusive ways
(supplying zero or two-or-more raises `AgentRunnerConfigError`):

- `model="provider:id"` — one string applied to every tier
  (`CHEAP` / `DEFAULT` / `HEAVY`).
- `models={ModelTier.CHEAP: …, ModelTier.DEFAULT: …, ModelTier.HEAVY: …}`
  — explicit per-tier mapping (string tier keys are coerced).
- `router=<custom Router>` — escape hatch for tests and fakes.

`AgentRunner` performs no network IO at construction time.

## Layer position

`agent` is a **sibling of `workflow`**, sitting above `workspace`; both
are below `harness`. It depends downward only:

- **workspace** — `Workspace`, `Run`, `RunContext`, `AssetCatalog`,
  `Folder`, … for on-disk session storage.

The agent does **not** import `molexp.workflow` or `molexp.harness` (they
are a sibling and an upstream layer); it never imports any application
layer (`plugins` / `server` / `cli` / `sweep`). These rules are
mechanically enforced by `tests/test_agent/test_import_guard.py`.

## SDK isolation

Two third-party SDKs sit behind import-boundary firewalls:

- `pydantic_ai` is confined to `src/molexp/agent/_pydanticai/` (the sole
  sanctioned `import pydantic_ai` site — `router.py`, `mcp.py`,
  `messages_codec.py`, …). `import molexp.agent` does **not** eagerly load
  it; the router is constructed lazily on the first `AgentRunner.run`, so
  call sites that never reach the LLM stay cold.
- `pydantic_graph` is **not** imported anywhere under `agent/`. The
  workflow graph engine is owned by `molexp.workflow`.

`tests/test_agent/test_import_guard.py` mechanically enforces both rules
(plus the public surface in `tests/test_agent/test_public_surface.py`).

## Sessions

A `Session` is the in-memory conversation handle. With a workspace,
`AgentRunner.session(id)` anchors it to an `AgentSession` `Folder` under
an `Agent` folder named after the loop, and backs it with a
`JsonlSessionStorage` writing `entries.jsonl` in that directory — so a
conversation survives across processes. Without a workspace, an
`InMemorySessionStorage` is used.

## Relationship to the harness

Pipeline orchestration is **not** in this layer. It moved up to
`molexp.harness` and is reached through the `agent.router.Router`
Protocol — the single sanctioned `harness → agent` import edge.

- `PlanMode` and `RunMode` are `molexp.harness.modes` `Mode` pipelines,
  not agent loops. `PlanMode()` constructs with **no** config object and
  runs as:

  ```python
  from molexp.harness.modes.plan import PlanMode

  result = await PlanMode().run(run=run, user_input=draft, gateway=gateway)
  ```

- The production entry point is the **`molexp plan [--execute]`** CLI
  (`src/molexp/cli/plan_cmd.py`): it drives `PlanMode` against a
  content-addressed `Run`, and with `--execute` chains `RunMode` on the
  same Run.

The harness reaches LLM reasoning through an `AgentGateway` whose
production implementation (`RouterBackedAgentGateway`) drives an
`agent.router.Router`. Workflow execution itself is owned by
`molexp.workflow`; the harness is a *consumer* of both layers, never an
alternate scheduler.

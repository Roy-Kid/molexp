# Agent

The agent layer turns natural-language research intent into LLM conversations that can read a workspace and call tools. It is a thin façade over [pydantic-ai](https://github.com/pydantic/pydantic-ai) plus the session, event, and on-disk plumbing that pydantic-ai does not provide.

There is no molexp-owned conversation loop. Chat is one `Router.complete_text`. Tool-using work is one ReAct (`Router.stream_agentic`). Plan is a harness workflow (see [Plan Mode](../guide/plan-mode.md)).

## Public surface

Four names:

```python
from molexp.agent import (
    AgentRunner,      # entry point — model + mode="text"|"agentic", call .run()
    AgentRunResult,   # returned by .run() — text, token usage, events
    AgentRuntime,     # frozen bundle: session + router + execution_env
    AgentSession,     # on-disk handle for a conversation
)
```

| `AgentRunner.mode` | Behavior | Use case |
|---|---|---|
| `"text"` | One `complete_text` — must not loop | Single-turn generation |
| `"agentic"` | One ReAct (`stream_agentic`) | Tool-using REPL / `molexp agent` |

## Quick example

```python
# docs: skip — requires an LLM API key
from molexp.agent import AgentRunner

runner = AgentRunner(model="anthropic:claude-sonnet-4-5", mode="text")
session = runner.session("chat-demo")  # persisted on disk
result = await runner.run(session, "summarize this dataset")
print(result.text)
```

The interactive loop is what `molexp agent` exposes as a terminal REPL — it streams the same events the web UI renders.

## Layer position

```
harness ──uses──→ agent ──uses──→ workspace
                  (agent and workflow are siblings — no edge between them)
```

The agent imports only `molexp.workspace`. It must **not** import `molexp.workflow`, `molexp.harness`, or any application layer. Pipeline orchestration lives in harness, reached through `molexp.agent.router.Router` — the single sanctioned `harness → agent` edge.

## SDK isolation

Two rules keep the agent layer honest:

- `pydantic_ai` is confined to `src/molexp/agent/_pydanticai/`. `import molexp.agent` does **not** eagerly load it — the router is built lazily on the first `.run()`.
- `pydantic_graph` is **never** imported under `agent/` — molexp dropped that dependency entirely.

## Sessions and events

`AgentSession` is the durable record of a conversation: append-only entries persisted through `JsonlSessionStorage` (on disk) or `InMemorySessionStorage` (in tests). Everything a loop observes flows out as an `AgentEvent` discriminated union through an `AsyncIteratorEventSink` — CLI renderer and server SSE stream consume the same contract.

## Next

- For the pipeline that drives the agent through PlanOrchestrator's two phases, see [Plan Mode](../guide/plan-mode.md).
- For the architecture-level import rules, see [Agent Layer Architecture](../architecture/agent.md).

# Agent

The agent layer turns natural-language research intent into LLM conversations that can read a workspace and call tools. It is a thin façade over [pydantic-ai](https://github.com/pydantic/pydantic-ai) plus the session, event, and on-disk plumbing that pydantic-ai does not provide.

**Loop vs Mode.** The agent-layer LLM-conversation concept is a **Loop**. **Mode** is reserved for the harness layer, which owns the multi-stage Plan pipeline (see [Plan Mode](../guide/plan-mode.md)).

## Public surface

Five names plus two concrete loops:

```python
from molexp.agent import (
    AgentRunner,      # entry point — construct with loop + model, call .run()
    AgentLoop,        # abstract base — async def run(*, runtime, sink, user_input)
    AgentRunResult,   # returned by .run() — text, token usage, optional failure
    AgentRuntime,     # frozen bundle a loop receives: session + router + execution_env
    AgentSession,     # on-disk handle for a conversation
)
from molexp.agent.loops import ChatLoop, InteractiveLoop
```

| Loop | Behavior | Use case |
|---|---|---|
| `ChatLoop` | One round-trip: user message → LLM response | Quick questions, single-turn generation |
| `InteractiveLoop` | Open-ended tool loop: LLM calls tools, sees results, continues | Planning, multi-step creation, exploration |

## Quick example

```python
# docs: skip — requires an LLM API key
from molexp.agent import AgentRunner
from molexp.agent.loops import ChatLoop, ChatLoopConfig

config = ChatLoopConfig(system_prompt="You are a helpful research assistant.")
runner = AgentRunner(loop=ChatLoop(config=config), model="anthropic:claude-sonnet-4-5")
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

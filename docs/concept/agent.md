# Agent

The agent layer turns natural-language research intent into structured
molexp artifacts. It is a *library* — it does not own the application
shell (no FastAPI routes, no CLI command tree); it sits between
`molexp.workflow` and the application that drives it.

## Public surface

The agent exposes exactly four user-visible names:

```python
from molexp.agent import (
    AgentRunner,
    AgentMode,
    AgentRunResult,
    AgentSession,
)
```

- `AgentRunner` — orchestration entry point. Takes a mode + a model
  string, lazily constructs a private `PydanticAIHarness` on first
  `.run()`, and injects it into the mode.
- `AgentMode` — abstract base. Subclasses implement
  `async def run(*, harness, session, user_input) -> AgentRunResult`.
  User code never constructs the harness.
- `AgentSession` — opaque per-conversation state. Holds the message
  history, mode-specific scratch state, and the session ID.
- `AgentRunResult` — frozen pydantic model returned by every
  `AgentRunner.run` call. Contains `text`, `usage`, optional
  `mode_state`, optional `failure`.

Concrete modes live in `molexp.agent.modes`:

- `ChatMode` — single-turn LLM round-trip via the harness.
- `PlanMode` — workflow-backed; drives a private multi-step plan
  graph through the public `molexp.workflow` API. Returns an
  `AgentRunResult` whose `mode_state["plan"]` carries the structured
  plan.
- `ReviewMode` — phase-2 placeholder.

```python
from molexp.agent import AgentRunner, AgentSession
from molexp.agent.modes import ChatMode, ChatModeConfig

mode = ChatMode(config=ChatModeConfig(system_prompt="…"))
runner = AgentRunner(mode=mode, model="openai:gpt-5.2")
result = await runner.run(AgentSession(), "summarize this dataset")
print(result.text)
```

## Layer position

`agent` is the top of the molexp dependency DAG. It depends on both
downstream layers:

- **workspace** — `Workspace`, `Run`, `RunContext`, `AssetCatalog`,
  `SubsystemStore`. Sessions live under
  `<workspace_root>/.subsystems/agent.sessions/` via workspace's
  generic subsystem storage.
- **workflow** — `Workflow`, `WorkflowSpec`, `Task`, `TaskContext`,
  `default_registry`, `Runnable`. PlanMode builds workflow specs and
  runs them through the standard `WorkflowSpec.execute(run=run)` API.

The agent does not import any sibling application layer
(`plugins` / `server` / `cli` / `sweep`). The agent stays a library.

## SDK isolation

Two third-party SDKs sit behind import-boundary firewalls:

- `pydantic_ai` is a private implementation detail confined to
  `src/molexp/agent/_pydanticai/`. `import molexp.agent` does **not**
  eagerly load it; the harness is constructed lazily on first
  `AgentRunner.run()`. This keeps the agent layer's import time
  light and lets call sites that don't actually need the LLM stay
  cold.
- `pydantic_graph` is **not** imported anywhere under `agent/`.
  Multi-step modes drive their workflows through the public
  `molexp.workflow` API, the sole sanctioned pg site in the project.

Import-boundary tests (`tests/test_agent/test_import_guard.py`)
mechanically enforce both rules.

## Sessions

`AgentSession` is the in-memory handle. Persistence to disk happens
through `SessionStore` (per-workspace `session.json` /
`messages.jsonl` / events) and `SessionCatalog` (the flat row index
under `<workspace_root>/.subsystems/agent.sessions/_index.json`).
`SessionCatalog.create / list / get / delete` keeps the on-disk
metadata file and the index row in sync.

The session catalog used to live under workspace as the workspace-side
session library; the 2026-05-09 rectification moved it up to the
agent layer because its schema (goal-summary projection, agent
status strings) is inherently agent-shaped. Workspace just vends the
`.subsystems/agent.sessions/` directory; the catalog interprets what
goes inside.

## What's not here

- **Tool execution semantics** — those live in
  `molexp.agent.tools.dispatcher`. The dispatcher decodes
  `ModelToolCall` requests from the harness and produces normalized
  `ToolResult` objects; both shapes are agent-internal data
  contracts in `molexp.agent.tools.spec`.
- **The LLM provider matrix** — pydantic-ai handles provider
  routing inside `_pydanticai/harness.py`. The agent layer's public
  API stays provider-neutral (a model string is the only handle).
- **Workflow execution** — owned by `molexp.workflow`. PlanMode is a
  *consumer* of that layer, not an alternate scheduler.

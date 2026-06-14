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
orchestration — the former PlanMode / ReviewMode pipelines moved to the
harness layer.)

Construction is plain Python — no factory functions
(`create_agent(...)` / `build_agent(...)` / `Agent(provider=...)`).
`AgentRunner` lazily builds the underlying pydantic-ai router on first
`.run()`, so `import molexp.agent` is cheap.

## Layer Boundaries

```
harness ──uses──▶ agent ──uses──▶ workspace
                  (agent and workflow are SIBLINGS — no edge between them)
```

`molexp.agent` may import from `molexp.workspace.*` (the storage layer
below it). It must **not** import `molexp.workflow` or `molexp.harness`
— workflow is a sibling and harness sits above; harness reaches the
agent layer through `molexp.agent.router` (the SDK-free Protocol
module). It must also not import sibling application layers
(`molexp.plugins`, `molexp.server`, `molexp.cli`, `molexp.sweep`).

### pydantic-ai firewall

The only places under `src/molexp/agent/` allowed to `import pydantic_ai`
are files inside `agent/_pydanticai/`:

| File | Purpose |
|---|---|
| `_pydanticai/router.py` | `PydanticAIRouter` — concrete `Router` implementation; one `Agent` instance per `(tier, schema | None)` |
| `_pydanticai/capability_probe.py` | `PydanticAICapabilityProbe` — two-agent probe (needs drafter + MCP-attached evidence gatherer) for the capability discovery gate |
| `_pydanticai/mcp.py` | `build_mcp_server` helper used by ChatMode tool injection |

The firewall is enforced by `tests/test_agent/test_import_guard.py`.
`import molexp.agent` does not eagerly load `pydantic_ai` — every
construction site is hidden behind a lazy import that fires only on the
first `AgentRunner.run()` call.

### pydantic-graph confinement

`pydantic_graph` is exclusively imported under
`src/molexp/workflow/_pydantic_graph/`. **Nothing under
`src/molexp/agent/` may import it.** PlanMode drives multi-step
workflows through the public `molexp.workflow` API, which is the sole
sanctioned pg site.

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

molexp retains ownership of the **workflow / session / provenance**
layer (PlanMode pipeline, `SessionCatalog`, on-disk evidence + assets)
because pydantic-ai does not cover those.

## Capability Discovery Gate

PlanMode's 13-node pipeline inserts a two-node capability-discovery pair
between IR compilation and codegen:

```
... → CompileTaskIR
    → DraftCapabilityNeeds → DiscoverCapabilities
    → GenerateWorkflowSkeleton
    → GenerateTaskTests / GenerateTaskImplementations
    → ValidateWorkspace → HumanReview → FinalHandoffCheck
```

The contract:

> LLM decides what needs discovery. Agent workflow performs discovery
> through molcrafts-molmcp. LLM uses discovered evidence. Compiler
> rejects unevidenced API usage. Anything pydantic-ai provides natively
> in *model-side execution* (tool dispatch, MCP, retries, message
> history, structured output) MUST use pydantic-ai; do not build
> parallel implementations under `molexp.agent`. Molexp retains
> ownership of the workflow / session / provenance layer (PlanMode
> pipeline, `SessionCatalog`, on-disk evidence + assets) — pydantic-ai
> does not cover these.

### Mechanism

1. `DraftCapabilityNeeds` — a pydantic-ai `Agent[None,
   CapabilityNeedReport]` (no tools) ingests the plan brief +
   workflow contract + per-task briefs and decides which Molcrafts
   APIs the experiment needs. Persisted to `capability/needs.yaml`.
2. `DiscoverCapabilities` — a second pydantic-ai `Agent[None,
   CapabilityEvidenceBatch]` mounts the molmcp MCP server through
   `Agent(toolsets=[MCPServerStdio(...)])`. pydantic-ai drives the
   tool-call loop end-to-end (listing, dispatch, retries, output
   parsing). Persisted to `capability/evidence.yaml` +
   `capability/missing.md`.
3. **Codegen contract** — `GenerateTaskTests` and
   `GenerateTaskImplementations` consume the evidence batch and:
   * augment the user prompt with the evidence appendix;
   * require the LLM to populate `evidence_refs` on the schema *and*
     emit a module-level `__capability_evidence__: tuple[str, ...]`
     literal whose set equals `evidence_refs`;
   * after writing, run `validate_codegen_evidence(source, batch)` to
     diff `ast_refs` ∪ `declared_refs` against the evidence batch's
     `api_ref` set;
   * raise `UnevidencedApiReference` on any miss.
4. **Repair-loop integration** — `drive_with_repair` catches
   `CapabilityDiscoveryRequired` and `UnevidencedApiReference` from
   the workflow runtime execution. The first maps to a re-run of both discovery
   nodes; the second maps to `DiscoverCapabilities` only on the first
   occurrence and escalates to both nodes from the second. Both
   exceptions subclass `molexp.workflow.WorkflowError` so the
   workflow runtime propagates them to the loop.
5. **Static post-check** — `ValidateWorkspace.capability_evidence_check`
   re-runs the dual-signal diff over every generated file at handoff
   time so a hand-edited workspace cannot smuggle unevidenced refs
   past the gate.

### `NullCapabilityProbe` blocks codegen

When no molmcp MCP server is configured, `AgentRunner` injects a
`NullCapabilityProbe` whose `discover()` raises
`CapabilityDiscoveryRequired` whenever `discovery_required=True`. This
is the load-bearing safety invariant: codegen MUST stop when discovery
cannot proceed; silently passing an empty evidence batch downstream
would let the LLM hallucinate Molcrafts API calls.

Pure-stdlib paths (`discovery_required=False`) are exempt — the probe
returns `CapabilityEvidenceBatch(discovery_skipped=True)` and codegen
skips the `__capability_evidence__` block requirement.

## Sessions

Session metadata + persistence lives at `molexp.agent.sessions`:

* `SessionMetadata` — pydantic data type;
* `SessionStore` — on-disk vendor under
  `<workspace>/.subsystems/agent.sessions/<session_id>/`;
* `SessionCatalog` — the in-memory catalog the runner queries.

These are molexp-native (pydantic-ai does not provide a session layer)
and are the canonical way to attach LLM transcripts to a workspace run.

# Development

Contributor-facing docs for working on `molexp` internals.

## Internals

- [Compiler](compiler.md) — DSL → `WorkflowSpec` → pydantic-graph trampoline, identity, caching
- [Task Protocols](task-protocols.md) — `Runnable` / `Streamable` structural contracts

## Active Specs

Design notes for in-flight or recently landed work. Status is tracked inside each document.

- [molcfg + Profiles](specs/molcfg-profiles.md) — profile-based configuration replacing `--dry-run`
- [Agent Harness Architecture](specs/agent-harness-architecture.md) — model-agnostic agent core with provider and tool plugins
- [Unified pydantic-graph Dispatch](specs/unified-pydantic-graph-dispatch.md) — merging sweep and backend into the outer graph
- [Full-screen Monitor](specs/fullscreen-monitor.md) — terminal dashboard for local and remote runs

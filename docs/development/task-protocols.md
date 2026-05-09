# Task Protocols

MolExp uses two structural protocols (`typing.Protocol`, `@runtime_checkable`) to integrate user or third-party classes into a workflow **without requiring a molexp import**. Any object whose method signature matches the protocol qualifies — no inheritance, no registration, no configuration.

```python
from typing import Protocol, AsyncIterator, runtime_checkable


@runtime_checkable
class Runnable(Protocol):
    """Batch task: produces a single value per execution."""
    async def execute(self, ctx) -> "Any": ...


@runtime_checkable
class Streamable(Protocol):
    """Streaming actor: yields a series of values."""
    async def run(self, ctx) -> AsyncIterator["Any"]: ...
```

These live in `molexp.workflow.protocols`.

## Why Structural (Not Nominal)?

The `ctx` argument is deliberately typed as `Any` so third-party code need not import molexp to satisfy the protocol. At runtime, molexp passes a concrete `TaskContext` / `ActorContext`; a third-party object can simply treat `ctx` as having the methods it cares about (`ctx.inputs`, `ctx.config`, …).

This makes it easy to drop in existing library components (e.g. data-pipeline nodes from another molcrafts package) without writing adapters:

```python
class ExternalProcessor:
    async def execute(self, ctx) -> dict:
        return {"processed": ctx.inputs}

from molexp.workflow import WorkflowBuilder
spec = WorkflowBuilder(name="pipeline").add(ExternalProcessor()).build()
```

## Relationship to `Task` / `Actor`

`molexp.workflow.Task` and `molexp.workflow.Actor` are **convenience base classes** that implement these protocols with helpful generics (`StateT`, `DepsT`, `InputT`, `OutputT`). Using them is optional but recommended when you want:

- Static type-checking of `ctx.inputs` / `ctx.state` / `ctx.deps`.
- An explicit declaration that this class is "meant as a molexp task".

At runtime, the compiler treats a `Task` subclass and a third-party `Runnable` object identically.

## What the Protocol Does *Not* Require

- **No configuration class.** The old `config_type` / Pydantic-config pattern is gone — configuration flows through `ctx.config` (a molcfg `ProfileConfig`) and per-instance attributes you set in `__init__`.
- **No registration step.** There is no `register_task(...)` / registry — the `WorkflowBuilder` / `workflow()` DSL is the source of truth.
- **No explicit input/output schema.** Type hints on `execute()` / `run()` are advisory; the runtime does not enforce them.

## Persistence

Workflows are **authored in Python and re-imported on each execution** — there is no JSON IR or on-disk workflow schema. Identity is captured at two levels:

- `Workflow.workflow_id` — deterministic topology hash (`name + task dependencies`).
- `TaskSnapshot` — per-task AST-normalized code hash + config hash.

Use `WorkflowSnapshotRef(source="train.py", git_commit="...")` (stored on `Experiment`) plus `config_hash` on `RunMetadata` to trace which code and config produced a run. The full replay path is: re-import `source`, rebuild the `Workflow`, activate the same molcfg profile.

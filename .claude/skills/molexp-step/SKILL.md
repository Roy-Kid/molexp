---
name: molexp-step
description: Develop a new workflow Step (batch) or Actor (streaming) for molexp's pydantic-graph workflow system.
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
argument-hint: <step/actor description>
---

Develop workflow step: $ARGUMENTS

## Steps

1. **Choose type**:
   - `Step` (batch) — runs once, returns result: `async def execute(ctx) -> OutputT`
   - `Actor` (streaming) — runs continuously, yields: `async def run(ctx) -> AsyncIterator[OutputT]`

2. **Define types**: `OutputT` as Pydantic model (never raw dict). `InputT` from upstream. `StateT` for shared state. `DepsT` for injected deps.

3. **Implement** using context:
   - `ctx.state` — shared mutable workflow state
   - `ctx.deps` — injected (workspace, run, services)
   - `ctx.inputs` — typed upstream output
   - Actors: `await ctx.receive()` / `await ctx.emit(channel, msg)`

4. **Integrate** via functional or OOP DSL:
   ```python
   # Functional
   @wf.step(depends_on=["upstream"])
   async def my_step(ctx: StepContext[S, D, InputT]) -> OutputT: ...

   # OOP
   wf.add(MyStep(), depends_on=["upstream"])
   ```

5. **Test** in `tests/workflow/`: isolate step first, then test in workflow spec.

6. **Verify**: `pytest tests/workflow/`

## Rules

- Return annotation determines execution type — no explicit flags
- Same-level steps parallelize automatically
- `parallel_map()` / `join()` for explicit fan-out/fan-in
- Never import from `_pydantic_graph/` directly

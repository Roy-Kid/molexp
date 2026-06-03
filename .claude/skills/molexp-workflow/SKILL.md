---
name: molexp-workflow
description: Author runnable molexp.workflow code as a decomposed task DAG — not one monolithic Task. Use whenever the user asks to write, generate, or scaffold a molexp workflow / pipeline (e.g. "build X as a molexp workflow", "wire these steps with molexp"). Enforces plan-before-code so the graph never collapses into a single task.
argument-hint: <what the workflow should build, in plain language>
user-invocable: true
---

# molexp-workflow

Generate `molexp.workflow` code that is an actual **task DAG** — many small
tasks wired with `depends_on` — instead of one giant `Task.execute()` that does
everything inline.

## Why this skill exists (read first — it changes how you write)

When you translate a request like *"build a polymer electrolyte and write a
LAMMPS file"* directly into code, the natural output is **one procedure**: do A,
then B, then C, all in a single `execute()`. That is the failure mode. It is
valid, it runs, and it throws away everything molexp's task model exists for:

- **per-task caching** (`TaskSnapshot` + `Caching`) — a monolith re-runs the
  whole thing on any edit; a DAG re-runs only the changed task and its descendants.
- **topology parallelism** — same-level tasks run concurrently for free; a
  monolith is strictly serial.
- **per-task retry / resume** — a 6-step monolith that dies at step 5 redoes
  steps 1–4; separate tasks resume at the failure boundary.
- **artifact lineage** — each task publishes its own products via
  `ctx.run_context.artifact.save(...)`; a monolith collapses the provenance.

The graph carries all the value, but the value is **invisible at the moment you
write the code**, while the cost of wiring (naming tasks, matching strings) is
fully visible. So the default drift is toward the monolith. This skill forces
the decomposition that the API does not.

The gold-standard reference in this ecosystem is
`/Users/roykid/work/molcrafts/polymer_electrolyte/build_flow.py` — one external
command per task, ~30 nodes, full `depends_on` fan-out. Read it when in doubt.

## The procedure — do these in order. Do not write `execute()` bodies first.

### Step 1 — Decompose on paper, before any code

Emit a **task table** and show it to the user. No Python yet. Columns:

| task name | does (one verb / one command) | consumes | produces |
|-----------|------------------------------|----------|----------|

One row per task. If a row's "does" column contains the word "and" joining two
distinct operations, **split it into two rows**. This single rule prevents most
collapses.

### Step 2 — Apply the cut rubric to the table

Cut a new task at **every** boundary below. A task is the smallest unit that is
*one* of each:

1. **one external command / one capability call** — `antechamber`, `parmchk2`,
   `tleap`, `sander`, one `molpack.pack`, one reader/writer. Never two external
   tools in one task.
2. **one independently cacheable result** — if an output could be reused by a
   later run unchanged, it deserves its own task so the cache can hit.
3. **one failure boundary** — if a step can fail and you would not want to redo
   the steps before it, cut before it.
4. **one fan-out element** — per-species, per-temperature, per-salt-level work
   is one task *per element* (a Python loop in `build_workflow()` that calls
   `.add(...)` N times), not one task that loops internally.

If two adjacent rows share all four (same tool, always run together, fail
together, no fan-out), merging them is fine. Bias toward splitting.

### Step 3 — Write the code following the API contract

Only now write Python. Two styles; prefer **OOP `Task` subclasses** when task
bodies are non-trivial (the polymer case), **`@wf.task` decorators** for short
pure functions.

```python
from molexp.workflow import Task, TaskContext, WorkflowBuilder

# OOP style — one class per table row. Parametrize with __init__, not by
# branching inside execute().
class AntechamberTask(Task):
    """Run antechamber: PDB -> typed/charged structure. ONE command."""
    def __init__(self, label: str, out_format: str) -> None:
        self.label = label
        self.out_format = out_format

    async def execute(self, ctx: TaskContext) -> dict:
        pdb = ctx.inputs["pdb"]          # see input contract below
        ...                              # exactly one external command
        return {"out": str(out)}         # return a small dict of paths/handles

def build_workflow():
    b = WorkflowBuilder(name="my_flow")
    b.add(EmbedTask("CAT"), name="embed_CAT")
    b.add(AntechamberTask("CAT", "mol2"), name="antechamber_CAT",
          depends_on=["embed_CAT"])     # wire the edge by upstream NAME
    for n in [54, 108, 216]:            # fan-out = a loop over .add(), not a loop in execute()
        b.add(BuildBoxTask(n), name=f"box_{n}", depends_on=["antechamber_CAT"])
    return b.build()

# run:  spec = build_workflow(); result = await spec.execute(run_context=ctx)
```

**`ctx.inputs` contract — the part agents get wrong:**

- **one** upstream  → `ctx.inputs` *is* that upstream's returned object.
  `square` depends on `load` → `ctx.inputs` is the list `load` returned.
- **many** upstreams → `ctx.inputs` is a **dict keyed by upstream task name**.
  `merge` depends on `["parse", "validate"]` → `ctx.inputs["parse"]`,
  `ctx.inputs["validate"]`.

So the string in `depends_on=[...]` and the key in `ctx.inputs[...]` **must be
the same name**. This double-entry is why a naming convention is mandatory.

**Naming convention (kills the string-matching bugs):** name every task
`f"{stage}_{label}"` — `embed_CAT`, `antechamber_CAT`, `parmchk2_CAT`. Build the
`depends_on` list from the same f-strings. Never hand-type a dependency name
that you did not also use as a `name=`.

**Other context handles:** `ctx.config` (JSON params), `ctx.state` (shared
mutable), `ctx.deps` (injected objects), `ctx.run_context` (the molexp Run —
use `ctx.run_context.artifact.save(name, path)` to publish products).

**Dynamic config from upstream** — when a task's parameter depends on an
upstream output, use `dependent_params`, do not reach into another task:

```python
b.add(MechTask(), name="mech", depends_on=["cooling"],
      dependent_params=lambda prev: {"T": 0.7 * prev["cooling"].output["Tg"]})
```

**Control flow** beyond a pure DAG (only if the table needs it):
`.control(src, to)`, `.branch(src, routes={label: target})`,
`.loop(body=[...], until=..., max_iters=...)`,
`.parallel(map_over=..., body=..., join=..., max_concurrency=...)`. For most
data pipelines, `depends_on` alone is enough — same-level tasks auto-parallelize.

### Step 4 — Self-check before declaring done

Reject your own output if any holds:

- [ ] Any `execute()` runs **two** external commands, or contains a `for` loop
      that should have been fan-out `.add()` calls.
- [ ] The whole workflow is **one** task (unless the request is genuinely a
      single command).
- [ ] A `depends_on` name has no matching `name=` on any `.add(...)`.
- [ ] A task reads another task's files by reaching outside `ctx.inputs`
      (hard-coded path instead of an upstream-published handle).
- [ ] A product is written to disk but never `artifact.save(...)`-published.

## Anti-pattern — the collapse this skill prevents

```python
# ❌ WRONG — one task does the entire pipeline. No caching, no parallel,
#    no resume, no lineage. This is what you produce by default; don't.
class ParametrizePolymerTask(Task):
    async def execute(self, ctx):
        m = embed(...)              # step 1
        ac = antechamber(m)         # step 2  (external command)
        frcmod = parmchk2(ac)       # step 3  (external command)
        prepi = prepgen(ac)         # step 4  (external command)
        chain = tleap(frcmod, prepi)# step 5  (external command)
        return sander(chain)        # step 6  (external command)
# Six external commands, six cache/failure boundaries, fused into one node.
```

The fix is Steps 1–3: six rows in the table → six tasks → five `depends_on`
edges. See `build_flow.py` for the fully expanded version.

## Lookup behavior

If asked about the molexp.workflow API, `Read` the actual source under
`src/molexp/workflow/` (builder.py, task.py, context.py) and quote it — do not
invent signatures. The public surface is `WorkflowBuilder`, `Task`, `Actor`,
`TaskContext`; never import private submodules.

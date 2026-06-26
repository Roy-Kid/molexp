# Plan & Run with the Agent Harness

`molexp plan` turns a natural-language experiment draft into a fully
verified, reviewed plan plus an execution report — in nine steps:
proposal → concrete spec → resolved capabilities → workflow IR →
tasks + per-task tests → input set → compile/dry-run → review →
execution report. With `--execute` it goes further: it runs the workflow
for real and writes a final report — all on a single content-addressed run.

This is the production entry point to the harness `PlanMode` pipeline. For
the stage-by-stage internals, see
[Plan Mode Architecture](../architecture/plan-mode.md).

## Prerequisites

PlanMode drives an LLM, so install the agent extra and configure a model:

```bash
pip install "molexp[agent]"
molexp config set agent.model anthropic:claude-sonnet-4-5
```

The model is read from `~/.molexp/config.json` (`agent.model`); the same
loader backs the CLI and the server. You can override it per invocation
with `--model`.

## Plan only

```bash
# Inline draft
molexp plan "screen three solvent ratios for electrolyte X and report conductivity"

# Or read the draft from a file
molexp plan --file draft.md
```

PlanMode runs nine steps — from saving the draft, through a concrete
spec, resolved capabilities, workflow-IR extraction, task binding +
per-task tests, the parameter-space input set, and a compile/dry-run, to
a single review gate — and stops at an **execution report** describing
where and how the workflow *would* run. It **does not execute** the
experiment. The run is filed under the `plans` / `plan` project /
experiment by default (override with `--project` / `--experiment`).

## Plan, then run

```bash
molexp plan --file draft.md --execute
```

`--execute` appends the real-execution tail onto the **same run**: after
the review gate it executes the workflow for real, then writes the final
report and audit. (The compile/dry-run and per-task tests in step 7
already gate the plan — red tests block the review gate.) Tests and the
workflow both run in **executor subprocesses**, so a generation bug can
never crash the planning process.

## What it produces

Everything lands under the run directory:

```text
runs/<run_id>/
├── artifacts/        # experiment report, workflow IR, generated source, tests, reports
└── harness.sqlite    # event log + artifact lineage
```

Because the run is **content-addressed from the draft**, re-issuing the
same draft resumes where it left off — completed stages are skipped, not
repeated. Change the draft and you get a new run.

## Approval gates

The plan ends at a review `ApprovalGate` (step 8); the `--execute` tail
adds a second gate over the final report. In non-interactive use an
auto-grant approver lets the pipeline proceed; in an interactive or
server-driven flow a human approves the verified plan (and, with
`--execute`, the final report) before the pipeline continues. Approval is
recorded in the run's event log alongside the machine-validation result,
so "a human approved it" and "machine validation passed" stay distinct.

## Chatting instead of planning

`molexp plan` is the batch, pipeline-driven path. For an interactive
conversation that can read your workspace and call tools, use the agent
REPL:

```bash
molexp agent
```

That drives the `InteractiveLoop` and streams the same events the web UI
renders. See the [Agent concept](../concept/agent.md) for the loop model.

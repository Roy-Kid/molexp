---
title: MolExp
description: Agent-assisted scientific-workflow platform for FAIR research
hide:
  - navigation
  - toc
hero:
  title: MolExp
  description: Build reproducible scientific workflows in Python. Define tasks as plain functions, let the engine handle the graph, and keep every run tracked on disk — with an optional LLM agent that plans, generates, and drives experiments.
  actions:
    - label: Get started
      href: "#start-here"
      style: primary
    - label: Concepts
      href: "#core-concepts"
    - label: Guides
      href: "guide/"
  install:
    label: Install
    command: pip install molexp
  badges:
    - img: https://img.shields.io/pypi/v/molexp
      href: https://pypi.org/project/molexp/
      alt: PyPI version
    - img: https://img.shields.io/badge/python-3.12%2B-blue
      href: https://pypi.org/project/molexp/
      alt: Python 3.12+
---

<h1 class="molcrafts-sr-only">MolExp</h1>

<div class="molcrafts-manual-home" markdown>

<section id="start-here" class="molcrafts-manual-section molcrafts-manual-section--compact" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Start here</span>

## Find the right page

Use this page as an index into the manual, not a marketing overview.

</div>

<nav class="molcrafts-manual-index" aria-label="Manual entry points">
  <a href="getting-started/quick-start/">
    <span>01</span>
    <strong>Run the quick start</strong>
    <em>One script, two tasks, one tracked run. See the whole lifecycle in 30 seconds.</em>
  </a>
  <a href="getting-started/first-workflow/">
    <span>02</span>
    <strong>Write your first workflow</strong>
    <em>Understand tasks, dependencies, compilation, and sync/async mixing.</em>
  </a>
  <a href="getting-started/tracked-runs/">
    <span>03</span>
    <strong>Track a run on disk</strong>
    <em>Workspace → Project → Experiment → Run. Persistence, sweeps, resume, rerun.</em>
  </a>
  <a href="getting-started/cli-and-profiles/">
    <span>04</span>
    <strong>Use the CLI and profiles</strong>
    <em>Replace asyncio.run() with molexp run. Add molcfg.yaml for execution variants.</em>
  </a>
  <a href="getting-started/start-from-ui/">
    <span>05</span>
    <strong>Start from the browser</strong>
    <em>Create projects, experiments, and runs through the UI. No Python required.</em>
  </a>
</nav>

</section>

<section id="representative-workflows" class="molcrafts-manual-section molcrafts-manual-section--stack" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">At a glance</span>

## What it looks like

The same tracked run, built up one concept at a time. Each block below is
self-contained — define the graph, keep the record, sweep parameters, then hand it
to the CLI.

</div>

</section>

<section id="snippet-workflow" class="molcrafts-manual-section molcrafts-manual-section--compact" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Workflow</span>

## Define the computation

Tasks are plain functions. Declare `depends_on` and the engine derives the graph,
runs independent tasks in parallel, and caches by content.

</div>

```python
from molexp.workflow import WorkflowCompiler

wf = WorkflowCompiler(name="sum")

@wf.task
def fetch(scale: float = 1.0) -> dict:
    return {"values": [1.0, 4.0, 9.0], "scale": scale}

@wf.task(depends_on=["fetch"])
def summarize(values: list[float], scale: float) -> float:
    return sum(values) * scale
```

</section>

<section id="snippet-workspace" class="molcrafts-manual-section molcrafts-manual-section--compact" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Workspace</span>

## Keep the record

`Workspace → Project → Experiment → Run`. Every run is a durable directory holding its
parameters, status, and outputs — nothing lives only in memory.

</div>

```python
import molexp as me

ws = me.Workspace("./lab", name="lab")
exp = ws.project("demo").experiment("sum")
run = exp.add_run(params={"scale": 2.0})

result = run.execute(wf)
print(run.status, result.outputs["summarize"])  # succeeded 28.0
```

</section>

<section id="snippet-sweep" class="molcrafts-manual-section molcrafts-manual-section--compact" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Sweep</span>

## Scan parameters

A sweep materializes one content-addressed run per cell and executes them in parallel.
Collapse the results to plain records and pick the best.

</div>

```python
scan = (
    ws.project("demo")
    .experiment("lr-scan")
    .sweep(wf, {"scale": [1.0, 2.0, 4.0]})
)
summary = scan.execute()
best = summary.min_by("summarize")
```

</section>

<section id="snippet-cli" class="molcrafts-manual-section molcrafts-manual-section--stack" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">CLI &amp; profiles</span>

## Run from the terminal

Register the experiment once, then let `molexp run` own discovery, profiles, resume /
rerun, and scheduler-backed execution.

</div>

```python
# train.py — register the experiment once
ws.project("demo").experiment("sum").run(
    wf.compile(), params={"scale": [1.0, 2.0]}
)
```

```bash
molexp run train.py --profile smoke
molexp run train.py --resume          # continue a failed run
molexp run train.py --rerun --fresh   # re-execute from scratch
```

</section>

<section id="core-concepts" class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Concepts</span>

## How it works

Four layers, each with a single job. They compose without coupling.

</div>

<dl class="molcrafts-feature-matrix">
  <div>
    <dt>Workflow</dt>
    <dd>Define computation as a graph of tasks. Decorator or class-based authoring. Automatic parallelism from dependencies. Content-addressed caching.</dd>
  </div>
  <div>
    <dt>Workspace</dt>
    <dd>Persistent hierarchy: Workspace → Project → Experiment → Run. Every run is a durable record with parameters, status, and outputs.</dd>
  </div>
  <div>
    <dt>Agent</dt>
    <dd>LLM conversation layer. Plan experiments, generate workflow code, and drive runs through tool calls — all recorded in a session on disk.</dd>
  </div>
  <div>
    <dt>Harness</dt>
    <dd>Experiment orchestrator. Draft specs → resolve capabilities → generate workflow code → compile → test → review. Nine auditable steps.</dd>
  </div>
  <div>
    <dt>Assets</dt>
    <dd>Every artifact, log, and checkpoint is tracked with content-addressed identity and lineage back to the run and task that produced it.</dd>
  </div>
  <div>
    <dt>Profiles</dt>
    <dd>Execution variants in YAML. One script, many shapes — smoke, production, dry-run — switched with a CLI flag.</dd>
  </div>
</dl>

</section>

<section id="ecosystem" class="molcrafts-manual-section molcrafts-manual-section--stack" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Ecosystem</span>

## Extends outward, stays light

`import molexp` stays lightweight — heavy integrations load only when you reach for
them. The core features connect to the wider molcrafts stack through optional extras
(`molexp[agent]`, `molexp[tensorboard]`) and to your own code through two independent
plugin channels.

</div>

<div class="molcrafts-manual-grid molcrafts-manual-grid--cols-3">
  <a href="guide/molq/">
    <strong>molq · scheduler bridge</strong>
    <em>Powers <code>molexp run</code> on the cluster: the same run submits to Slurm, PBS, or LSF. Only the transport changes — workflow and record stay identical.</em>
  </a>
  <a href="getting-started/cli-and-profiles/">
    <strong>molcfg · run profiles</strong>
    <em>Backs the profile system. <code>molcfg.yaml</code> holds execution variants, switched with one <code>--profile</code> flag.</em>
  </a>
  <a href="architecture/plan-mode/">
    <strong>molmcp · capabilities</strong>
    <em>Grounds the agent harness. PlanMode discovers the full toolchain through molmcp, then binds the minimal subset an experiment needs.</em>
  </a>
  <a href="concept/plugins/">
    <strong>molvis · visualization</strong>
    <em>A built-in UI plugin renders molecular structures in the browser — one of the in-tree bundles alongside <code>core</code> and <code>metrics</code>.</em>
  </a>
  <a href="plugins/">
    <strong>CLI plugins</strong>
    <em>Ship <code>molexp &lt;yourcmd&gt;</code> subcommands from any pip package via the <code>molexp.cli_plugins</code> entry point.</em>
  </a>
  <a href="plugins/">
    <strong>UI plugins</strong>
    <em>Contribute a dynamically-imported React bundle to the SPA via the independent <code>molexp.ui_plugins</code> channel.</em>
  </a>
</div>

</section>

<section id="doc-map" class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Map</span>

## Documentation map

The sections below mirror the navigation tree so the homepage works as a
compact table of contents for returning users.

</div>

<div class="molcrafts-doc-map">
  <section>
    <h3><a href="getting-started/">Getting Started</a></h3>
    <p>Quick start, first workflow, tracked runs, CLI and profiles, and the browser-based UI walkthrough.</p>
  </section>
  <section>
    <h3><a href="concept/">Concepts</a></h3>
    <p>Workflow and workspace models, agent layer, asset reproducibility, and plugin architecture.</p>
  </section>
  <section>
    <h3><a href="guide/">Guides</a></h3>
    <p>Task authoring, control flow, sweeps, workspace API, persistence, profiles, plan mode, server lifecycle, and scheduler bridge.</p>
  </section>
  <section>
    <h3><a href="architecture/">Architecture</a></h3>
    <p>Layer boundaries, import rules, agent firewall, plan-mode pipeline, and workflow engine design.</p>
  </section>
</div>

</section>

<section id="three-pillars" class="molcrafts-manual-section molcrafts-manual-section--stack" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Design</span>

## Three pillars

MolExp is not a grab-bag of features. Every subsystem serves one of these goals.

</div>

<div class="molcrafts-manual-grid molcrafts-manual-grid--cols-3">
  <a href="concept/workflow/">
    <strong>Reproducible compute</strong>
    <em>Content-addressed caching, deterministic run ids, and per-task snapshots mean the same inputs always produce the same identity.</em>
  </a>
  <a href="concept/workspace/">
    <strong>Durable records</strong>
    <em>Every run is a directory with parameters, provenance, outputs, and execution history. Nothing lives only in memory.</em>
  </a>
  <a href="concept/agent/">
    <strong>Agent-assisted science</strong>
    <em>An LLM agent plans experiments, generates code, and drives runs. Every decision is auditable — the agent's session is a first-class workspace object.</em>
  </a>
</div>

</section>

<section class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Quick reference</span>

## Key APIs

Common entry points you will reach for most often.

</div>

<div class="molcrafts-manual-list">
  <a href="concept/workflow/">
    <strong>WorkflowCompiler</strong>
    <em>Define tasks with @wf.task. Compile to a frozen graph.</em>
  </a>
  <a href="getting-started/tracked-runs/">
    <strong>Workspace → Project → Experiment → Run</strong>
    <em>Create the persistent hierarchy. Execute, sweep, resume, rerun.</em>
  </a>
  <a href="guide/task-and-actor/">
    <strong>Task · Actor</strong>
    <em>Two task shapes: sync/async function (decorator) or reusable class.</em>
  </a>
  <a href="guide/control-flow/">
    <strong>Branch · Loop · Parallel</strong>
    <em>Control flow primitives that compose into any DAG shape.</em>
  </a>
  <a href="guide/sweeps/">
    <strong>Sweep · RunSet</strong>
    <em>Grid-search parameters, execute in parallel, summarize with to_records().</em>
  </a>
  <a href="guide/plan-mode/">
    <strong>PlanMode</strong>
    <em>Nine-step agent-driven experiment pipeline: draft → spec → code → test → review.</em>
  </a>
</div>

</section>

</div>

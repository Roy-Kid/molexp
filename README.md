<div align="center">

<h1>
  <img src=".github/assets/moko.svg" alt="" height="48" align="absmiddle">
  &nbsp;molexp
</h1>

<p><strong>An agent-assisted scientific-workflow platform for FAIR research</strong></p>

<p>
  <a href="https://img.shields.io/github/actions/workflow/status/MolCrafts/molexp/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI"><img src="https://img.shields.io/github/actions/workflow/status/MolCrafts/molexp/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI" alt="CI"></a>
  <a href="https://pypi.org/project/molexp/"><img src="https://img.shields.io/pypi/v/molexp?style=flat-square&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/molexp/"><img src="https://img.shields.io/pypi/pyversions/molexp?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <img src="https://img.shields.io/badge/license-BSD--3--Clause-18432B?style=flat-square" alt="License">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square" alt="Ruff"></a>
</p>

<p>
  <a href="https://molcrafts.github.io/molexp/"><b>Documentation</b></a> &nbsp;&middot;&nbsp;
  <a href="#quick-start"><b>Quick start</b></a> &nbsp;&middot;&nbsp;
  <a href="#molcrafts-ecosystem"><b>Ecosystem</b></a>
</p>

</div>

molexp turns a Python script of typed tasks into a tracked, reproducible experiment. It pairs a content-hashed workflow engine with a file-system-backed `Workspace → Project → Experiment → Run` hierarchy, profile-driven run variants, and optional cluster submission — then layers on an audited orchestration harness (Plan/Run stage pipelines, artifact lineage, approval gates), an optional LLM agent that can plan and drive those workflows, and a FastAPI server with a bundled React UI.

> **Under active development.** Public APIs may change between minor releases.

## Vision

Research computation is rarely a single program — it is the same idea run many times, with different parameters, on different machines, until something works. The artifacts of that effort usually scatter: a script in one place, its outputs in another, the parameters in a notebook, the "which run was the good one" in someone's memory. molexp exists to close that gap, so the definition of an experiment and the record of every execution are one connected object instead of folklore.

It aspires to make reproducibility the default rather than a discipline. You write ordinary typed Python; molexp captures the workflow's content hash, the resolved configuration, the artifacts, the errors, and the execution history, and writes them down atomically as the run happens. The same workflow runs locally for a smoke test and on a cluster for the real thing, without changing the science — only where the worker process launches.

What that unlocks is a research workflow you can trust and revisit: experiments that re-run exactly, runs that can be compared and resumed, and a workspace that stays browsable — from the CLI or a web UI — long after the original author has moved on.

## Capabilities

| Module | Capability |
|--------|------------|
| `molexp.workflow`   | Typed task-graph engine — `WorkflowCompiler` (decorator + OOP + protocol styles) compiles to a frozen, content-hashed `CompiledWorkflow`; `WorkflowRuntime` executes it with topology-driven parallelism, IR export, contract validation |
| `molexp.workspace`  | File-system storage primitive — `Workspace → Project → Experiment → Run` `Folder` hierarchy, content-addressed assets, atomic JSON I/O, run lifecycle |
| `molexp.config`     | In-code process-global config — a live `molcfg.Config` for runtime values such as LLM API keys, registered in code (never from env) |
| `molexp.profile`    | File-based per-run config — `molcfg.yaml` loading and named profiles; resolves `defaults` + `profiles` into an immutable, content-hashed `ProfileConfig` |
| `molexp.agent`      | Optional LLM layer — `AgentRunner` / `AgentLoop` (`ChatLoop` one round-trip, `InteractiveLoop` emergent tool loop) with persisted `AgentSession`s, built on PydanticAI (lazy-loaded) |
| `molexp.harness`    | Experiment orchestrator — audited `PlanMode` / `RunMode` stage pipelines (artifact lineage, approval gates, executors) over a content-addressed Run; the production `molexp plan [--execute]` entry point that lets an LLM agent plan, generate, and drive a workflow |
| `molexp.server`     | FastAPI app — REST routes for workspace, projects, experiments, runs, assets, execution, plus SSE streaming and bundled-SPA serving |
| `molexp.cli`        | `molexp` command-line entry point — workspace init/info, run/execute, project / experiment / run / asset / target / session subcommands |
| `molexp.plugins`    | On-demand capability registry — `submit_molq` scheduler bridge (SLURM / PBS / LSF) and `gh` GitHub client; core stays dependency-light |
| `molexp.git`        | Thin async wrappers over the `git` binary — `ensure_clone` / `fetch` / `push` and `GitWorktreeManager` for per-experiment working dirs |
| `ui/`               | React 19 + Rsbuild three-panel web client — navigation tree, entity viewers, inspector; compiled ahead of time and bundled into the wheel |

## Install

```bash
pip install molexp
```

Requires Python >= 3.12. Core depends on `pydantic`, `pydantic-graph`, `typer`, `rich`, `fastapi`, `uvicorn`, and the MolCrafts libraries `mollog`, `molcfg`, and `molq`. Optional extras: `molexp[agent]` adds the PydanticAI LLM harness; `molexp[all]` and `molexp[dev]` pull everything for development.

## Quick start

```python
import asyncio

from molexp.workflow import TaskContext, WorkflowCompiler, WorkflowRuntime

wf = WorkflowCompiler(name="demo")


@wf.task
async def fetch(ctx: TaskContext) -> list[float]:
    return [1.0, 4.0, 9.0]


@wf.task(depends_on=["fetch"])
async def reduce(ctx: TaskContext) -> float:
    return sum(ctx.inputs)


result = asyncio.run(WorkflowRuntime().execute(wf.compile()))
print(result.outputs)  # {'fetch': [1.0, 4.0, 9.0], 'reduce': 14.0}
```

Attaching a workflow to a tracked `Workspace` experiment (`ws.project(...).experiment(...).run(wf.compile(), params=...)`), running it with `molcfg` profiles via `molexp run`, and submitting to a cluster are covered in the docs.

## Documentation

- [Getting Started](https://molcrafts.github.io/molexp/getting-started/) — runnable first workflow, tracked runs, CLI and profiles
- [Concepts](https://molcrafts.github.io/molexp/concept/) — the workflow / workspace / plugin mental model
- [Guide](https://molcrafts.github.io/molexp/guide/) — task & actor authoring, runtime, assets, server, molq
- [Architecture](https://molcrafts.github.io/molexp/architecture/) — layer boundaries the code preserves
- [Development](https://molcrafts.github.io/molexp/development/) — compiler internals, task protocols, active specs

## MolCrafts ecosystem

| Project | Role |
|---------|------|
| [molpy](https://github.com/MolCrafts/molpy)     | Python toolkit — the shared molecular data model & workflow layer |
| [molrs](https://github.com/MolCrafts/molrs)     | Rust core — molecular data structures & compute kernels (native + WASM) |
| [molpack](https://github.com/MolCrafts/molpack) | Packmol-grade molecular packing (Rust + Python) |
| [molvis](https://github.com/MolCrafts/molvis)   | WebGL molecular visualization & editing |
| **molexp** | Agent-assisted scientific-workflow platform for FAIR research — this repo |
| [molnex](https://github.com/MolCrafts/molnex)   | Molecular machine-learning framework |
| [molq](https://github.com/MolCrafts/molq)       | Unified job queue — local / SLURM / PBS / LSF |
| [molcfg](https://github.com/MolCrafts/molcfg)   | Layered configuration library |
| [mollog](https://github.com/MolCrafts/mollog)   | Structured logging, stdlib-compatible |
| [molhub](https://github.com/MolCrafts/molhub)   | Molecular dataset hub |
| [molmcp](https://github.com/MolCrafts/molmcp)   | MCP server for the ecosystem |
| [molrec](https://github.com/MolCrafts/molrec)   | Atomistic record specification |

## Contributing

Contributions are welcome — see the [development docs](https://molcrafts.github.io/molexp/development/) to get started.

## License

BSD-3-Clause — see [LICENSE](LICENSE).

<hr>

<div align="center">
<sub>Crafted with 💚 by <a href="https://github.com/MolCrafts">MolCrafts</a></sub>
</div>

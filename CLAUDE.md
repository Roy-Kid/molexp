# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend (Python)

```bash
# Install in editable mode (no frontend build — Python-only dev loop)
pip install -e .

# Run all tests
pytest tests/

# Run specific test module
pytest tests/workspace/test_workspace.py

# Run single test
pytest tests/workspace/test_workspace.py::test_workspace_creation

# Start server (serves bundled UI if src/molexp/dist/ is populated, otherwise API-only)
molexp serve /path/to/workspace --port 8000

# Initialize a workspace
molexp init [path]

# Release: build frontend into src/molexp/dist/, then build the wheel
cd ui && npm run build && cd ..
python -m build --wheel
```

### Frontend (TypeScript/React)

```bash
cd ui

# Dev server (localhost:5173, requires running backend)
npm run dev

# Dev with mock API (no backend needed)
npm run dev:mock

# Production build
npm run build

# Run frontend tests
npm test

# Regenerate TypeScript API client from openapi.json
npm run generate:api
```

### After changing the FastAPI backend

1. Regenerate `openapi.json` (start server, fetch `/api/openapi.json`)
2. Regenerate the TypeScript client: `cd ui && npm run generate:api`
3. Update MSW mock handlers in `ui/mocks/handlers/` if new endpoints added

## Architecture

molexp is a workflow-and-agent platform for research experiment management, built on pydantic-graph and PydanticAI.

```
WorkflowSpec → pydantic-graph Compiler → Runtime → Workspace → FastAPI → React UI
                                                       ↑
                                            AgentService (PydanticAI)
```

### Five Core Layers

#### 1. Workflow Layer (`src/molexp/workflow/`)

Single OOP entry point — `Workflow` — supports both decorator and builder styles on the same class:

**Decorator style** (function-as-task):
```python
wf = Workflow(name="pipeline")

@wf.task
async def fetch(ctx: TaskContext[State, Deps, None]) -> FetchResult: ...

@wf.task(depends_on=["fetch"])
async def process(ctx: TaskContext[State, Deps, FetchResult]) -> ProcessResult: ...

spec = wf.build()
result = await spec.execute(run=run)
```

**OOP style** (add `Task` instances):
```python
wf = Workflow(name="pipeline").add(FetchTask()).add(ProcessTask(), depends_on=["fetch"]).build()
result = await wf.execute(run=run)
```

Both styles can be mixed on the same `Workflow` instance. Control-flow helpers are methods, not free functions: `@wf.parallel_map(...)` and `@wf.join(...)`.

Key abstractions:
- `Task` — Batch execution (`async def execute(ctx) -> OutputT`); implements the `Runnable` protocol
- `Actor` — Streaming execution (`async def run(ctx) -> AsyncIterator[OutputT]`); implements the `Streamable` protocol
- `TaskContext` / `ActorContext` — Typed context with `state`, `deps`, `inputs`, `config`, optional workspace `run_context`
- `WorkflowSpec` — Compiled spec with deterministic `workflow_id` (topology hash)
- `WorkflowRuntime` — Abstract runtime; `GraphWorkflowRuntime` (in `workflow/_pydantic_graph/`) is the default, created lazily by `WorkflowSpec`

Internal `_pydantic_graph/` compiles specs into pydantic-graph IR with topological levels for automatic parallelization. Never import from `_pydantic_graph/` directly.

Supporting modules: `cache.py` (LRU content-addressed), `snapshot.py` (AST-normalized code hashing), `protocols.py` (structural `Runnable` / `Streamable` protocols for zero-import third-party integration).

#### 2. Agent Layer (`src/molexp/plugins/agent_pydanticai/`)

Goal-driven autonomous execution built on PydanticAI, exposed as an **optional plugin** (loaded lazily via `molexp.plugins.registry`). Install with `pip install molexp[agent]`. Public API:

```python
from molexp.plugins.agent_pydanticai import AgentService, Goal

service = AgentService.from_workspace("./lab")
session = await service.run(Goal(description="...", constraints=[...]))
```

- `AgentService` — Entry point, creates sessions from workspace
- `AgentRuntime` — Abstract runtime (PydanticAI implementation in `_pydantic_ai/`)
- `Tool` / `@agent_tool` — Tool definitions with approval levels
- `ApprovalPolicy` — Glob-pattern-based tool approval control
- Session events: `PlanCreatedEvent`, `ToolCallEvent`, `ObservationEvent`, etc.

Internal `_pydantic_ai/` handles PydanticAI integration. Never import directly.

#### 3. Workspace Layer (`src/molexp/workspace/`)

File-system-backed hierarchical state: `Workspace → Project → Experiment → Run`

- Each level owns an `AssetLibrary` (content-addressed, deduplication)
- `RunContext` (context manager) owns execution lifecycle: claim ownership → append `ExecutionRecord` → execute → exit with `RunStatus`
- `ParamSpace` (`GridSpace`, `UniformSpace`) for parameter sweeps (expanded at script level; one combination = one `Experiment`)
- `ResumePolicy` protocol for resumable execution
- All metadata writes are atomic (temp-file + `os.rename`)
- Constructors are side-effect-free, but child factories (`ws.project(...)`, `project.experiment(...)`, `exp.run(...)`) materialize immediately and are idempotent (get-or-create by slug/ID)

**On-disk layout** — per-attempt artifacts live under `executions/<exec_id>/`; cross-attempt state stays at the run level:

```
workspace_root/
├── workspace.json
├── projects.json                          # container index
└── projects/<project_id>/
    ├── project.json
    ├── experiments.json                   # container index
    └── experiments/<experiment_id>/
        ├── experiment.json
        ├── runs.json                      # container index
        └── runs/run-<id>/
            ├── run.json                   # status, params, execution_history
            ├── assets.json                # run-scoped manifest (all attempts)
            ├── artifacts/                 # per-run final products
            ├── .ckpt/                     # per-run resume checkpoints
            ├── cache/                     # per-run user-domain cache
            └── executions/                # per-attempt artifacts
                ├── executions.json        # container index
                └── exec-<run_id>[-N]/
                    ├── execution.json     # per-attempt metadata
                    ├── workflow.json      # pydantic-graph state
                    ├── stdout.log         # scheduler stdout
                    ├── stderr.log         # scheduler stderr
                    ├── error.txt          # exception trace (on failure)
                    ├── logs/<name>.log    # named log streams (ctx.log("name"))
                    └── jobs/<uuid>/       # molq scheduler manifests
```

Container indices (`*.json`) are local conveniences rebuilt on materialize/delete; the global catalog at `.catalog/` remains authoritative for cross-cutting queries.

#### 4. Server Layer (`src/molexp/server/`)

FastAPI app, all routes under `/api`:

| Route module | Endpoints |
|---|---|
| `agent.py` | Session CRUD, SSE event streaming, approval |
| `project.py` | Project CRUD |
| `experiment.py` | Experiment CRUD |
| `run.py` | Run CRUD |
| `workspace.py` | Workspace open/list/folders |
| `execution.py` | Execution planning/status |
| `registry.py` | Available task types |
| `asset.py` | Asset management |

- `dependencies.py` — FastAPI DI: `get_workspace()`, `get_settings()`
- `manager.py` — `ServerManager` for lifecycle (start/stop/status/logs)
- Production: serves SPA from `static_dir`; Dev: API-only + CORS for localhost:5173

#### 5. UI Layer (`ui/src/`)

React 19 + Rsbuild, three-panel layout:

- **Left**: Navigation tree (projects, experiments, runs)
- **Center**: Content viewer (dispatched by entity type via `registry.ts`)
- **Right**: Inspector/metadata panel

Key patterns:
- `registry.ts` — Maps entity types to renderer components
- `state/useWorkspaceState.ts` — Zustand store for workspace state
- `state/useUrlState.ts` — URL-based routing state
- `state/api.ts` — Wraps auto-generated OpenAPI client
- `api/generated/` — **Never edit manually**; regenerate with `npm run generate:api`
- `mocks/handlers/` — MSW handlers for `dev:mock` mode; keep in sync with API changes
- `resolvers/` — Entity resolution for rendering dispatch

### Packaging & Frontend Serving

The React frontend is **compiled ahead of time by npm** and bundled inside the Python package — matching the `molvis` release workflow. `pip install` / `python -m build` **never** invokes npm.

```
ui/src/  →  (cd ui && npm run build)  →  src/molexp/dist/  →  (hatchling)  →  wheel
```

- **`ui/package.json`** `build` script runs rsbuild and copies `ui/dist/.` into `src/molexp/dist/` — one command does both halves
- **`pyproject.toml`** uses `hatchling` and declares `[tool.hatch.build] artifacts = ["src/molexp/dist/**"]` so the wheel ships the bundle
- **`src/molexp/dist/`** is gitignored (except `.gitkeep`); it is populated on demand by `cd ui && npm run build`
- **Runtime**: `create_app()` uses `importlib.resources.files("molexp") / "dist"` to locate the bundled assets. If empty, the server runs API-only.
- **Release**: `cd ui && npm run build && cd .. && python -m build --wheel`
- **Dev mode**: Run backend (`molexp serve --port 8000`) and frontend (`cd ui && npm run dev`) separately
- **Production** (`molexp serve`): serves API + bundled SPA from the installed package

### Key Patterns

- **Topology-driven parallelism**: Tasks are grouped into levels by the dependency graph; same-level tasks run in parallel automatically.
- **Sweep-level fan-out**: `molexp.sweep.run_sweep` owns the outer `(experiment × Run)` loop via a single-node pydantic-graph with a bounded `jobs` semaphore.
- **Content-addressed caching**: `TaskSnapshot` uses AST-normalized code hash + config hash; whitespace/comment changes don't invalidate cache.
- **Atomic persistence**: All JSON writes use temp-file + `os.rename` for crash safety.
- **Internal convention**: Prefixed `_pydantic_graph/` (in `workflow/`) and `_pydantic_ai/` (in `plugins/agent_pydanticai/`) are private implementation details; the public API is the parent package's `__init__.py`.

### Adding a New Workflow Task

1. Subclass `Task` (batch) or `Actor` (streaming) — or use any third-party object whose method signature matches the `Runnable` / `Streamable` protocol (no molexp import required).
2. Implement `execute()` / `run()` with a typed return annotation.
3. Add to the workflow via the `Workflow` decorators (`@wf.task` / `@wf.actor`) or `.add()` for Task/Actor instances.
4. The compiler auto-detects batch vs streaming from the return annotation / `Streamable` runtime check.

### Adding a New API Route

1. Add route handler in `src/molexp/server/routes/<module>.py`
2. Register router in `src/molexp/server/routes/__init__.py`
3. Add request/response Pydantic schemas to `src/molexp/server/schemas/`
4. Regenerate openapi.json and TS client: `cd ui && npm run generate:api`
5. Add MSW mock handler in `ui/mocks/handlers/` for `dev:mock`

### Adding a New UI Renderer

1. Create component in `ui/src/app/renderers/<Name>Viewer.tsx`
2. Register in `ui/src/app/renderers/registerRenderers.ts`
3. Add entity type mapping in `ui/src/app/registry.ts`
4. Add test fixture in `ui/src/__fixtures__/`

### Test Organization

Tests mirror source structure:
```
tests/
├── agent/          → src/molexp/agent/
├── server/         → src/molexp/server/
├── workflow/       → src/molexp/workflow/
└── workspace/      → src/molexp/workspace/
```

Each test directory has `conftest.py` for shared fixtures. Use `conftest.py` at directory level, not standalone fixture files.

## Skills (`.claude/skills/`)

| Skill | Trigger | Purpose |
|---|---|---|
| `/molexp-plan` | User | Natural language → structured technical spec |
| `/molexp-add` | User | Full feature implementation (plan → TDD → cross-layer wiring → verify) |
| `/molexp-add-api` | User | API endpoint (route + schema + client regen + MSW mock) |
| `/molexp-add-ui` | User | Frontend mechanics (renderer + state + mock + test) — invokes `molexp-designer` for post-impl polish |
| `/molexp-add-task` | User | Workflow Task/Actor development |
| `/molexp-add-tool` | User | PydanticAI agent tool development |
| `/molexp-design` | User | Frontend visual/UX polish: info density, design-system tokens, a11y, empty/error states |
| `/molexp-test` | User | TDD testing with coverage analysis |
| `/molexp-review` | Auto/User | Architecture + performance + UI design review (layer compliance, async safety, caching, I/O, concurrency, design system) |

> Naming follows the conventions in `molmcp/docs/concepts/naming.md`: `<domain>-<phase>[-<scope>]` with closed phase vocabulary.

### Agents (delegated, not user-invoked)

| Agent | Axis |
|---|---|
| `molexp-architect` | 5-layer compliance, module boundaries |
| `molexp-optimizer` | Async, I/O, serialization, React perf |
| `molexp-tester` | Test authoring (RED/GREEN/REFACTOR) |
| `molexp-documenter` | Google-style docstrings, JSDoc, OpenAPI descriptions |
| `molexp-designer` | UI visual quality, info density, tokens, a11y |

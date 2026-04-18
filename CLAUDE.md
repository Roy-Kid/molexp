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

# Start server (serves bundled UI if src/molexp/_webapp/ is populated, otherwise API-only)
molexp serve --workspace /path/to/workspace --port 8000

# Initialize a workspace
molexp init [path]

# Release: build frontend into src/molexp/_webapp/, then build the wheel
npm run build:ui
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

Two equivalent APIs for defining task graphs:

**Functional DSL** (decorator-based):
```python
wf = workflow(name="pipeline")

@wf.step
async def fetch(ctx: StepContext[State, Deps, None]) -> FetchResult: ...

@wf.step(depends_on=["fetch"])
async def process(ctx: StepContext[State, Deps, FetchResult]) -> ProcessResult: ...

spec = wf.build()
result = await spec.execute(run=run)
```

**OOP DSL** (builder-based):
```python
wf = WorkflowBuilder(name="pipeline").add(FetchStep()).add(ProcessStep(), depends_on=["fetch"]).build()
result = await wf.execute(run=run)
```

Key abstractions:
- `Step` — Batch execution (`async def execute(ctx) -> OutputT`)
- `Actor` — Streaming execution (`async def run(ctx) -> AsyncIterator[OutputT]`)
- `StepContext` / `ActorContext` — Typed context with state, deps, inputs
- `WorkflowSpec` — Compiled spec with deterministic `workflow_id` (topology hash)
- `WorkflowRuntime` — Abstract runtime; `GraphWorkflowRuntime` backed by pydantic-graph

Internal `_pydantic_graph/` compiles specs into pydantic-graph IR with topological levels for automatic parallelization. Never import from `_pydantic_graph/` directly.

Supporting modules: `cache.py` (LRU content-addressed), `persistence.py` (run store adapter), `snapshot.py` (AST-normalized code hashing).

#### 2. Agent Layer (`src/molexp/agent/`)

Goal-driven autonomous execution built on PydanticAI. Public API:

```python
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
- `RunContext` manages execution lifecycle (enter → execute → exit with status)
- `ParamSpace` (`GridSpace`, `UniformSpace`) for parameter sweeps
- `ResumePolicy` protocol for resumable execution
- All metadata writes are atomic (temp-file + `os.rename`)
- Constructors are side-effect-free; call `materialize()` to create dirs/files

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
ui/src/  →  (npm run build:ui)  →  src/molexp/_webapp/  →  (hatchling)  →  wheel
```

- **Root `package.json`** exposes `npm run build:ui`, which builds the `molexp-ui` workspace and copies `ui/dist/.` into `src/molexp/_webapp/`
- **`pyproject.toml`** uses `hatchling` and declares `[tool.hatch.build] artifacts = ["src/molexp/_webapp/**"]` so the wheel ships the bundle
- **`src/molexp/_webapp/`** is gitignored (except `.gitkeep`); it is populated on demand by `npm run build:ui`
- **Runtime**: `create_app()` uses `importlib.resources.files("molexp") / "_webapp"` to locate the bundled assets. If empty, the server runs API-only.
- **Release**: `npm run build:ui && python -m build --wheel`
- **Dev mode**: Run backend (`molexp serve --port 8000`) and frontend (`npm run dev` from repo root, or `cd ui && npm run dev`) separately
- **Production** (`molexp serve`): serves API + bundled SPA from the installed package

### Key Patterns

- **Topology-driven parallelism**: Steps grouped into levels by dependency graph; same-level steps run in parallel automatically
- **Content-addressed caching**: `TaskSnapshot` uses AST-normalized code hash + config hash; whitespace/comment changes don't invalidate cache
- **Atomic persistence**: All JSON writes use temp-file + `os.rename` for crash safety
- **Internal convention**: Prefixed `_pydantic_graph/` and `_pydantic_ai/` are private implementation details; public API is the parent package's `__init__.py`

### Adding a New Workflow Step

1. Subclass `Step` (batch) or `Actor` (streaming) in appropriate module
2. Implement `execute()` / `run()` with typed return annotation
3. Add to workflow via functional DSL (`@wf.step`) or OOP builder (`.add()`)
4. Compiler auto-detects execution type from return annotation

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
| `/molexp-impl` | User | Full feature implementation (plan → TDD → cross-layer wiring → verify) |
| `/molexp-spec` | User | Natural language → structured technical spec |
| `/molexp-api` | User | API endpoint (route + schema + client regen + MSW mock) |
| `/molexp-ui` | User | Frontend mechanics (renderer + state + mock + test) — invokes `molexp-designer` for post-impl polish |
| `/molexp-design` | User | Frontend visual/UX polish: info density, design-system tokens, a11y, empty/error states |
| `/molexp-step` | User | Workflow Step/Actor development |
| `/molexp-test` | User | TDD testing with coverage analysis |
| `/molexp-review` | Auto/User | Architecture + performance + UI design review (layer compliance, async safety, caching, I/O, concurrency, design system) |
| `/molexp-agent-tool` | User | PydanticAI agent tool development |

### Agents (delegated, not user-invoked)

| Agent | Axis |
|---|---|
| `molexp-architect` | 5-layer compliance, module boundaries |
| `molexp-optimizer` | Async, I/O, serialization, React perf |
| `molexp-tester` | Test authoring (RED/GREEN/REFACTOR) |
| `molexp-documenter` | Google-style docstrings, JSDoc, OpenAPI descriptions |
| `molexp-designer` | UI visual quality, info density, tokens, a11y |

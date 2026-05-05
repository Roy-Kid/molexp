# Project Map — molexp

A quick orientation aid for agents arriving at this repository. The
authoritative architecture description lives under `## Architecture`
in `CLAUDE.md`; this file is a thin index, not a duplicate.

## Top-level layout

```
molexp/
├── src/molexp/              # Python package (the wheel ships this)
│   ├── workflow/            # Layer 1 — pydantic-graph compiler & runtime
│   ├── plugins/             # Layer 2 — agent plugin (PydanticAI), MCP tools
│   ├── workspace/           # Layer 3 — Workspace → Project → Experiment → Run
│   ├── server/              # Layer 4 — FastAPI app, routes under /api
│   └── dist/                # bundled UI assets (gitignored, populated by `cd ui && npm run build`)
├── ui/                      # Layer 5 — React 19 + rsbuild frontend
│   ├── src/
│   ├── mocks/               # MSW handlers for `npm run dev:mock`
│   └── package.json         # `build` produces both ui/dist and src/molexp/dist
├── tests/                   # mirrors src/molexp/ structure, conftest.py per directory
├── docs/                    # public-facing docs (concept, development, getting-started, guide)
├── .agent/                  # passive internal context (this directory)
├── .claude/
│   └── specs/               # active in-flight specs (`/mol:spec` writes; `/mol:impl` ticks + deletes)
├── .github/workflows/       # ci.yml, publish.yml
├── pyproject.toml           # hatchling, declares src/molexp/dist/** as wheel artifacts
├── package.json             # TS root config (dev tooling shim)
└── CLAUDE.md                # mol_project frontmatter + thin router + full architecture prose
```

## Five layers (summary)

The detailed layer rules live in `CLAUDE.md ## Architecture`. The
short version:

1. **Workflow** (`src/molexp/workflow/`) — `Workflow` is the single OOP entry point; supports decorator and builder styles. Internal `_pydantic_graph/` is private.
2. **Agent** (`src/molexp/plugins/agent_pydanticai/`) — optional plugin; `pip install molexp[agent]`. Internal `_pydantic_ai/` is private.
3. **Workspace** (`src/molexp/workspace/`) — file-system-backed hierarchy; atomic writes; per-attempt artifacts under `executions/`.
4. **Server** (`src/molexp/server/`) — FastAPI; routes under `/api`; serves bundled SPA when `src/molexp/dist/` is populated.
5. **UI** (`ui/src/`) — React 19 + rsbuild, three-panel layout, OpenAPI-generated client (never edit `api/generated/`).

## Build pipeline

```
ui/src/  →  cd ui && npm run build  →  src/molexp/dist/  →  hatchling  →  wheel
```

`pip install` and `python -m build` never invoke npm. The frontend is
compiled ahead of time and bundled into the Python package.

## Test layout

`tests/` mirrors `src/molexp/`. Each test directory has its own
`conftest.py`. Frontend tests live alongside source under `ui/src/`.

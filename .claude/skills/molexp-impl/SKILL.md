---
name: molexp-impl
description: Full feature implementation for molexp — use when implementing a feature, fixing a bug, or making a significant code change that spans multiple layers.
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Bash, Grep, Glob, Agent
argument-hint: <feature description or spec file path>
---

Implement $ARGUMENTS in the molexp project.

**Execution discipline**: Before writing any code, enter **Plan Mode** to lay out the full plan, then create **Tasks** for each phase below. Update task status as work progresses (`in_progress` → `completed`). This enforces a structured, auditable workflow — the agent must not skip phases or jump ahead without completing prior tasks.

## Phase 1: Understand

Read the relevant source files to understand what exists. If `$ARGUMENTS` is a file path, read it as a spec. Determine which layers are affected:

- `src/molexp/workspace/` — data models, file-system state
- `src/molexp/workflow/` — task graph, steps, runtime
- `src/molexp/agent/` — PydanticAI agent tools/service
- `src/molexp/server/` — FastAPI routes, schemas
- `ui/src/` — React renderers, state, components

Create an implementation plan with discrete steps. Present it and wait for confirmation.

## Phase 2: Implement (TDD)

For each step:

1. Write test in `tests/<module>/` — must FAIL first
2. Implement minimal code to pass — files < 800 lines, functions < 50 lines
3. Refactor — no duplication, immutable patterns

## Phase 3: Cross-layer wiring

If multiple layers changed, wire them bottom-up:

1. Workspace/Workflow core logic
2. Server routes + schemas in `src/molexp/server/schemas/` and `routes/`
3. Register new route in `routes/__init__.py`
4. Regenerate TS client: `cd ui && npm run generate:api`
5. UI renderers in `ui/src/app/renderers/`, register in `registry.ts`
6. MSW mocks in `ui/mocks/handlers/`

## Phase 4: Verify

Run `pytest tests/<module>/` for each affected module. Run `cd ui && npx tsc --noEmit` if UI changed. Review the diff for layer violations, missing tests, mutation, hardcoded values.

## Rules

- Module = Feature. No standalone micro-utility files.
- All imports at file top. No inline/deferred imports.
- `_pydantic_graph/` and `_pydantic_ai/` are private — never import from outside parent package.
- `ui/src/api/generated/` is auto-generated — never edit manually.
- Atomic JSON writes: temp-file + `os.rename`.
- Immutable: return new objects, never mutate.

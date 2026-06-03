---
name: ui-nav-rearchitecture
description: In-progress UI navigation rearchitecture — entity model foundation landed, legacy nav to be replaced (no backward-compat)
metadata:
  type: project
---

The molexp UI (`ui/src/app/`) is being rearchitected to fix: pages not cross-linked, two competing taxonomies (`LeftPanelView` vs `SemanticObjectType`), two rendering paths, and renderer sprawl. User directive (2026-06-03): **no backward-compat, delete legacy code**.

**Foundation landed (`ui/src/app/entities/`):** the new single source of truth.
- `kinds.ts` — `EntityRef` + `ENTITY_META` (icon/label/accent per kind).
- `paths.ts` — `entityPath(ref, snapshot)` is the one URL builder.
- `relations.ts` — `resolveRelations(ref, snapshot)` declares the workspace graph edges (snapshot-only; async producer edges still TODO).
- `catalog.ts` + `GlobalCommandPalette.tsx` — ⌘K jump-to-anything across all kinds.
- `RelatedPanel.tsx` — wired into `panels/RightPanel.tsx`; gives every entity clickable cross-links.

**Done (2 phases, all green — tsc 0 / 243 tests / build ok / 0 new lint):**
1. Connectivity: `RelatedPanel` (in `panels/RightPanel.tsx`) + ⌘K `GlobalCommandPalette` (in `AppShell`); ContextBar search relabeled "Filter current list".
2. Breadcrumb lifted to the shell (`entities/breadcrumb.ts` + `entities/Breadcrumb.tsx`, rendered in `AppShell`'s center-top bar). Stripped breadcrumb from EntityPage/EntityHeader + 8 renderers + the molq plugin viewer. `useNavigationState` slimmed 580→294 lines (removed `buildBreadcrumbs`/`buildContextMeta`/`navigateUp`). Dead files deleted: `TopBar`, `ExecutionConsole`.

**Phase 3 (workflow + runs, all green — tsc 0 / 253 tests / build ok / biome 15, down 1):**
- Workflow: the layer was already DRY (shared `FlowgramCanvas` + `flowgram-document.ts`; 3 wrappers = 3 real sources: entity-graph editable / inline run preview / status-annotated file). Removed ~60 lines of **dead** workspace-file branch from `WorkflowGraphViewer` (it's only ever mounted by `WorkflowViewer` for `workflow` entities, so its file branch was unreachable). `WorkflowFileViewer` kept (distinct JSON format).
- Runs: the dashboard is a legitimate cross-run analytics view (KPI/Gantt/failing-experiments), NOT a `RunViewer` duplicate. Only real fix: centralized the run URL — new `runPath(projectId, experimentId, runId)` in `entities/paths.ts`, used by both `entityPath` and `RunsPage.navigateToRun`.
- Added `entities/__tests__/relations.test.ts` (10 tests: relations/paths/catalog).
- Verified the app boots: `MOLEXP_USE_MOCK=true npm run dev` → serves HTTP 200 on :3000, no compile errors. (Still no pixel-level check — no Playwright.)

**Original review OVERSTATED the workflow + runs "sprawl".** On close reading both are reasonably factored; the right move was targeted dead-code removal + entity-model consistency + tests, NOT a destructive rewrite. `WorkflowInspector` (per-node metadata) is also NOT redundant with `MetadataInspector`.

**Gotchas:**
- `npm run typecheck` piped through grep can FALSE-NEGATIVE because tsc colorizes; the reliable gate is `npx tsc --noEmit 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -c "error TS"`. (rsbuild build does NOT typecheck, so build-green ≠ types-green.)
- Removing breadcrumb props left unused `snapshot` params / imports; whole-`src` greps must include `src/plugins` (the molq plugin consumes `useNavigationState`).
- Repo has 16 pre-existing biome errors at HEAD (App.tsx conditional hooks, etc.) — not ours; don't chase them.

Toolchain: this is an HPC env where the node path MOVES between sessions (saw `~/.nvm/.../v26.2.0` then `~/.local/x86_64/nvm/versions/node/v26.3.0`). Resolve it fresh each session: `zsh -lic 'command -v node'` or `find ~/.local ~/.nvm -name node -type f`. Deps hoist to repo-root `node_modules` (not `ui/`). `npm run typecheck` / `npm test` / `npm run build` from `ui/`. Dev: `MOLEXP_USE_MOCK=true npm run dev` serves on :3000.

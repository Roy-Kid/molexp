---
name: molexp-ui
description: Develop a frontend component, renderer, or UI feature for molexp's React UI — focuses on mechanics (state wiring, registration, mocks). For visual polish or design audits use /molexp-design.
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Bash, Grep, Glob, Agent
argument-hint: <component or feature description>
---

Develop UI feature: $ARGUMENTS

## Steps

1. **Read context**: Check `ui/src/app/registry.ts` (renderer dispatch), existing renderers in `ui/src/app/renderers/`, state in `ui/src/app/state/`, generated types in `ui/src/api/generated/models/`.

2. **Implement component**:
   - Renderer → `ui/src/app/renderers/<Name>Viewer.tsx`
   - UI primitive → `ui/src/components/ui/` (Radix-based, no business logic)
   - Register renderer in `registerRenderers.ts` and `registry.ts`

3. **Wire state**:
   - Zustand stores: `useWorkspaceState`, `useUrlState`
   - API calls through `ui/src/app/state/api.ts` — never call generated services directly
   - Entity resolution: URL state → resolver → renderer

4. **Mock data** (for `dev:mock` mode):
   - Data in `ui/mocks/db/index.ts`
   - Handler in `ui/mocks/handlers/`
   - Register in `ui/mocks/handlers/index.ts`

5. **Test**: Add `.test.ts` file, fixtures in `ui/src/__fixtures__/`

6. **Verify**:
   ```bash
   cd ui && npx tsc --noEmit && npm test && npm run dev:mock
   ```

7. **Design pass**: invoke the `molexp-designer` agent on the changed renderer/files. Apply CRITICAL and HIGH fixes it reports (token violations, missing states, density regressions). For deeper polish, use `/molexp-design`.

## Stack

React 19, TypeScript strict, Rsbuild, Radix UI, `@xyflow/react` (graphs), Monaco (editor), Vitest.

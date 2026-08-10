# MolExp Workflow Preview (VS Code)

Preview a molexp `workflow.json` IR as an interactive DAG, reusing the same
workflow canvas the web UI ships.

## Layout

```
apps/
  web/src/components/workflow/   ← shadcn-style, app-decoupled module
  vsc-ext/                       ← this extension
```

The build aliases `@/` to `../web/src`, so the webview imports the *exact* UI
components that `molexp serve` uses.

## Scripts (repo root)

```bash
npm run build:vsc-ext   # production bundle
npm run dev:vsc-ext     # watch mode
npm run typecheck:vsc-ext
```

## Local

```bash
cd apps/vsc-ext
npm run build
# then F5 / "Run Extension" from the VS Code host
```

# MolExp Workflow Preview (VSCode extension)

Preview a molexp `workflow.json` IR as an interactive DAG, **reusing the molexp
UI workflow components** — the same flowgram canvas + shadcn chrome that
`molexp serve` renders. No running server required; the extension reads the file
directly.

## How it works

```
molexp/
  ui/src/components/workflow/   ← shadcn-style, app-decoupled module (Phase 1)
      workflow-preview.tsx      ← <WorkflowPreview source={...} />
      workflow-graph.tsx, flowgram-canvas.tsx, flowgram-document.ts, …
  vscode-ext/                   ← this extension
      src/extension.ts          ← CustomTextEditorProvider for *workflow.json
      webview/main.tsx          ← mounts <WorkflowPreview> in a webview
      build.mjs                 ← esbuild + @tailwindcss/postcss
```

The build aliases `@/` to `../ui/src`, so the webview imports the *exact* UI
components rather than a fork. Third-party deps (react, `@flowgram.ai`, radix)
resolve from the hoisted `../node_modules`.

## Develop

```bash
cd molexp/vscode-ext
npm install            # installs @types/vscode only; the rest resolve from ../node_modules
npm run build          # -> dist/extension.js, dist/webview.js, dist/webview.css, dist/theme.css
```

Then press **F5** in VSCode (Extension Development Host) and open any
`workflow.json` (e.g. `projects/<p>/experiments/<e>/workflow.json`). It opens in
the **MolExp Workflow Preview** custom editor. Use *Reopen Editor With…* to
switch between the raw JSON and the preview.

## Notes

- Read-only: the webview never writes back to the file.
- The theme follows the editor (light / dark) by mirroring VSCode's `vscode-dark`
  body class onto the UI's `.dark` token palette.
- Requires the sibling `molexp/ui` package and its installed `node_modules`.

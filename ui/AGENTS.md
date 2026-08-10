# AGENTS.md

You are an expert in JavaScript, Rsbuild, and web application development. You write maintainable, performant, and accessible code.

## Commands

**From repo root** (molvis-style `verb:package`):

| Script | What |
|--------|------|
| `npm run dev:ui` | UI with **MSW mock** + seeded showcase (default frontend work) |
| `npm run dev:api` | UI against a real API (`/api` proxy; start API separately or use `molexp serve --dev`) |
| `npm run build:ui` | Production build → `src/molexp/dist/` |
| `npm run preview:ui` | Preview the production build |
| `npm run typecheck:ui` / `test:ui` / `lint` / `lint:fix` / `format` | Checks |
| `npm run generate:api` | Regenerate OpenAPI client |

**From `ui/`** (leaf verbs):

`dev` (mock) · `dev:api` (real proxy) · `build` · `preview` · `typecheck` · `test` · `test:watch` · `lint` · `lint:fix` · `format` · `generate:api`

## Docs

- Rsbuild: https://rsbuild.rs/llms.txt
- Rspack: https://rspack.rs/llms.txt

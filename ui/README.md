# Molexp UI

Single-page application for Molexp that mirrors the Workspace and Workflow split. The UI is registry-driven and renders editors/viewers/inspectors based on semantic object type, file kind, and content type.

## Development

Install dependencies (repo root or `ui/`):

```bash
npm install
```

| Script | Backend | Notes |
|--------|---------|--------|
| `npm run dev` | Real API (`/api` proxy) | Needs `molexp serve` (or equivalent) on the API port |
| `npm run dev:page` | **MSW mock** | No Python server; opens the seeded Protein Folding showcase |

```bash
# Real backend (UI only; start the API separately)
npm run dev

# Mock showcase (recommended for UI work without a server)
npm run dev:page
```

Build / preview:

```bash
npm run build
npm run preview
```

## API Client Generation

Auto-generated client from the backend OpenAPI spec (`src/api/generated` — do not hand-edit).

1. Dump the spec (repo root, Python env active):
   ```bash
   python scripts/dump_openapi.py
   ```
2. Regenerate:
   ```bash
   npm run generate:api
   ```

## Mock layer

`dev:page` enables MSW. See [`mocks/README.md`](mocks/README.md) for architecture, seeding, and handler overrides.


## Registry System

New editors or viewers are added by implementing a renderer and registering it in `ui/src/app/renderers/registerRenderers.ts`. The layout is fixed; registrations only affect which panels render for a selection.

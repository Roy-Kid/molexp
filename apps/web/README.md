# Molexp Web (`apps/web`)

Single-page application for Molexp that mirrors the Workspace and Workflow split. The UI is registry-driven and renders editors/viewers/inspectors based on semantic object type, file kind, and content type.

## Development

Install once from the **repo root** (npm workspaces):

```bash
npm install
```

| Root script | Leaf (`cd apps/web`) | Backend | Notes |
|-------------|----------------------|---------|--------|
| `npm run dev:web` | `npm run dev` | **MSW mock** | Default frontend work; opens the seeded Protein Folding showcase |
| `npm run dev:api` | `npm run dev:api` | Real API (`/api` proxy) | Needs `molexp serve` (or equivalent) on the API port; also what `molexp serve --dev` starts |

```bash
# Mock showcase (no Python server)
npm run dev:web

# Real backend (start the API separately, or use molexp serve --dev)
npm run dev:api
```

Build / check / preview (repo root):

```bash
npm run build:web
npm run typecheck:web
npm run test:web
npm run lint
npm run preview:web
```

## API Client Generation

Auto-generated client from the backend OpenAPI spec (`src/api/generated` — do not hand-edit).

1. Dump the spec (repo root, Python env active):
   ```bash
   python scripts/dump_openapi.py
   ```
2. Regenerate (repo root):
   ```bash
   npm run generate:api
   ```

## Mock layer

`dev` / `dev:web` enables MSW. See [`mocks/README.md`](mocks/README.md) for architecture, seeding, and handler overrides.

# Molexp UI

Single-page application for Molexp that mirrors the Workspace and Workflow split. The UI is registry-driven and renders editors/viewers/inspectors based on semantic object type, file kind, and content type.

## Development

Install dependencies:

```bash
npm install
```

Run the dev server:

```bash
npm run dev
```

Build for production:

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```

## API Client Generation
This project uses an auto-generated API client based on the backend OpenAPI specification.

To regenerate the client:

1. Ensure the Python backend environment is active.
2. Generate the `openapi.json` spec:
   ```bash
   cd ../../molexp
   python3 -c "from src.molexp.server.app import app; import json; print(json.dumps(app.openapi()))" > openapi.json
   ```
3. Run the generator script:
   ```bash
   cd ui
   npm run generate:api
   ```
   *Note: This command uses `openapi-typescript-codegen` (Node.js) instead of the Java-based generator.*

The client source is generated into `src/api/generated`. Do not modify these files manually.

## Development with Mocks

The UI includes an MSW (Mock Service Worker) based API mocking layer for local development and testing without a real backend.

### Enabling Mocks

To enable API mocking in development:

1. Run the dev server with the mock flag:
   ```bash
   npm run dev:mock
   ```

2. All API requests will be intercepted by MSW

### Disabling Mocks

To connect to the real backend, run the dev server without `--mock`.

### Mock Layer Documentation

See [`mocks/README.md`](../mocks/README.md) for:
- Architecture overview
- How to seed custom data for tests
- How to override handlers for specific scenarios
- API coverage details


## Registry System

New editors or viewers are added by implementing a renderer and registering it in `ui/src/app/renderers/registerRenderers.ts`. The layout is fixed; registrations only affect which panels render for a selection.

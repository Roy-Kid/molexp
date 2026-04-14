# MSW Mock Layer

This directory contains the MSW (Mock Service Worker) API mocking layer for the molexp frontend.

## Architecture Overview

The mock layer is organized into three main components:

### 1. Database (`db/`)
- **In-memory storage** for workspace data (projects, experiments, runs, assets, files)
- **Session-scoped persistence**: Data survives across requests within a browser tab or test file
- **Reset utilities**: `resetDatabase()` for test isolation

### 2. Handlers (`handlers/`)
- **API-centric organization**: One file per domain (projects, experiments, runs, assets, workspace, execution, registry)
- **Data-driven**: Handlers read from and write to the database, not hardcoded responses
- **Stateful**: Supports read and write operations (create project, update file, set run status)

### 3. Utilities (`utils/`)
- **Seed functions**: `seedWorkspace(config)` to initialize custom data
- **File operations**: `writeFile()`, `readFile()`, `deleteFile()`
- **Entity creation**: `addProject()`, `addExperiment()`, `addRun()`
- **Run management**: `setRunStatus()`

## Directory Structure

```
ui/mocks/
├── db/
│   └── index.ts          # In-memory database with seed data
├── handlers/
│   ├── projects.ts       # Project CRUD operations
│   ├── experiments.ts    # Experiment management
│   ├── runs.ts           # Run lifecycle
│   ├── assets.ts         # Asset management
│   ├── workspace.ts      # File tree, read/write operations
│   ├── execution.ts      # Run execution (placeholder)
│   ├── registry.ts       # Task registry (placeholder)
│   └── index.ts          # Central export
├── utils/
│   └── index.ts          # Helper functions
├── browser.ts            # MSW browser worker setup
├── node.ts               # MSW Node server setup
└── README.md             # This file
```

## Development Usage

### Enabling Mocks in Development

1. Run the development server with the mock flag:
   ```bash
   npm run dev -- --mock
   ```

2. All API requests will be intercepted by MSW. Check the browser console for `[MSW]` prefixed logs.

### Disabling Mocks

Run the dev server without `--mock` to connect to the real backend.

## Test Usage

### Basic Test Setup

Tests can use MSW through a shared setup file wired into `Rstest`:

```typescript
import { beforeAll, afterEach, afterAll } from '@rstest/core';
import { server } from './mocks/node';
import { resetDatabase } from './mocks/db';

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => {
  server.resetHandlers();
  resetDatabase();
});
afterAll(() => server.close());
```

### Seeding Custom Data

```typescript
import { seedWorkspace } from '../../../mocks/utils';

describe('ProjectList', () => {
  it('renders projects from mock API', async () => {
    seedWorkspace({
      projects: [
        { id: 'test-1', name: 'Test Project 1' },
        { id: 'test-2', name: 'Test Project 2' },
      ],
    });

    render(<ProjectList />);
    
    expect(await screen.findByText('Test Project 1')).toBeInTheDocument();
  });
});
```

### Overriding Handlers for Specific Tests

```typescript
import { server } from '../../../mocks/node';
import { http, HttpResponse } from 'msw';

it('handles API errors', async () => {
  server.use(
    http.get('/api/projects', () => {
      return HttpResponse.json(
        { message: 'Internal server error' },
        { status: 500 }
      );
    })
  );

  render(<ProjectList />);
  
  expect(await screen.findByText('Error loading projects')).toBeInTheDocument();
});
```

## API Coverage

The mock layer covers all API endpoints used by the frontend:

### Projects
- `GET /api/projects` - List all projects
- `GET /api/projects/:id` - Get project by ID
- `POST /api/projects` - Create new project
- `DELETE /api/projects/:id` - Delete project

### Experiments
- `GET /api/projects/:projectId/experiments` - List experiments for a project

### Runs
- `GET /api/projects/:projectId/experiments/:experimentId/runs` - List runs for an experiment

### Assets
- `GET /api/assets` - List all assets

### Workspace Files
- `GET /api/workspace/info` - Get workspace metadata
- `GET /api/workspace/files?path=...` - List files in directory
- `GET /api/workspace/file?path=...` - Read file content (text)
- `GET /api/workspace/file/blob?path=...` - Read file content (binary)
- `POST /api/workspace/open` - Set workspace path
- `POST /api/workspace/directories` - Create directory
- `PUT /api/workspace/files` - Write file content

### Execution (Placeholder)
- `POST /api/execute` - Trigger run execution
- `GET /api/runs/:id/status` - Poll run status

### Plugins / Registry
- `GET /api/plugins` - List available UI plugins
- `GET /api/tasks` - Task registry placeholder
- `GET /api/tasks/:id` - Task detail placeholder

## Maintenance

### When the API Changes

1. **Update handlers**: Modify the corresponding handler file in `handlers/`
2. **Update database schema**: If new fields are added, update the seed data in `db/index.ts`
3. **Update types**: Ensure TypeScript types in `src/app/types.ts` match the API

### Adding New Endpoints

1. Create or update the appropriate handler file in `handlers/`
2. Add the handler to the exports in `handlers/index.ts`
3. If needed, add database accessors in `db/index.ts`

### Example: Adding a New Endpoint

```typescript
// handlers/projects.ts
export const projectHandlers = [
  // ... existing handlers

  // PATCH /api/projects/:id - Update project
  http.patch(`${API_BASE}/projects/:id`, async ({ params, request }) => {
    const { id } = params;
    const updates = await request.json();
    
    const project = getProject(id as string);
    if (!project) {
      return HttpResponse.json(
        { message: `Project ${id} not found` },
        { status: 404 }
      );
    }

    const updated = { ...project, ...updates };
    setProject(updated);

    return HttpResponse.json(updated);
  }),
];
```

## Troubleshooting

### Mocks not working in development
- Check that the dev server was started with `--mock`
- Look for `[MSW]` logs in browser console
- Verify the service worker is registered in DevTools → Application → Service Workers

### Tests failing with network errors
- Ensure `setupTests.ts` is properly configured
- Check that `server.listen()` is called in `beforeAll`
- Verify handlers are reset in `afterEach`

### Mock data not persisting
- Database is session-scoped (browser tab or test file)
- Use `seedWorkspace()` to initialize data for each test
- Call `resetDatabase()` in `afterEach` for test isolation

## Why Mocks are Outside `src/`

Mock files are placed in `ui/mocks/` (parallel to `ui/src/`) to:
- **Separate concerns**: Test infrastructure is distinct from production code
- **Avoid pollution**: Mocks don't appear in production bundles
- **Clear organization**: Easy to identify and maintain test-only code

The production app imports mocks dynamically only when the mock flag is enabled:
```typescript
// ui/src/index.tsx
if (__USE_MOCK__) {
  await import("../mocks/browser").then(m => m.start());
}
```

# Web UI Guide

## Overview

The molexp Web UI provides a VS Code Explorer-style interface for browsing and managing your workspace, projects, experiments, and runs.

## Quick Start

### 1. Start the Server

```bash
# From the molexp root directory
./start_server.sh
```

This will:
- Create sample workspace data (if needed)
- Start the API server on http://localhost:8000
- Start the UI dev server on http://localhost:5173

### 2. Access the UI

Open your browser to: **http://localhost:5173**

### 3. Navigate to Workspace

Click on **"Workspace"** in the left sidebar to access the explorer view.

## Features

### Workspace Explorer (Left Panel)

- **Tree View**: Hierarchical display of Projects → Experiments → Runs
- **Status Indicators**: Visual icons showing run status:
  - ✅ Green checkmark: Succeeded
  - ❌ Red X: Failed
  - 🔵 Blue spinner: Running
  - ⏰ Gray clock: Pending
- **Collapsible Nodes**: Click to expand/collapse project and experiment folders
- **Auto-Expand**: First two levels auto-expand for quick access

### Detail Panel (Right Panel)

Click any item in the explorer to view detailed information:

#### Project Details
- Name, description, owner
- Creation date
- Tags
- List of experiments

#### Experiment Details
- Name, description
- Workflow file path
- Git commit (if available)
- Parameter space definition
- List of recent runs

#### Run Details
- Run ID (timestamp-based)
- Status and timestamps
- Parameters used
- Workflow snapshot
- Input/Output assets
- Execution context (environment, dependencies, hardware)

#### Asset Details
- Asset ID and type
- File format and size
- Content hash (for deduplication)
- Producer run
- Metadata

## API Endpoints

The API server provides RESTful endpoints:

### Workspace
- `GET /api/workspace/info` - Get workspace summary
- `GET /api/workspace/tree` - Get complete tree structure

### Projects
- `GET /api/projects` - List all projects
- `GET /api/projects/{project_id}` - Get project details
- `POST /api/projects` - Create new project
- `DELETE /api/projects/{project_id}` - Delete project

### Experiments
- `GET /api/projects/{project_id}/experiments` - List experiments
- `GET /api/projects/{project_id}/experiments/{experiment_id}` - Get experiment details
- `POST /api/projects/{project_id}/experiments` - Create experiment
- `DELETE /api/projects/{project_id}/experiments/{experiment_id}` - Delete experiment

### Runs
- `GET /api/projects/{project_id}/experiments/{experiment_id}/runs` - List runs
- `GET /api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}` - Get run details
- `POST /api/projects/{project_id}/experiments/{experiment_id}/runs` - Create run
- `PATCH /api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/status` - Update status

### Assets
- `GET /api/assets` - List all assets
- `GET /api/assets/{asset_id}` - Get asset details

### Documentation
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)

## Development

### API Server Only

```bash
export MOLEXP_WORKSPACE=/path/to/workspace
python -m uvicorn molexp.api.server:app --reload
```

### UI Only

```bash
cd ui
npm run dev
```

### Create Sample Data

```bash
python create_sample_data.py
```

This creates a test workspace at `/tmp/molexp_ui_test` with:
- 2 projects
- 3 experiments
- 7 runs with various statuses

## Architecture

### Frontend (React + TypeScript)
- **WorkspaceExplorer.tsx**: Tree view component
- **DetailPanel.tsx**: Detail display component
- **Workspace.tsx**: Main page layout

### Backend (FastAPI + Python)
- **server.py**: API endpoints
- **workspace.py**: Workspace management
- **models.py**: Pydantic data models
- **repositories/**: Data persistence layer

## Customization

### Change Workspace Location

```bash
export MOLEXP_WORKSPACE=/your/workspace/path
```

### API Port

Edit `start_server.sh` or run manually:

```bash
python -m uvicorn molexp.api.server:app --port 8080
```

### UI Port

Edit `ui/rsbuild.config.ts`:

```typescript
export default {
  server: {
    port: 3000,
  },
};
```

## Troubleshooting

### API not connecting

1. Check API is running: `curl http://localhost:8000/health`
2. Check CORS settings in `server.py`
3. Verify workspace path is set correctly

### Empty workspace

1. Create sample data: `python create_sample_data.py`
2. Or create manually via CLI: `molexp project create ...`

### UI not loading

1. Install dependencies: `cd ui && npm install`
2. Check for TypeScript errors: `npm run build`
3. Clear cache: `rm -rf ui/node_modules ui/dist`

## Next Steps

- Create your own projects via CLI or API
- Run workflows and track execution
- Browse asset deduplication in action
- Explore run parameters and results

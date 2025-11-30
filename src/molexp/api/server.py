"""Extended FastAPI server with Project-Experiment-Run workspace support."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import workspace components
try:
    from molexp.workspace import Workspace
    from molexp.models import RunStatus, AssetType, Asset, AssetFile
    from molexp.id_utils import generate_asset_id, compute_content_hash
    WORKSPACE_AVAILABLE = True
except ImportError:
    WORKSPACE_AVAILABLE = False

# Import task graph components for JSON IR
try:
    from molexp.task_graph_compiler import TaskGraphCompiler
    from molexp.workflow_registry import get_workflow_registry
    TASK_GRAPH_AVAILABLE = True
except ImportError:
    TASK_GRAPH_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(title="MolExp API", version="0.2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workspace
def get_workspace() -> Workspace:
    """Get workspace instance."""
    if not WORKSPACE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Workspace module not available")
    
    workspace_path = os.environ.get("MOLEXP_WORKSPACE", str(Path.cwd()))
    return Workspace.from_path(workspace_path)


# ============================================================================
# Workspace Endpoints
# ============================================================================

@app.get("/api/workspace/info")
def get_workspace_info():
    """Get workspace information."""
    workspace = get_workspace()
    projects = workspace.list_projects()
    assets = workspace.list_assets()
    
    return {
        "root": str(workspace.root),
        "projectCount": len(projects),
        "assetCount": len(assets),
    }


@app.get("/api/dashboard/stats")
def get_dashboard_stats():
    """Get dashboard statistics for Overview page."""
    workspace = get_workspace()
    projects = workspace.list_projects()
    assets = workspace.list_assets()
    
    # Count total experiments and runs
    total_experiments = 0
    total_runs = 0
    
    for project in projects:
        experiments = workspace.list_experiments(project.project_id)
        total_experiments += len(experiments)
        
        for exp in experiments:
            runs = workspace.list_runs(project.project_id, exp.experiment_id)
            total_runs += len(runs)
    
    # Collect recent experiments
    recent_experiments = []
    for project in projects:
        experiments = workspace.list_experiments(project.project_id)
        for exp in experiments:
            recent_experiments.append({
                "id": exp.experiment_id,
                "name": exp.name,
                "status": "Active",  # Placeholder, could derive from runs
                "details": f"Project: {project.name}"
            })
    
    # Sort by some criteria if possible, or just take first 5
    recent_experiments = recent_experiments[:5]

    return {
        "totalExperiments": total_experiments,
        "activeWorkflows": total_runs, # Using runs count as proxy for active workflows
        "dataUsage": f"{len(assets) * 1.5:.1f} MB", # Mock calculation
        "computeHours": f"{total_runs * 0.5:.1f}h", # Mock calculation
        "recentExperiments": recent_experiments
    }



@app.get("/api/workspace/tree")
def get_workspace_tree():
    """Get complete workspace tree structure (VS Code Explorer style)."""
    workspace = get_workspace()
    projects = workspace.list_projects()
    
    tree_items = []
    
    for project in projects:
        experiments = workspace.list_experiments(project.project_id)
        
        experiment_items = []
        for exp in experiments:
            runs = workspace.list_runs(project.project_id, exp.experiment_id)
            
            run_items = []
            for run in runs:
                run_items.append({
                    "id": f"{project.project_id}/{exp.experiment_id}/{run.run_id}",
                    "name": run.run_id,
                    "type": "run",
                    "status": run.status.value,
                    "created": run.created_at.isoformat(),
                    "finished": run.finished_at.isoformat() if run.finished_at else None,
                    "parameters": run.parameters,
                })
            
            experiment_items.append({
                "id": f"{project.project_id}/{exp.experiment_id}",
                "name": exp.name,
                "type": "experiment",
                "experimentId": exp.experiment_id,
                "workflow": exp.workflow_template.source,
                "created": exp.created_at.isoformat(),
                "runCount": len(runs),
                "children": run_items,
            })
        
        tree_items.append({
            "id": project.project_id,
            "name": project.name,
            "type": "project",
            "projectId": project.project_id,
            "owner": project.owner,
            "tags": project.tags,
            "created": project.created_at.isoformat(),
            "experimentCount": len(experiments),
            "children": experiment_items,
        })
    
    return {
        "id": "workspace",
        "name": "Workspace",
        "type": "workspace",
        "children": tree_items,
    }


# ============================================================================
# Project Endpoints
# ============================================================================

class ProjectCreate(BaseModel):
    project_id: str
    name: str
    description: str = ""
    owner: str = ""
    tags: list[str] = []


@app.get("/api/projects")
def list_projects():
    """List all projects."""
    workspace = get_workspace()
    projects = workspace.list_projects()
    
    return [
        {
            "id": p.project_id,
            "projectId": p.project_id,
            "name": p.name,
            "description": p.description,
            "owner": p.owner,
            "tags": p.tags,
            "created": p.created_at.isoformat(),
            "config": p.config,
        }
        for p in projects
    ]


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    """Get project details."""
    workspace = get_workspace()
    project = workspace.get_project(project_id)
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    experiments = workspace.list_experiments(project_id)
    
    return {
        "id": project.project_id,
        "projectId": project.project_id,
        "name": project.name,
        "description": project.description,
        "owner": project.owner,
        "tags": project.tags,
        "created": project.created_at.isoformat(),
        "config": project.config,
        "experimentCount": len(experiments),
        "experiments": [
            {
                "id": e.experiment_id,
                "name": e.name,
                "created": e.created_at.isoformat(),
            }
            for e in experiments
        ],
    }


@app.post("/api/projects")
def create_project(project: ProjectCreate):
    """Create a new project."""
    workspace = get_workspace()
    
    try:
        new_project = workspace.create_project(
            project_id=project.project_id,
            name=project.name,
            description=project.description,
            owner=project.owner,
            tags=project.tags,
        )
        
        return {
            "id": new_project.project_id,
            "projectId": new_project.project_id,
            "name": new_project.name,
            "description": new_project.description,
            "owner": new_project.owner,
            "tags": new_project.tags,
            "created": new_project.created_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str):
    """Delete a project."""
    workspace = get_workspace()
    
    try:
        workspace.delete_project(project_id)
        return {"message": "Project deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Experiment Endpoints
# ============================================================================

class ExperimentCreate(BaseModel):
    experiment_id: str
    name: str
    workflow_source: str
    description: str = ""
    parameter_space: dict[str, Any] = {}


@app.get("/api/projects/{project_id}/experiments")
def list_experiments(project_id: str):
    """List experiments in a project."""
    workspace = get_workspace()
    experiments = workspace.list_experiments(project_id)
    
    return [
        {
            "id": e.experiment_id,
            "experimentId": e.experiment_id,
            "projectId": e.project_id,
            "name": e.name,
            "description": e.description,
            "workflow": e.workflow_template.source,
            "created": e.created_at.isoformat(),
            "parameterSpace": e.parameter_space,
        }
        for e in experiments
    ]


@app.get("/api/projects/{project_id}/experiments/{experiment_id}")
def get_experiment(project_id: str, experiment_id: str):
    """Get experiment details."""
    workspace = get_workspace()
    experiment = workspace.get_experiment(project_id, experiment_id)
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    runs = workspace.list_runs(project_id, experiment_id)
    
    return {
        "id": experiment.experiment_id,
        "experimentId": experiment.experiment_id,
        "projectId": experiment.project_id,
        "name": experiment.name,
        "description": experiment.description,
        "workflow": experiment.workflow_template.source,
        "workflowType": experiment.workflow_template.type,
        "gitCommit": experiment.workflow_template.git_commit,
        "created": experiment.created_at.isoformat(),
        "parameterSpace": experiment.parameter_space,
        "defaultInputs": [
            {
                "assetId": ref.asset_id,
                "role": ref.role,
            }
            for ref in experiment.default_inputs
        ],
        "runCount": len(runs),
        "runs": [
            {
                "id": r.run_id,
                "status": r.status.value,
                "created": r.created_at.isoformat(),
                "parameters": r.parameters,
            }
            for r in runs
        ],
    }


@app.post("/api/projects/{project_id}/experiments")
def create_experiment(project_id: str, experiment: ExperimentCreate):
    """Create a new experiment."""
    workspace = get_workspace()
    
    try:
        new_exp = workspace.create_experiment(
            project_id=project_id,
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            workflow_source=experiment.workflow_source,
            description=experiment.description,
            parameter_space=experiment.parameter_space,
        )
        
        return {
            "id": new_exp.experiment_id,
            "experimentId": new_exp.experiment_id,
            "projectId": new_exp.project_id,
            "name": new_exp.name,
            "description": new_exp.description,
            "workflow": new_exp.workflow_template.source,
            "created": new_exp.created_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/projects/{project_id}/experiments/{experiment_id}")
def delete_experiment(project_id: str, experiment_id: str):
    """Delete an experiment."""
    workspace = get_workspace()
    
    try:
        workspace.delete_experiment(project_id, experiment_id)
        return {"message": "Experiment deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Run Endpoints
# ============================================================================

class RunCreate(BaseModel):
    parameters: dict[str, Any]
    workflow_file: str
    git_commit: str | None = None


@app.get("/api/projects/{project_id}/experiments/{experiment_id}/runs")
def list_runs(project_id: str, experiment_id: str):
    """List runs in an experiment."""
    workspace = get_workspace()
    runs = workspace.list_runs(project_id, experiment_id)
    
    return [
        {
            "id": r.run_id,
            "runId": r.run_id,
            "projectId": r.project_id,
            "experimentId": r.experiment_id,
            "status": r.status.value,
            "created": r.created_at.isoformat(),
            "finished": r.finished_at.isoformat() if r.finished_at else None,
            "parameters": r.parameters,
            "workflow": r.workflow_snapshot.workflow_file,
        }
        for r in runs
    ]


@app.get("/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}")
def get_run(project_id: str, experiment_id: str, run_id: str):
    """Get run details."""
    workspace = get_workspace()
    run = workspace.get_run(project_id, experiment_id, run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Get asset references
    asset_refs = workspace.get_asset_refs(project_id, experiment_id, run_id)
    
    # Get context
    context = workspace.get_run_context(project_id, experiment_id, run_id)
    
    return {
        "id": run.run_id,
        "runId": run.run_id,
        "projectId": run.project_id,
        "experimentId": run.experiment_id,
        "status": run.status.value,
        "created": run.created_at.isoformat(),
        "finished": run.finished_at.isoformat() if run.finished_at else None,
        "parameters": run.parameters,
        "workflow": {
            "file": run.workflow_snapshot.workflow_file,
            "gitCommit": run.workflow_snapshot.git_commit,
            "serializedGraph": run.workflow_snapshot.serialized_graph,
        },
        "executorInfo": run.executor_info,
        "workingDir": run.working_dir,
        "logsDir": run.logs_dir,
        "assetRefs": {
            "inputs": [
                {
                    "assetId": ref.asset_id,
                    "role": ref.role,
                    "producerRunId": ref.producer_run_id,
                    "accessedAt": ref.accessed_at.isoformat() if ref.accessed_at else None,
                }
                for ref in (asset_refs.inputs if asset_refs else [])
            ],
            "outputs": [
                {
                    "assetId": ref.asset_id,
                    "role": ref.role,
                    "producerRunId": ref.producer_run_id,
                    "producedAt": ref.produced_at.isoformat() if ref.produced_at else None,
                }
                for ref in (asset_refs.outputs if asset_refs else [])
            ],
        },
        "context": {
            "environment": context.environment if context else {},
            "dependencies": context.dependencies if context else {},
            "hardware": context.hardware if context else {},
        } if context else None,
    }


@app.post("/api/projects/{project_id}/experiments/{experiment_id}/runs")
def create_run(project_id: str, experiment_id: str, run: RunCreate):
    """Create a new run."""
    workspace = get_workspace()
    
    try:
        new_run = workspace.create_run(
            project_id=project_id,
            experiment_id=experiment_id,
            parameters=run.parameters,
            workflow_file=run.workflow_file,
            git_commit=run.git_commit,
        )
        
        return {
            "id": new_run.run_id,
            "runId": new_run.run_id,
            "projectId": new_run.project_id,
            "experimentId": new_run.experiment_id,
            "status": new_run.status.value,
            "created": new_run.created_at.isoformat(),
            "parameters": new_run.parameters,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.patch("/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/status")
def update_run_status(project_id: str, experiment_id: str, run_id: str, status: dict[str, str]):
    """Update run status."""
    workspace = get_workspace()
    run = workspace.get_run(project_id, experiment_id, run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    try:
        run.status = RunStatus(status.get("status", run.status.value))
        if status.get("status") in ["succeeded", "failed", "cancelled"]:
            run.finished_at = datetime.now()
        
        workspace.update_run(run)
        
        return {
            "id": run.run_id,
            "status": run.status.value,
            "finished": run.finished_at.isoformat() if run.finished_at else None,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Asset Endpoints
# ============================================================================

@app.get("/api/assets")
def list_assets(limit: int = 100):
    """List all assets."""
    workspace = get_workspace()
    assets = workspace.list_assets()[:limit]
    
    return [
        {
            "id": a.asset_id,
            "assetId": a.asset_id,
            "type": a.type.value,
            "format": a.format,
            "size": a.size_bytes,
            "contentHash": a.content_hash,
            "created": a.created_at.isoformat(),
            "producerRunId": a.producer_run_id,
            "tags": a.tags,
            "metadata": a.metadata,
        }
        for a in assets
    ]


@app.get("/api/assets/{asset_id}")
def get_asset(asset_id: str):
    """Get asset details."""
    workspace = get_workspace()
    asset = workspace.get_asset(asset_id)
    
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    return {
        "id": asset.asset_id,
        "assetId": asset.asset_id,
        "type": asset.type.value,
        "format": asset.format,
        "size": asset.size_bytes,
        "contentHash": asset.content_hash,
        "mimeType": asset.mime_type,
        "created": asset.created_at.isoformat(),
        "producerRunId": asset.producer_run_id,
        "tags": asset.tags,
        "metadata": asset.metadata,
        "files": [
            {
                "path": f.path,
                "size": f.size,
                "hash": f.hash,
            }
            for f in asset.files
        ],
    }


@app.post("/api/assets/upload")
async def upload_asset(file: UploadFile = File(...)):
    """Upload a new asset."""
    workspace = get_workspace()
    
    # Create temp file to store upload
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    
    try:
        # Compute hash
        content_hash = compute_content_hash(tmp_path)
        
        # Check if already exists
        existing_id = workspace.assets.exists(content_hash)
        if existing_id:
            # Return existing asset
            asset = workspace.get_asset(existing_id)
            return {
                "id": asset.asset_id,
                "assetId": asset.asset_id,
                "type": asset.type.value,
                "format": asset.format,
                "size": asset.size_bytes,
                "contentHash": asset.content_hash,
                "created": asset.created_at.isoformat(),
                "producerRunId": asset.producer_run_id,
                "tags": asset.tags,
                "metadata": asset.metadata,
            }
            
        # Create new asset
        asset_id = generate_asset_id()
        asset = Asset(
            asset_id=asset_id,
            type=AssetType.OTHER,
            format=Path(file.filename).suffix.lstrip(".") or "dat",
            content_hash=content_hash,
            size_bytes=tmp_path.stat().st_size,
            created_at=datetime.now(),
            files=[
                AssetFile(
                    path=file.filename,
                    size=tmp_path.stat().st_size,
                    hash=content_hash
                )
            ],
            metadata={"original_filename": file.filename},
            tags=[],
            producer_run_id=None
        )
        
        # Store in workspace
        workspace.store_asset(asset, tmp_path)
        
        return {
            "id": asset.asset_id,
            "assetId": asset.asset_id,
            "type": asset.type.value,
            "format": asset.format,
            "size": asset.size_bytes,
            "contentHash": asset.content_hash,
            "created": asset.created_at.isoformat(),
            "producerRunId": asset.producer_run_id,
            "tags": asset.tags,
            "metadata": asset.metadata,
        }
        
    finally:
        # Cleanup temp file
        if tmp_path.exists():
            tmp_path.unlink()


@app.get("/api/assets/{asset_id}/download")
def download_asset(asset_id: str):
    """Download asset content."""
    workspace = get_workspace()
    asset = workspace.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    # Resolve file path
    # Note: This relies on FileSystemAssetRepo implementation details
    asset_dir = workspace.assets.root / asset_id / "data"
    if not asset_dir.exists():
        raise HTTPException(status_code=404, detail="Asset data not found")
    
    files = list(asset_dir.iterdir())
    if not files:
        raise HTTPException(status_code=404, detail="Asset data is empty")
    
    # For now, just serve the first file
    file_path = files[0]
    filename = asset.metadata.get("original_filename") or file_path.name
    
    return StreamingResponse(
        open(file_path, "rb"),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "workspace_available": WORKSPACE_AVAILABLE,
        "task_graph_available": TASK_GRAPH_AVAILABLE,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("molexp.api.server:app", host="0.0.0.0", port=8000, reload=True)

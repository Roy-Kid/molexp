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


from molexp.workspace import Workspace
from molexp.workspace.scanner import FolderScanner
from molexp.models import RunStatus, AssetType, Asset, AssetFile
from molexp.utils.id import generate_asset_id, compute_content_hash

from molexp.ir.loader import load_workflow_from_json
from molexp.ir.engine import WorkflowEngine

from molexp.workflow.plugin import get_node_registry, load_plugins

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


# ============================================================================
# Workspace Folder Storage (in-memory)
# ============================================================================

class WorkspaceFolderStore:
    """In-memory storage for workspace folders."""
    def __init__(self):
        self.folders: dict[str, dict[str, Any]] = {}
    
    def add(self, folder_id: str, path: str, name: str) -> None:
        self.folders[folder_id] = {
            "id": folder_id,
            "path": path,
            "name": name,
            "added_at": datetime.now().isoformat()
        }
    
    def remove(self, folder_id: str) -> bool:
        if folder_id in self.folders:
            del self.folders[folder_id]
            return True
        return False
    
    def get(self, folder_id: str) -> dict[str, Any] | None:
        return self.folders.get(folder_id)
    
    def list_all(self) -> list[dict[str, Any]]:
        return list(self.folders.values())

# Global workspace folder store
workspace_folders = WorkspaceFolderStore()


# Startup event: Load plugins
@app.on_event("startup")
async def startup_event():
    """Load node plugins at startup."""
    load_plugins()
    registry = get_node_registry()
    node_count = len(registry.list_all())
    print(f"✓ Loaded {node_count} node types from plugins")


# Initialize workspace
def get_workspace() -> Workspace:
    """Get workspace instance."""
    workspace_path = os.environ.get("MOLEXP_WORKSPACE", str(Path.cwd()))
    return Workspace.from_path(workspace_path)


# ============================================================================
# Node Plugin Endpoints
# ============================================================================

@app.get("/api/nodes")
def list_nodes():
    """List all available node types from plugins.
    
    Returns:
        Dictionary with all node definitions including metadata and config schemas
    """
    registry = get_node_registry()
    return registry.to_dict()


@app.get("/api/nodes/{node_id}")
def get_node(node_id: str):
    """Get details for a specific node type.
    
    Args:
        node_id: Node identifier (e.g., "io.write_file")
        
    Returns:
        Node definition with metadata and config schema
    """
    registry = get_node_registry()
    registration = registry.get(node_id)
    
    if not registration:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    
    return registration.to_dict()


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
                    "indexed": True,
                    "kind": "run",
                    "schema_version": run.schema_version,
                    "status": run.status.value,
                    "created": run.created_at.isoformat(),
                    "finished": run.finished_at.isoformat() if run.finished_at else None,
                    "parameters": run.parameters,
                })
            
            # Scan for workflow files
            exp_dir = workspace.root / "projects" / project.project_id / "experiments" / exp.experiment_id
            workflow_items = []
            if exp_dir.exists():
                for item in exp_dir.iterdir():
                    if item.is_file() and item.name.endswith(".flow"):
                        workflow_items.append({
                            "id": f"workspace:projects/{project.project_id}/experiments/{exp.experiment_id}/{item.name}",
                            "name": item.name,
                            "type": "file",
                            "path": f"projects/{project.project_id}/experiments/{exp.experiment_id}/{item.name}",
                            "size": item.stat().st_size,
                        })

            experiment_items.append({
                "id": f"{project.project_id}/{exp.experiment_id}",
                "name": exp.name,
                "type": "experiment",
                "indexed": True,
                "kind": "experiment",
                "schema_version": exp.schema_version,
                "experimentId": exp.experiment_id,
                "workflow": exp.workflow_template.source,
                "created": exp.created_at.isoformat(),
                "runCount": len(runs),
                "children": workflow_items + run_items,
            })
        
        tree_items.append({
            "id": project.project_id,
            "name": project.name,
            "type": "project",
            "indexed": True,
            "kind": "project",
            "schema_version": project.schema_version,
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
# Workspace Folder Endpoints
# ============================================================================

class WorkspaceFolderAdd(BaseModel):
    path: str
    name: str | None = None


@app.get("/api/workspace/folders")
def list_workspace_folders():
    """List all workspace folders."""
    return workspace_folders.list_all()


@app.post("/api/workspace/folders")
def add_workspace_folder(folder: WorkspaceFolderAdd):
    """Add a workspace folder."""
    folder_path = Path(folder.path).resolve()
    
    # Validate path exists and is a directory
    if not folder_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {folder.path}")
    
    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {folder.path}")
    
    # Check if already added
    for existing in workspace_folders.list_all():
        if Path(existing["path"]).resolve() == folder_path:
            raise HTTPException(status_code=400, detail="Folder already added to workspace")
    
    # Generate ID and add
    folder_id = str(uuid4())[:8]
    folder_name = folder.name or folder_path.name
    
    workspace_folders.add(folder_id, str(folder_path), folder_name)
    
    return {
        "id": folder_id,
        "path": str(folder_path),
        "name": folder_name,
        "added_at": datetime.now().isoformat()
    }


@app.delete("/api/workspace/folders/{folder_id}")
def remove_workspace_folder(folder_id: str):
    """Remove a workspace folder."""
    if not workspace_folders.remove(folder_id):
        raise HTTPException(status_code=404, detail="Folder not found")
    
    return {"message": "Folder removed"}


@app.get("/api/workspace/folders/{folder_id}/browse")
def browse_workspace_folder(folder_id: str, path: str = ""):
    """Browse contents of a workspace folder.
    
    Args:
        folder_id: Workspace folder ID
        path: Relative path within the folder (optional)
    """
    folder = workspace_folders.get(folder_id)
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    
    # Construct full path
    base_path = Path(folder["path"])
    if path:
        full_path = (base_path / path).resolve()
    else:
        full_path = base_path
    
    # Security check: ensure path is within folder
    try:
        full_path.relative_to(base_path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace folder")
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    
    if not full_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    # List directory contents
    entries = []
    try:
        for item in sorted(full_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            entry = {
                "name": item.name,
                "path": str(item.relative_to(base_path)),
                "type": "directory" if item.is_dir() else "file",
            }
            
            if item.is_file():
                try:
                    entry["size"] = item.stat().st_size
                except:
                    entry["size"] = 0
            
            entries.append(entry)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    return {
        "path": path,
        "entries": entries
    }


@app.get("/api/workspace/files/content")
def read_workspace_file(folder_id: str, path: str):
    """Read content of a file in a workspace folder.
    
    Args:
        folder_id: Workspace folder ID
        path: Relative path to the file
    """
    folder = workspace_folders.get(folder_id)
    if folder_id == "workspace":
        workspace = get_workspace()
        base_path = workspace.root
    elif not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    else:
        # Construct full path
        base_path = Path(folder["path"])
    full_path = (base_path / path).resolve()
    
    # Security check: ensure path is within folder
    try:
        full_path.relative_to(base_path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace folder")
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not full_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    
    # Read file content
    # Limit size for now (e.g., 1MB)
    MAX_SIZE = 1024 * 1024
    if full_path.stat().st_size > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large to preview")
        
    try:
        # Try to read as text
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Binary files not supported for preview")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FileContentUpdate(BaseModel):
    folder_id: str
    path: str
    content: str


@app.put("/api/workspace/files/content")
def write_workspace_file(update: FileContentUpdate):
    """Write content to a file in a workspace folder.
    
    Args:
        update: File update data
    """
    folder = workspace_folders.get(update.folder_id)
    if update.folder_id == "workspace":
        workspace = get_workspace()
        base_path = workspace.root
    elif not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    else:
        # Construct full path
        base_path = Path(folder["path"])
    full_path = (base_path / update.path).resolve()
    
    # Security check: ensure path is within folder
    try:
        full_path.relative_to(base_path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside workspace folder")
    
    # Ensure parent directory exists
    if not full_path.parent.exists():
        raise HTTPException(status_code=404, detail="Parent directory does not exist")
        
    # if not full_path.exists():
    #     raise HTTPException(status_code=404, detail="File not found")
    
    if full_path.exists() and not full_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
        
    try:
        # Write file content
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(update.content)
        return {"message": "File saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DirectoryCreate(BaseModel):
    folder_id: str
    path: str

@app.post("/api/workspace/files/directory")
def create_workspace_directory(data: DirectoryCreate):
    """Create a directory in a workspace folder."""
    folder = workspace_folders.get(data.folder_id)
    if data.folder_id == "workspace":
        workspace = get_workspace()
        base_path = workspace.root
    elif not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    else:
        base_path = Path(folder["path"])
    
    full_path = (base_path / data.path).resolve()
    
    try:
        full_path.relative_to(base_path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
        
    if full_path.exists():
        raise HTTPException(status_code=400, detail="Path already exists")
        
    try:
        full_path.mkdir(parents=True, exist_ok=True)
        return {"message": "Directory created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str):
    """Delete a project."""
    workspace = get_workspace()
    
    workspace.delete_project(project_id)
    return {"message": "Project deleted"}


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


@app.delete("/api/projects/{project_id}/experiments/{experiment_id}")
def delete_experiment(project_id: str, experiment_id: str):
    """Delete an experiment."""
    workspace = get_workspace()
    
    workspace.delete_experiment(project_id, experiment_id)
    return {"message": "Experiment deleted"}


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



@app.patch("/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/status")
def update_run_status(project_id: str, experiment_id: str, run_id: str, status: dict[str, str]):
    """Update run status."""
    workspace = get_workspace()
    run = workspace.get_run(project_id, experiment_id, run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run.status = RunStatus(status.get("status", run.status.value))
    if status.get("status") in ["succeeded", "failed", "cancelled"]:
        run.finished_at = datetime.now()
    
    workspace.update_run(run)
    
    return {
        "id": run.run_id,
        "status": run.status.value,
        "finished": run.finished_at.isoformat() if run.finished_at else None,
    }

@app.post("/api/executions")
def create_generic_execution(run_data: dict):
    """Create a new execution in the default playground project."""
    workspace = get_workspace()
    
    # Ensure default project and experiment exist
    project_id = "playground"
    experiment_id = "default"
    
    if not workspace.get_project(project_id):
        workspace.create_project(project_id, "Playground", "Default project for ad-hoc runs")
        
    if not workspace.get_experiment(project_id, experiment_id):
        workspace.create_experiment(project_id, experiment_id, "Default Experiment", "adhoc")
        
    # Create the run
    # Extract workflow snapshot if present
    workflow_snapshot = run_data.get("workflowSnapshot")
    workflow_file = "workflow.json" # Placeholder
    
    # If we have a snapshot, we might want to save it or use it directly
    # For now, we'll pass it through if the workspace supports it, 
    # or just create the run structure.
    
    new_run = workspace.create_run(
        project_id=project_id,
        experiment_id=experiment_id,
        parameters={},
        workflow_file=workflow_file
    )
    
    # If snapshot provided, update the run with it
    if workflow_snapshot:
        import json
        # We need to manually update the snapshot since create_run might not take it directly
        # depending on the internal API. Assuming we can update it:
        new_run.workflow_snapshot.serialized_graph = json.dumps(workflow_snapshot)
        workspace.update_run(new_run)

    return {
        "id": new_run.run_id,
        "runId": new_run.run_id,
        "projectId": new_run.project_id,
        "experimentId": new_run.experiment_id,
        "status": new_run.status.value,
        "created": new_run.created_at.isoformat(),
        "name": run_data.get("name", new_run.run_id)
    }

@app.post("/api/projects/{project_id}/experiments/{experiment_id}/runs/plan")
def preview_run_plan(run: RunCreate):
    """Preview execution plan for a workflow."""
    # Load workflow from file or string
    # For preview, we might receive the JSON string directly in run.workflow_file if it's a draft
    # But RunCreate expects a file path. 
    # Let's assume for now the UI sends a temporary file path or we extend RunCreate.
    # Actually, the UI usually sends the JSON content for preview.
    # Let's create a specific model for PlanRequest.
    pass

class PlanRequest(BaseModel):
    workflow_json: str
    targets: list[str] | None = None

@app.post("/api/plan")
def get_execution_plan(request: PlanRequest):
    """Get execution plan for a workflow definition."""
    from molexp.ir.loader import load_workflow_from_json
    from molexp.ir.compiler import plan_execution, ValidationError
    
    workflow_ir = load_workflow_from_json(request.workflow_json)
    
    # Override targets if provided
    plan = plan_execution(workflow_ir, targets=request.targets)
    
    return {
        "plan": plan,
        "nodeCount": len(plan)
    }
@app.post("/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/start")
def start_run(project_id: str, experiment_id: str, run_id: str):
    """Start run execution."""
    workspace = get_workspace()
    run = workspace.get_run(project_id, experiment_id, run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Check if run is already finished or running
    if run.status in [RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.RUNNING]:
        raise HTTPException(status_code=400, detail=f"Run is already {run.status.value}")

    # Update status to running
    run.status = RunStatus.RUNNING
    workspace.update_run(run)

    # TODO: This is a simplified synchronous execution
    # In production, this should be offloaded to a background worker
    
    # If we have a serialized graph, use it
    if run.workflow_snapshot.serialized_graph:
        import json
        from molexp.ir.loader import load_workflow_from_json
        from molexp.ir.engine import WorkflowEngine
        from molexp.ir.compiler import compile_workflow, plan_execution, ValidationError
        
        workflow_ir = load_workflow_from_json(run.workflow_snapshot.serialized_graph)
        
        # Validate/Compile
        compile_workflow(workflow_ir)
        # Calculate execution plan (topological sort of required subgraph)
        execution_plan = plan_execution(workflow_ir)
            
        engine = WorkflowEngine()
        # Execute only the planned nodes
        status_map = engine.execute(workflow_ir, run_id=run.run_id, node_ids=execution_plan)
        
        # Check if all PLANNED nodes succeeded
        all_succeeded = all(status_map.get(nid) == "SUCCEEDED" for nid in execution_plan)
        run.status = RunStatus.SUCCEEDED if all_succeeded else RunStatus.FAILED
    else:
        # SIMULATION FOR NOW to avoid complex dynamic loading issues
        import time
        time.sleep(1)
        run.status = RunStatus.SUCCEEDED
    
    run.finished_at = datetime.now()
    workspace.update_run(run)
    
    return {
        "id": run.run_id,
        "status": run.status.value,
        "finished": run.finished_at.isoformat() if run.finished_at else None,
    }



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
        "ir_available": IR_AVAILABLE,
    }


# ============================================================================
# Indexed Folder Endpoints
# ============================================================================

@app.get("/api/entities/{kind}/{entity_id}")
def get_entity_metadata(kind: str, entity_id: str):
    """Get metadata for any indexed entity.
    
    Args:
        kind: Entity kind (project, experiment, run, asset)
        entity_id: Entity identifier
        
    Returns:
        Entity metadata including kind, indexed status, and full entity data
    """
    workspace = get_workspace()
    scanner = FolderScanner(workspace.root)
    
    # Construct path based on kind
    if kind == "project":
        folder_path = workspace.root / "projects" / entity_id
    elif kind == "asset":
        folder_path = workspace.root / "assets" / entity_id
    elif kind == "experiment":
        # For experiment, we need project_id in the path
        # This is a simplified version - in production you'd need to search
        raise HTTPException(
            status_code=400, 
            detail="Experiment requires project_id. Use /api/projects/{project_id}/experiments/{experiment_id}"
        )
    elif kind == "run":
        # Similar issue for runs
        raise HTTPException(
            status_code=400,
            detail="Run requires project_id and experiment_id. Use the specific run endpoint"
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown entity kind: {kind}")
    
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Entity not found")
    
    entity_info = scanner.scan_folder(folder_path)
    
    if not entity_info:
        raise HTTPException(status_code=404, detail="Entity not found or not indexed")
    
    return {
        "kind": entity_info["kind"],
        "indexed": True,
        "path": entity_info["path"],
        "metadata": entity_info["entity"].model_dump(mode="json"),
    }


@app.get("/api/workspace/classify")
def classify_folder(path: str):
    """Classify a folder as an indexed entity or generic folder.
    
    Args:
        path: Relative path from workspace root
        
    Returns:
        Classification info including indexed status and kind
    """
    workspace = get_workspace()
    scanner = FolderScanner(workspace.root)
    
    folder_path = workspace.root / path
    
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    entity_info = scanner.scan_folder(folder_path)
    
    if entity_info:
        return {
            "indexed": True,
            "kind": entity_info["kind"],
            "path": entity_info["path"],
            "metadata": entity_info["entity"].model_dump(mode="json"),
        }
    else:
        return {
            "indexed": False,
            "kind": "folder",
            "path": str(folder_path.relative_to(workspace.root)),
        }


@app.get("/api/workspace/scan")
def scan_workspace():
    """Scan entire workspace and return all indexed entities.
    
    Returns:
        List of all indexed entities with their metadata
    """
    workspace = get_workspace()
    scanner = FolderScanner(workspace.root)
    
    entities = scanner.scan_workspace()
    
    return {
        "total": len(entities),
        "entities": [
            {
                "kind": e["kind"],
                "path": e["path"],
                "metadata": e["entity"].model_dump(mode="json"),
            }
            for e in entities
        ],
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("molexp.api.server:app", host="0.0.0.0", port=8000, reload=True)

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid

from .models import Project, Experiment, Run, RunStatus, AssetRef, AssetMeta

class FileSystemManager:
    def __init__(self, root_path: str):
        self.root = Path(root_path).resolve()
        self.projects_dir = self.root / "projects"
        self.assets_dir = self.root / "assets"
        self.assets_objects_dir = self.assets_dir / "objects"
        self.assets_index_dir = self.assets_dir / "index"
        self.sys_dir = self.root / ".sys"

    def init_repo(self):
        """Initialize the repository structure."""
        for d in [self.projects_dir, self.assets_objects_dir, self.assets_index_dir, self.sys_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Create a basic config file if not exists
        config_path = self.sys_dir / "config.json"
        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump({"created_at": str(datetime.now())}, f, indent=2)

    def _save_model(self, path: Path, model: Any):
        """Helper to save a Pydantic model to JSON."""
        with open(path, "w") as f:
            f.write(model.model_dump_json(indent=2))

    def create_project(self, name: str, slug: str, description: Optional[str] = None) -> Project:
        project_dir = self.projects_dir / slug
        if project_dir.exists():
            raise FileExistsError(f"Project '{slug}' already exists.")
        
        project_dir.mkdir(parents=True)
        (project_dir / "experiments").mkdir()

        project = Project(
            id=str(uuid.uuid4()),
            slug=slug,
            name=name,
            description=description,
            created_at=datetime.now()
        )
        
        self._save_model(project_dir / "project.json", project)
        return project

    def get_project(self, slug: str) -> Project:
        path = self.projects_dir / slug / "project.json"
        if not path.exists():
            raise FileNotFoundError(f"Project '{slug}' not found.")
        
        with open(path, "r") as f:
            data = json.load(f)
        return Project(**data)

    def create_experiment(self, project_slug: str, name: str, slug: str, 
                          workflow_template: Dict[str, Any] = {}, 
                          parameter_space: Dict[str, Any] = {}) -> Experiment:
        
        project = self.get_project(project_slug) # Verify project exists
        exp_dir = self.projects_dir / project_slug / "experiments" / slug
        if exp_dir.exists():
            raise FileExistsError(f"Experiment '{slug}' already exists in project '{project_slug}'.")
        
        exp_dir.mkdir(parents=True)
        (exp_dir / "runs").mkdir()

        experiment = Experiment(
            id=str(uuid.uuid4()),
            slug=slug,
            project_id=project.id,
            name=name,
            workflow_template=workflow_template,
            parameter_space=parameter_space
        )

        self._save_model(exp_dir / "experiment.json", experiment)
        return experiment

    def get_experiment(self, project_slug: str, slug: str) -> Experiment:
        path = self.projects_dir / project_slug / "experiments" / slug / "experiment.json"
        if not path.exists():
            raise FileNotFoundError(f"Experiment '{slug}' not found in project '{project_slug}'.")
        
        with open(path, "r") as f:
            data = json.load(f)
        return Experiment(**data)

    def create_run(self, project_slug: str, experiment_slug: str, 
                   parameters: Dict[str, Any], name: Optional[str] = None) -> Run:
        
        project = self.get_project(project_slug)
        experiment = self.get_experiment(project_slug, experiment_slug)

        # Generate Run ID: Timestamp + Short UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        run_id = f"{timestamp}_{short_uuid}"
        
        run_dir = self.projects_dir / project_slug / "experiments" / experiment_slug / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "artifacts").mkdir()

        run = Run(
            id=run_id,
            project_id=project.id,
            experiment_id=experiment.id,
            name=name or run_id,
            status=RunStatus.PENDING,
            created_at=datetime.now(),
            parameters=parameters,
            workflow_snapshot=experiment.workflow_template, # Snapshot the template
            working_dir=str(run_dir)
        )

        self._save_model(run_dir / "run.json", run)
        
        # Save context separately as requested in design
        context = {
            "resolved_params": parameters,
            "env": dict(os.environ), # Be careful with secrets, maybe filter later
            "cwd": os.getcwd()
        }
        with open(run_dir / "context.json", "w") as f:
            json.dump(context, f, indent=2, default=str)
            
        # Initialize empty asset refs
        with open(run_dir / "asset_refs.json", "w") as f:
            json.dump([], f, indent=2)

        return run

    def get_run(self, project_slug: str, experiment_slug: str, run_id: str) -> Run:
        path = self.projects_dir / project_slug / "experiments" / experiment_slug / "runs" / run_id / "run.json"
        if not path.exists():
            raise FileNotFoundError(f"Run '{run_id}' not found.")
        
        with open(path, "r") as f:
            data = json.load(f)
        return Run(**data)
        
    def list_projects(self) -> List[Project]:
        projects = []
        if not self.projects_dir.exists():
            return []
        for p in self.projects_dir.iterdir():
            if p.is_dir() and (p / "project.json").exists():
                projects.append(self.get_project(p.name))
        return projects

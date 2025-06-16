"""
Project management for organizing multiple experiments.
"""

from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime
from pathlib import Path
import yaml
import json

from pydantic import BaseModel, Field, ConfigDict
from .experiment import Experiment
from .param import Param, ParamSpace, ParamSampler
from .logging_config import get_logger

logger = get_logger("project")


class ProjectConfig(BaseModel):
    """Project-level configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(..., description="Project name")
    description: str = Field(default="", description="Project description")
    version: str = Field(default="1.0.0", description="Project version")
    author: str = Field(default="", description="Project author")
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list, description="Project tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Project(BaseModel):
    """Project class for managing multiple experiments."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    config: ProjectConfig = Field(..., description="Project configuration")
    experiments: Dict[str, Experiment] = Field(
        default_factory=dict, 
        description="Experiments in this project"
    )
    shared_resources: Dict[str, Any] = Field(
        default_factory=dict,
        description="Shared resources across experiments"
    )
    base_path: Optional[Path] = Field(
        default=None,
        description="Base directory path for the project"
    )
    
    def __init__(self, name: str, base_path: Optional[str | Path] = None, **kwargs):
        """Initialize project with name and optional base path."""
        if 'config' not in kwargs:
            kwargs['config'] = ProjectConfig(name=name)
        elif isinstance(kwargs['config'], dict):
            kwargs['config']['name'] = name
            kwargs['config'] = ProjectConfig(**kwargs['config'])
        
        # Set up base path
        if base_path is not None:
            kwargs['base_path'] = Path(base_path).resolve()
        else:
            # Default to current directory with project name
            kwargs['base_path'] = Path.cwd() / name
        
        super().__init__(**kwargs)
        
        # Create project directory structure
        self._create_project_structure()
        logger.info(f"Created project: {self.config.name} at {self.base_path}")
    
    def _create_project_structure(self) -> None:
        """Create the project directory structure."""
        if self.base_path:
            # Create main project directory
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.base_path / "experiments").mkdir(exist_ok=True)
            (self.base_path / "shared").mkdir(exist_ok=True)
            (self.base_path / "data").mkdir(exist_ok=True)
            (self.base_path / "results").mkdir(exist_ok=True)
            
            logger.info(f"Created project directory structure at {self.base_path}")
    
    def get_project_file_path(self) -> Path:
        """Get the path for the main project YAML file."""
        if self.base_path:
            return self.base_path / f"{self.config.name}.yaml"
        return Path(f"{self.config.name}.yaml")
    
    def get_experiments_dir(self) -> Path:
        """Get the experiments directory path."""
        if self.base_path:
            return self.base_path / "experiments"
        return Path("experiments")
    
    # Experiment Management
    def add_experiment(self, experiment: Experiment) -> None:
        """Add an experiment to the project."""
        if experiment.name in self.experiments:
            logger.warning(f"Experiment '{experiment.name}' already exists, replacing")
        
        # Set experiment's base path if not already set
        if hasattr(experiment, 'base_path') and experiment.base_path is None:
            exp_path = self.get_experiments_dir() / experiment.name
            experiment.base_path = exp_path
            experiment._create_experiment_structure()
        
        self.experiments[experiment.name] = experiment
        logger.info(f"Added experiment '{experiment.name}' to project '{self.config.name}'")
    
    def remove_experiment(self, name: str) -> None:
        """Remove an experiment from the project."""
        if name not in self.experiments:
            raise ValueError(f"Experiment '{name}' not found in project")
        
        del self.experiments[name]
        logger.info(f"Removed experiment '{name}' from project '{self.config.name}'")
    
    def get_experiment(self, name: str) -> Experiment:
        """Get an experiment by name."""
        if name not in self.experiments:
            raise ValueError(f"Experiment '{name}' not found in project")
        return self.experiments[name]
    
    def list_experiments(self) -> List[str]:
        """List all experiment names in the project."""
        return list(self.experiments.keys())
    
    # Batch Operations
    def create_parameter_study(self, 
                             base_experiment: str,
                             param_space: ParamSpace,
                             sampler: ParamSampler,
                             name_template: str = "{base_name}_param_{index}") -> List[str]:
        """Create multiple experiments for parameter study."""
        if base_experiment not in self.experiments:
            raise ValueError(f"Base experiment '{base_experiment}' not found")
        
        base_exp = self.experiments[base_experiment]
        created_experiments = []
        
        # Convert sampler generator to list to get count
        param_combinations = list(sampler.sample(param_space))
        logger.info(f"Creating parameter study based on '{base_experiment}' with {len(param_combinations)} combinations")
        
        for i, params in enumerate(param_combinations):
            exp_name = name_template.format(base_name=base_experiment, index=i, **params)
            
            # Clone base experiment
            new_exp_data = base_exp.model_dump()
            new_exp_data['name'] = exp_name
            new_exp = Experiment(**new_exp_data)
            
            # Store parameters for this experiment
            new_exp.metadata['parameters'] = params
            
            self.add_experiment(new_exp)
            created_experiments.append(exp_name)
        
        logger.info(f"Created {len(created_experiments)} experiments for parameter study")
        return created_experiments
    
    def batch_execute(self, experiment_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute multiple experiments in batch."""
        from .executor import ExperimentExecutor
        
        if experiment_names is None:
            experiment_names = self.list_experiments()
        
        results = {}
        logger.info(f"Starting batch execution of {len(experiment_names)} experiments")
        
        for exp_name in experiment_names:
            experiment = None
            try:
                logger.info(f"Executing experiment: {exp_name}")
                experiment = self.get_experiment(exp_name)
                executor = ExperimentExecutor(experiment)
                
                # Use parameters from metadata if available
                params = None
                if experiment.metadata and 'parameters' in experiment.metadata:
                    params = Param(experiment.metadata['parameters'])
                
                result = executor.run(param=params)
                results[exp_name] = {
                    'status': 'completed',
                    'result': result,
                    'experiment': experiment
                }
                logger.info(f"Completed experiment: {exp_name}")
                
            except Exception as e:
                logger.error(f"Failed to execute experiment '{exp_name}': {e}")
                results[exp_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'experiment': experiment
                }
        
        completed_count = len([r for r in results.values() if r['status'] == 'completed'])
        failed_count = len([r for r in results.values() if r['status'] == 'failed'])
        logger.info(f"Batch execution completed: {completed_count} succeeded, {failed_count} failed")
        return results
    
    # Resource Management
    def add_shared_resource(self, name: str, resource: Any) -> None:
        """Add a shared resource that can be used across experiments."""
        self.shared_resources[name] = resource
        logger.debug(f"Added shared resource: {name}")
    
    def get_shared_resource(self, name: str) -> Any:
        """Get a shared resource by name."""
        if name not in self.shared_resources:
            raise ValueError(f"Shared resource '{name}' not found")
        return self.shared_resources[name]
    
    # Analysis and Reporting
    def get_project_summary(self) -> Dict[str, Any]:
        """Get a summary of the project."""
        total_tasks = 0
        for exp in self.experiments.values():
            if exp.task_pool is not None:
                total_tasks += len(exp.task_pool.tasks)
        
        summary = {
            'project_name': self.config.name,
            'description': self.config.description,
            'version': self.config.version,
            'created_at': self.config.created_at,
            'experiment_count': len(self.experiments),
            'total_tasks': total_tasks,
            'experiments': {}
        }
        
        for name, exp in self.experiments.items():
            task_count = len(exp.task_pool.tasks) if exp.task_pool is not None else 0
            has_dependencies = False
            if exp.task_pool is not None:
                has_dependencies = any(task.deps for task in exp.task_pool.tasks.values())
            
            exp_metadata = exp.metadata
            
            summary['experiments'][name] = {
                'task_count': task_count,
                'has_dependencies': has_dependencies,
                'metadata': exp_metadata
            }
        
        return summary
    
    # Serialization
    def to_yaml(self, file_path: Optional[Path] = None) -> str:
        """Serialize project to YAML format."""
        # 优先级：file_path参数 > self.base_path > 当前目录
        if file_path is not None:
            out_path = Path(file_path)
        elif self.base_path:
            out_path = self.base_path / f"{self.config.name}.yaml"
        else:
            out_path = Path(f"{self.config.name}.yaml")
        
        # Custom serialization to handle complex objects
        experiments_data = {}
        for name, exp in self.experiments.items():
            exp_data = {
                'name': exp.name,
                'readme': exp.readme,
                'metadata': exp.metadata,
                'task_pool': {
                    'name': exp.task_pool.name if exp.task_pool else None,
                    'tasks': {
                        task_name: task.model_dump(exclude={'base_path'})
                        for task_name, task in exp.task_pool.tasks.items()
                    } if exp.task_pool else {}
                } if exp.task_pool else None
            }
            experiments_data[name] = exp_data
        
        data = {
            'config': self.config.model_dump(),
            'experiments': experiments_data,
            'shared_resources': self.shared_resources
        }
        yaml_str = yaml.dump(data, default_flow_style=False, indent=2)
        
        if out_path:
            out_path.write_text(yaml_str)
            logger.info(f"Saved project to: {out_path}")
        
        return yaml_str
    
    @classmethod
    def from_yaml(cls, yaml_input: str | Path) -> "Project":
        """Load project from YAML format."""
        if isinstance(yaml_input, Path):
            if not yaml_input.exists():
                raise FileNotFoundError(f"File not found: {yaml_input}")
            yaml_str = yaml_input.read_text()
            logger.info(f"Loading project from: {yaml_input}")
        else:
            yaml_str = yaml_input
        
        data = yaml.safe_load(yaml_str)
        
        # Reconstruct config
        config_data = data.get('config', {})
        config = ProjectConfig(**config_data)
        
        # Reconstruct experiments
        experiments = {}
        for exp_name, exp_data in data.get('experiments', {}).items():
            # Create experiment
            experiment = Experiment(
                name=exp_data['name'],
                readme=exp_data.get('readme', ''),
                metadata=exp_data.get('metadata', {})
            )
            
            # Reconstruct task pool if present
            task_pool_data = exp_data.get('task_pool')
            if task_pool_data:
                from .pool import TaskPool
                from .task import Task, ShellTask, HamiltonTask, LocalTask, RemoteTask
                
                task_pool = TaskPool(name=task_pool_data['name'])
                
                # Reconstruct tasks
                for task_name, task_data in task_pool_data.get('tasks', {}).items():
                    # Determine task type and create appropriate task
                    task_type = task_data.get('__class__', 'Task')
                    if 'commands' in task_data:
                        task = ShellTask(**task_data)
                    elif 'modules' in task_data:
                        task = HamiltonTask(**task_data)
                    elif task_type == 'LocalTask':
                        task = LocalTask(**task_data)
                    elif task_type == 'RemoteTask':
                        task = RemoteTask(**task_data)
                    else:
                        task = Task(**task_data)
                    
                    task_pool.add_task(task)
                
                experiment.set_task_pool(task_pool)
            
            experiments[exp_name] = experiment
        
        # Create project
        project = cls(name=config.name, config=config)
        project.experiments = experiments
        project.shared_resources = data.get('shared_resources', {})
        
        return project
    
    def to_json(self, file_path: Optional[Path] = None) -> str:
        """Serialize project to JSON format."""
        # Custom serialization to handle complex objects
        experiments_data = {}
        for name, exp in self.experiments.items():
            exp_data = {
                'name': exp.name,
                'readme': exp.readme,
                'metadata': exp.metadata,
                'task_pool': {
                    'name': exp.task_pool.name if exp.task_pool else None,
                    'tasks': {
                        task_name: task.model_dump()
                        for task_name, task in exp.task_pool.tasks.items()
                    } if exp.task_pool else {}
                } if exp.task_pool else None
            }
            experiments_data[name] = exp_data
        
        data = {
            'config': self.config.model_dump(),
            'experiments': experiments_data,
            'shared_resources': self.shared_resources
        }
        json_str = json.dumps(data, indent=2, default=str)
        
        if file_path:
            file_path.write_text(json_str)
            logger.info(f"Saved project to: {file_path}")
        
        return json_str
    
    @classmethod
    def from_json(cls, json_input: str | Path) -> "Project":
        """Load project from JSON format."""
        if isinstance(json_input, Path):
            if not json_input.exists():
                raise FileNotFoundError(f"File not found: {json_input}")
            json_str = json_input.read_text()
            logger.info(f"Loading project from: {json_input}")
        else:
            json_str = json_input
        
        data = json.loads(json_str)
        
        # Reconstruct config
        config_data = data.get('config', {})
        config = ProjectConfig(**config_data)
        
        # Reconstruct experiments
        experiments = {}
        for exp_name, exp_data in data.get('experiments', {}).items():
            # Create experiment
            experiment = Experiment(
                name=exp_data['name'],
                readme=exp_data.get('readme', ''),
                metadata=exp_data.get('metadata', {})
            )
            
            # Reconstruct task pool if present
            task_pool_data = exp_data.get('task_pool')
            if task_pool_data:
                from .pool import TaskPool
                from .task import Task, ShellTask, HamiltonTask, LocalTask, RemoteTask
                
                task_pool = TaskPool(name=task_pool_data['name'])
                
                # Reconstruct tasks
                for task_name, task_data in task_pool_data.get('tasks', {}).items():
                    # Determine task type and create appropriate task
                    if 'commands' in task_data:
                        task = ShellTask(**task_data)
                    elif 'modules' in task_data:
                        task = HamiltonTask(**task_data)
                    elif task_data.get('__class__') == 'LocalTask':
                        task = LocalTask(**task_data)
                    elif task_data.get('__class__') == 'RemoteTask':
                        task = RemoteTask(**task_data)
                    else:
                        task = Task(**task_data)
                    
                    task_pool.add_task(task)
                
                experiment.set_task_pool(task_pool)
            
            experiments[exp_name] = experiment
        
        # Create project
        project = cls(name=config.name, config=config)
        project.experiments = experiments
        project.shared_resources = data.get('shared_resources', {})
        
        return project
    
    # Iterator support
    def experiments_iter(self) -> Iterator[Experiment]:
        """Iterate over experiments in the project."""
        return iter(self.experiments.values())
    
    def __len__(self) -> int:
        """Return number of experiments in the project."""
        return len(self.experiments)
    
    def __contains__(self, experiment_name: str) -> bool:
        """Check if experiment exists in project."""
        return experiment_name in self.experiments
    
    def __repr__(self) -> str:
        """String representation of the project."""
        return f"Project(name='{self.config.name}', experiments={len(self.experiments)})"

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from pathlib import Path
from .pool import TaskPool
from .logging_config import get_logger

logger = get_logger("experiment")


class Experiment(BaseModel):
    """
    Represents an experiment definition with tasks (pure definition, no execution).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    readme: str = ""
    task_pool: Optional[TaskPool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    base_path: Optional[Path] = Field(
        default=None,
        description="Base directory path for the experiment"
    )
    
    def __init__(self, name: str, base_path: Optional[str | Path] = None, **kwargs):
        """Initialize experiment with name and optional base path."""
        kwargs.update(name=name)
        
        # Set up base path
        if base_path is not None:
            kwargs['base_path'] = Path(base_path).resolve()
        
        super().__init__(**kwargs)
        
        # Create experiment directory structure if base_path is set
        if self.base_path:
            self._create_experiment_structure()
        
        logger.info(f"Created experiment: {self.name}" + (f" at {self.base_path}" if self.base_path else ""))
    
    def _create_experiment_structure(self) -> None:
        """Create the experiment directory structure."""
        if self.base_path:
            # Create main experiment directory
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.base_path / "tasks").mkdir(exist_ok=True)
            (self.base_path / "data").mkdir(exist_ok=True)
            (self.base_path / "results").mkdir(exist_ok=True)
            (self.base_path / "config").mkdir(exist_ok=True)
            
            logger.info(f"Created experiment directory structure at {self.base_path}")
    
    def get_experiment_file_path(self) -> Path:
        """Get the path for the experiment YAML file."""
        if self.base_path:
            return self.base_path / f"{self.name}.yaml"
        return Path(f"{self.name}_experiment.yaml")
    
    def get_tasks_dir(self) -> Path:
        """Get the tasks directory path."""
        if self.base_path:
            return self.base_path / "tasks"
        return Path("tasks")

    def set_task_pool(self, task_pool: TaskPool) -> None:
        """Set the task pool for this experiment"""
        logger.info(f"Setting task pool for experiment '{self.name}' with {len(task_pool.tasks)} tasks")
        self.task_pool = task_pool

    def get_task_pool(self) -> Optional[TaskPool]:
        """Get the task pool for this experiment"""
        return self.task_pool
    
    def add_task(self, task) -> None:
        """Add a task to the experiment's task pool"""
        if self.task_pool is None:
            logger.debug(f"Creating new task pool for experiment '{self.name}'")
            self.task_pool = TaskPool(name=f"{self.name}_tasks")
        
        # Set task's base path if not already set and experiment has a base path
        if hasattr(task, 'base_path') and task.base_path is None and self.base_path:
            task_path = self.get_tasks_dir() / task.name
            task.base_path = task_path
            if hasattr(task, '_create_task_structure'):
                task._create_task_structure()
        
        logger.debug(f"Adding task '{task.name}' to experiment '{self.name}'")
        self.task_pool.add_task(task)
    
    def get_execution_order(self) -> list[str]:
        """Get the execution order of tasks using TaskGraph"""
        if self.task_pool is None:
            logger.warning(f"Experiment '{self.name}' has no task pool, returning empty execution order")
            return []
        
        logger.debug(f"Computing execution order for experiment '{self.name}' with {len(self.task_pool.tasks)} tasks")
        # Use TaskGraph for execution order
        from .graph import TaskGraph
        task_graph = TaskGraph(self.task_pool)
        execution_order = task_graph.topological_sort()
        logger.info(f"Execution order computed for experiment '{self.name}': {execution_order}")
        return execution_order
    
    def validate_experiment(self) -> None:
        """Validate the experiment definition using TaskGraph"""
        if self.task_pool is not None:
            logger.info(f"Validating experiment '{self.name}' with {len(self.task_pool.tasks)} tasks")
            from .graph import TaskGraph
            task_graph = TaskGraph(self.task_pool)
            task_graph.validate_dependencies()
            logger.info(f"Experiment '{self.name}' validation successful")
        else:
            logger.warning(f"Experiment '{self.name}' has no task pool to validate")

    def __post_init__(self):
        """
        Post-initialization hook to set default values or perform additional setup.
        """
    
    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Export experiment to YAML"""
        import yaml
        from pathlib import Path
        # 优先级：path参数 > self.base_path > 当前目录
        if path is not None:
            out_path = Path(path)
        elif self.base_path:
            out_path = self.base_path / f"{self.name}.yaml"
        else:
            out_path = Path(f"{self.name}_experiment.yaml")
        # Export experiment data
        experiment_data = {
            "name": self.name,
            "readme": self.readme,
            "metadata": self.metadata,
            "task_pool": {
                "name": self.task_pool.name,
                "tasks": {name: task.model_dump(exclude={'base_path'}) for name, task in self.task_pool.tasks.items()}
            } if self.task_pool else None
        }
        data = yaml.safe_dump(experiment_data, sort_keys=False)
        if isinstance(out_path, Path):
            with out_path.open("w") as f:
                f.write(data)
        return data
    
    @classmethod
    def from_yaml(cls, source: str | Path):
        """Load experiment from YAML"""
        import yaml
        from pathlib import Path
        
        # Determine if source is a file path or YAML content
        if isinstance(source, Path) or (isinstance(source, str) and len(source) < 255 and not '\n' in source):
            # Likely a file path
            try:
                path = Path(source)
                if path.exists():
                    content = path.read_text()
                else:
                    content = str(source)  # Treat as YAML string
            except (OSError, ValueError):
                content = str(source)  # Treat as YAML string
        else:
            # Definitely YAML content
            content = str(source)
        data = yaml.safe_load(content)
        
        experiment = cls(name=data["name"], readme=data.get("readme", ""))
        
        if data.get("task_pool"):
            task_pool_data = data["task_pool"]
            task_pool = TaskPool(task_pool_data["name"])
            
            for task_name, task_data in task_pool_data["tasks"].items():
                from .task import Task
                task = Task(**task_data)
                task_pool.add_task(task)
            
            experiment.set_task_pool(task_pool)
        
        return experiment
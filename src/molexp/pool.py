"""
TaskPool: Pure CRUD management for Task objects.
No dependency logic - only task storage and management.
"""

from typing import Dict, List, Optional
from pathlib import Path
import yaml
from .task import Task
from .logging_config import get_logger

logger = get_logger("pool")


class TaskPool:
    """TaskPool manages task CRUD operations (no dependency logic)"""
    
    def __init__(self, name: str):
        self.name = name
        self.tasks: Dict[str, Task] = {}
        logger.debug(f"Created TaskPool '{name}'")
    
    def add_task(self, task: Task) -> None:
        """Add a task to the pool"""
        if task.name in self.tasks:
            error_msg = f"Task '{task.name}' already exists in task pool '{self.name}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.tasks[task.name] = task
        logger.debug(f"Added task '{task.name}' to pool '{self.name}' (total: {len(self.tasks)})")
    
    def remove_task(self, task_name: str) -> None:
        """Remove a task from the pool"""
        if task_name not in self.tasks:
            error_msg = f"Task '{task_name}' not found in task pool '{self.name}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
        del self.tasks[task_name]
        logger.debug(f"Removed task '{task_name}' from pool '{self.name}' (remaining: {len(self.tasks)})")
    
    def get_task(self, task_name: str) -> Optional[Task]:
        """Get a task by name"""
        return self.tasks.get(task_name)
    
    def list_tasks(self) -> List[str]:
        """List all task names"""
        return list(self.tasks.keys())
    
    def update_task(self, task: Task) -> None:
        """Update an existing task"""
        if task.name not in self.tasks:
            raise ValueError(f"Task '{task.name}' not found in task pool")
        self.tasks[task.name] = task
    
    def get_tasks_by_deps(self, dependency: str) -> List[str]:
        """Get tasks that depend on a specific task"""
        dependents = []
        for name, task in self.tasks.items():
            if dependency in task.deps:
                dependents.append(name)
        return dependents
    
    def task_count(self) -> int:
        """Get total number of tasks"""
        return len(self.tasks)
    
    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Export task pool to YAML"""
        if path is None:
            path = Path(f"{self.name}_tasks.yaml")
        
        pool_data = {
            "name": self.name,
            "tasks": {name: task.model_dump() for name, task in self.tasks.items()}
        }
        
        data = yaml.safe_dump(pool_data, sort_keys=False)
        if isinstance(path, Path):
            with path.open("w") as f:
                f.write(data)
        return data
    
    @classmethod
    def from_yaml(cls, source: str | Path):
        """Load task pool from YAML"""
        path = Path(source) if isinstance(source, (str, Path)) and Path(source).exists() else None
        content = Path(path).read_text() if path else str(source)
        data = yaml.safe_load(content)
        
        pool = cls(name=data["name"])
        for task_name, task_data in data["tasks"].items():
            task = Task(**task_data)
            pool.add_task(task)
        
        return pool
    
    def __repr__(self) -> str:
        return f"TaskPool(name='{self.name}', tasks={len(self.tasks)})"

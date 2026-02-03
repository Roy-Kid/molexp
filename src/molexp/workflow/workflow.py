"""Workflow abstraction for representing workflow graphs.

This module contains the Workflow class for representing workflows as Pydantic models.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .task import Task

from .link import Link
from .registry import get_task_id, get_task_class
from .task import TaskConfig


class WorkflowMetadata(BaseModel):
    """Optional metadata for workflows (uses Pydantic for validation)."""

    label: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    custom: dict[str, str | int | float | bool] = Field(default_factory=dict)


class Workflow(BaseModel):
    """Workflow graph containing tasks and links.
    
    This is the primary abstraction for workflows in molexp. Workflows are
    Pydantic models that can be serialized to/from JSON/YAML files.
    
    Attributes:
        workflow_id: Unique workflow identifier (auto-generated)
        name: Optional workflow name
        task_configs: List of serialized task configurations
        links: List of Link instances between tasks
        metadata: Optional workflow metadata
        _tasks: Private cache of Task objects (not serialized)
    """
    
    workflow_id: str = Field(default_factory=lambda: f"workflow_{uuid4().hex[:8]}")
    name: str | None = None
    task_configs: list[TaskConfig] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    metadata: WorkflowMetadata = Field(default_factory=WorkflowMetadata)
    
    # Private runtime cache (not serialized)
    _tasks: list[Task] | None = None
    
    model_config = {"arbitrary_types_allowed": True}
    
    @classmethod
    def from_tasks(
        cls,
        tasks: list[Task],
        links: list[Link],
        name: str | None = None,
    ) -> Workflow:
        """Build Workflow from Task objects.
        
        Args:
            tasks: List of Task instances
            links: List of Link instances
            name: Optional workflow name
            
        Returns:
            Workflow instance with tasks serialized to task_configs
        """
        task_configs = []
        for task in tasks:
            # Try to get registered task_type_id; fall back for unregistered tasks
            try:
                task_type_id = get_task_id(task.__class__)
            except (KeyError, ValueError):
                # Unregistered task — use class name as placeholder.
                # The workflow can still be executed immediately but not
                # deserialized from disk without registration.
                task_type_id = task.__class__.__qualname__
            phase = getattr(task, "phase", None)
            task_configs.append(
                TaskConfig(
                    task_id=task.task_id,
                    task_type=task_type_id,
                    config=task.config.model_dump(),
                    phase=phase,
                )
            )
        
        workflow = cls(
            name=name,
            task_configs=task_configs,
            links=links,
        )
        workflow._tasks = tasks  # Cache runtime objects
        return workflow
    
    def get_tasks(self) -> list[Task]:
        """Get or restore Task objects.
        
        Returns:
            List of Task instances
        """
        if self._tasks is not None:
            return self._tasks
        
        # Restore tasks from configs using global registry
        tasks = []
        for task_config in self.task_configs:
            task_class = get_task_class(task_config.task_type)
            task = task_class(**task_config.config)
            task.task_id = task_config.task_id  # Restore ID
            tasks.append(task)
        
        self._tasks = tasks  # Cache
        return tasks
    
    def save(self, path: Path, format: str = "auto") -> None:
        """Save workflow to file.
        
        Args:
            path: Path to save the workflow
            format: File format ('json', 'yaml', or 'auto' to detect from extension)
            
        Raises:
            ValueError: If format is unsupported
        """
        if format == "auto":
            format = "yaml" if path.suffix in (".yaml", ".yml") else "json"
        
        if format == "json":
            content = self.model_dump_json(indent=2, exclude={"_tasks"})
        elif format == "yaml":
            import yaml
            content = yaml.dump(
                self.model_dump(exclude={"_tasks"}),
                default_flow_style=False,
                sort_keys=False,
            )
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    
    @classmethod
    def load(cls, path: Path) -> Workflow:
        """Load workflow from file.
        
        Args:
            path: Path to the workflow file
            
        Returns:
            Workflow instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file extension is unsupported
        """
        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {path}")
        
        content = path.read_text()
        
        if path.suffix == ".json":
            return cls.model_validate_json(content)
        elif path.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(content)
            return cls.model_validate(data)
        else:
            raise ValueError(
                f"Unsupported file extension: {path.suffix}. "
                f"Use .json, .yaml, or .yml"
            )
    
    def __repr__(self) -> str:
        return f"Workflow(id={self.workflow_id!r}, tasks={len(self.task_configs)}, links={len(self.links)})"

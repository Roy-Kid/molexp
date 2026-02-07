"""Task registry for managing registered task types.

This is the single source of truth for task registration. It combines:
- Plugin metadata (label, category, ports) for API/UI
- Execution mapping (task_type_id <-> task_class) for workflow serialization
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from ..task import Task
from mollog import get_logger
from .metadata import TaskMetadata

logger = get_logger(__name__)


class TaskRegistration:
    """Registration entry for a task type.

    Attributes:
        task_type_id: Unique task identifier
        task_class: Task class
        config_class: Configuration model class
        metadata: Task metadata
    """

    def __init__(
        self,
        task_type_id: str,
        task_class: Type[Task],
        config_class: Type[BaseModel],
        metadata: TaskMetadata,
    ):
        self.task_type_id = task_type_id
        self.task_class = task_class
        self.config_class = config_class
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses.

        Returns:
            Dictionary with task metadata and config schema
        """
        return {
            "id": self.task_type_id,
            "label": self.metadata.label,
            "category": self.metadata.category,
            "description": self.metadata.description,
            "inputs": [port.model_dump(mode="json") for port in self.metadata.inputs],
            "outputs": [port.model_dump(mode="json") for port in self.metadata.outputs],
            "icon": self.metadata.icon,
            "tags": self.metadata.tags,
            "config_schema": self.config_class.model_json_schema(),
        }


class TaskRegistry:
    """Central registry for all task types.

    This registry is populated via entry points or direct registration and
    provides both metadata for API/UI and class lookup for workflow execution.
    """

    def __init__(self):
        self._tasks: Dict[str, TaskRegistration] = {}
        self._class_to_id: Dict[Type[Task], str] = {}
        self._loaded = False

    @staticmethod
    def derive_task_id(task_class: Type[Task]) -> str:
        """Derive a deterministic task type ID from a class."""
        return f"{task_class.__module__}.{task_class.__qualname__}"

    # ========== Registration ==========

    def register(
        self,
        task_type_id: str,
        task_class: Type[Task],
        metadata: TaskMetadata,
        config_class: Optional[Type[BaseModel]] = None,
    ) -> str:
        """Register a task type.

        Args:
            task_type_id: Unique task identifier (e.g., "io.read_file")
            task_class: Task class
            metadata: Task metadata
            config_class: Configuration class (defaults to task_class.config_type)

        Returns:
            The registered task type ID.

        Raises:
            ValueError: If task_type_id collides or task_class lacks config_type
        """
        if task_type_id in self._tasks:
            existing = self._tasks[task_type_id]
            if existing.task_class is not task_class:
                raise ValueError(
                    f"Task type ID '{task_type_id}' already registered to "
                    f"{existing.task_class.__name__}, cannot register {task_class.__name__}"
                )
            return task_type_id  # Already registered, idempotent

        # Check for class already registered under different ID
        prior_id = self._class_to_id.get(task_class)
        if prior_id is not None and prior_id != task_type_id:
            raise ValueError(
                f"{task_class.__name__} already registered as '{prior_id}', "
                f"cannot register as '{task_type_id}'"
            )

        # Resolve config class
        if config_class is None:
            if not hasattr(task_class, "config_type"):
                raise ValueError(
                    f"Task class {task_class.__name__} must have 'config_type' attribute "
                    f"or config_class must be provided"
                )
            config_class = task_class.config_type

        registration = TaskRegistration(
            task_type_id=task_type_id,
            task_class=task_class,
            config_class=config_class,
            metadata=metadata,
        )

        self._tasks[task_type_id] = registration
        self._class_to_id[task_class] = task_type_id
        setattr(task_class, "task_type_id", task_type_id)

        logger.info(f"Registered task: {task_type_id} ({task_class.__name__})")
        return task_type_id

    def register_task(
        self,
        task_class: Type[Task],
        task_type_id: str | None = None,
    ) -> str:
        """Register a task class without rich metadata.

        This is the simpler registration path for programmatic use (no plugin
        metadata). A minimal TaskMetadata is auto-generated.

        Args:
            task_class: Task class to register
            task_type_id: Explicit type ID (optional). If omitted, derived from module path.

        Returns:
            The registered task type ID.
        """
        if task_type_id is None:
            task_type_id = self.derive_task_id(task_class)

        # Check if already registered
        if task_class in self._class_to_id:
            return self._class_to_id[task_class]

        # Auto-generate minimal metadata
        label = task_class.__name__.replace("Task", "")
        parts = task_type_id.split(".")
        category = parts[0] if len(parts) > 1 else "general"

        metadata = TaskMetadata(
            label=label,
            category=category,
            description=task_class.__doc__ or f"{label} task",
            inputs=[],
            outputs=[],
        )

        return self.register(
            task_type_id=task_type_id,
            task_class=task_class,
            metadata=metadata,
        )

    # ========== Lookup ==========

    def has(self, task_type_id: str) -> bool:
        """Check if a task type ID is registered."""
        return task_type_id in self._tasks

    def get(self, task_type_id: str) -> Optional[TaskRegistration]:
        """Get registration for a task type ID.

        Args:
            task_type_id: Task identifier

        Returns:
            Registration entry or None if not found
        """
        return self._tasks.get(task_type_id)

    def get_task_class(self, task_type_id: str) -> Type[Task]:
        """Get task class by type ID.

        Args:
            task_type_id: Task identifier

        Returns:
            Task class

        Raises:
            ValueError: If task type not registered
        """
        reg = self._tasks.get(task_type_id)
        if reg is None:
            raise ValueError(
                f"Task type '{task_type_id}' not registered. "
                f"Available: {sorted(self._tasks.keys())}"
            )
        return reg.task_class

    def get_task_id(self, task_class: Type[Task]) -> str:
        """Get the registered task type ID for a class.

        Args:
            task_class: Task class

        Returns:
            Registered task type ID

        Raises:
            ValueError: If class not registered
        """
        if task_class not in self._class_to_id:
            raise ValueError(
                f"{task_class.__name__} is not registered. "
                "Register with register_task() before serialization."
            )
        return self._class_to_id[task_class]

    # ========== Listing ==========

    def list_all(self) -> List[TaskRegistration]:
        """List all registered tasks."""
        return list(self._tasks.values())

    def list_ids(self) -> list[str]:
        """List all registered task type IDs."""
        return sorted(self._tasks.keys())

    def list_by_category(self, category: str) -> List[TaskRegistration]:
        """List tasks in a specific category."""
        return [
            reg for reg in self._tasks.values() if reg.metadata.category == category
        ]

    # ========== Export ==========

    def to_dict(self) -> dict[str, Any]:
        """Export all tasks as dictionary for API."""
        return {"nodes": [reg.to_dict() for reg in self._tasks.values()]}

    # ========== Lifecycle ==========

    def mark_loaded(self) -> None:
        """Mark registry as loaded (plugins have been discovered)."""
        self._loaded = True

    def is_loaded(self) -> bool:
        """Check if plugins have been loaded."""
        return self._loaded


# Global registry instance
_global_registry: Optional[TaskRegistry] = None


def get_task_registry() -> TaskRegistry:
    """Get the global task registry instance.

    Returns:
        Global TaskRegistry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = TaskRegistry()
    return _global_registry


def register(task_type_id: str | None = None, metadata: Optional[TaskMetadata] = None) -> Callable:
    """Decorator to register a task class.

    If metadata is not provided, minimal metadata will be auto-generated.
    If task_type_id is omitted, a deterministic ID is derived from module path.

    Args:
        task_type_id: Unique task identifier
        metadata: Optional task metadata

    Returns:
        Decorator function
    """

    def decorator(task_class: Type[Task]) -> Type[Task]:
        resolved_id = task_type_id or TaskRegistry.derive_task_id(task_class)

        # Auto-generate minimal metadata if not provided
        if metadata is None:
            parts = resolved_id.split(".")
            category = parts[0] if len(parts) > 1 else "general"
            label = task_class.__name__.replace("Task", "")

            auto_metadata = TaskMetadata(
                label=label,
                category=category,
                description=task_class.__doc__ or f"{label} task",
                inputs=[],
                outputs=[],
            )
            final_metadata = auto_metadata
        else:
            final_metadata = metadata

        registry = get_task_registry()
        registry.register(
            task_type_id=resolved_id,
            task_class=task_class,
            metadata=final_metadata,
        )

        return task_class

    return decorator

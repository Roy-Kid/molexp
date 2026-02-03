"""Task registry convenience functions.

This module re-exports registry functions from the unified plugin registry.
All registration state lives in ``workflow.plugin.registry.TaskRegistry``.
"""

from __future__ import annotations

from typing import Type

from .task import Task
from .plugin.registry import TaskRegistry, get_task_registry


def register_task(task_class: Type[Task], task_type_id: str | None = None) -> str:
    """Register a task class in the global registry."""
    return get_task_registry().register_task(task_class, task_type_id=task_type_id)


def get_task_class(task_type_id: str) -> Type[Task]:
    """Get a task class by exact type ID."""
    return get_task_registry().get_task_class(task_type_id)


def get_task_id(task_class: Type[Task]) -> str:
    """Get the registered task type ID for a class."""
    return get_task_registry().get_task_id(task_class)


def list_registered_tasks() -> list[str]:
    """List all registered task type IDs."""
    return get_task_registry().list_ids()

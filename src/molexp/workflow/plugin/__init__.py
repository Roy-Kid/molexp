"""Plugin system for molexp tasks.

This package provides the infrastructure for discovering and loading
task plugins via Python entry points.
"""

from .loader import load_plugins
from .metadata import TaskMetadata, PortMetadata
from .registry import TaskRegistry, get_task_registry

__all__ = [
    "TaskMetadata",
    "PortMetadata",
    "TaskRegistry",
    "get_task_registry",
    "load_plugins",
]

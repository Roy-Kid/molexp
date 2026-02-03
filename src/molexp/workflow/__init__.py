"""Unified workflow system for molexp.

This module provides the core abstractions for building and executing workflows:
- Task: Base abstraction for executable units
- Control flow: Built-in tasks for workflow logic
- Compiler: Workflow compilation and validation
- Engine: Workflow execution
"""

# Core workflow abstractions
from .protocol import TaskProtocol
from .task import Task, TaskConfig

# Compiler
from .compiler import WorkflowCompiler

# Engine
from .engine import WorkflowEngine

# Models and Registry
from .link import Link
from .workflow import Workflow, WorkflowMetadata
from .registry import (
    TaskRegistry,
    get_task_registry,
    register_task,
    get_task_class,
    get_task_id,
    list_registered_tasks,
)

__all__ = [
    # Core abstractions
    "TaskProtocol",
    "Task",
    "TaskConfig",
    # Compiler
    "WorkflowCompiler",
    # Engine
    "WorkflowEngine",
    # Models
    "Workflow",
    "WorkflowMetadata",
    "Link",
    # Registry
    "TaskRegistry",
    "get_task_registry",
    "register_task",
    "get_task_class",
    "get_task_id",
    "list_registered_tasks",
]

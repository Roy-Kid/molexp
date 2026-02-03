"""Task execution status enum."""

from enum import Enum


class TaskStatus(str, Enum):
    """Status of a task in the workflow engine.

    Inherits from str for JSON serialization compatibility.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

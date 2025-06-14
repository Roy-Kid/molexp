"""
Task dispatch system for handling different task types with different execution backends.
"""

from .base import TaskDispatcher
from .shell import ShellSubmitter
from .local import LocalSubmitter

__all__ = ['TaskDispatcher', 'ShellSubmitter', 'LocalSubmitter']

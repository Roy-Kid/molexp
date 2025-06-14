"""
Local task submitter for executing local shell tasks.
"""

from typing import Any, Optional, TYPE_CHECKING
from .shell import ShellSubmitter

if TYPE_CHECKING:
    from ..task import Task, LocalTask
    from ..param import Param


class LocalSubmitter(ShellSubmitter):
    """Submitter for executing local tasks (inherits from ShellSubmitter)"""
    
    def __init__(self, shell: str = "/bin/bash", timeout: Optional[int] = None):
        """
        Initialize LocalSubmitter
        
        Args:
            shell: Shell to use for execution (default: /bin/bash)
            timeout: Timeout in seconds for command execution
        """
        super().__init__(shell=shell, timeout=timeout)
    
    def can_handle(self, task: "Task") -> bool:
        """Check if this is a LocalTask"""
        from ..task import LocalTask
        return isinstance(task, LocalTask)
    
    def submit(self, task: "Task", param: Optional["Param"] = None) -> Any:
        """
        Execute local task - delegates to parent ShellSubmitter
        
        Args:
            task: LocalTask to execute
            param: Parameters for template rendering
            
        Returns:
            Dictionary with execution results
        """
        from ..task import LocalTask
        
        if not isinstance(task, LocalTask):
            raise ValueError(f"LocalSubmitter can only handle LocalTask, got {type(task)}")
        
        # Use parent ShellSubmitter logic
        result = super().submit(task, param)
        
        # Update task type in result
        result["task_type"] = "LocalTask"
        
        return result

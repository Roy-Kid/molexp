"""
Base classes for the task dispatch system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..task import Task
    from ..param import Param

# Import logger
from ..logging_config import get_logger
logger = get_logger("dispatch")


class TaskSubmitter(ABC):
    """Base class for task submitters that handle specific task types"""
    
    @abstractmethod
    def can_handle(self, task: "Task") -> bool:
        """Check if this submitter can handle the given task type"""
        pass
    
    @abstractmethod
    def submit(self, task: "Task", param: Optional["Param"] = None) -> Any:
        """Submit the task for execution and return the result"""
        pass


class TaskDispatcher:
    """
    Dispatcher that routes tasks to appropriate submitters based on task type.
    """
    
    def __init__(self):
        self.submitters: list[TaskSubmitter] = []
        logger.debug("TaskDispatcher initialized")
    
    def register_submitter(self, submitter: TaskSubmitter) -> None:
        """Register a new task submitter"""
        self.submitters.append(submitter)
        logger.info(f"Registered submitter: {type(submitter).__name__}")
    
    def execute_task(self, task: "Task", param: Optional["Param"] = None) -> Any:
        """
        Execute a task by finding the appropriate submitter.
        
        Args:
            task: The task to execute
            param: Optional parameters for execution
            
        Returns:
            Task execution result
            
        Raises:
            ValueError: If no submitter can handle the task type
        """
        logger.debug(f"Dispatching task '{task.name}' of type {type(task).__name__}")
        
        for submitter in self.submitters:
            if submitter.can_handle(task):
                logger.info(f"Task '{task.name}' handled by {type(submitter).__name__}")
                return submitter.submit(task, param)
        
        # If no submitter found, fall back to simulation
        logger.warning(f"No submitter found for task '{task.name}' (type: {type(task).__name__}), falling back to simulation")
        return self._simulate_task_execution(task, param)
    
    def _simulate_task_execution(self, task: "Task", param: Optional["Param"] = None) -> Any:
        """Fallback simulation for unsupported task types"""
        return {
            "task_name": task.name,
            "task_type": type(task).__name__,
            "args": task.args,
            "kwargs": task.kwargs,
            "outputs": task.outputs,
            "param": dict(param) if param else None,
            "status": "simulated",
            "message": f"No submitter found for task type {type(task).__name__}"
        }

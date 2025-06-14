"""
Execution layer: Executor and ExperimentExecutor classes for task execution.
Executor handles actual task execution, ExperimentExecutor provides experiment-level functionality.
"""

from typing import Any, Dict, List, Set, TYPE_CHECKING, Optional
from collections import defaultdict
from .graph import TaskGraph
from .task import Task
from .dispatch.base import TaskDispatcher
from .dispatch.shell import ShellSubmitter
from .dispatch.local import LocalSubmitter
from .logging_config import get_logger

if TYPE_CHECKING:
    from .experiment import Experiment

logger = get_logger("executor")


class TaskStatus:
    """Task execution status enum"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Executor:
    """
    Executor is responsible for actual execution, using TaskGraph for dependency and execution order.
    """
    
    def __init__(self, task_graph: TaskGraph):
        """Initialize Executor with TaskGraph"""
        logger.info(f"Initializing Executor with {len(task_graph.task_pool.tasks)} tasks")
        self.task_graph = task_graph
        self.task_status: Dict[str, str] = {}
        self.execution_results: Dict[str, Any] = {}
        
        # Initialize task dispatch system
        self.dispatcher = TaskDispatcher()
        self._setup_default_submitters()
        
        # Initialize task status
        for task_name in self.task_graph.task_pool.tasks:
            self.task_status[task_name] = TaskStatus.PENDING
        
        logger.debug(f"Executor initialized with tasks: {list(self.task_graph.task_pool.tasks.keys())}")
    
    def _setup_default_submitters(self):
        """Setup default task submitters"""
        # Register submitters in order of preference
        self.dispatcher.register_submitter(LocalSubmitter())
        self.dispatcher.register_submitter(ShellSubmitter())
        logger.debug("Default task submitters registered")
    
    def register_submitter(self, submitter):
        """Register a custom task submitter"""
        self.dispatcher.register_submitter(submitter)
        logger.info(f"Custom submitter registered: {type(submitter).__name__}")
    
    def get_executable_tasks(self) -> List[str]:
        """Get list of tasks that can be executed (all dependencies completed)"""
        completed_tasks = {
            name for name, status in self.task_status.items() 
            if status == TaskStatus.COMPLETED
        }
        
        ready = []
        for task_name in self.task_graph.task_pool.tasks:
            if self.task_status[task_name] == TaskStatus.PENDING:
                # Use TaskGraph to check if task is ready
                if task_name in self.task_graph.get_ready_tasks(completed_tasks):
                    ready.append(task_name)
        
        return ready
    
    def mark_task_running(self, task_name: str) -> None:
        """Mark a task as running"""
        if task_name not in self.task_graph.task_pool.tasks:
            raise ValueError(f"Task '{task_name}' not found in task pool")
        self.task_status[task_name] = TaskStatus.RUNNING
        logger.info(f"Task '{task_name}' marked as running")
    
    def mark_task_completed(self, task_name: str, result: Any = None) -> None:
        """Mark a task as completed with optional result"""
        if task_name not in self.task_graph.task_pool.tasks:
            raise ValueError(f"Task '{task_name}' not found in task pool")
        self.task_status[task_name] = TaskStatus.COMPLETED
        if result is not None:
            self.execution_results[task_name] = result
        logger.info(f"Task '{task_name}' marked as completed")
    
    def mark_task_failed(self, task_name: str, error: Any = None) -> None:
        """Mark a task as failed with optional error info"""
        if task_name not in self.task_graph.task_pool.tasks:
            raise ValueError(f"Task '{task_name}' not found in task pool")
        self.task_status[task_name] = TaskStatus.FAILED
        if error is not None:
            self.execution_results[task_name] = {"error": str(error)}
        logger.warning(f"Task '{task_name}' marked as failed: {error}")
    
    def get_execution_status(self) -> Dict[str, int]:
        """Get overall execution status summary"""
        status_count = defaultdict(int)
        for status in self.task_status.values():
            status_count[status] += 1
        return dict(status_count)
    
    def is_execution_completed(self) -> bool:
        """Check if all tasks are completed"""
        return all(status == TaskStatus.COMPLETED for status in self.task_status.values())
    
    def is_execution_failed(self) -> bool:
        """Check if any task has failed"""
        return any(status == TaskStatus.FAILED for status in self.task_status.values())
    
    def reset_execution(self) -> None:
        """Reset all tasks to pending status"""
        for task_name in self.task_graph.task_pool.tasks:
            self.task_status[task_name] = TaskStatus.PENDING
        self.execution_results.clear()
    
    def run(self, param=None) -> Dict[str, Any]:
        """Execute the workflow with given parameters"""
        from .param import Param
        
        if param is None:
            param = Param()
        
        logger.info(f"Starting workflow execution with {len(self.task_graph.task_pool.tasks)} tasks")
        logger.debug(f"Execution parameters: {dict(param) if param else 'None'}")
        
        # Reset execution state
        self.reset_execution()
        
        # Get execution order from TaskGraph
        execution_order = self.task_graph.topological_sort()
        logger.info(f"Execution order: {execution_order}")
        
        results = {}
        
        try:
            while not self.is_execution_completed() and not self.is_execution_failed():
                # Get tasks that can be executed now
                executable_tasks = self.get_executable_tasks()
                
                if not executable_tasks:
                    if not self.is_execution_completed():
                        error_msg = "No executable tasks found but execution not completed (possible deadlock)"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    break
                
                logger.debug(f"Executing tasks: {executable_tasks}")
                
                # Execute all executable tasks (could be parallel in future)
                for task_name in executable_tasks:
                    task = self.task_graph.task_pool.tasks[task_name]
                    
                    try:
                        logger.info(f"Starting execution of task '{task_name}'")
                        self.mark_task_running(task_name)
                        
                        # Execute the task
                        result = self._execute_task(task, param)
                        
                        self.mark_task_completed(task_name, result)
                        results[task_name] = result
                        logger.info(f"Task '{task_name}' completed successfully")
                        
                    except Exception as e:
                        logger.error(f"Task '{task_name}' failed with error: {e}")
                        self.mark_task_failed(task_name, e)
                        results[task_name] = {"error": str(e)}
                        # Continue execution of other tasks unless specified otherwise
                        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            results["execution_error"] = str(e)
        
        status = self.get_execution_status()
        logger.info(f"Workflow execution completed. Status: {status}")
        return results
    
    def _execute_task(self, task: Task, param) -> Any:
        """Execute a single task using dispatch system"""
        return self.dispatcher.execute_task(task, param)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution"""
        return {
            "task_count": len(self.task_graph.task_pool.tasks),
            "status_summary": self.get_execution_status(),
            "completed": self.is_execution_completed(),
            "failed": self.is_execution_failed(),
            "results": self.execution_results
        }
    
    def __repr__(self) -> str:
        return f"Executor(tasks={len(self.task_graph.task_pool.tasks)}, completed={self.get_execution_status().get('completed', 0)})"


"""
ExperimentExecutor for running complete experiments.
This class wraps an Executor and provides experiment-level functionality.
"""

from typing import Any, Dict, Optional, List, TYPE_CHECKING
from collections import defaultdict
from .pool import TaskPool
from .graph import TaskGraph
from .executor import Executor, TaskStatus
from .task import Task

if TYPE_CHECKING:
    from .experiment import Experiment


class ExperimentExecutor:
    """Experiment executor that runs complete experiments using an internal Executor"""
    
    def __init__(self, experiment: "Experiment", name: Optional[str] = None):
        """Initialize experiment executor with an experiment"""
        if experiment is None:
            raise ValueError("Experiment is required")
        
        if not hasattr(experiment, 'task_pool') or experiment.task_pool is None:
            raise ValueError("Experiment must have a task_pool")
        
        self.experiment = experiment
        self.name = name or f"{experiment.name}_executor"
        
        # Create TaskGraph and internal Executor
        self.task_graph = TaskGraph(experiment.task_pool)
        self.executor = Executor(self.task_graph)
    
    @property
    def task_pool(self) -> TaskPool:
        """Get the task pool from the experiment"""
        if self.experiment.task_pool is None:
            raise RuntimeError("Experiment task_pool is None")
        return self.experiment.task_pool
    
    @property
    def task_status(self) -> Dict[str, str]:
        """Get task status from internal executor"""
        return self.executor.task_status
    
    @property
    def execution_results(self) -> Dict[str, Any]:
        """Get execution results from internal executor"""
        return self.executor.execution_results
    
    def get_executable_tasks(self) -> List[str]:
        """Get list of tasks that can be executed (delegate to executor)"""
        return self.executor.get_executable_tasks()
    
    def mark_task_running(self, task_name: str) -> None:
        """Mark a task as running (delegate to executor)"""
        self.executor.mark_task_running(task_name)
    
    def mark_task_completed(self, task_name: str, result: Any = None) -> None:
        """Mark a task as completed (delegate to executor)"""
        self.executor.mark_task_completed(task_name, result)
    
    def mark_task_failed(self, task_name: str, error: Any = None) -> None:
        """Mark a task as failed (delegate to executor)"""
        self.executor.mark_task_failed(task_name, error)
    
    def get_execution_status(self) -> Dict[str, int]:
        """Get overall execution status summary (delegate to executor)"""
        return self.executor.get_execution_status()
    
    def is_execution_completed(self) -> bool:
        """Check if all tasks are completed (delegate to executor)"""
        return self.executor.is_execution_completed()
    
    def is_execution_failed(self) -> bool:
        """Check if any task has failed (delegate to executor)"""
        return self.executor.is_execution_failed()
    
    def reset_execution(self) -> None:
        """Reset all tasks to pending status (delegate to executor)"""
        self.executor.reset_execution()
    
    def run(self, param=None) -> Dict[str, Any]:
        """Execute the experiment with given parameters"""
        from .param import Param
        
        if param is None:
            param = Param()
        
        # Validate experiment before execution
        self.experiment.validate_experiment()
        
        # Delegate to internal executor
        return self.executor.run(param)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution"""
        return {
            "name": self.name,
            "experiment_name": self.experiment.name,
            "task_count": len(self.task_pool.tasks),
            "status_summary": self.get_execution_status(),
            "completed": self.is_execution_completed(),
            "failed": self.is_execution_failed(),
            "results": self.execution_results
        }
    
    def __repr__(self) -> str:
        return f"ExperimentExecutor(name='{self.name}', experiment='{self.experiment.name}', tasks={len(self.task_pool.tasks)})"

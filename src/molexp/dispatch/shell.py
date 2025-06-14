"""
Shell task submitter for executing shell commands.
"""

import subprocess
from typing import Any, Dict, Optional, TYPE_CHECKING
from .base import TaskSubmitter

if TYPE_CHECKING:
    from ..task import Task, ShellTask
    from ..param import Param

# Import logger
from ..logging_config import get_logger
logger = get_logger("dispatch.shell")


class ShellSubmitter(TaskSubmitter):
    """Submitter for executing shell tasks"""
    
    def __init__(self, shell: str = "/bin/bash", timeout: Optional[int] = None):
        """
        Initialize ShellSubmitter
        
        Args:
            shell: Shell to use for execution (default: /bin/bash)
            timeout: Timeout in seconds for command execution
        """
        self.shell = shell
        self.timeout = timeout
        logger.info(f"ShellSubmitter initialized with shell: {shell}, timeout: {timeout}")
    
    def can_handle(self, task: "Task") -> bool:
        """Check if this is a ShellTask"""
        from ..task import ShellTask
        can_handle = isinstance(task, ShellTask)
        logger.debug(f"ShellSubmitter can handle task '{task.name}': {can_handle}")
        return can_handle
    
    def submit(self, task: "Task", param: Optional["Param"] = None) -> Any:
        """
        Execute shell task commands
        
        Args:
            task: ShellTask to execute
            param: Parameters for template rendering
            
        Returns:
            Dictionary with execution results
        """
        from ..task import ShellTask
        
        if not isinstance(task, ShellTask):
            raise ValueError(f"ShellSubmitter can only handle ShellTask, got {type(task)}")
        
        shell_task: "ShellTask" = task
        logger.info(f"Executing shell task '{task.name}' with {len(shell_task.commands)} commands")
        
        # Prepare parameters for template rendering
        template_params = {}
        if param:
            template_params.update(param)
        
        # Render commands with parameters
        try:
            rendered_commands = shell_task.render_commands(**template_params)
            logger.debug(f"Rendered {len(rendered_commands)} commands for task '{task.name}'")
        except ValueError as e:
            logger.error(f"Template rendering failed for task '{task.name}': {e}")
            return {
                "task_name": task.name,
                "status": "failed",
                "error": f"Template rendering failed: {e}",
                "commands": shell_task.commands
            }
        
        # Execute commands
        results = []
        for i, command in enumerate(rendered_commands):
            try:
                logger.debug(f"Executing command {i+1}/{len(rendered_commands)}: {command}")
                result = self._execute_command(command)
                results.append({
                    "command_index": i,
                    "command": command,
                    "returncode": result["returncode"],
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                    "success": result["returncode"] == 0
                })
                if result["returncode"] == 0:
                    logger.debug(f"Command {i+1} executed successfully")
                else:
                    logger.warning(f"Command {i+1} failed with return code {result['returncode']}")
            except Exception as e:
                logger.error(f"Command {i+1} failed with exception: {e}")
                results.append({
                    "command_index": i,
                    "command": command,
                    "error": str(e),
                    "success": False
                })
        
        # Determine overall success
        all_success = all(r.get("success", False) for r in results)
        status = "completed" if all_success else "failed"
        
        logger.info(f"Shell task '{task.name}' {status} ({len([r for r in results if r.get('success')])}/{len(results)} commands successful)")
        
        return {
            "task_name": task.name,
            "task_type": "ShellTask",
            "status": status,
            "commands_executed": len(rendered_commands),
            "results": results,
            "success": all_success
        }
    
    def _execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute a single shell command
        
        Args:
            command: Command to execute
            
        Returns:
            Dictionary with execution results
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                executable=self.shell,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired as e:
            return {
                "returncode": -1,
                "stdout": e.stdout or "",
                "stderr": f"Command timed out after {self.timeout} seconds: {e.stderr or ''}"
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}"
            }

"""Workflow execution engine with parallel execution and failure propagation."""

import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing_extensions import Any
from rich.console import Console

from mollog import get_logger

logger = get_logger(__name__)
console = Console()

from molexp.workflow.status import TaskStatus
from molexp.workflow.task import Task
from molexp.workflow.workflow import Workflow
from molexp.workspace import Context, RunStatus, RunContext
from molexp.workflow.compiler import WorkflowCompiler


# ============================================================================
# Execution context is now part of Run.context
# Helper functions for execution context management
def mark_succeeded(context: "Context", task_id: str, result: dict[str, Any]) -> None:
    """Mark task as succeeded and store result."""
    context.tasks[task_id] = TaskStatus.SUCCEEDED
    context.results[task_id] = result


def mark_failed(context: "Context", task_id: str, error: Exception) -> None:
    """Mark task as failed and store error."""
    context.tasks[task_id] = TaskStatus.FAILED
    # Store both error message and full traceback
    context.errors[task_id] = {
        "message": str(error),
        "type": type(error).__name__,
        "traceback": traceback.format_exc()
    }


def mark_cancelled(context: "Context", task_id: str) -> None:
    """Mark task as cancelled."""
    context.tasks[task_id] = TaskStatus.CANCELLED
    context.errors[task_id] = {
        "message": "Cancelled due to upstream failure",
        "type": "Cancelled",
        "traceback": None
    }


def is_failed_or_cancelled(context: "Context", task_id: str) -> bool:
    """Check if task has failed or been cancelled."""
    status = context.tasks.get(task_id)
    return status in (TaskStatus.FAILED, TaskStatus.CANCELLED)


def print_task_error(task_id: str, error_info: dict[str, Any]) -> None:
    """Print task error with background colors but no borders for easy copying."""
    console.print()
    
    # Print error header with background color
    console.print(f"[bold white on red] ❌ TASK FAILED: {task_id} [/bold white on red]")
    console.print(f"[bold red]Error Type:[/bold red] {error_info['type']}")
    console.print(f"[bold red]Message:[/bold red] {error_info['message']}")
    
    # Print traceback with light red background
    if error_info.get('traceback'):
        console.print()
        console.print("[bold yellow]Full Traceback:[/bold yellow]")
        
        # Split traceback into lines and print each with light red background
        traceback_lines = error_info['traceback'].strip().split('\n')
        for line in traceback_lines:
            # Use light red background for the entire traceback
            console.print(f"[on rgb(60,20,20)]{line}[/on rgb(60,20,20)]")
    console.print()
# ============================================================================
# Workflow Engine
# ============================================================================


class WorkflowEngine:
    """Executes compiled workflows with parallel execution and failure propagation.

    The engine:
    - Accepts a Workflow object at construction
    - Compiles it using WorkflowCompiler
    - Executes tasks with named inputs/outputs
    - Supports parallel execution of independent tasks
    - Propagates failures to dependent tasks
    """

    def __init__(
        self,
        workflow: "Workflow",
        max_workers: int = 4,
        task_timeout: float | None = None,
    ) -> None:
        """Initialize the workflow engine.

        Args:
            workflow: Workflow object to execute
            max_workers: Maximum number of concurrent task executions
            task_timeout: Timeout in seconds for each task (None = no timeout)
        """
        self.workflow = workflow
        self.max_workers = max_workers
        self.task_timeout = task_timeout

        # Compile workflow
        self.compiled = WorkflowCompiler().compile(workflow)

        # Get dependency graph from compiled workflow
        self._adj, self._in_degree_template = self.compiled.get_dependency_graph()

        # Get tasks from workflow (will use cached _tasks if available)
        # Import registry here to avoid circular dependency
        tasks = workflow.get_tasks()

        # Build task map from workflow
        self._task_map = {task.task_id: task for task in tasks}
        self._thread_executor: ThreadPoolExecutor | None = None

        # Submitor registry for external task submission
        self._submitors: dict[str, Any] = {}  # backend_name -> submitor instance

    def _compute_subgraph(self, phases: list[str] | None = None) -> tuple[set[str], dict[str, int]]:
        """Compute the workflow graph to execute, optionally filtered by phase.

        Args:
            phases: If provided, only tasks whose phase is in this list
                    (or whose phase is None) will be included.

        Returns:
            Tuple of (tasks_to_execute, in_degree_map)
        """
        if phases is None:
            tasks_to_execute = set(self._task_map.keys())
        else:
            # Build a task_id -> phase lookup from task_configs
            phase_map = {tc.task_id: tc.phase for tc in self.workflow.task_configs}
            tasks_to_execute = {
                task_id for task_id in self._task_map
                if phase_map.get(task_id) is None or phase_map.get(task_id) in phases
            }

        # Build in_degree map counting only edges within the subgraph
        in_degree = {task_id: 0 for task_id in tasks_to_execute}
        for task_id in tasks_to_execute:
            for link in self.workflow.links:
                if link.target == task_id and link.source in tasks_to_execute:
                    in_degree[task_id] += 1

        return tasks_to_execute, in_degree

    def execute(
        self,
        run_context: "RunContext",
        phases: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        """Execute the workflow with RunContext.

        Args:
            run_context: RunContext wrapper (not pure Context)
            phases: If provided, only execute tasks whose phase matches
                    (or tasks with phase=None). Pass None to run all tasks.
            **kwargs: Flat keyword arguments that tasks can pick from based on their input requirements

        Returns:
            Dict mapping task IDs to their output dicts
        """
        self.run_context = run_context
        self._phases = phases

        # Stage 1: Before execution - enrich context with workflow
        self._before_execute()

        try:
            # Stage 2: Execute tasks
            results = self._execute(**kwargs)

            # Stage 3: After execution (success)
            self._after_execute(results, success=True)

            return results

        except Exception as e:
            # Stage 3: After execution (failure)
            self._after_execute(None, success=False, error=e)
            raise
    
    def _before_execute(self):
        """Enrich context with workflow information and load persisted results."""
        # Use RunContext method to set workflow
        self.run_context.set_workflow(self.workflow)

        # Load persisted results from run.json (enables cross-phase input resolution)
        self._load_persisted_results()

        # Initialize task statuses in internal context
        for task_config in self.workflow.task_configs:
            self.run_context.context.tasks[task_config.task_id] = TaskStatus.PENDING

        # Save updated context
        self.run_context._save_context()
        self._init_execution_metadata()

    def _load_persisted_results(self):
        """Load previously persisted context.results from run.json."""
        import json as _json

        run_json = self.run_context.work_dir / "run.json"
        if not run_json.exists():
            return

        try:
            with open(run_json, "r") as f:
                data = _json.load(f)
            persisted_results = data.get("context", {}).get("results", {})
            if persisted_results:
                # Merge: persisted results are available but won't overwrite fresh ones
                for key, value in persisted_results.items():
                    if key not in self.run_context.context.results:
                        self.run_context.context.results[key] = value
        except Exception:
            pass  # If loading fails, proceed without persisted results
    
    def _execute(self, **kwargs: Any) -> dict[str, dict[str, Any]]:
        """Execute workflow tasks.
        
        Args:
            **kwargs: Flat keyword arguments for task dependency injection
            
        Returns:
            Dict mapping task IDs to their output dicts
        """
        # Access internal context for execution
        context = self.run_context.context
        
        # Set context status to RUNNING
        context.status["run"] = RunStatus.RUNNING
        
        # Compute workflow graph to execute (filtered by phase if set)
        tasks_to_execute, in_degree = self._compute_subgraph(self._phases)
        
        # Initialize execution queue with tasks that have no dependencies in subgraph
        queue = [task_id for task_id in tasks_to_execute if in_degree[task_id] == 0]
        active_futures: dict[Future, str] = {}
        
        logger.info(f"Starting workflow {self.workflow.workflow_id}, executing {len(tasks_to_execute)} tasks")
        
        # Create executor
        self._thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        try:
            while queue or active_futures:
                # Submit ready tasks
                while queue:
                    task_id = queue.pop(0)
                    
                    # Skip if already cancelled
                    if is_failed_or_cancelled(context, task_id):
                        continue
                    
                    task = self._task_map[task_id]
                    context.tasks[task_id] = TaskStatus.RUNNING
                    
                    # Resolve inputs from upstream tasks and global kwargs
                    task_inputs = self._resolve_inputs(task_id, kwargs, context)
                    self._validate_inputs(task_id, task_inputs)
                    
                    # Submit for execution
                    future = self._thread_executor.submit(
                        self._execute_task,
                        task_id,
                        task,
                        task_inputs,
                    )
                    active_futures[future] = task_id
                
                # Wait for any completion
                if not active_futures:
                    break
                
                done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
                
                for future in done:
                    task_id = active_futures.pop(future)
                    
                    try:
                        result = future.result(timeout=self.task_timeout)
                        mark_succeeded(context, task_id, result)
                        logger.info(f"Task {task_id} succeeded")
                        
                        # Unlock dependents (only those in subgraph)
                        for neighbor in self._adj[task_id]:
                            if neighbor in tasks_to_execute:
                                in_degree[neighbor] -= 1
                                if in_degree[neighbor] == 0:
                                    queue.append(neighbor)
                    
                    except Exception as e:
                        self._handle_task_failure(task_id, e, tasks_to_execute, in_degree, context)
            
            # Return results
            return context.results
        
        finally:
            self._thread_executor.shutdown(wait=True)
            self._thread_executor = None
    
    def _after_execute(self, results: dict | None, success: bool, error: Exception | None = None):
        """Update context with execution results.
        
        Args:
            results: Execution results (None if failed)
            success: Whether execution succeeded
            error: Exception if failed
        """
        context = self.run_context.context
        
        if success:
            # Use RunContext method to bind results
            if results:
                for key, value in results.items():
                    self.run_context.set_result(key, value)
            
            # Check if workflow succeeded or failed based on task statuses
            has_failures = any(
                status in (TaskStatus.FAILED, TaskStatus.CANCELLED)
                for status in context.tasks.values()
            )
            
            if has_failures:
                context.status['run'] = RunStatus.FAILED
                logger.error(f"Workflow {self.workflow.workflow_id} failed")
            else:
                context.status['run'] = RunStatus.SUCCEEDED
                logger.info(f"Workflow {self.workflow.workflow_id} succeeded")
        else:
            context.status['run'] = RunStatus.FAILED
            if error:
                context.errors['engine'] = {
                    "type": type(error).__name__,
                    "message": str(error)
                }
            logger.error(f"Workflow execution failed with error: {error}")
        
        # Save updated context
        self.run_context._save_context()

    def _init_execution_metadata(self) -> None:
        """Initialize execution metadata in context."""
        context = self.run_context.context
        context.execution = {}

    def register_submitor(
        self,
        name: str,
        submitor: "SubmitorProtocol",
    ) -> None:
        """Register a submitor backend for task submission.

        Args:
            name: Backend name (e.g., "molq", "molq-slurm")
            submitor: Submitor instance implementing SubmitorProtocol

        Example:
            >>> from molq.submit import get_submitor
            >>> engine = WorkflowEngine(workflow)
            >>> submitor = get_submitor("local", "local")
            >>> engine.register_submitor("molq", submitor)
        """
        from molexp.workflow.submitor import SubmitorProtocol

        # Validate submitor implements protocol
        if not hasattr(submitor, "submit") or not hasattr(submitor, "query"):
            raise TypeError(
                f"Submitor must implement SubmitorProtocol (submit, query, cancel methods)"
            )

        self._submitors[name] = submitor
        logger.info(f"Registered submitor '{name}': {submitor}")

    def _get_submitor(self, backend: str) -> Any:
        """Get registered submitor for backend name.

        Args:
            backend: Backend name

        Returns:
            Submitor instance

        Raises:
            ValueError: If backend not registered
        """
        if backend not in self._submitors:
            raise ValueError(
                f"Submitor backend '{backend}' not configured. "
                f"Call engine.register_submitor('{backend}', submitor) first."
            )
        return self._submitors[backend]

    def _execute_task(
        self,
        task_id: str,
        task: Task,
        task_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a task via execute() or submit() based on submittable attribute.

        Args:
            task_id: Task identifier
            task: Task instance
            task_inputs: Resolved input arguments

        Returns:
            Task outputs dict
        """
        # Dispatch based on submittable attribute
        if task.submittable is None:
            # Local execution
            return task.execute(ctx=self.run_context, **task_inputs)
        else:
            # External submission
            backend = task.submittable
            submitor = self._get_submitor(backend)
            return self._handle_generator_submission(
                task, submitor, task_id, task_inputs
            )

    def _handle_generator_submission(
        self,
        task: Task,
        submitor: "SubmitorProtocol",
        task_id: str,
        task_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle generator-based task submission via submitor.

        Args:
            task: Task instance
            submitor: Submitor implementing SubmitorProtocol
            task_id: Task identifier
            task_inputs: Task inputs

        Returns:
            Task outputs dict

        Raises:
            TypeError: If yielded value is not a JobSpec
            RuntimeError: If submission fails
        """
        import time

        generator = task.submit(ctx=self.run_context, **task_inputs)
        result = None
        submitted_jobs = []  # Track all submitted jobs for this task

        try:
            job_spec = next(generator)
            while True:
                # Validate JobSpec (can be dict or Pydantic model)
                if not isinstance(job_spec, (dict, object)):
                    raise TypeError(
                        f"Task {task_id} yielded {type(job_spec)}, expected JobSpec"
                    )

                # Extract execution spec to check blocking behavior
                execution_spec = self._get_execution_spec(job_spec)
                block = execution_spec.get("block", True) if isinstance(execution_spec, dict) else getattr(execution_spec, "block", True)

                # Submit via submitor
                logger.info(f"Submitting job for task {task_id} via {task.submittable}")
                job_id = submitor.submit(job_spec)
                logger.info(f"Job {job_id} submitted for task {task_id}")

                # Track submission
                submitted_jobs.append(job_id)

                # Record submission in context.execution
                if "submissions" not in self.run_context.context.execution:
                    self.run_context.context.execution["submissions"] = {}
                if task_id not in self.run_context.context.execution["submissions"]:
                    self.run_context.context.execution["submissions"][task_id] = []
                self.run_context.context.execution["submissions"][task_id].append({
                    "job_id": job_id,
                    "backend": task.submittable,
                    "block": block,
                })

                # Monitor job if blocking is enabled
                if block:
                    self._monitor_job(task_id, job_id, submitor)

                # Send job_id back to generator
                job_spec = generator.send(job_id)

        except StopIteration as e:
            result = e.value

        if result is None:
            raise RuntimeError(
                f"Task {task_id} submit() generator did not return a result"
            )

        return result

    def _get_execution_spec(self, job_spec: Any) -> Any:
        """Extract execution spec from JobSpec."""
        if isinstance(job_spec, dict):
            return job_spec.get("execution", {})
        else:
            return getattr(job_spec, "execution", None)

    def _monitor_job(
        self,
        task_id: str,
        job_id: int,
        submitor: "SubmitorProtocol",
        interval: int = 2,
    ) -> None:
        """Monitor a submitted job until completion.

        Args:
            task_id: Task identifier
            job_id: Job ID to monitor
            submitor: Submitor to query status from
            interval: Polling interval in seconds
        """
        import time

        logger.info(f"Monitoring job {job_id} for task {task_id}")
        last_status = None

        while True:
            # Query job status
            statuses = submitor.query(job_id)
            job_status = statuses.get(job_id)

            if job_status is None:
                logger.warning(f"Job {job_id} not found in query results")
                break

            # Display status if changed
            current_status = job_status.status.name
            if current_status != last_status:
                self._display_job_status(task_id, job_id, job_status)
                last_status = current_status

            # Check if finished
            if job_status.is_finish:
                if job_status.status.name == "FAILED":
                    logger.error(f"Job {job_id} for task {task_id} failed")
                    raise RuntimeError(f"Job {job_id} failed")
                else:
                    logger.info(f"Job {job_id} for task {task_id} completed successfully")
                break

            time.sleep(interval)

    def _display_job_status(
        self,
        task_id: str,
        job_id: int,
        job_status: Any,
    ) -> None:
        """Display job status using molexp's logging system.

        Args:
            task_id: Task identifier
            job_id: Job ID
            job_status: JobStatus object from molq
        """
        status_name = job_status.status.name
        job_name = getattr(job_status, "name", "")

        # Color-coded status display
        status_colors = {
            "PENDING": "yellow",
            "RUNNING": "blue",
            "COMPLETED": "green",
            "FAILED": "red",
            "FINISHED": "green",
        }
        color = status_colors.get(status_name, "white")

        # Log with appropriate level
        if status_name == "FAILED":
            logger.error(f"Task {task_id} | Job {job_id} [{job_name}]: {status_name}")
        elif status_name in ("COMPLETED", "FINISHED"):
            logger.info(f"Task {task_id} | Job {job_id} [{job_name}]: {status_name}")
        else:
            logger.info(f"Task {task_id} | Job {job_id} [{job_name}]: {status_name}")

        # Also display to console using rich if available
        try:
            console.print(
                f"[bold]{task_id}[/bold] | Job {job_id} [{job_name}]: [{color}]{status_name}[/{color}]"
            )
        except Exception:
            pass  # Fallback to logger only if console fails

    def _resolve_inputs(
        self,
        task_id: str,
        global_kwargs: dict[str, Any],
        context: "Context",
    ) -> dict[str, Any]:
        """Resolve inputs for a task using explicit link mappings."""
        task_inputs: dict[str, Any] = {}
        
        # Get target task to check its input requirements
        target_task = self._task_map.get(task_id)
        if not target_task or not target_task.inputs:
            return task_inputs
        
        # 1. Pick from global kwargs based on task input names
        for input_name in target_task.inputs.keys():
            if input_name in global_kwargs:
                task_inputs[input_name] = global_kwargs[input_name]
        
        # 2. Collect outputs from upstream tasks via links
        for link in self.workflow.links:
            if link.target == task_id:
                source_task_id = link.source
                # Get outputs from source task results
                if source_task_id in context.results:
                    source_outputs = context.results[source_task_id]
                    # Mapping is guaranteed to exist (auto-generated by compiler if None)
                    if link.mapping:
                        for output_name, input_name in link.mapping.items():
                            if output_name not in source_outputs:
                                raise ValueError(
                                    f"Link mapping references missing output '{output_name}' "
                                    f"from task '{source_task_id}'"
                                )
                            task_inputs[input_name] = source_outputs[output_name]
        
        return task_inputs

    def _validate_inputs(self, task_id: str, task_inputs: dict[str, Any]) -> None:
        """Fail fast if required inputs are missing."""
        target_task = self._task_map.get(task_id)
        if not target_task or not target_task.inputs:
            return
        missing = [
            input_name
            for input_name in target_task.inputs.keys()
            if input_name not in task_inputs
        ]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(
                f"Task '{task_id}' missing required inputs: {missing_list}"
            )

    def _handle_task_failure(
        self,
        task_id: str,
        error: Exception,
        tasks_to_execute: set[str] | None = None,
        in_degree: dict[str, int] | None = None,
        context: "Context | None" = None,
    ) -> None:
        """Handle task failure and propagate to dependents."""
        if context is None:
            context = self.run_context.context
        
        mark_failed(context, task_id, error)
        
        # Print beautiful error message with full traceback
        print_task_error(task_id, context.errors[task_id])
        
        # Propagate failure to all dependents (only in subgraph if provided)
        self._propagate_failure(task_id, tasks_to_execute, context)

    def _propagate_failure(
        self,
        failed_task_id: str,
        tasks_to_execute: set[str] | None = None,
        context: "Context | None" = None,
    ) -> None:
        """Propagate failure to all dependent tasks."""
        if context is None:
            context = self.run_context.context
        
        queue = [failed_task_id]
        visited = set()
        
        while queue:
            task_id = queue.pop(0)
            if task_id in visited:
                continue
            visited.add(task_id)
            
            # Cancel all dependents (only in subgraph if provided)
            for dependent_id in self._adj.get(task_id, []):
                if tasks_to_execute is None or dependent_id in tasks_to_execute:
                    if not is_failed_or_cancelled(context, dependent_id):
                        mark_cancelled(context, dependent_id)
                        logger.warning(f"⚠️  Task {dependent_id} cancelled due to upstream failure")
                        queue.append(dependent_id)

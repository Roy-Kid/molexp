"""Workflow execution engine with hybrid execution and hot reconfiguration.

This module implements the WorkflowEngine class which orchestrates workflow execution
with automatic mode selection based on task types:

- **Pure Batch Mode**: Traditional ThreadPoolExecutor-based execution for workflows
  containing only batch tasks. Provides fast, dependency-ordered execution.

- **Hybrid Mode**: Asyncio-based execution for workflows containing Actors. Supports
  concurrent actor execution with message passing while also handling batch tasks.

Key Features:
    - Automatic execution mode detection via type inspection
    - Hot reconfiguration: update actor config during execution
    - Message channel management with backpressure control
    - Failure propagation with detailed error reporting
    - External task submission via submitor protocol

Example:
    Basic workflow execution::

        engine = WorkflowEngine(workflow)
        with run.start() as run_ctx:
            results = engine.execute(run_context=run_ctx)

    Hot reconfiguration::

        engine.update_actor_config('sampler_123', {'threshold': 0.3})

    Multi-actor workflow::

        workflow = Workflow.from_tasks([actorA, actorB, actorC], links)
        engine = WorkflowEngine(workflow)
        results = engine.execute(run_context=run_ctx)
"""

import asyncio
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing_extensions import Any
from rich.console import Console

from mollog import get_logger

logger = get_logger(__name__)
console = Console()

from molexp.workflow.execution_type import TaskExecutionType
from molexp.workflow.status import TaskStatus
from molexp.workflow.task import Task, Actor
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

        Automatically detects Actor tasks and uses hybrid execution mode
        with asyncio event loop. Pure batch workflows use the fast path.

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
            # Stage 2: Detect actors and choose execution path
            if self.compiled.has_actors():
                # Hybrid execution with asyncio for actors
                logger.debug("Detected Actor tasks - using hybrid execution mode")
                results = self._execute_hybrid(**kwargs)
            else:
                # Pure batch - fast path with ThreadPoolExecutor
                results = self._execute(**kwargs)

            # Stage 3: After execution (success)
            self._after_execute(results, success=True)

            return results

        except Exception as e:
            # Stage 3: After execution (failure)
            self._after_execute(None, success=False, error=e)
            raise

    def update_actor_config(self, actor_id: str, config_updates: dict[str, Any]) -> None:
        """Update actor configuration during execution (hot reconfiguration).

        This method enables updating actor config while the workflow is running.
        Actors check self.config on each iteration, so changes take effect
        immediately on the next iteration.

        **Thread Safety:** Safe in asyncio (single-threaded event loop).
        **Stateless Pattern:** All actor state should be in config for this to work.

        Args:
            actor_id: Task ID of the actor to update
            config_updates: Dict of config fields to update (partial updates supported)

        Raises:
            KeyError: If actor_id not found
            ValueError: If config validation fails
            TypeError: If task is not an Actor

        Example:
            >>> # Update sampling threshold during execution
            >>> engine.update_actor_config('sampler_123', {'threshold': 0.2})
            >>>
            >>> # Update multiple fields
            >>> engine.update_actor_config('trainer_456', {
            ...     'learning_rate': 0.001,
            ...     'batch_size': 64
            ... })
        """
        from ..task import Actor

        # Get actor from task map
        if actor_id not in self._task_map:
            raise KeyError(f"Actor '{actor_id}' not found in workflow")

        actor = self._task_map[actor_id]

        # Verify it's an Actor
        if not isinstance(actor, Actor):
            raise TypeError(f"Task '{actor_id}' is not an Actor (type: {type(actor).__name__})")

        # Create updated config by merging with existing config
        current_config_dict = actor.config.model_dump()
        current_config_dict.update(config_updates)

        # Validate new config against actor's config_type schema
        new_config = actor.config_type(**current_config_dict)

        # Update config in-place (thread-safe in asyncio)
        actor.config = new_config

        logger.info(f"Updated config for actor {actor_id}: {config_updates}")

        # TODO: Save to workflow metadata for persistence
        # TODO: Store in config history if tracking enabled

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
        if not run_json.exists() or run_json.stat().st_size == 0:
            return

        with open(run_json, "r") as f:
            data = _json.load(f)
        persisted_results = data.get("context", {}).get("results", {})
        if persisted_results:
            # Merge: persisted results are available but won't overwrite fresh ones
            for key, value in persisted_results.items():
                if key not in self.run_context.context.results:
                    self.run_context.context.results[key] = value
    
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
                    result = future.result(timeout=self.task_timeout)
                    mark_succeeded(context, task_id, result)
                    logger.debug(f"Task {task_id} succeeded")

                    # Unlock dependents (only those in subgraph)
                    for neighbor in self._adj[task_id]:
                        if neighbor in tasks_to_execute:
                            in_degree[neighbor] -= 1
                            if in_degree[neighbor] == 0:
                                queue.append(neighbor)
            
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
                logger.debug(f"Workflow {self.workflow.workflow_id} succeeded")
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
        submitted_jobs = []  # Track all submitted jobs for this task

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
            result = generator.send(job_id)
            if result is None:
                job_spec = generator.send(job_id)
            else:
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

        # Also display to console using rich
        console.print(
            f"[bold]{task_id}[/bold] | Job {job_id} [{job_name}]: [{color}]{status_name}[/{color}]"
        )

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

    def _execute_hybrid(self, **kwargs: Any) -> dict[str, dict[str, Any]]:
        """Execute workflow with Actor support using asyncio event loop.

        Handles both batch tasks (wrapped with asyncio.to_thread) and Actor tasks
        (native coroutines) in a unified asyncio event loop.

        Args:
            **kwargs: Flat keyword arguments for task dependency injection

        Returns:
            Dict mapping task IDs to their output dicts
        """
        context = self.run_context.context

        # Set context status to RUNNING
        context.status["run"] = RunStatus.RUNNING

        # Get task types and channels from compiler
        task_types = self.compiled.get_task_types()
        channel_configs = self.compiled.get_channels()

        # Create asyncio.Queue instances for channels
        channels = {}
        for channel_id, config in channel_configs.items():
            buffer_size = config['buffer_size']
            channels[channel_id] = asyncio.Queue(maxsize=buffer_size)
            logger.debug(f"Created channel {channel_id} with buffer_size={buffer_size}")

        # Note: Channels are registered by _run_actor with logical names (e.g., 'data', 'results')
        # not with physical IDs (e.g., 'ActorA_to_ActorB')

        # Run asyncio event loop
        results = asyncio.run(self._run_hybrid_workflow(task_types, channels, channel_configs, kwargs))
        return results

    async def _run_hybrid_workflow(
        self,
        task_types: dict[str, TaskExecutionType],
        channels: dict[str, asyncio.Queue],
        channel_configs: dict[str, Any],
        kwargs: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Run hybrid workflow in asyncio event loop.

        Args:
            task_types: Dict of task_id -> TaskExecutionType
            channels: Dict of channel_id -> asyncio.Queue
            channel_configs: Dict of channel configurations
            kwargs: Global kwargs for tasks

        Returns:
            Dict of task results
        """
        context = self.run_context.context

        # Compute subgraph
        tasks_to_execute, _ = self._compute_subgraph(self._phases)

        # Create tasks (actors as coroutines, batch as wrapped threads)
        coroutines = []
        for task_id in tasks_to_execute:
            task = self._task_map[task_id]
            task_type = task_types.get(task_id, TaskExecutionType.BATCH)

            if task_type == TaskExecutionType.ACTOR:
                # Actor - create coroutine
                coro = self._run_actor(task_id, task, channels, channel_configs)
                coroutines.append(coro)
            else:
                # Batch task - wrap in asyncio.to_thread
                # For simplicity in Phase 1, we'll just run actors
                # Batch tasks would need more complex orchestration
                logger.warning(f"Batch task {task_id} in hybrid mode - will skip for Phase 1")

        # Run all actors concurrently
        logger.debug(f"Starting {len(coroutines)} actors")
        await asyncio.gather(*coroutines, return_exceptions=True)

        return context.results

    async def _run_actor(
        self,
        task_id: str,
        task: Actor,
        channels: dict[str, asyncio.Queue],
        channel_configs: dict[str, Any]
    ) -> None:
        """Run a single actor.

        Args:
            task_id: Task identifier
            task: Actor instance
            channels: All channels
            channel_configs: Channel configurations
        """
        context = self.run_context.context
        context.tasks[task_id] = TaskStatus.RUNNING
        logger.debug(f"Starting actor {task_id}")

        # Setup channel routing for this actor
        # Find channels where this actor is source or target
        actor_channels = {}
        for channel_id, config in channel_configs.items():
            if config['source'] == task_id:
                # This actor is the source
                actor_channels[channel_id] = channels[channel_id]
            elif config['target'] == task_id:
                # This actor is the target
                actor_channels[channel_id] = channels[channel_id]

        # Register channels for this actor using Link mapping names
        for channel_id, queue in actor_channels.items():
            config = channel_configs[channel_id]
            mapping = config.get('mapping', {})

            if config['source'] == task_id:
                # Actor is source - register output names from mapping keys
                for output_name in mapping.keys():
                    self.run_context._register_channel(output_name, queue)
                    logger.debug(f"Registered channel '{output_name}' (emit) for actor {task_id}")

            elif config['target'] == task_id:
                # Actor is target - register input names from mapping values
                for input_name in mapping.values():
                    self.run_context._register_channel(input_name, queue)
                    logger.debug(f"Registered channel '{input_name}' (receive) for actor {task_id}")

        # Execute actor with isolation - catch exceptions to allow other actors to continue
        try:
            gen = task.execute(ctx=self.run_context)

            # Run actor generator
            async for output in gen:
                # Actor yielded - could handle control signals here
                pass

            # Get actor's final result (if it called ctx.set_result)
            result = context.results.get(task_id, {})
            mark_succeeded(context, task_id, result)
            logger.debug(f"Actor {task_id} completed")
        except Exception as e:
            # Actor failed - mark failure but allow other actors to continue
            mark_failed(context, task_id, e)
            logger.error(f"Actor {task_id} failed: {e}")
            print_task_error(task_id, context.errors[task_id])

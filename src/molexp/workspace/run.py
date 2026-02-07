"""Run entity and execution context with actor message passing support.

This module provides the Run and RunContext classes for workflow execution:

1. **Run**: Represents a single execution instance within an experiment. Tracks
   parameters, metadata, and execution state. Persisted via materialize().

2. **RunContext**: Primary execution context interface for tasks and actors. Provides:
   - Lifecycle management (__enter__/__exit__)
   - Result binding (set_result/get_result)
   - Checkpoint creation
   - **Message passing**: emit() and receive() for actor communication
   - **Channel management**: Dynamic channel registration and monitoring

Message Passing (Hybrid Mode):
    For workflows containing Actors, RunContext provides async message passing:
    - `await ctx.receive(channel)`: Receive message from named channel (blocks if empty)
    - `await ctx.emit(channel, msg)`: Send message to named channel (backpressure support)
    - `ctx.get_channel_depths()`: Monitor channel queue sizes

Example:
    Batch task execution::

        with run.start() as ctx:
            result = task.execute(ctx=ctx, input_value=42)
            ctx.set_result('task_id', result)

    Actor message passing::

        async def execute(self, ctx, **inputs):
            data = await ctx.receive('input_channel')
            result = self.process(data)
            await ctx.emit('output_channel', result)
            yield
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from .experiment import Experiment

from .base import _save_metadata, _load_metadata, _reconstruct
from .context import Context
from .metadata import RunMetadata, ExperimentMetadata, ProjectMetadata, WorkspaceMetadata
from .utils import generate_id

from molexp.workflow.status import TaskStatus as RunStatus


class RunContext:
    """Primary execution context interface.
    
    Encapsulates:
    - Pure Context data model (internal)
    - Lifecycle management (__enter__/__exit__)
    - Result binding methods
    - Checkpoint creation
    - Context serialization
    
    Users interact with RunContext, not Context directly.
    
    Attributes:
        run: The Run being executed (can be None for standalone mode)
        work_dir: Run working directory
        artifacts_dir: Directory for artifacts
        _context: Internal Context data model
    """
    
    def __init__(self, run: Run):
        self.run = run
        
        # Compute work_dir from hierarchy
        self.work_dir = (
            self.run.experiment.project.workspace.root / "projects" /
            self.run.experiment.project.id / "experiments" /
            self.run.experiment.id / "runs" / self.run.id
        )
        
        # Artifacts directory
        self.artifacts_dir = self.work_dir / "artifacts"
        
        # Internal: Pure Context BaseModel
        self._context = Context(
            run_id=self.run.id,
            experiment_id=self.run.experiment.id,
            project_id=self.run.experiment.project.id,
            work_dir=self.work_dir,
            artifacts_dir=self.artifacts_dir,
        )
        
        # Runtime state
        self._start_time = None

        # Actor message passing infrastructure
        self._channels: dict[str, Any] = {}  # channel_name -> asyncio.Queue
        self._channel_routing: dict[tuple[str, str], str] = {}  # (actor_id, channel_name) -> target_channel
    
    # ========== Lifecycle ==========
    
    def __enter__(self) -> "RunContext":
        """Initialize execution and save initial state."""
        # Create directory structure
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)

        # Load existing results from previous phases before overwriting
        self._load_existing_results()

        # Set status to running
        self.run.status = RunStatus.RUNNING
        self._start_time = datetime.now()

        # Serialize initial context to run.json
        self._save_context()

        return self  # Return RunContext, not Context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize execution and save final state.
        
        Status is determined by:
        1. Python exceptions (exc_type is not None)
        2. Workflow execution status (self._context.status)
        
        This ensures that workflow-level failures (task failures, cancellations)
        are properly reflected in the Run status, even when no exception is raised.
        """
        if exc_type is None:
            # No Python exception, check workflow execution status
            workflow_status = self._context.status.get('run')
            if workflow_status == RunStatus.FAILED:
                # Workflow failed (tasks failed or were cancelled)
                self.run.status = RunStatus.FAILED
                self.run.metadata.finished_at = datetime.now()
            else:
                # Success
                self.run.status = RunStatus.SUCCEEDED
                self.run.metadata.finished_at = datetime.now()
        else:
            # Python exception occurred
            self.run.status = RunStatus.FAILED
            self.run.metadata.finished_at = datetime.now()
            
            # Save error summary to metadata
            self.run.metadata.error = {
                "type": exc_type.__name__,
                "message": str(exc_val),
                "timestamp": datetime.now().isoformat(),
            }
            
            # Save full traceback to error files
            self._save_error_details(exc_type, exc_val, exc_tb)
        
        # Serialize final context to run.json
        self._save_context()
        
        return False  # Don't suppress exceptions
    
    # ========== Public API ==========
    
    def set_result(self, key: str, value: Any) -> None:
        """Bind a result to the context.
        
        Args:
            key: Result key
            value: Result value (will be serialized)
        """
        self._context.results[key] = value
    
    def get_result(self, key: str) -> Any:
        """Get a result from the context.
        
        Args:
            key: Result key
            
        Returns:
            Result value or None if not found
        """
        return self._context.results.get(key)
    
    def set_workflow(self, workflow: BaseModel | dict) -> None:
        """Set workflow data in context.

        Args:
            workflow: Workflow object (Pydantic model or dict)
        """
        if isinstance(workflow, BaseModel):
            self._context.workflow = workflow.model_dump(exclude={"_tasks"})
        elif isinstance(workflow, dict):
            self._context.workflow = workflow
        else:
            raise TypeError("Workflow must be Pydantic BaseModel or dict")
    
    def checkpoint(self, name: str | None = None) -> str:
        """Create checkpoint of current context state.
        
        Args:
            name: Optional checkpoint name
            
        Returns:
            Checkpoint ID
        """
        from .checkpoint import generate_checkpoint_id
        
        ckpt_id = generate_checkpoint_id()
        ckpt_dir = self.work_dir / ".ckpt"
        ckpt_dir.mkdir(exist_ok=True)
        
        ckpt_file = ckpt_dir / f"{ckpt_id}.json"
        with open(ckpt_file, 'w') as f:
            json.dump({
                "ckpt_id": ckpt_id,
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "context": self._context.model_dump(mode='json')
            }, f, indent=2, default=str)
        
        return ckpt_id
    
    def save_artifact(self, name: str, data: Any) -> Path:
        """Save an artifact to the artifacts directory.
        
        Args:
            name: Artifact filename
            data: Data to save (dict will be saved as JSON)
            
        Returns:
            Path to saved artifact
        """
        artifact_path = self.artifacts_dir / name
        
        if isinstance(data, dict):
            with open(artifact_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            # Handle other types as text
            with open(artifact_path, 'w') as f:
                f.write(str(data))
        
        return artifact_path
    
    # ========== Internal Methods ==========

    def _load_existing_results(self):
        """Load results from an existing run.json into the context.

        This enables cross-phase workflows where phase 2 needs outputs
        produced by phase 1. Called during __enter__ before the initial
        save overwrites run.json.
        """
        run_json = self.work_dir / "run.json"
        if not run_json.exists() or run_json.stat().st_size == 0:
            return

        with open(run_json, "r") as f:
            data = json.load(f)
        persisted = data.get("context", {}).get("results", {})
        for key, value in persisted.items():
            if key not in self._context.results:
                self._context.results[key] = value

    def _save_context(self):
        """Serialize context to run.json."""
        run_json = self.work_dir / "run.json"
        context_dict = self._context.model_dump(mode='json')

        with open(run_json, 'w') as f:
            json.dump({
                "id": self.run.id,
                "status": self.run.status,
                "parameters": self.run.parameters,
                "assets": self.run.assets,
                "created_at": self.run.metadata.created_at.isoformat(),
                "updated_at": datetime.now().isoformat(),
                "context": context_dict
            }, f, indent=2, default=str)
    
    def _save_error_details(self, exc_type, exc_val, exc_tb):
        """Save detailed error information to files."""
        # Format traceback
        tb_lines = traceback.format_exception(exc_type, exc_val, exc_tb)
        tb_text = ''.join(tb_lines)
        
        # Save as text in artifacts
        error_txt = self.artifacts_dir / "error.txt"
        with open(error_txt, 'w') as f:
            f.write(f"Error occurred at: {datetime.now().isoformat()}\n")
            f.write(f"Exception type: {exc_type.__name__}\n")
            f.write(f"Exception message: {str(exc_val)}\n\n")
            f.write("Traceback:\n")
            f.write(tb_text)
        
        # Save as JSON in artifacts
        error_json = self.artifacts_dir / "error.json"
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "type": exc_type.__name__,
            "message": str(exc_val),
            "traceback": tb_lines,
        }
        with open(error_json, 'w') as f:
            json.dump(error_data, f, indent=2)
    
    # ========== Internal Access (for Engine) ==========
    
    @property
    def context(self) -> Context:
        """Access internal Context (for engine use only).

        Note: Prefer using RunContext methods over direct context access.
        """
        return self._context

    # ========== Asset Registration ==========

    def register_asset(
        self,
        name: str,
        src: Path | str,
        action: str = "copy",
        meta: dict | None = None
    ):
        """Register an asset at experiment level.
        
        Assets are important outputs that may be reused by other runs.
        
        Args:
            name: Asset name
            src: Source path
            action: Import action (copy, move, symlink, hardlink)
            meta: Optional metadata
            
        Returns:
            Created Asset
        """
        asset = self.run.experiment.assets.import_asset(
            name=name,
            src=src,
            action=action,
            meta=meta or {}
        )
        
        # Track reference in run.assets
        if "assets" not in self.run.assets:
            self.run.assets["assets"] = {}
        self.run.assets["assets"][name] = asset.id
        self.run.save()
        
        return asset
    
    def get_asset(self, name: str, scope: str = "project"):
        """Get an asset by name from a specific scope.

        Args:
            name: Asset name to look up
            scope: Scope to search in ('experiment', 'project', or 'workspace')

        Returns:
            Asset object if found, None otherwise

        Raises:
            ValueError: If scope is not recognized
        """
        from .asset import AssetLibrary

        if scope == "experiment":
            return self.run.experiment.assets.get_asset(name)
        elif scope == "project":
            return self.run.experiment.project.assets.get_asset(name)
        elif scope == "workspace":
            return self.run.experiment.project.workspace.assets.get_asset(name)
        else:
            raise ValueError(
                f"Unknown scope: {scope!r}. Use 'experiment', 'project', or 'workspace'."
            )

    def find_asset(self, name: str):
        """Find an asset by name, searching upward through scopes.

        Search order: experiment → project → workspace.
        Returns the first match found.

        Args:
            name: Asset name to look up

        Returns:
            Asset object if found, None otherwise
        """
        for scope in ("experiment", "project", "workspace"):
            asset = self.get_asset(name, scope=scope)
            if asset is not None:
                return asset
        return None

    def get_artifact_path(self, name: str) -> Path:
        """Get path to an artifact."""
        return self.artifacts_dir / name

    # ========== Actor Message Passing API ==========

    async def receive(self, channel: str) -> Any:
        """Receive one message from named channel (for Actors).

        Blocks if channel is empty until a message arrives.
        Channel names are dynamic and routed by workflow Links.

        Args:
            channel: Channel name to receive from

        Returns:
            Message from the channel

        Raises:
            KeyError: If channel does not exist (not connected in workflow)

        Example:
            >>> async def execute(self, ctx, **inputs):
            ...     data = await ctx.receive('data')
            ...     result = self.process(data)
            ...     await ctx.emit('results', result)
        """
        if channel not in self._channels:
            raise KeyError(
                f"Channel '{channel}' not found. Available channels: {list(self._channels.keys())}"
            )

        queue = self._channels[channel]
        return await queue.get()

    async def emit(self, channel: str, message: Any) -> None:
        """Send message to named channel (for Actors).

        Blocks if channel is full until space is available (backpressure).
        Channel names are dynamic and routed by workflow Links.

        If channel doesn't exist (no consumer connected), message is silently dropped.

        Args:
            channel: Channel name to send to
            message: Message to send

        Example:
            >>> async def execute(self, ctx, **inputs):
            ...     data = await ctx.receive('data')
            ...     result = self.process(data)
            ...     await ctx.emit('results', result)
        """
        if channel not in self._channels:
            # Channel not connected - drop message silently
            # This allows actors to produce output without requiring consumers
            return

        queue = self._channels[channel]
        await queue.put(message)

    def get_channel_depths(self) -> dict[str, int]:
        """Get current depth (qsize) of all channels.

        Useful for monitoring backpressure and debugging.

        Returns:
            Dict mapping channel names to current queue sizes

        Example:
            >>> depths = ctx.get_channel_depths()
            >>> print(depths)  # {'data': 45, 'results': 12}
        """
        return {
            name: queue.qsize()
            for name, queue in self._channels.items()
        }

    def _register_channel(self, name: str, queue: Any) -> None:
        """Register a channel (for engine use only).

        Args:
            name: Channel name
            queue: asyncio.Queue instance
        """
        self._channels[name] = queue

    @classmethod
    def from_run_dir(cls, path: Path) -> RunContext:
        """Reconstruct a RunContext from an existing run directory.

        Reads run.json and rebuilds the full Workspace → Project → Experiment → Run
        hierarchy by walking the filesystem upward. Parent entities are reconstructed
        lazily using metadata files found on disk.

        Args:
            path: Path to the run directory (must contain run.json)

        Returns:
            RunContext with restored hierarchy and context data

        Raises:
            FileNotFoundError: If run.json or parent metadata files are missing
        """
        path = Path(path).resolve()
        run_json_path = path / "run.json"
        if not run_json_path.exists():
            raise FileNotFoundError(f"run.json not found in {path}")

        # Read run.json (RunContext format: {id, status, parameters, assets, context: {...}})
        with open(run_json_path, "r") as f:
            run_data = json.load(f)

        # Walk directory hierarchy upward:
        # run_dir / .. = runs/ / .. = experiment_dir / .. = experiments/ / .. = project_dir / .. = projects/ / .. = workspace_root
        run_dir = path
        runs_dir = run_dir.parent          # .../runs/
        experiment_dir = runs_dir.parent    # .../experiments/{experiment_id}/
        experiments_dir = experiment_dir.parent  # .../experiments/
        project_dir = experiments_dir.parent     # .../projects/{project_id}/
        projects_dir = project_dir.parent        # .../projects/
        workspace_root = projects_dir.parent     # workspace root

        # Reconstruct Workspace
        from .workspace import Workspace
        ws_meta_path = workspace_root / "workspace.json"
        if not ws_meta_path.exists():
            raise FileNotFoundError(
                f"workspace.json not found at {ws_meta_path}. "
                f"Cannot reconstruct workspace hierarchy from {path}"
            )
        ws_meta = _load_metadata(WorkspaceMetadata, ws_meta_path)
        workspace = _reconstruct(Workspace, {
            "root": workspace_root,
            "metadata": ws_meta,
            "_assets_lib": None,
        })

        # Reconstruct Project
        from .project import Project
        proj_meta_path = project_dir / "project.json"
        if not proj_meta_path.exists():
            raise FileNotFoundError(
                f"project.json not found at {proj_meta_path}. "
                f"Cannot reconstruct project from {path}"
            )
        proj_meta = _load_metadata(ProjectMetadata, proj_meta_path)
        project = _reconstruct(Project, {
            "workspace": workspace,
            "metadata": proj_meta,
            "_assets_lib": None,
        })

        # Reconstruct Experiment
        from .experiment import Experiment
        exp_meta_path = experiment_dir / "experiment.json"
        if not exp_meta_path.exists():
            raise FileNotFoundError(
                f"experiment.json not found at {exp_meta_path}. "
                f"Cannot reconstruct experiment from {path}"
            )
        exp_meta = _load_metadata(ExperimentMetadata, exp_meta_path)
        experiment = _reconstruct(Experiment, {
            "project": project,
            "metadata": exp_meta,
            "_assets_lib": None,
        })

        # Reconstruct Run
        run_meta = RunMetadata(
            id=run_data["id"],
            parameters=run_data.get("parameters", {}),
            assets=run_data.get("assets", {}),
            created_at=run_data.get("created_at", datetime.now().isoformat()),
            updated_at=run_data.get("updated_at", datetime.now().isoformat()),
            status=run_data.get("status", "pending"),
        )
        run = _reconstruct(Run, {
            "experiment": experiment,
            "metadata": run_meta,
        })

        # Build RunContext without calling __init__ (to avoid recomputing work_dir)
        ctx = cls.__new__(cls)
        ctx.run = run
        ctx.work_dir = run_dir
        ctx.artifacts_dir = run_dir / "artifacts"
        ctx._start_time = None

        # Restore context from run.json if available
        context_data = run_data.get("context", {})
        ctx._context = Context(
            run_id=run.id,
            experiment_id=experiment.id,
            project_id=project.id,
            work_dir=run_dir,
            artifacts_dir=run_dir / "artifacts",
            tasks=context_data.get("tasks", {}),
            results=context_data.get("results", {}),
            status=context_data.get("status", {}),
            errors=context_data.get("errors", {}),
            workflow=context_data.get("workflow"),
            execution=context_data.get("execution", {}),
        )

        return ctx


class Run:
    """Single execution instance with runtime behavior.
    
    Construction is side-effect free and auto-generates metadata.
    Filesystem operations happen explicitly via materialize().
    
    Example:
        >>> # User provides only what they care about
        >>> run = Run(
        ...     experiment=experiment,
        ...     parameters={"lr": 0.001, "batch_size": 32}
        ... )
        >>> 
        >>> # System fields auto-generated
        >>> assert run.id  # UUID auto-generated
        >>> assert run.metadata.created_at  # Timestamp auto-generated
        >>> 
        >>> # No filesystem side effects yet
        >>> # Explicitly materialize to disk
        >>> run.materialize()  # NOW directories/files are created
        >>> 
        >>> # Execute workflow with context manager
        >>> with run.context() as ctx:
        ...     # Workflow execution
        ...     engine = WorkflowEngine(workflow)
        ...     engine.execute(workflow, ctx)
    """
    
    def __init__(
        self,
        experiment: Experiment,
        parameters: dict[str, Any] | None = None,
        assets: dict[str, Any] | None = None,
        id: str | None = None,
    ):
        """Initialize run with user inputs and auto-generate metadata.

        Args:
            experiment: Parent experiment (runtime dependency)
            parameters: Execution parameters (user input)
            assets: Input/output assets (user input)
            id: Optional custom run ID (if None, auto-generates UUID)
        """
        self.experiment = experiment

        # Auto-generate metadata with system fields
        self.metadata = RunMetadata(
            id=id or generate_id(),  # Auto-generated UUID if not provided
            parameters=parameters or {},
            assets=assets or {},
            created_at=datetime.now(),  # Auto-generated timestamp
            updated_at=datetime.now(),
            status=RunStatus.PENDING,
        )
    
    # Property proxies for convenient access to metadata fields
    
    @property
    def id(self) -> str:
        """Run identifier."""
        return self.metadata.id
    
    @property
    def parameters(self) -> dict[str, Any]:
        """Execution parameters."""
        return self.metadata.parameters
    
    @property
    def assets(self) -> dict[str, Any]:
        """Input/output assets."""
        return self.metadata.assets
    
    @property
    def status(self) -> str:
        """Run execution status."""
        return self.metadata.status
    
    @status.setter
    def status(self, value: str) -> None:
        """Update run status."""
        self.metadata.status = value
        self.metadata.updated_at = datetime.now()
    
    def materialize(self) -> None:
        """Create filesystem structure and persist metadata.

        This is the explicit side-effect method that:
        - Creates run directory
        - Writes metadata JSON file
        - Initializes any subdirectories
        """
        run_dir = (
            self.experiment.workspace.root / "projects" /
            self.experiment.project.id / "experiments" /
            self.experiment.id / "runs" / self.id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        _save_metadata(self.metadata, run_dir / "run.json")
    
    def save(self) -> None:
        """Save updated metadata to disk.

        Updates the metadata file without recreating directory structure.
        """
        run_dir = (
            self.experiment.workspace.root / "projects" /
            self.experiment.project.id / "experiments" /
            self.experiment.id / "runs" / self.id
        )
        self.metadata.updated_at = datetime.now()
        _save_metadata(self.metadata, run_dir / "run.json")
    
    @property
    def ctx(self) -> RunContext:
        """Lazily-created RunContext, available without entering the context manager."""
        if not hasattr(self, '_run_context') or self._run_context is None:
            self._run_context = RunContext(self)
        return self._run_context

    @property
    def work_dir(self) -> Path:
        """Run working directory."""
        return self.ctx.work_dir

    def start(self) -> RunContext:
        """Start the run and return execution context manager.

        Usage:
            >>> with run.start() as ctx:
            ...     engine.execute(run_context=ctx, ...)

        Returns:
            RunContext: Context manager for run execution
        """
        return self.ctx
    

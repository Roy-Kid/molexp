"""``Mode`` — harness multi-stage orchestration base, executed eagerly.

A ``Mode`` declares an ordered list of :class:`Stage`s and runs them
**eagerly, one task at a time** against a ``workspace.Run``: each stage runs
through the shared :func:`~molexp.harness.core.stage_task.run_stage_bracketed`
audit bracket, its produced artifact id is persisted to a per-run completion
ledger, and the next stage runs. That eager, task-by-task model is what makes
caching and resume fall out cleanly:

- **Caching** — a stage whose ``name`` is already in the run's completion
  ledger (same ``Mode`` + same ``user_input``) is skipped; its prior artifact
  is reused. So re-running an identical pipeline re-invokes no stage bodies.
- **Resume** — because the ledger is updated after *each* stage, a pipeline
  that fails mid-way leaves the completed stages recorded; re-running skips
  them and resumes from the failed stage.

Contrast with :class:`molexp.agent.AgentLoop`: that is an LLM-conversation
coroutine streaming ``AgentEvent`` to a sink; ``harness.Mode`` is a multi-stage
experiment-orchestration mode. The agent side was renamed loop → so the
``Mode`` name now belongs unambiguously to the harness layer.

The eager loop here drives stages directly through the harness audit helper.
A future workflow-engine *eager execution mode* (a clean task-by-task public
stepper on ``molexp.workflow`` — its current ``iter()`` is a raw per-level
passthrough) can back this loop without changing the ``Mode`` contract; that
is deferred to a workflow-layer spec.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.core.stage_task import run_stage_bracketed
from molexp.harness.schemas import ArtifactRef, ModeResult
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore
from molexp.workspace.utils import derive_execution_id

if TYPE_CHECKING:
    from pathlib import Path

    from molexp.harness.gateways.gateway import AgentGateway

__all__ = ["Mode"]


class Mode(ABC):
    """Base class for harness multi-stage orchestration modes.

    Subclasses pin :attr:`name` and implement :meth:`stages`; the base owns
    eager task-by-task execution, the per-run completion ledger (caching +
    resume), and :class:`ModeResult` assembly.
    """

    name: ClassVar[str]

    @abstractmethod
    def stages(self, user_input: Any) -> list[Stage]:  # noqa: ANN401 — user_input is mode-defined
        """Return the ordered stages for ``user_input`` (linear dependency chain)."""
        raise NotImplementedError

    async def run(
        self,
        *,
        run: Any,  # noqa: ANN401 — workspace.Run (imported lazily-free; duck-typed run_dir/id)
        user_input: Any,  # noqa: ANN401
        gateway: AgentGateway | None = None,
    ) -> ModeResult:
        """Run the declared stages eagerly on ``run`` and return a :class:`ModeResult`.

        Args:
            run: The ``workspace.Run`` the pipeline executes under (provides
                ``run_dir`` for the stores + completion ledger and ``id``).
            user_input: The mode's input (e.g. a natural-language draft); part
                of the completion-ledger cache key.
            gateway: Optional :class:`AgentGateway` wired into the
                ``HarnessRunContext`` for LLM-backed stages.

        Returns:
            A :class:`ModeResult` carrying the per-stage artifacts + the final.

        Raises:
            ValueError: If :meth:`stages` returns an empty list.
            StageExecutionError: If a stage fails (completed stages remain in
                the ledger so a re-run resumes from the failed stage).
        """
        stages = list(self.stages(user_input))
        if not stages:
            raise ValueError(f"Mode[{self.name!r}].stages returned no stages")

        ctx = self._build_ctx(run, gateway)
        ledger_path = self._ledger_path(run, user_input)
        completed: dict[str, str] = self._load_ledger(ledger_path)

        for stage in stages:
            if stage.name in completed:
                continue  # cache hit / resume — skip the stage body
            ref = await run_stage_bracketed(ctx, stage)
            completed[stage.name] = ref.id
            self._write_ledger(ledger_path, completed)

        stage_artifacts: tuple[ArtifactRef, ...] = tuple(
            ctx.artifact_store.get_ref(completed[stage.name]) for stage in stages
        )
        return ModeResult(
            mode_name=self.name,
            run_id=run.id,
            execution_id=derive_execution_id(run.id, run.run_dir / "executions"),
            stage_artifacts=stage_artifacts,
            final_artifact=stage_artifacts[-1] if stage_artifacts else None,
        )

    # ----------------------------------------------------------- internals

    def _build_ctx(self, run: Any, gateway: AgentGateway | None) -> HarnessRunContext:  # noqa: ANN401
        artifact_store = FileArtifactStore(root=run.run_dir / "artifacts")
        db_path = run.run_dir / "harness.sqlite"
        return HarnessRunContext(
            run_id=run.id,
            workspace_root=run.run_dir,
            artifact_store=artifact_store,
            event_log=SQLiteEventLog(path=db_path),
            provenance_store=SQLiteProvenanceStore(path=db_path, artifact_store=artifact_store),
            agent_gateway=gateway,
        )

    def _ledger_path(self, run: Any, user_input: Any) -> Path:  # noqa: ANN401
        digest = hashlib.sha256(
            json.dumps(user_input, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return run.run_dir / ".mode_ledger" / f"{self.name}-{digest}.json"

    @staticmethod
    def _load_ledger(path: Path) -> dict[str, str]:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def _write_ledger(path: Path, completed: dict[str, str]) -> None:
        from molexp.workspace import atomic_write_json

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, completed)

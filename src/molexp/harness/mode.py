"""``Mode`` — harness multi-stage orchestration base, executed eagerly.

A ``Mode`` declares an ordered list of :class:`Stage`s and runs them
**eagerly, one task at a time** against a ``workspace.Run``: each stage runs
through the shared
:func:`~molexp.harness.core.stage_runner.run_stage_bracketed` audit bracket,
its produced artifact id is persisted to a per-run completion ledger, and the
next stage runs. That eager, task-by-task model is what makes caching and
resume fall out cleanly:

- **Caching** — a stage is skipped (its prior artifact reused) only when the
  ledger holds an entry for the same ``Mode`` + same ``user_input`` whose
  recorded :func:`~molexp.harness.core.fingerprint.stage_fingerprint` matches
  the stage's *current* code and whose artifact is still present in the
  store. Re-running an identical pipeline re-invokes no stage bodies.
- **Resume** — because the ledger is updated after *each* stage, a pipeline
  that fails mid-way leaves the completed stages recorded; re-running skips
  them and resumes from the failed stage.
- **Verified, never stale** — an entry whose fingerprint mismatches (stage
  code changed), whose artifact is gone, or which predates fingerprinting
  (legacy ledgers) is dropped with a warning and the stage recomputes.
  Mirrors the workflow layer's seed-validation law: warn + recompute, never
  silently reuse, never error.

Contrast with :class:`molexp.agent.AgentLoop`: that is an LLM-conversation
coroutine streaming ``AgentEvent`` to a sink; ``harness.Mode`` is a
multi-stage experiment-orchestration mode. "Mode" belongs to the harness
layer; "Loop" to the agent layer.

The eager loop drives stages directly through the harness audit bracket —
the harness pipeline is linear by design and does not run on the
``molexp.workflow`` engine. If a future mode ever needs DAG-shaped stages,
that becomes a workflow-layer spec; the ``Mode`` contract here stays as-is.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict

from mollog import get_logger

from molexp.harness.core.fingerprint import stage_fingerprint
from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.core.stage_runner import run_stage_bracketed
from molexp.harness.errors import ArtifactNotFoundError
from molexp.harness.schemas import ArtifactRef, ModeResult
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore
from molexp.workspace.utils import derive_execution_id

if TYPE_CHECKING:
    from pathlib import Path

    from molexp.harness.gateways.gateway import AgentGateway

__all__ = ["Mode"]

_LOG = get_logger(__name__)


class _LedgerEntry(TypedDict):
    """One completed-stage record: produced artifact + producing code identity.

    ``fingerprint`` is None for entries loaded from pre-fingerprint ledgers —
    such entries are unverifiable and always recompute.
    """

    artifact: str
    fingerprint: str | None


class Mode(ABC):
    """Base class for harness multi-stage orchestration modes.

    Subclasses pin :attr:`name` and implement :meth:`stages`; the base owns
    eager task-by-task execution, the per-run completion ledger (verified
    caching + resume), and :class:`ModeResult` assembly.
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
        completed: dict[str, _LedgerEntry] = self._load_ledger(ledger_path)

        for stage in stages:
            fingerprint = stage_fingerprint(stage)
            if self._entry_valid(ctx, completed.get(stage.name), stage.name, fingerprint):
                continue  # verified cache hit / resume — skip the stage body
            completed.pop(stage.name, None)
            ref = await run_stage_bracketed(ctx, stage)
            completed[stage.name] = {"artifact": ref.id, "fingerprint": fingerprint}
            self._write_ledger(ledger_path, completed, run_id=run.id)

        stage_artifacts: tuple[ArtifactRef, ...] = tuple(
            ctx.artifact_store.get_ref(completed[stage.name]["artifact"]) for stage in stages
        )
        return ModeResult(
            mode_name=self.name,
            run_id=run.id,
            execution_id=derive_execution_id(run.id, run.run_dir / "executions"),
            stage_artifacts=stage_artifacts,
            final_artifact=stage_artifacts[-1] if stage_artifacts else None,
        )

    # ----------------------------------------------------------- internals

    def _entry_valid(
        self,
        ctx: HarnessRunContext,
        entry: _LedgerEntry | None,
        stage_name: str,
        fingerprint: str,
    ) -> bool:
        """Return True iff ``entry`` may be reused for the current stage code.

        Three drop conditions, each logged at warning level so the recompute
        is visible: no recorded fingerprint (pre-fingerprint ledger),
        fingerprint mismatch (stage code changed), artifact missing from the
        store. Dropping recomputes the stage — never errors, never reuses.
        """
        if entry is None:
            return False
        recorded = entry.get("fingerprint")
        if recorded is None:
            _LOG.warning(
                f"[mode:{self.name}] ledger entry for stage {stage_name!r} predates "
                "fingerprinting — recomputing"
            )
            return False
        if recorded != fingerprint:
            _LOG.warning(
                f"[mode:{self.name}] stage {stage_name!r} code changed since the ledger "
                "entry was written — recomputing"
            )
            return False
        artifact_id = entry["artifact"]
        try:
            ctx.artifact_store.get_ref(artifact_id)
        except ArtifactNotFoundError:
            _LOG.warning(
                f"[mode:{self.name}] stage {stage_name!r} ledger artifact {artifact_id!r} "
                "is gone from the store — recomputing"
            )
            return False
        return True

    def _build_ctx(self, run: Any, gateway: AgentGateway | None) -> HarnessRunContext:  # noqa: ANN401
        artifact_store = FileArtifactStore(root=run.run_dir / "artifacts")
        db_path = run.run_dir / "harness.sqlite"
        return HarnessRunContext(
            run_id=run.id,
            workspace_root=run.run_dir,
            artifact_store=artifact_store,
            event_log=SQLiteEventLog(path=db_path),
            lineage_store=SQLiteArtifactLineageStore(path=db_path, artifact_store=artifact_store),
            agent_gateway=gateway,
        )

    def _ledger_path(self, run: Any, user_input: Any) -> Path:  # noqa: ANN401
        digest = hashlib.sha256(
            json.dumps(user_input, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return run.run_dir / ".mode_ledger" / f"{self.name}-{digest}.json"

    @staticmethod
    def _load_ledger(path: Path) -> dict[str, _LedgerEntry]:
        """Return the ``{stage_name: entry}`` map from ``path``.

        Accepts all historical ledger shapes — the current fingerprinted
        document (``stages`` values are ``{"artifact": ..., "fingerprint":
        ...}`` dicts), the previous self-describing document (``stages``
        values are bare artifact-id strings), and the original flat
        ``{stage_name: artifact_id}`` mapping. Entries without a fingerprint
        are normalized to ``fingerprint=None`` so :meth:`_entry_valid` drops
        them (recompute) instead of trusting an unverifiable artifact; an
        entry with no usable artifact id at all is dropped outright.
        """
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        stages = raw.get("stages", raw)
        if not isinstance(stages, dict):
            return {}
        normalized: dict[str, _LedgerEntry] = {}
        for name, value in stages.items():
            if isinstance(value, dict):
                artifact = value.get("artifact")
                fingerprint = value.get("fingerprint")
                if isinstance(artifact, str):
                    normalized[name] = {
                        "artifact": artifact,
                        "fingerprint": fingerprint if isinstance(fingerprint, str) else None,
                    }
            elif isinstance(value, str):
                normalized[name] = {"artifact": value, "fingerprint": None}
        return normalized

    def _write_ledger(self, path: Path, completed: dict[str, _LedgerEntry], *, run_id: str) -> None:
        """Persist the completion ledger as a self-describing document.

        Besides the per-stage ``{"artifact": ..., "fingerprint": ...}``
        entries the ledger names the ``workspace.Run`` (``run_id``) and the
        mode that produced it — the Run-side pointer into the harness
        artifact world: anything reading a run directory can discover which
        pipeline ran, which artifact each stage produced, and which stage
        code produced it (the artifacts themselves live under
        ``run_dir/artifacts``, events + lineage in ``run_dir/harness.sqlite``).
        """
        from molexp.workspace import atomic_write_json

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"run_id": run_id, "mode": self.name, "stages": completed})

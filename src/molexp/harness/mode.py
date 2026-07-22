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
- **Rejected, never pinned** — when a stage fails with a *persisted* failure
  (:class:`StagePersistedFailureError`, i.e. a validation report whose
  ``parent_ids`` name the rejected artifact(s)), the ledger entries for the
  nearest recorded producers of those artifacts — and every entry at a later
  pipeline position — are evicted from the ledger file before the failure
  propagates. Without this, resume would skip the producer forever while the
  deterministic validator re-rejects its pinned artifact: a permanent
  deadlock. Re-running upstream stages is acceptable; re-failing on the same
  rejected artifact is not.

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
from molexp.harness.core.stage_runner import run_stage_bracketed, take_stage_duration
from molexp.harness.errors import ArtifactNotFoundError, StagePersistedFailureError
from molexp.harness.schemas import ModeResult, PlanArtifactRef
from molexp.harness.schemas.mode_result import StageTiming
from molexp.harness.store.approval_store import SQLiteApprovalStore
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore
from molexp.workspace.utils import derive_execution_id

if TYPE_CHECKING:
    from pathlib import Path

    from molexp.harness.gateways.gateway import AgentGateway
    from molexp.harness.registry.capability_registry import CapabilityRegistry

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
        capability_registry: CapabilityRegistry | None = None,
    ) -> ModeResult:
        """Run the declared stages eagerly on ``run`` and return a :class:`ModeResult`.

        Args:
            run: The ``workspace.Run`` the pipeline executes under (provides
                ``run_dir`` for the stores + completion ledger and ``id``).
            user_input: The mode's input (e.g. a natural-language draft); part
                of the completion-ledger cache key.
            gateway: Optional :class:`AgentGateway` wired into the
                ``HarnessRunContext`` for LLM-backed stages.
            capability_registry: Optional :class:`CapabilityRegistry` wired into
                the ``HarnessRunContext``. When present, ``ValidateBoundWorkflow``
                grounds binding against it (unknown-capability / call-shape /
                backend checks); when ``None`` those checks are skipped.

        Returns:
            A :class:`ModeResult` carrying the per-stage artifacts + the final.

        Raises:
            ValueError: If :meth:`stages` returns an empty list.
            StageExecutionError: If a stage fails. Completed stages remain in
                the ledger so a re-run resumes from the failed stage — except
                the producers of an artifact a validator *rejected*
                (:class:`StagePersistedFailureError`), which are evicted so
                the re-run regenerates the rejected artifact instead of
                deadlocking on it (see :meth:`_evict_rejected_producers`).
        """
        stages = list(self.stages(user_input))
        if not stages:
            raise ValueError(f"Mode[{self.name!r}].stages returned no stages")

        ctx = self._build_ctx(run, gateway, capability_registry)
        ledger_path = self._ledger_path(run, user_input)
        completed: dict[str, _LedgerEntry] = self._load_ledger(ledger_path)
        timings: list[StageTiming] = []

        for stage in stages:
            fingerprint = stage_fingerprint(stage)
            if self._entry_valid(ctx, completed.get(stage.name), stage.name, fingerprint):
                timings.append(
                    StageTiming(stage=stage.name, duration_s=0.0, status="skipped")
                )
                _LOG.info(f"stage skip name={stage.name} reason=ledger_hit")
                continue  # verified cache hit / resume — skip the stage body
            completed.pop(stage.name, None)
            try:
                ref = await run_stage_bracketed(ctx, stage)
            except StagePersistedFailureError as exc:
                # A validator rejected an upstream artifact: evict its ledger
                # producers so the next run regenerates instead of deadlocking
                # on the pinned rejected artifact (see module docstring).
                if self._evict_rejected_producers(
                    ctx,
                    stages,
                    completed,
                    failed_stage=stage.name,
                    rejected_ref=exc.persisted_ref,
                ):
                    self._write_ledger(ledger_path, completed, run_id=run.id)
                # Best-effort duration for the failed stage (may be absent).
                fail_dt = take_stage_duration() or 0.0
                timings.append(
                    StageTiming(stage=stage.name, duration_s=fail_dt, status="failed")
                )
                self._log_timing_summary(timings)
                raise
            except Exception:
                fail_dt = take_stage_duration() or 0.0
                if fail_dt:
                    timings.append(
                        StageTiming(stage=stage.name, duration_s=fail_dt, status="failed")
                    )
                self._log_timing_summary(timings)
                raise
            dt = take_stage_duration() or 0.0
            timings.append(StageTiming(stage=stage.name, duration_s=dt, status="ok"))
            completed[stage.name] = {"artifact": ref.id, "fingerprint": fingerprint}
            self._write_ledger(ledger_path, completed, run_id=run.id)

        stage_artifacts: tuple[PlanArtifactRef, ...] = tuple(
            ctx.artifact_store.get_ref(completed[stage.name]["artifact"]) for stage in stages
        )
        result = ModeResult(
            mode_name=self.name,
            run_id=run.id,
            execution_id=derive_execution_id(run.id, run.run_dir / "executions"),
            stage_artifacts=stage_artifacts,
            final_artifact=stage_artifacts[-1] if stage_artifacts else None,
            stage_timings=tuple(timings),
        )
        self._log_timing_summary(timings)
        return result

    def _log_timing_summary(self, timings: list[StageTiming]) -> None:
        """Log stages sorted by duration so operators see the hot path.

        Includes ``ok`` *and* ``failed`` stages (skipped stay at 0 and are
        listed last). A failed sequential codegen that burned 30 min must
        appear in the summary — not vanish because status != ok.
        """
        if not timings:
            return
        timed = [t for t in timings if t.status in ("ok", "failed")]
        total = sum(t.duration_s for t in timed)
        skipped = sum(1 for t in timings if t.status == "skipped")
        failed = sum(1 for t in timings if t.status == "failed")
        _LOG.info(
            f"mode timing summary mode={self.name} stages={len(timings)} "
            f"ok={sum(1 for t in timings if t.status == 'ok')} "
            f"failed={failed} skipped={skipped} total_s={total:.2f}"
        )
        for t in sorted(timed, key=lambda x: x.duration_s, reverse=True):
            pct = (100.0 * t.duration_s / total) if total > 0 else 0.0
            _LOG.info(
                f"  timing stage={t.stage} status={t.status} "
                f"duration_s={t.duration_s:.2f} pct={pct:.0f}"
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

    def _evict_rejected_producers(
        self,
        ctx: HarnessRunContext,
        stages: list[Stage],
        completed: dict[str, _LedgerEntry],
        *,
        failed_stage: str,
        rejected_ref: PlanArtifactRef,
    ) -> bool:
        """Evict the ledger entries pinning artifacts a validator just rejected.

        ``rejected_ref`` is the persisted failure report; its ``parent_ids``
        name the artifact(s) the failed stage consumed and rejected. From
        those ids the artifact ancestry (``PlanArtifactRef.parent_ids``) is
        walked upward to the *nearest* ledger-recorded producers — a rejected
        artifact may itself be unledgered (e.g. a RepairLoop attempt), in
        which case its inputs' producers are the ones pinning stale state.
        Each matched producer is evicted, along with every ledger entry at a
        later pipeline position: the pipeline is linear, so downstream
        entries may have consumed the evicted output and keeping them would
        re-create the very cross-stage drift being repaired. ``completed``
        is pruned in place; the caller persists the ledger on ``True``.

        Only reachable for :class:`StagePersistedFailureError` — a generic
        failure identifies no rejected artifact, and the failed stage itself
        was never recorded, so plain resume-from-the-failed-stage applies.
        """
        producer_by_artifact = {entry["artifact"]: name for name, entry in completed.items()}
        matched: set[str] = set()
        frontier: list[str] = list(rejected_ref.parent_ids)
        seen: set[str] = set()
        while frontier:
            artifact_id = frontier.pop()
            if artifact_id in seen:
                continue
            seen.add(artifact_id)
            producer = producer_by_artifact.get(artifact_id)
            if producer is not None:
                matched.add(producer)  # nearest ledgered producer — stop this path
                continue
            try:
                parents = ctx.artifact_store.get_ref(artifact_id).parent_ids
            except ArtifactNotFoundError:
                continue
            frontier.extend(parents)
        if not matched:
            return False

        order = {stage.name: index for index, stage in enumerate(stages)}
        positions = [order[name] for name in matched if name in order]
        evicted = set(matched)
        if positions:
            earliest = min(positions)
            evicted.update(name for name in completed if order.get(name, -1) >= earliest)
        for name in evicted:
            completed.pop(name, None)
        _LOG.warning(
            f"[mode:{self.name}] stage {failed_stage!r} rejected artifact(s) pinned by the "
            f"completion ledger — evicting {sorted(evicted)} so the next run regenerates "
            "them instead of re-failing on the same artifact"
        )
        return True

    def _build_ctx(
        self,
        run: Any,  # noqa: ANN401
        gateway: AgentGateway | None,
        capability_registry: CapabilityRegistry | None = None,
    ) -> HarnessRunContext:
        artifact_store = FileArtifactStore(root=run.run_dir / "artifacts")
        db_path = run.run_dir / "harness.sqlite"
        return HarnessRunContext(
            run_id=run.id,
            workspace_root=run.run_dir,
            artifact_store=artifact_store,
            event_log=SQLiteEventLog(path=db_path),
            lineage_store=SQLiteArtifactLineageStore(path=db_path, artifact_store=artifact_store),
            agent_gateway=gateway,
            capability_registry=capability_registry,
            approval_store=SQLiteApprovalStore(path=db_path),
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

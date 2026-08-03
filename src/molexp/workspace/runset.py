"""RunSet — an ordered batch of sibling Runs, plus its result summary.

``Experiment.sweep(workflow, params)`` seeds one content-addressed Run per
parameter cell and returns a :class:`RunSet`; ``RunSet.execute`` drives every
``pending`` run through the exact same execution path as ``molexp run``
(RunContext lifecycle → status machine, ``_ops`` sidecar, heartbeat →
workflow engine), reached via the :func:`~molexp.workspace.run.set_run_executor`
inversion seam — the workspace layer never imports workflow. One run's
failure never interrupts its siblings; the summary reports it honestly.

:class:`RunSetResult` is the analysis-friendly summary: one record per run
(params flattened + per-task outputs + ``run_id`` / ``status`` / ``error``),
with pure-Python ``min_by`` / ``max_by``. ``to_records()`` rows are plain
dicts — feed them to whatever analysis stack you use; molexp deliberately
bridges to none.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

from molexp._typing import JSONValue, TaskOutput

from .execution_results import read_completed_node_outputs
from .models import RunStatus
from .run import Run, RunWorkflowExecutor, require_run_executor

__all__ = ["RunRecord", "RunSet", "RunSetResult"]


@dataclass(frozen=True)
class RunRecord:
    """One run's row in a :class:`RunSetResult` (immutable)."""

    run_id: str
    status: str
    params: Mapping[str, JSONValue]
    outputs: Mapping[str, TaskOutput]
    error: str | None = None

    def to_record(self) -> dict[str, object]:
        """Flatten to one analysis row: params + task outputs + identity.

        Task outputs shadow same-named params; the reserved identity keys
        (``run_id`` / ``status`` / ``error``) always win.
        """
        row: dict[str, object] = {**self.params, **self.outputs}
        row["run_id"] = self.run_id
        row["status"] = self.status
        row["error"] = self.error
        return row


@dataclass(frozen=True)
class RunSetResult:
    """Immutable batch-execution summary — one :class:`RunRecord` per run."""

    entries: tuple[RunRecord, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self.entries)

    @property
    def failed(self) -> tuple[RunRecord, ...]:
        """The records whose run did not succeed (honest failure surface)."""
        return tuple(rec for rec in self.entries if rec.status == RunStatus.FAILED.value)

    def to_records(self) -> list[dict[str, object]]:
        """One flat dict per run: params + per-task outputs + identity keys."""
        return [rec.to_record() for rec in self.entries]

    def _rows_with(self, key: str) -> list[dict[str, object]]:
        rows = [row for row in self.to_records() if row.get(key) is not None]
        if not rows:
            raise ValueError(
                f"no record carries a comparable {key!r} value — "
                f"available keys: {sorted({k for row in self.to_records() for k in row})}"
            )
        return rows

    def min_by(self, key: str) -> dict[str, object]:
        """The record with the smallest *key* value (params or task output)."""
        return min(self._rows_with(key), key=lambda row: row[key])

    def max_by(self, key: str) -> dict[str, object]:
        """The record with the largest *key* value (params or task output)."""
        return max(self._rows_with(key), key=lambda row: row[key])


class RunSet(Sequence[Run]):
    """An ordered, immutable batch of sibling :class:`Run` objects.

    Built by ``Experiment.sweep`` (which remembers the swept workflow) or
    ``Experiment.runs`` (workflow resolved from the experiment's binding at
    execute time). A plain sequence for inspection; :meth:`execute` /
    :meth:`collect` produce the :class:`RunSetResult` summary.
    """

    def __init__(self, runs: Iterable[Run], *, workflow: object | None = None) -> None:
        self._runs: tuple[Run, ...] = tuple(runs)
        self._workflow = workflow

    # ── Sequence protocol ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._runs)

    @overload
    def __getitem__(self, index: int) -> Run: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Run, ...]: ...

    def __getitem__(self, index: int | slice) -> Run | tuple[Run, ...]:
        return self._runs[index]

    def __iter__(self) -> Iterator[Run]:
        return iter(self._runs)

    def __repr__(self) -> str:
        return f"RunSet({len(self._runs)} runs)"

    @property
    def runs(self) -> tuple[Run, ...]:
        return self._runs

    # ── Execution ────────────────────────────────────────────────────────

    def execute(self, *, parallel: int = 1) -> RunSetResult:
        """Execute every ``pending`` run; return the honest batch summary.

        Reuses the single sanctioned execution path (``molexp run``'s):
        each run goes through its RunContext lifecycle + the workflow
        engine, via the workflow layer's registered run executor. Runs
        outside the ``pending`` domain (succeeded / failed / cancelled /
        running) are left alone — retrying a failure stays an explicit
        per-run verb — and appear in the summary with their current status
        and persisted outputs. A failing run is recorded (``status`` +
        ``error``) and never interrupts its siblings.

        Args:
            parallel: Maximum number of runs executing concurrently (≥ 1).

        Raises:
            ValueError: ``parallel`` < 1.
            RuntimeError: Called with the workflow layer not imported, or
                from inside a running event loop.
        """
        if parallel < 1:
            raise ValueError(f"parallel must be >= 1, got {parallel}")
        executor = require_run_executor()  # fail fast before opening a loop
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "RunSet.execute() was called from inside a running event "
                "loop; drive the runs with run.aexecute(...) there instead."
            )
        return asyncio.run(self._execute(executor, parallel=parallel))

    async def _execute(self, executor: RunWorkflowExecutor, parallel: int) -> RunSetResult:
        semaphore = asyncio.Semaphore(parallel)

        async def _one(run: Run) -> RunRecord:
            if run.status != RunStatus.PENDING.value:
                return self._record_for(run)
            async with semaphore:
                try:
                    result = await executor.aexecute(run, self._workflow)
                except Exception as exc:
                    return self._record_for(run, exc=exc)
            return self._record_for(run, outputs=dict(getattr(result, "outputs", {}) or {}))

        entries = await asyncio.gather(*(_one(run) for run in self._runs))
        return RunSetResult(entries=tuple(entries))

    def collect(self) -> RunSetResult:
        """Summarize the runs' current on-disk state without executing.

        Reads each run's status and its latest execution's persisted
        completed-node outputs — the read-back path for a later session
        (``experiment.runs().collect()``).
        """
        return RunSetResult(entries=tuple(self._record_for(run) for run in self._runs))

    # ── Record building ──────────────────────────────────────────────────

    def _record_for(
        self,
        run: Run,
        *,
        outputs: dict[str, TaskOutput] | None = None,
        error: str | None = None,
        exc: Exception | None = None,
    ) -> RunRecord:
        if outputs is None:
            # A RunFailedError carries the partial WorkflowResult (duck-read:
            # workspace never imports the workflow layer's types).
            partial = getattr(getattr(exc, "result", None), "outputs", None)
            outputs = dict(partial) if partial else self._persisted_outputs(run)
        if error is None:
            # Prefer the concise persisted record ("Type: message"); fall back
            # to the caught exception's rendering for non-run failures.
            error = self._persisted_error(run)
            if error is None and exc is not None:
                error = f"{type(exc).__name__}: {exc}"
        return RunRecord(
            run_id=run.id,
            status=run.status,
            params=dict(run.parameters),
            outputs=outputs,
            error=error,
        )

    @staticmethod
    def _persisted_outputs(run: Run) -> dict[str, TaskOutput]:
        """Latest execution's completed-node outputs (lossy records dropped)."""
        history = run.read_ops().executions
        if not history:
            return {}
        execution_id = history[-1].execution_id
        if not execution_id:
            return {}
        records = read_completed_node_outputs(Path(str(run.run_dir)), execution_id)
        return {name: rec.value for name, rec in records.items() if not rec.lossy}

    @staticmethod
    def _persisted_error(run: Run) -> str | None:
        error = getattr(run.metadata, "error", None)
        if error is None:
            return None
        return f"{error.type}: {error.message}"

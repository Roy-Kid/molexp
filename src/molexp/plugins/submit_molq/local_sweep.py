"""Local subprocess-based sweep dispatch via molq.

Mirrors the cluster path: each replica runs in its own ``molexp execute``
subprocess spawned through molq's ``local`` scheduler.  The subprocess starts
with ``cwd=exec_dir`` so cwd-relative output from user tasks (``logs/...``,
tensorboard runs, framework checkpoints) lands inside the per-attempt
directory and is isolated across retries — matching the cluster backend
behaviour.

Bounded concurrency is provided by an :class:`asyncio.Semaphore` keyed on
the ``--jobs`` flag.  A failure in one replica is captured and reported but
does not cancel its peers.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any


async def run_local_sweep(
    replicas: list[Any],
    *,
    jobs: int = 1,
) -> dict[str, Exception]:
    """Run *replicas* concurrently, each as a ``molexp execute`` subprocess.

    Args:
        replicas: Iterable of objects exposing ``mol_run`` (a workspace ``Run``).
            The ``experiment`` attribute, if present, is unused here — the
            worker rebuilds it from ``run_dir/run.json``.
        jobs: Maximum concurrent subprocesses (clamped to ``>= 1``).

    Returns:
        ``{replica_id: Exception}`` for replicas whose worker subprocess
        exited non-zero or whose ``Submitor.submit`` raised.  Empty dict on
        full success.
    """
    if not replicas:
        return {}

    from molq import Cluster, JobExecution, Submitor

    from molexp.workflow import make_execution_id

    sem = asyncio.Semaphore(max(1, jobs))
    failures: dict[str, Exception] = {}

    def _submit_and_wait(mol_run: Any, run_dir: Path, exec_dir: Path, execution_id: str) -> Any:
        """Submit and wait for the worker subprocess.  Runs in a thread.

        Submitor construction opens a SQLite store synchronously and
        ``handle.wait`` blocks on a polling loop — both block the event
        loop if invoked directly from a coroutine.  Calling this entire
        helper via :func:`asyncio.to_thread` lets concurrent replicas
        actually run in parallel.
        """
        jobs_dir = exec_dir / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        with Submitor(
            Cluster(name="local", scheduler="local"),
            jobs_dir=str(jobs_dir),
        ) as submitor:
            handle = submitor.submit_job(
                argv=[
                    sys.executable,
                    "-m",
                    "molexp.cli",
                    "execute",
                    str(run_dir),
                    "--execution-id",
                    execution_id,
                ],
                execution=JobExecution(
                    cwd=str(exec_dir),
                    output_file=str(exec_dir / "stdout.log"),
                    error_file=str(exec_dir / "stderr.log"),
                ),
                metadata={
                    "run_id": mol_run.id,
                    "run_dir": str(run_dir),
                    "execution_id": execution_id,
                },
            )
            return handle.wait()

    async def _run_one(replica: Any) -> None:
        mol_run = replica.mol_run
        rid = str(getattr(mol_run, "id", id(mol_run)))
        run_dir = Path(mol_run.run_dir)
        execution_id = make_execution_id(mol_run.id, run_dir)
        exec_dir = run_dir / "executions" / execution_id

        async with sem:
            try:
                record = await asyncio.to_thread(
                    _submit_and_wait, mol_run, run_dir, exec_dir, execution_id
                )
            except Exception as exc:
                failures[rid] = exc
                return

            exit_code = getattr(record, "exit_code", None)
            if exit_code is not None and exit_code != 0:
                failures[rid] = RuntimeError(f"worker for run {rid!r} exited with code {exit_code}")

    await asyncio.gather(*(_run_one(r) for r in replicas))
    return failures

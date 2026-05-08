"""Stage-in / stage-out for remote compute targets.

When a Run executes on a non-local :class:`~molexp.workspace.ComputeTarget`,
its working directory must exist on the remote filesystem before the worker
starts and its outputs must come back to the local workspace afterwards.
This module owns those two transfers via the molq :class:`~molq.transport.Transport`.

The local↔remote contract is intentionally narrow:

* **Stage-in** mirrors the local ``run_dir`` (run.json, experiment metadata,
  workspace metadata, the workflow source) into ``target_run_dir`` on the
  transport's filesystem.  It excludes ``executions/`` and Python bytecode.
* **Stage-out** mirrors the remote ``executions/<exec_id>/`` plus any new
  files under ``artifacts/``, ``assets/`` and ``.ckpt/`` back to the local
  ``run_dir``.  Run metadata (``run.json``) is also pulled so the local
  status reflects what the worker reported.

Both operations are no-ops when the target's working dir resolves to the
local ``run_dir`` (i.e. a local target with no scratch-root override) so the
``--local`` path stays free of needless rsync calls.

Stage-out **must be idempotent** — molq's reconciler may fire the terminal
callback more than once during recovery.  rsync makes the file pull
self-deduplicating; downstream asset registration uses asset_id-based
deduping.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

from molq.transport import Transport, TransportError

from molexp.workspace.targets import target_run_dir

if TYPE_CHECKING:
    from molexp.workspace import ComputeTarget, Run


_RSYNC_EXCLUDES = ("__pycache__", "*.pyc", "*.pyo", ".DS_Store")


def stage_in(transport: Transport, mol_run: Run, target: ComputeTarget) -> None:
    """Mirror the local run dir to the target's filesystem.

    No-op when source and destination are the same path (local target with no
    scratch override).  Skips ``executions/`` because the worker writes that
    on the target side and we don't want to overwrite remote state with stale
    local copies.
    """
    src = str(Path(mol_run.run_dir).resolve())
    workspace = mol_run.experiment.project.workspace
    dst = target_run_dir(target, workspace, mol_run)
    if src == dst:
        return

    transport.mkdir(dst, parents=True, exist_ok=True)
    transport.upload(
        src,
        dst,
        recursive=True,
        exclude=(*_RSYNC_EXCLUDES, "executions"),
    )


def stage_out(
    transport: Transport,
    mol_run: Run,
    target: ComputeTarget,
    execution_id: str,
) -> None:
    """Mirror the per-attempt artifacts and updated run metadata back locally.

    Idempotent — safe to call multiple times for the same execution.
    Downstream asset registration uses content-addressed asset_ids so re-pulls
    are deduplicated.
    """
    workspace = mol_run.experiment.project.workspace
    remote_run = target_run_dir(target, workspace, mol_run)
    local_run = str(Path(mol_run.run_dir).resolve())
    if remote_run == local_run:
        return

    # Per-attempt directory (always pulled; small).
    remote_exec = f"{remote_run}/executions/{execution_id}"
    local_exec = str(Path(local_run) / "executions" / execution_id)
    Path(local_exec).mkdir(parents=True, exist_ok=True)
    try:
        transport.download(
            remote_exec,
            local_exec,
            recursive=True,
            exclude=_RSYNC_EXCLUDES,
        )
    except TransportError:
        # The exec dir may not exist yet if submit failed before the worker
        # started — leave the local copy untouched and surface no error.
        return

    # Top-level artifacts the run may have produced.
    for sub in ("artifacts", ".ckpt", "assets.json"):
        remote_path = f"{remote_run}/{sub}"
        local_path = str(Path(local_run) / sub)
        try:
            if not transport.exists(remote_path):
                continue
        except TransportError:
            continue
        try:
            transport.download(
                remote_path,
                local_path,
                recursive=sub != "assets.json",
                exclude=_RSYNC_EXCLUDES,
            )
        except TransportError:
            continue

    # Run metadata — always pulled so the local copy reflects worker outcomes.
    with contextlib.suppress(TransportError):
        transport.download(
            f"{remote_run}/run.json",
            str(Path(local_run) / "run.json"),
        )


__all__ = ["stage_in", "stage_out"]

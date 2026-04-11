"""Typer command functions for each molq scheduler backend.

Each function is registered as a sub-command of ``molexp run`` by
:func:`molexp.plugins.submit_molq.register_commands`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer


# ── Shared helpers ───────────────────────────────────────────────────────────

_SCRIPT_ARG = Annotated[
    Path,
    typer.Argument(
        help="Python script with me.entry(project) call.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
]


def _run_backend(
    *,
    scheduler: str,
    script: Path,
    resume: bool,
    workspace: Path | None,
    cluster: str | None,
    resources: dict,
    scheduling: dict,
) -> None:
    """Build a submit handler and execute the parameter sweep."""
    from molexp.cli import _execute_sweep
    from molexp.plugins.submit_molq.submit import make_submit_handler

    handler = make_submit_handler(
        scheduler=scheduler,
        cluster=cluster,
        resources=resources,
        scheduling=scheduling,
    )
    _execute_sweep(
        script=script,
        dry_run=False,
        resume=resume,
        workspace=workspace,
        run_handler=handler,
        mode_label=f"[magenta]{scheduler}[/magenta]",
    )


# ── SLURM ────────────────────────────────────────────────────────────────────


def run_slurm(
    script: _SCRIPT_ARG,
    # ── Execution mode ──
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume existing dry-run runs for remote submission."),
    ] = False,
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root (overrides project config)."),
    ] = None,
    # ── Resources (all Optional, NO defaults) ──
    partition: Annotated[
        Optional[str],
        typer.Option("--partition", "-p", help="SLURM partition."),
    ] = None,
    cpus: Annotated[
        Optional[int],
        typer.Option("--cpus", help="CPU cores per job."),
    ] = None,
    mem: Annotated[
        Optional[str],
        typer.Option("--mem", help="Memory per job (e.g. 8G, 512M)."),
    ] = None,
    time: Annotated[
        Optional[str],
        typer.Option("--time", "-t", help="Wall-clock time limit (e.g. 12:00:00, 2h30m)."),
    ] = None,
    gpus: Annotated[
        Optional[int],
        typer.Option("--gpus", help="GPUs per job."),
    ] = None,
    gpu_type: Annotated[
        Optional[str],
        typer.Option("--gpu-type", help="GPU type constraint (e.g. a100)."),
    ] = None,
    account: Annotated[
        Optional[str],
        typer.Option("--account", "-A", help="SLURM account."),
    ] = None,
    qos: Annotated[
        Optional[str],
        typer.Option("--qos", help="SLURM QOS."),
    ] = None,
    cluster: Annotated[
        Optional[str],
        typer.Option("--cluster", help="molq cluster name."),
    ] = None,
) -> None:
    """Submit runs to a SLURM cluster via molq."""
    _run_backend(
        scheduler="slurm",
        script=script,
        resume=resume,
        workspace=workspace,
        cluster=cluster,
        resources={"cpus": cpus, "mem": mem, "gpus": gpus, "gpu_type": gpu_type, "time": time},
        scheduling={"queue": partition, "account": account, "qos": qos},
    )


# ── PBS ──────────────────────────────────────────────────────────────────────


def run_pbs(
    script: _SCRIPT_ARG,
    # ── Execution mode ──
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume existing dry-run runs for remote submission."),
    ] = False,
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root (overrides project config)."),
    ] = None,
    # ── Resources (all Optional, NO defaults) ──
    queue: Annotated[
        Optional[str],
        typer.Option("--queue", "-q", help="PBS queue."),
    ] = None,
    cpus: Annotated[
        Optional[int],
        typer.Option("--cpus", help="CPU cores per job."),
    ] = None,
    mem: Annotated[
        Optional[str],
        typer.Option("--mem", help="Memory per job (e.g. 8G, 512M)."),
    ] = None,
    time: Annotated[
        Optional[str],
        typer.Option("--time", "-t", help="Wall-clock time limit (e.g. 12:00:00, 2h30m)."),
    ] = None,
    gpus: Annotated[
        Optional[int],
        typer.Option("--gpus", help="GPUs per job."),
    ] = None,
    account: Annotated[
        Optional[str],
        typer.Option("--account", "-A", help="PBS account."),
    ] = None,
    cluster: Annotated[
        Optional[str],
        typer.Option("--cluster", help="molq cluster name."),
    ] = None,
) -> None:
    """Submit runs to a PBS/Torque cluster via molq."""
    _run_backend(
        scheduler="pbs",
        script=script,
        resume=resume,
        workspace=workspace,
        cluster=cluster,
        resources={"cpus": cpus, "mem": mem, "gpus": gpus, "time": time},
        scheduling={"queue": queue, "account": account},
    )


# ── LSF ──────────────────────────────────────────────────────────────────────


def run_lsf(
    script: _SCRIPT_ARG,
    # ── Execution mode ──
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume existing dry-run runs for remote submission."),
    ] = False,
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root (overrides project config)."),
    ] = None,
    # ── Resources (all Optional, NO defaults) ──
    queue: Annotated[
        Optional[str],
        typer.Option("--queue", "-q", help="LSF queue."),
    ] = None,
    cpus: Annotated[
        Optional[int],
        typer.Option("--cpus", help="CPU cores per job."),
    ] = None,
    mem: Annotated[
        Optional[str],
        typer.Option("--mem", help="Memory per job (e.g. 8G, 512M)."),
    ] = None,
    time: Annotated[
        Optional[str],
        typer.Option("--time", "-t", help="Wall-clock time limit (e.g. 12:00:00, 2h30m)."),
    ] = None,
    gpus: Annotated[
        Optional[int],
        typer.Option("--gpus", help="GPUs per job."),
    ] = None,
    gpu_type: Annotated[
        Optional[str],
        typer.Option("--gpu-type", help="GPU type constraint (e.g. a100)."),
    ] = None,
    account: Annotated[
        Optional[str],
        typer.Option("--account", "-A", help="LSF account/project."),
    ] = None,
    cluster: Annotated[
        Optional[str],
        typer.Option("--cluster", help="molq cluster name."),
    ] = None,
) -> None:
    """Submit runs to an LSF cluster via molq."""
    _run_backend(
        scheduler="lsf",
        script=script,
        resume=resume,
        workspace=workspace,
        cluster=cluster,
        resources={"cpus": cpus, "mem": mem, "gpus": gpus, "gpu_type": gpu_type, "time": time},
        scheduling={"queue": queue, "account": account},
    )

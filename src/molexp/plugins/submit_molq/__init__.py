"""molq submission plugin — registers scheduler sub-commands under ``molexp run``.

When molq is installed, this plugin adds ``slurm``, ``pbs``, and ``lsf``
sub-commands.  If molq is not installed the import fails silently and the
CLI only offers the built-in ``local`` backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer


def register_commands(run_app: typer.Typer) -> None:
    """Register molq scheduler sub-commands on *run_app*."""
    from molexp.plugins.submit_molq.commands import run_lsf, run_pbs, run_slurm

    run_app.command(name="slurm", help="Submit runs to a SLURM cluster.")(run_slurm)
    run_app.command(name="pbs", help="Submit runs to a PBS/Torque cluster.")(run_pbs)
    run_app.command(name="lsf", help="Submit runs to an LSF cluster.")(run_lsf)

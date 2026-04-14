"""``molexp run`` — execute workflows defined by a Python script."""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import Annotated, Any, Callable, Optional

import typer

from molexp.config import MolCfg, ProfileConfig, load_molcfg
from molexp.config.loader import find_default_config

from . import app
from ._common import (
    console,
    deterministic_run_id,
    reap_zombie_run,
    rprint,
)

RunHandler = Callable[[Path, Any, Any, Any], None]


# ── Config / profile resolution ─────────────────────────────────────────────


def _resolve_profile(
    config_path: Path | None, profile: str | None
) -> ProfileConfig:
    """Load molcfg and resolve the requested profile.

    - If *config_path* is given it must exist.
    - Otherwise we look for a default ``molcfg.yaml`` / ``.yml`` / ``.json``
      in the current working directory.
    - If no file is found and no *profile* is requested, return an
      empty :class:`ProfileConfig` (defaults-only, name=None).
    - If no file is found but *profile* is requested, abort.
    """
    resolved_path: Path | None
    if config_path is not None:
        resolved_path = config_path
    else:
        resolved_path = find_default_config()

    if resolved_path is None:
        if profile is not None:
            rprint(
                f"[red]Error:[/red] --profile {profile!r} was requested but "
                "no config file was found (searched for ./molcfg.yaml / .yml / .json)."
            )
            raise typer.Exit(1)
        return ProfileConfig({}, name=None)

    try:
        cfg: MolCfg = load_molcfg(resolved_path)
    except Exception as exc:
        rprint(f"[red]Error:[/red] failed to load config {resolved_path}: {exc}")
        raise typer.Exit(1)

    try:
        return cfg.resolve(profile)
    except KeyError as exc:
        rprint(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


# ── Sweep driver ────────────────────────────────────────────────────────────


def _execute_sweep(
    *,
    script: Path,
    profile_cfg: ProfileConfig,
    resume: bool,
    workspace: Path | None,
    run_handler: RunHandler,
    mode_label: str,
    suppress_ok: bool = False,
) -> tuple[int, list[Any]]:
    """Walk each workspace's experiments and dispatch one run per replica.

    Idempotent: re-running with the same script produces deterministic
    run IDs (derived from parameters + active profile) so repeated
    invocations skip already-executed replicas unless ``--resume`` is
    set, in which case non-succeeded runs of the matching profile are
    replayed.
    """
    from molexp.entry import load_workspaces

    try:
        workspaces = load_workspaces(script)
    except Exception as exc:
        rprint(f"[red]Error importing {script.name}:[/red] {exc}")
        rprint(traceback.format_exc(), end="")
        raise typer.Exit(1)

    if not workspaces:
        rprint(
            "[red]Error:[/red] No me.entry() call found in script. "
            "Add [bold]me.entry(workspace)[/bold] at module level."
        )
        raise typer.Exit(1)

    total_dispatched = 0
    dispatched_runs: list[Any] = []

    for ws in workspaces:
        if workspace is not None:
            override = Path(workspace).resolve()
            if override != ws.root:
                rprint(
                    f"[yellow]Warning:[/yellow] --workspace {override} differs from "
                    f"script's Workspace root {ws.root}; using the script's workspace."
                )

        for project in ws.list_projects():
            for exp in project.list_experiments():
                if exp.workflow is None:
                    rprint(
                        f"[red]Error:[/red] Experiment {exp.name!r} has no workflow. "
                        "Call experiment.set_workflow(fn) before me.entry()."
                    )
                    raise typer.Exit(1)

                seeds = exp.get_seeds()
                total = exp.n_replicas
                label = f"[cyan]resume[/cyan] + {mode_label}" if resume else mode_label
                profile_display = profile_cfg.name or "(defaults)"
                rprint(
                    f"\n[bold]Experiment:[/bold] {exp.name}"
                    f"\n  Script:    {script}"
                    f"\n  Workspace: {ws.root}"
                    f"\n  Project:   {project.name}"
                    f"\n  Profile:   {profile_display}"
                    f"\n  Runs:      {total} replicas"
                    f"\n  Mode:      {label}"
                )

                created_runs: list[tuple[Any, int]] = []

                for replica_idx, seed in enumerate(seeds):
                    run_params = {**exp.params, "seed": seed, "replica": replica_idx}
                    # Profile name and config hash are folded into the ID
                    # so different profiles of the same replica do not
                    # collide under the same run directory.
                    id_seed = dict(run_params)
                    if profile_cfg.name is not None:
                        id_seed["_profile"] = profile_cfg.name
                        id_seed["_config_hash"] = profile_cfg.content_hash()
                    run_id = deterministic_run_id(id_seed)

                    existing = exp.get_run(run_id)
                    if existing is not None and existing.status == "running":
                        if reap_zombie_run(existing):
                            rprint(
                                f"  [yellow]![/yellow] {exp.id}  seed={seed} "
                                "(stale 'running' run reaped → failed)"
                            )

                    if resume:
                        if existing is None:
                            rprint(f"  [dim]- {exp.id}  seed={seed} (no existing run, skipped)[/dim]")
                            continue
                        # Resume all non-succeeded runs that match this profile.
                        if existing.status == "succeeded":
                            rprint(f"  [dim]- {exp.id}  seed={seed} (already succeeded, skipped)[/dim]")
                            continue
                        if existing.metadata.profile != profile_cfg.name:
                            rprint(
                                f"  [dim]- {exp.id}  seed={seed} (profile mismatch: "
                                f"{existing.metadata.profile!r} vs {profile_cfg.name!r}, skipped)[/dim]"
                            )
                            continue
                        mol_run = existing
                    elif existing is not None:
                        if existing.status in ("succeeded", "running"):
                            rprint(f"  [dim]- {exp.id}  seed={seed} ({existing.status}, skipped)[/dim]")
                            continue
                        mol_run = existing
                    else:
                        mol_run = exp.run(parameters=run_params, id=run_id)

                    created_runs.append((mol_run, seed))
                    icon = "[cyan]>[/cyan]" if resume else "[dim]o[/dim]"
                    rprint(f"  {icon} {exp.id}  seed={seed}")

                for mol_run, seed in created_runs:
                    run_handler(script, mol_run, exp, project)
                    dispatched_runs.append(mol_run)
                    rprint(f"  [cyan]>[/cyan] dispatched {exp.id}  seed={seed}")

                total_dispatched += len(created_runs)
                if not suppress_ok:
                    verb = "resumed" if resume else "completed"
                    rprint(f"\n[green]OK[/green] {len(created_runs)} runs {verb}.")

    return total_dispatched, dispatched_runs


# ── Background launcher ─────────────────────────────────────────────────────


def _launch_bg(
    *,
    script: Path,
    config_path: Path | None,
    profile: str | None,
    resume: bool,
    workspace: Path | None,
) -> None:
    """Fork a detached subprocess that re-runs the sweep without ``--bg``."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "molexp.cli", "run", str(script), "--local"]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    if profile is not None:
        cmd.extend(["--profile", profile])
    if resume:
        cmd.append("--resume")
    if workspace is not None:
        cmd.extend(["--workspace", str(workspace)])

    ws_root = Path(workspace).resolve() if workspace else Path.cwd()
    ws_root.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        stdout=open(ws_root / "molexp_bg.log", "a"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    rprint(
        f"[green]OK[/green] Launched in background (pid={proc.pid}). "
        f"Logs: [bold]{ws_root / 'molexp_bg.log'}[/bold]"
    )


# ── Typer argument / option aliases ─────────────────────────────────────────

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


# ── Command ─────────────────────────────────────────────────────────────────


@app.command()
def run(
    script: _SCRIPT_ARG,
    # ── Execution backend ──────────────────────────────────────────────────
    local: Annotated[
        bool,
        typer.Option("--local", help="Run locally, sequentially (default).", rich_help_panel="Execution Backend"),
    ] = False,
    slurm: Annotated[
        bool,
        typer.Option("--slurm", help="Submit to a SLURM cluster via molq.", rich_help_panel="Execution Backend"),
    ] = False,
    pbs: Annotated[
        bool,
        typer.Option("--pbs", help="Submit to a PBS/Torque cluster via molq.", rich_help_panel="Execution Backend"),
    ] = False,
    lsf: Annotated[
        bool,
        typer.Option("--lsf", help="Submit to an LSF cluster via molq.", rich_help_panel="Execution Backend"),
    ] = False,
    scheduler: Annotated[
        Optional[str],
        typer.Option(
            "--scheduler",
            help="Submit via a molq scheduler backend (e.g. local, slurm, pbs, lsf).",
            rich_help_panel="Execution Backend",
        ),
    ] = None,
    # ── Config / profile ───────────────────────────────────────────────────
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help=(
                "Path to molcfg file (YAML or JSON). "
                "Defaults to ./molcfg.yaml / .yml / .json if present."
            ),
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    profile: Annotated[
        Optional[str],
        typer.Option(
            "--profile",
            help=(
                "Named molcfg profile to activate (e.g. 'dry-run', 'smoke'). "
                "Profile names with '-' are normalized to '_'."
            ),
        ),
    ] = None,
    # ── Common options ─────────────────────────────────────────────────────
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help=(
                "Re-execute non-succeeded runs whose profile matches the "
                "one selected by --profile."
            ),
        ),
    ] = False,
    bg: Annotated[
        bool,
        typer.Option("--bg", help="Run in background (local only). Logs to <workspace>/molexp_bg.log."),
    ] = False,
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root (overrides project config)."),
    ] = None,
    # ── HPC options (slurm / pbs / lsf) ───────────────────────────────────
    cpus: Annotated[
        Optional[int],
        typer.Option("--cpus", help="CPU cores per job.", rich_help_panel="HPC Options"),
    ] = None,
    mem: Annotated[
        Optional[str],
        typer.Option("--mem", help="Memory per job (e.g. 8G, 512M).", rich_help_panel="HPC Options"),
    ] = None,
    time: Annotated[
        Optional[str],
        typer.Option("--time", "-t", help="Wall-clock time limit (e.g. 12:00:00, 2h30m).", rich_help_panel="HPC Options"),
    ] = None,
    gpus: Annotated[
        Optional[int],
        typer.Option("--gpus", help="GPUs per job.", rich_help_panel="HPC Options"),
    ] = None,
    gpu_type: Annotated[
        Optional[str],
        typer.Option("--gpu-type", help="GPU type constraint (e.g. a100).", rich_help_panel="HPC Options"),
    ] = None,
    account: Annotated[
        Optional[str],
        typer.Option("--account", "-A", help="Account / project name.", rich_help_panel="HPC Options"),
    ] = None,
    cluster: Annotated[
        Optional[str],
        typer.Option("--cluster", help="molq cluster name.", rich_help_panel="HPC Options"),
    ] = None,
    # ── SLURM-specific ─────────────────────────────────────────────────────
    partition: Annotated[
        Optional[str],
        typer.Option("--partition", "-p", help="SLURM partition.", rich_help_panel="SLURM Options"),
    ] = None,
    qos: Annotated[
        Optional[str],
        typer.Option("--qos", help="SLURM QOS.", rich_help_panel="SLURM Options"),
    ] = None,
    # ── PBS / LSF-specific ─────────────────────────────────────────────────
    queue: Annotated[
        Optional[str],
        typer.Option("--queue", "-q", help="PBS/LSF queue name.", rich_help_panel="PBS / LSF Options"),
    ] = None,
    # ── Monitor ────────────────────────────────────────────────────────────
    block: Annotated[
        bool,
        typer.Option(
            "--block",
            help=(
                "Block after cluster submission and open the interactive "
                "run monitor until all jobs finish (press q to close)."
            ),
            rich_help_panel="HPC Options",
        ),
    ] = False,
) -> None:
    """Execute the workflow(s) defined by *script*."""
    # ── Backend selection ──────────────────────────────────────────────────
    legacy_schedulers = [n for n, flag in (("slurm", slurm), ("pbs", pbs), ("lsf", lsf)) if flag]
    if len(legacy_schedulers) > 1:
        rprint(
            f"[red]Error:[/red] Specify at most one of "
            f"{', '.join('--' + n for n in legacy_schedulers)}."
        )
        raise typer.Exit(1)

    selected_scheduler = scheduler or (legacy_schedulers[0] if legacy_schedulers else None)
    if local and selected_scheduler is not None:
        rprint(
            "[red]Error:[/red] Specify at most one backend flag "
            "(--local, --scheduler, --slurm, --pbs, --lsf)."
        )
        raise typer.Exit(1)

    is_local = selected_scheduler is None

    if bg and not is_local:
        rprint("[red]Error:[/red] --bg is only valid with --local.")
        raise typer.Exit(1)

    profile_cfg = _resolve_profile(config, profile)

    # ── Local execution ────────────────────────────────────────────────────
    if is_local:
        if bg:
            _launch_bg(
                script=script,
                config_path=config,
                profile=profile,
                resume=resume,
                workspace=workspace,
            )
            return

        profile_label = f"[cyan]{profile_cfg.name}[/cyan]" if profile_cfg.name else "[dim](defaults)[/dim]"
        label = f"[green]local[/green] profile={profile_label}"

        def _local_handler(
            script_: Path, mol_run: Any, experiment: Any, _project: Any,
        ) -> None:
            asyncio.run(
                experiment.workflow.execute(run=mol_run, profile_config=profile_cfg)
            )

        _execute_sweep(
            script=script,
            profile_cfg=profile_cfg,
            resume=resume,
            workspace=workspace,
            run_handler=_local_handler,
            mode_label=label,
        )
        return

    # ── Cluster backends ───────────────────────────────────────────────────
    try:
        from molexp.plugins.submit_molq.metadata import supported_schedulers
        from molexp.plugins.submit_molq.submit import make_submit_handler
    except ImportError:
        rprint(
            "[red]Error:[/red] molq is not installed. "
            "Install it to use cluster backends: [bold]pip install molq[/bold]"
        )
        raise typer.Exit(1)

    available_schedulers = supported_schedulers()
    if not available_schedulers:
        rprint(
            "[red]Error:[/red] molq is not installed. "
            "Install it to use cluster backends: [bold]pip install molq[/bold]"
        )
        raise typer.Exit(1)
    if available_schedulers and selected_scheduler not in available_schedulers:
        supported_text = ", ".join(available_schedulers)
        rprint(
            f"[red]Error:[/red] Unsupported molq scheduler: {selected_scheduler!r}. "
            f"Available: {supported_text}"
        )
        raise typer.Exit(1)

    selected_queue = partition if partition is not None else queue
    handler = make_submit_handler(
        scheduler=selected_scheduler,
        cluster=cluster,
        resources={"cpus": cpus, "mem": mem, "gpus": gpus, "gpu_type": gpu_type, "time": time},
        scheduling={"queue": selected_queue, "account": account, "qos": qos},
    )
    profile_label = f"[cyan]{profile_cfg.name}[/cyan]" if profile_cfg.name else "[dim](defaults)[/dim]"
    mode_label = f"[magenta]{selected_scheduler}[/magenta] profile={profile_label}"

    n, submitted = _execute_sweep(
        script=script,
        profile_cfg=profile_cfg,
        resume=resume,
        workspace=workspace,
        run_handler=handler,
        mode_label=mode_label,
        suppress_ok=True,
    )

    if n == 0:
        verb = "resumed" if resume else "submitted"
        rprint(f"\n[dim]No runs {verb}.[/dim]")
        return

    if block and submitted:
        from molexp.monitor import RunMonitor
        rprint(f"\n[dim]Submitted {n} runs. Opening monitor… (press q to close)[/dim]")
        RunMonitor(title=f"{script.stem}  [{mode_label}]").watch(submitted)
        rprint(f"\n[dim]Monitor closed. {n} runs are still executing (if any).[/dim]")
        rprint(f"[dim]Reopen with:  molexp watch --workspace {workspace or '.'} [/dim]")
    else:
        verb = "resumed" if resume else "submitted"
        rprint(f"\n[green]OK[/green] {n} runs {verb}.")
        if submitted:
            rprint(f"[dim]Open the monitor with:  molexp watch --workspace {workspace or '.'} [/dim]")


# Silence unused-import lint (console is re-exported for other modules).
_ = console

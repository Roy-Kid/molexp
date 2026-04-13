"""Command-line interface for molexp."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any, Callable, List, Optional

import typer
import uvicorn
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from molexp.plugins.submit_molq.metadata import normalize_executor_info
from molexp.workspace import Workspace

app = typer.Typer(
    name="molexp",
    help="Molecular experiment workflow management",
    no_args_is_help=True,
)

console = Console()


# ── Internal helpers ──────────────────────────────────────────────────────────


def _params_to_id(params: dict[str, Any]) -> str:
    """Compact, filesystem-safe experiment ID from a parameter dict."""
    parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            formatted = f"{v:.0e}".replace("+", "")
            parts.append(f"{k}-{formatted}")
        else:
            parts.append(f"{k}-{v}")
    return "_".join(parts)


def _params_to_label(params: dict[str, Any]) -> str:
    """Human-readable experiment label."""
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def _deterministic_run_id(params: dict[str, Any]) -> str:
    """Generate a deterministic run ID from parameters.

    Same parameters always produce the same ID, making run creation
    idempotent across repeated ``molexp run`` invocations.
    """
    import hashlib

    raw = "|".join(f"{k}={v!r}" for k, v in sorted(params.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _pid_alive(pid: int) -> bool:
    """Return ``True`` if a process with *pid* is running on this host.

    Uses ``os.kill(pid, 0)`` which only checks existence; it does not
    actually deliver a signal.  Returns ``False`` for ``pid <= 0`` and
    for processes we do not own (PermissionError is treated as "alive"
    — the process exists, we just can't signal it).
    """
    import os

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reap_zombie_run(run: Any) -> bool:
    """Mark a stale ``RUNNING`` run as ``FAILED`` if no live owner remains.

    Returns ``True`` when the run was reaped (i.e. its status was flipped
    from ``"running"`` to ``"failed"``), ``False`` when the run appears to
    still be owned by a live process.
    """
    import platform
    from datetime import datetime

    from molexp.workspace.models import ErrorInfo
    from molexp.workspace.run import RunStatus

    labels = dict(run.metadata.labels)
    pid_str = labels.get("pid")
    host = labels.get("host")
    same_host = host == platform.node()

    if same_host and pid_str and pid_str.isdigit() and _pid_alive(int(pid_str)):
        return False

    now = datetime.now()
    for key in ("pid", "host", "heartbeat"):
        labels.pop(key, None)
    run._update_metadata(
        status=RunStatus.FAILED,
        finished_at=now,
        labels=labels,
        error=ErrorInfo(
            type="ZombieRun",
            message=(
                f"Run was left in 'running' state by a prior invocation "
                f"(pid={pid_str or '?'} host={host or '?'}) that did not "
                "finish cleanly.  Automatically marked FAILED."
            ),
            timestamp=now,
        ),
    )
    return True


def _run_executor_info(run: Any) -> dict[str, str]:
    """Return normalized executor metadata for a workspace run."""
    return normalize_executor_info(run.metadata.executor_info, run.metadata.labels)


# ============ Run Command ============


# ── Shared sweep logic ───────────────────────────────────────────────────────

# run_handler signature: (script, mol_run, exp_spec, project_spec) -> None
RunHandler = Callable[[Path, Any, Any, Any], None]


def _execute_sweep(
    *,
    script: Path,
    dry_run: bool,
    resume: bool,
    workspace: Path | None,
    run_handler: RunHandler,
    mode_label: str,
    suppress_ok: bool = False,
) -> tuple[int, list[Any]]:
    """Iterate a parameter sweep and call *run_handler* for each run.

    This contains the shared project/experiment/run iteration logic used
    by both the built-in ``local`` sub-command and plugin-provided backends.

    Returns:
        ``(total_dispatched, dispatched_runs)`` — count and list of Run objects
        that were passed to *run_handler*.
    """
    from molexp.entry import load_projects

    # ── Validate flag combinations ──────────────────────────────────────
    if resume and dry_run:
        rprint("[red]Error:[/red] --resume and --dry-run are mutually exclusive.")
        raise typer.Exit(1)

    # ── Load the script (triggers me.entry() calls) ─────────────────────
    try:
        projects = load_projects(script)
    except Exception as exc:
        rprint(f"[red]Error importing {script.name}:[/red] {exc}")
        raise typer.Exit(1)

    if not projects:
        rprint(
            "[red]Error:[/red] No me.entry() call found in script. "
            "Add [bold]me.entry(project)[/bold] at module level."
        )
        raise typer.Exit(1)

    # ── Process each registered project ─────────────────────────────────
    total_dispatched = 0
    dispatched_runs: list[Any] = []
    for project_spec in projects:
        ws_root = Path(workspace).resolve() if workspace else project_spec.workspace_root.resolve()
        ws = Workspace.from_path(ws_root)

        ws_project = ws.get_project(project_spec.name)
        if ws_project is None:
            ws_project = ws.create_project(name=project_spec.name)
            rprint(f"[green]OK[/green] Created project: {project_spec.name}")

        for exp_spec in project_spec.experiments:
            if exp_spec.workflow is None:
                rprint(
                    f"[red]Error:[/red] Experiment {exp_spec.name!r} has no workflow. "
                    "Call experiment.set_workflow(fn) before me.entry()."
                )
                raise typer.Exit(1)

            workflow = exp_spec.workflow
            seeds = exp_spec.get_seeds()

            param_iter: list[dict[str, Any]]
            if exp_spec.params is not None:
                param_iter = list(exp_spec.params)
            else:
                param_iter = [{}]

            total = len(param_iter) * exp_spec.n_replicas
            if resume:
                label = f"[cyan]resume[/cyan] + {mode_label}"
            else:
                label = mode_label
            rprint(
                f"\n[bold]Experiment:[/bold] {exp_spec.name}"
                f"\n  Script:    {script}"
                f"\n  Workspace: {ws.root}"
                f"\n  Project:   {project_spec.name}"
                f"\n  Runs:      {len(param_iter)} combos x {exp_spec.n_replicas} replicas = {total}"
                f"\n  Mode:      {label}"
            )

            created_runs: list[tuple[Any, dict[str, Any], int]] = []

            for params in param_iter:
                if params:
                    exp_id = _params_to_id(params)
                    exp_label = _params_to_label(params)
                else:
                    exp_id = exp_spec.name
                    exp_label = exp_spec.name

                ws_exp = ws_project.get_experiment(exp_id)
                if ws_exp is None:
                    ws_exp = ws_project.create_experiment(
                        name=exp_label,
                        id=exp_id,
                        workflow_source=str(script),
                        parameter_space=dict(params) if params else {},
                    )

                for replica_idx, seed in enumerate(seeds):
                    run_params = {**params, "seed": seed, "replica": replica_idx}
                    run_id = _deterministic_run_id(run_params)

                    existing = ws_exp.get_run(run_id)
                    if existing is not None and existing.status == "running":
                        if _reap_zombie_run(existing):
                            rprint(
                                f"  [yellow]![/yellow] {exp_id}  seed={seed} "
                                "(stale 'running' run reaped → failed)"
                            )
                    if resume:
                        if existing is None:
                            rprint(f"  [dim]- {exp_id}  seed={seed} (no existing run, skipped)[/dim]")
                            continue
                        status = existing.status
                        if status != "dry_run":
                            rprint(f"  [dim]- {exp_id}  seed={seed} ({status}, skipped)[/dim]")
                            continue
                        mol_run = existing
                    elif existing is not None:
                        status = existing.status
                        if status in ("succeeded", "running", "dry_run"):
                            rprint(f"  [dim]- {exp_id}  seed={seed} ({status}, skipped)[/dim]")
                            continue
                        mol_run = existing
                    else:
                        mol_run = ws_exp.create_run(
                            parameters=run_params, id=run_id,
                        )

                    created_runs.append((mol_run, params, seed))
                    if resume:
                        icon = "[cyan]>[/cyan]"
                    elif dry_run:
                        icon = "[yellow]~[/yellow]"
                    else:
                        icon = "[dim]o[/dim]"
                    rprint(f"  {icon} {exp_id}  seed={seed}")

            # ── Execute or submit ───────────────────────────────────────
            for mol_run, params, seed in created_runs:
                exp_id = _params_to_id(params) if params else exp_spec.name
                run_handler(script, mol_run, exp_spec, project_spec)
                dispatched_runs.append(mol_run)
                rprint(f"  [cyan]>[/cyan] dispatched {exp_id}  seed={seed}")

            total_dispatched += len(created_runs)
            if not suppress_ok:
                if resume:
                    verb = "resumed"
                elif dry_run:
                    verb = "completed in dry-run mode"
                else:
                    verb = "completed"
                rprint(f"\n[green]OK[/green] {len(created_runs)} runs {verb}.")

    return total_dispatched, dispatched_runs


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
    # ── Common options ─────────────────────────────────────────────────────
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Execute in dry-run mode (local only). Tasks still run and can "
                "branch on ctx.dry_run; runs appear in the UI with a [dry-run] badge."
            ),
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Resume existing dry-run runs. Only 'dry_run' status runs are executed.",
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
    no_watch: Annotated[
        bool,
        typer.Option(
            "--no-watch",
            help="Skip the interactive monitor after molq submission (useful for CI/scripts).",
            rich_help_panel="HPC Options",
        ),
    ] = False,
) -> None:
    """Run or schedule a parameter sweep defined in a Python script."""
    # ── Validate backend flags ─────────────────────────────────────────────
    legacy_schedulers = [
        name for name, enabled in (("slurm", slurm), ("pbs", pbs), ("lsf", lsf))
        if enabled
    ]
    if len(legacy_schedulers) > 1:
        rprint(
            "[red]Error:[/red] Specify at most one backend flag "
            "(--local, --scheduler, --slurm, --pbs, --lsf)."
        )
        raise typer.Exit(1)
    if scheduler and legacy_schedulers:
        rprint(
            "[red]Error:[/red] Use either --scheduler or one legacy backend flag "
            "(--slurm/--pbs/--lsf), not both."
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
    if dry_run and not is_local:
        rprint("[red]Error:[/red] --dry-run is only valid with --local.")
        raise typer.Exit(1)

    # ── Local execution ────────────────────────────────────────────────────
    if is_local:
        if bg:
            _launch_bg(script=script, dry_run=dry_run, resume=resume, workspace=workspace)
            return

        label = "[yellow]dry-run[/yellow]" if dry_run else "[green]local[/green]"

        def _local_handler(
            script_: Path, mol_run: Any, exp_spec: Any, _project_spec: Any,
        ) -> None:
            asyncio.run(exp_spec.workflow.execute(run=mol_run, dry_run=dry_run))

        _execute_sweep(
            script=script,
            dry_run=dry_run,
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
    mode_label = f"[magenta]{selected_scheduler}[/magenta]"

    n, submitted = _execute_sweep(
        script=script,
        dry_run=False,
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

    if not no_watch and submitted:
        from molexp.monitor import RunMonitor
        rprint(f"\n[dim]Submitted {n} runs. Opening monitor… (press q to close)[/dim]")
        RunMonitor(title=f"{script.stem}  [{mode_label}]").watch(submitted)
        rprint(f"\n[dim]Monitor closed. {n} runs are still executing.[/dim]")
        rprint(f"[dim]Reopen with:  molexp watch --workspace {workspace or '.'} [/dim]")
    else:
        verb = "resumed" if resume else "submitted"
        rprint(f"\n[green]OK[/green] {n} runs {verb}.")


def _launch_bg(
    *,
    script: Path,
    dry_run: bool,
    resume: bool,
    workspace: Path | None,
) -> None:
    """Fork a detached subprocess that re-runs the sweep without ``--bg``."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "molexp.cli", "run", str(script), "--local"]
    if dry_run:
        cmd.append("--dry-run")
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
        start_new_session=True,
    )
    rprint(f"[green]OK[/green] Background PID {proc.pid}")
    rprint(f"  Log: {ws_root / 'molexp_bg.log'}")


# ============ Server Command ============


@app.command()
def serve(
    workdir: Annotated[
        Path,
        typer.Option(
            "--workdir",
            "-w",
            help="Workspace directory path",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Server port"),
    ] = 8000,
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Server host"),
    ] = "localhost",
) -> None:
    """Start the MolExp server (API + bundled web UI)."""
    import os

    # Auto-discover workspace: if cwd has no workspace.json, check ./workspace
    resolved = Path(workdir).resolve()
    if not (resolved / "workspace.json").exists():
        candidate = resolved / "workspace"
        if (candidate / "workspace.json").exists():
            resolved = candidate
            rprint(f"[dim]Auto-detected workspace at {resolved}[/dim]")
        else:
            rprint(
                f"[yellow]Warning:[/yellow] No workspace.json found in {resolved}. "
                "Run [bold]molexp init[/bold] or use [bold]--workdir[/bold]."
            )

    os.chdir(resolved)

    rprint(f"[bold]Serving Workspace:[/bold] {workdir}")

    from molexp.server.app import _find_bundled_webapp, create_app

    webapp = _find_bundled_webapp()
    if webapp is None:
        rprint(f"[cyan]->[/cyan] API at http://{host}:{port}/api  (no bundled UI)")
        rprint(
            "[dim]  Build a wheel to include the frontend, "
            "or run the frontend dev server separately:[/dim]"
        )
        rprint(f"[dim]  cd ui && npm run dev -- --api-port={port}[/dim]")
    else:
        rprint(f"[cyan]->[/cyan] http://{host}:{port}")

    application = create_app()
    uvicorn.run(application, host=host, port=port, log_level="info")


# ============ Workspace Commands ============


@app.command()
def init(
    path: Annotated[
        Optional[Path],
        typer.Argument(help="Workspace path (default: current directory)"),
    ] = None,
) -> None:
    """Initialize a new workspace."""
    workspace_path = path or Path.cwd()
    ws = Workspace.from_path(workspace_path)

    rprint(f"[green]OK[/green] Initialized workspace at: {ws.root}")
    rprint(f"  - Projects directory: {ws.root / 'projects'}")
    rprint(f"  - Assets directory: {ws.root / 'assets'}")


@app.command()
def info(
    path: Annotated[
        Optional[Path],
        typer.Option("--path", "-p", help="Workspace path"),
    ] = None,
) -> None:
    """Show workspace information."""
    ws = _get_workspace(path)

    projects = ws.list_projects()

    total_experiments = 0
    total_runs = 0
    run_status_counts: dict[str, int] = {
        "pending": 0,
        "running": 0,
        "succeeded": 0,
        "failed": 0,
        "cancelled": 0,
        "dry_run": 0,
    }

    for project in projects:
        experiments = project.list_experiments()
        total_experiments += len(experiments)

        for experiment in experiments:
            runs = experiment.list_runs()
            total_runs += len(runs)

            for r in runs:
                status = str(r.status).lower()
                if status in run_status_counts:
                    run_status_counts[status] += 1

    rprint(f"[bold]Workspace:[/bold] {ws.root}")
    rprint("\n[bold]Statistics:[/bold]")
    rprint(f"  Projects: {len(projects)}")
    rprint(f"  Experiments: {total_experiments}")
    rprint(f"  Runs: {total_runs}")

    if total_runs > 0:
        rprint("\n[bold]Run Status:[/bold]")
        for status, count in run_status_counts.items():
            if count > 0:
                color = {
                    "succeeded": "green",
                    "failed": "red",
                    "running": "yellow",
                    "pending": "blue",
                    "cancelled": "gray",
                    "dry_run": "cyan",
                }.get(status, "white")
                rprint(f"  [{color}]{status.capitalize()}[/{color}]: {count}")


# ============ Watch Command ============


@app.command()
def watch(
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root (default: current directory)."),
    ] = None,
    project: Annotated[
        Optional[str],
        typer.Option("--project", "-p", help="Filter by project name or ID."),
    ] = None,
    experiment: Annotated[
        Optional[str],
        typer.Option("--experiment", "-e", help="Filter by experiment name or ID."),
    ] = None,
    refresh: Annotated[
        float,
        typer.Option("--refresh", "-r", help="Refresh interval in seconds."),
    ] = 2.0,
) -> None:
    """Reopen the full-screen run monitor for an existing workspace."""
    ws_root = Path(workspace).resolve() if workspace else Path.cwd()

    try:
        from molexp.workspace import Workspace as _Workspace
        ws = _Workspace.load(ws_root)
    except FileNotFoundError:
        rprint(f"[red]Error:[/red] No workspace found at {ws_root}")
        rprint("  Run [bold]molexp init[/bold] to create one, or pass [bold]--workspace[/bold].")
        raise typer.Exit(1)

    # Collect matching runs
    runs: list[Any] = []
    for proj in ws.list_projects():
        if project and proj.id != project and proj.name != project:
            continue
        for exp in proj.list_experiments():
            if experiment and exp.id != experiment and exp.name != experiment:
                continue
            runs.extend(exp.list_runs())

    if not runs:
        rprint("[yellow]No runs found[/yellow] — check --project / --experiment filters.")
        raise typer.Exit(0)

    try:
        from molexp.monitor import RunMonitor
    except ImportError:
        rprint("[red]Error:[/red] molq is not installed. Install it to use the monitor.")
        raise typer.Exit(1)

    title = ws.name
    if experiment:
        title = f"{ws.name} / {experiment}"
    elif project:
        title = f"{ws.name} / {project}"

    rprint(f"[dim]Watching {len(runs)} runs. Press q to close.[/dim]")
    RunMonitor(title=title, refresh_interval=refresh).watch(runs)
    rprint(f"\n[dim]Monitor closed. Runs are still executing (if any).[/dim]")


# ============ Project Commands ============

project_app = typer.Typer(help="Project management commands")
app.add_typer(project_app, name="project")


@project_app.command("create")
def project_create(
    name: Annotated[str, typer.Argument(help="Project name")],
    path: Annotated[
        Optional[Path], typer.Option("--path", "-p", help="Workspace path")
    ] = None,
) -> None:
    """Create a new project."""
    ws = _get_workspace(path)

    try:
        project = ws.create_project(name=name)
        rprint(f"[green]OK[/green] Created project: {project.id}")
        rprint(f"  Name: {project.name}")
    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@project_app.command("list")
def project_list(
    path: Annotated[
        Optional[Path], typer.Option("--path", "-p", help="Workspace path")
    ] = None,
) -> None:
    """List all projects."""
    ws = _get_workspace(path)
    projects = ws.list_projects()

    if not projects:
        rprint("[yellow]No projects found[/yellow]")
        return

    table = Table(title="Projects")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Owner")
    table.add_column("Tags")
    table.add_column("Created")

    for project in projects:
        table.add_row(
            project.id,
            project.name,
            project.owner,
            ", ".join(project.tags),
            project.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@project_app.command("info")
def project_info(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    path: Annotated[
        Optional[Path], typer.Option("--path", "-p", help="Workspace path")
    ] = None,
) -> None:
    """Show project information."""
    ws = _get_workspace(path)
    project = ws.get_project(project_id)

    if not project:
        rprint(f"[red]Error:[/red] Project not found: {project_id}")
        raise typer.Exit(1)

    rprint(f"[bold]Project:[/bold] {project.id}")
    rprint(f"  Name: {project.name}")
    rprint(f"  Description: {project.description}")
    rprint(f"  Owner: {project.owner}")
    rprint(f"  Tags: {', '.join(project.tags)}")
    rprint(f"  Created: {project.created_at}")

    experiments = project.list_experiments()
    rprint(f"  Experiments: {len(experiments)}")


# ============ Experiment Commands ============

experiment_app = typer.Typer(help="Experiment management commands")
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("create")
def experiment_create(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    name: Annotated[str, typer.Option("--name", "-n", help="Experiment name")],
    path: Annotated[
        Optional[Path], typer.Option("--path", "-p", help="Workspace path")
    ] = None,
) -> None:
    """Create a new experiment."""
    ws = _get_workspace(path)

    try:
        project = ws.get_project(project_id)
        if not project:
            rprint(f"[red]Error:[/red] Project not found: {project_id}")
            raise typer.Exit(1)

        experiment = project.create_experiment(name=name)
        rprint(f"[green]OK[/green] Created experiment: {experiment.id}")
        rprint(f"  Name: {experiment.name}")
        rprint(f"  Project: {project_id}")
    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@experiment_app.command("list")
def experiment_list(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    path: Annotated[
        Optional[Path], typer.Option("--path", "-p", help="Workspace path")
    ] = None,
) -> None:
    """List all experiments in a project."""
    ws = _get_workspace(path)
    project = ws.get_project(project_id)

    if not project:
        rprint(f"[red]Error:[/red] Project not found: {project_id}")
        raise typer.Exit(1)

    experiments = project.list_experiments()

    if not experiments:
        rprint(f"[yellow]No experiments found in project: {project_id}[/yellow]")
        return

    table = Table(title=f"Experiments in {project_id}")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Created")

    for exp in experiments:
        table.add_row(
            exp.id,
            exp.name,
            exp.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


# ============ Run Commands ============

run_app = typer.Typer(help="Run management commands")
app.add_typer(run_app, name="runs")


@run_app.command("create")
def run_create(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    params: Annotated[
        Optional[str],
        typer.Option("--params", help="Parameters JSON string or file path"),
    ] = None,
    path: Annotated[
        Optional[Path],
        typer.Option("--path", "-p", help="Workspace path"),
    ] = None,
) -> None:
    """Create a new run."""
    ws = _get_workspace(path)

    parameters: dict = {}
    if params:
        params_path = Path(params)
        if params_path.exists():
            parameters = json.loads(params_path.read_text())
        else:
            try:
                parameters = json.loads(params)
            except json.JSONDecodeError:
                rprint(f"[red]Error:[/red] Invalid JSON in parameters: {params}")
                raise typer.Exit(1)

    try:
        project = ws.get_project(project_id)
        if not project:
            rprint(f"[red]Error:[/red] Project not found: {project_id}")
            raise typer.Exit(1)

        experiment = project.get_experiment(experiment_id)
        if not experiment:
            rprint(f"[red]Error:[/red] Experiment not found: {experiment_id}")
            raise typer.Exit(1)

        r = experiment.create_run(parameters=parameters)
        rprint(f"[green]OK[/green] Created run: {r.id}")
        rprint(f"  Project: {project_id}")
        rprint(f"  Experiment: {experiment_id}")
        rprint(f"  Status: {r.status}")
        if parameters:
            rprint(f"  Parameters: {json.dumps(parameters, indent=2)}")

    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@run_app.command("list")
def run_list(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    path: Annotated[
        Optional[Path], typer.Option("--path", "-p", help="Workspace path")
    ] = None,
) -> None:
    """List all runs in an experiment."""
    ws = _get_workspace(path)

    project = ws.get_project(project_id)
    if not project:
        rprint(f"[red]Error:[/red] Project not found: {project_id}")
        raise typer.Exit(1)

    experiment = project.get_experiment(experiment_id)
    if not experiment:
        rprint(f"[red]Error:[/red] Experiment not found: {experiment_id}")
        raise typer.Exit(1)

    runs = experiment.list_runs()

    if not runs:
        rprint(
            f"[yellow]No runs found in {project_id}/{experiment_id}[/yellow]"
        )
        return

    table = Table(title=f"Runs in {project_id}/{experiment_id}")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Created")

    for r in runs:
        status = str(r.status).lower()
        status_color = {
            "succeeded": "green",
            "failed": "red",
            "running": "yellow",
            "pending": "blue",
            "cancelled": "gray",
            "dry_run": "cyan",
        }.get(status, "white")

        table.add_row(
            r.id,
            f"[{status_color}]{status}[/{status_color}]",
            r.metadata.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        )

    console.print(table)


# Terminal statuses — runs in these states are not cancellable
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "dry_run"})


@run_app.command("cancel")
def run_cancel(
    run_ids: Annotated[
        Optional[List[str]],
        typer.Argument(
            help="Run IDs to cancel. Omit to use --project/--experiment with --all or --status."
        ),
    ] = None,
    project_id: Annotated[
        Optional[str],
        typer.Option("--project", "-p", help="Project ID (required in experiment-scope mode)."),
    ] = None,
    experiment_id: Annotated[
        Optional[str],
        typer.Option("--experiment", "-e", help="Experiment ID (required in experiment-scope mode)."),
    ] = None,
    all_runs: Annotated[
        bool,
        typer.Option("--all", help="Cancel all non-terminal runs in the experiment."),
    ] = False,
    status_filter: Annotated[
        Optional[str],
        typer.Option("--status", help="Comma-separated statuses to filter (e.g. 'pending,running')."),
    ] = None,
    scheduler: Annotated[
        str,
        typer.Option(
            "--scheduler",
            help="Fallback molq scheduler backend when run metadata lacks executor info.",
        ),
    ] = "slurm",
    cluster: Annotated[
        Optional[str],
        typer.Option("--cluster", help="molq cluster name (default: 'default')."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt."),
    ] = False,
    path: Annotated[
        Optional[Path],
        typer.Option("--path", help="Workspace path."),
    ] = None,
) -> None:
    """Cancel one or more scheduled runs.

    Two modes:

    \b
    # Cancel specific runs by ID
    molexp runs cancel <run_id1> <run_id2> --yes

    \b
    # Cancel all non-terminal runs in an experiment
    molexp runs cancel --project proj1 --experiment exp1 --all

    \b
    # Cancel only running jobs, with confirmation
    molexp runs cancel --project proj1 --experiment exp1 --status running
    """
    from molexp.workspace.run import RunStatus

    ws = _get_workspace(path)

    # ── Resolve target runs ──────────────────────────────────────────────────
    target_runs = []

    if run_ids:
        # Direct run-ID mode: scan workspace to locate each run.
        for rid in run_ids:
            found = None
            for proj in ws.list_projects():
                for exp in proj.list_experiments():
                    r = exp.get_run(rid)
                    if r is not None:
                        found = r
                        break
                if found:
                    break
            if found is None:
                rprint(f"[yellow]Warning:[/yellow] Run {rid!r} not found — skipping.")
            else:
                target_runs.append(found)
    else:
        # Experiment-scope mode: --project and --experiment are required.
        if not project_id or not experiment_id:
            rprint(
                "[red]Error:[/red] Provide run IDs as arguments, or both "
                "--project and --experiment options."
            )
            raise typer.Exit(1)

        project = ws.get_project(project_id)
        if not project:
            rprint(f"[red]Error:[/red] Project not found: {project_id}")
            raise typer.Exit(1)

        experiment = project.get_experiment(experiment_id)
        if not experiment:
            rprint(f"[red]Error:[/red] Experiment not found: {experiment_id}")
            raise typer.Exit(1)

        candidates = experiment.list_runs()

        if all_runs:
            target_runs = [r for r in candidates if r.status not in _TERMINAL_STATUSES]
        elif status_filter:
            allowed = {s.strip().lower() for s in status_filter.split(",")}
            target_runs = [r for r in candidates if r.status in allowed]
        else:
            rprint(
                "[red]Error:[/red] Specify --all or --status when using "
                "--project/--experiment mode."
            )
            raise typer.Exit(1)

    if not target_runs:
        rprint("[yellow]No runs matched the criteria — nothing to cancel.[/yellow]")
        raise typer.Exit(0)

    # ── Skip already-terminal runs (only relevant in direct-ID mode) ─────────
    already_terminal = [r for r in target_runs if r.status in _TERMINAL_STATUSES]
    target_runs = [r for r in target_runs if r.status not in _TERMINAL_STATUSES]

    for r in already_terminal:
        rprint(f"[yellow]Skipping[/yellow] {r.id} — already terminal: {r.status}")

    if not target_runs:
        rprint("[yellow]All matched runs are already in a terminal state.[/yellow]")
        raise typer.Exit(0)

    # ── Confirmation table ───────────────────────────────────────────────────
    table = Table(title=f"Runs to cancel ({len(target_runs)})")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Scheduler", style="magenta")
    table.add_column("molq_job_id", style="dim")
    table.add_column("scheduler_job_id", style="dim")

    for r in target_runs:
        executor_info = _run_executor_info(r)
        table.add_row(
            r.id,
            r.status,
            executor_info.get("scheduler") or scheduler,
            executor_info.get("job_id") or "—",
            executor_info.get("scheduler_job_id") or "—",
        )

    console.print(table)

    if not yes:
        confirm = typer.prompt(f"\nCancel {len(target_runs)} job(s)? [y/N]", default="N")
        if confirm.strip().lower() not in ("y", "yes"):
            rprint("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    # ── Lazy-load molq Submitor ──────────────────────────────────────────────
    submitor_cache: dict[tuple[str, str], Any] = {}
    molq_available = True
    Submitor = None
    try:
        from molq import Submitor  # type: ignore[import]
    except ImportError:
        molq_available = False
        rprint(
            "[yellow]Warning:[/yellow] molq is not installed — will only update "
            "workspace state without calling the scheduler.\n"
            "  Install with: [bold]pip install molq[/bold]"
        )

    # ── Cancel each run ──────────────────────────────────────────────────────
    cancelled = 0
    errors = 0

    try:
        for r in target_runs:
            executor_info = _run_executor_info(r)
            molq_id = executor_info.get("job_id")
            scheduler_job_id = executor_info.get("scheduler_job_id")
            run_scheduler = executor_info.get("scheduler") or scheduler
            run_cluster = executor_info.get("cluster_name") or cluster or "default"

            if Submitor is not None and molq_available:
                if molq_id and run_scheduler:
                    cache_key = (run_scheduler, run_cluster)
                    submitor = submitor_cache.get(cache_key)
                    if submitor is None:
                        submitor = Submitor(cluster_name=run_cluster, scheduler=run_scheduler)
                        submitor_cache[cache_key] = submitor
                    try:
                        submitor.cancel(molq_id)
                    except Exception as exc:
                        rprint(
                            f"  [yellow]Warning:[/yellow] scheduler cancel failed for "
                            f"{r.id} (molq_job_id={molq_id}, scheduler={run_scheduler}): {exc}"
                        )
                        errors += 1
                elif scheduler_job_id and run_scheduler:
                    cache_key = (run_scheduler, run_cluster)
                    submitor = submitor_cache.get(cache_key)
                    if submitor is None:
                        submitor = Submitor(cluster_name=run_cluster, scheduler=run_scheduler)
                        submitor_cache[cache_key] = submitor
                    try:
                        submitor._scheduler_impl.cancel(scheduler_job_id)
                    except Exception as exc:
                        rprint(
                            f"  [yellow]Warning:[/yellow] scheduler cancel failed for "
                            f"{r.id} (scheduler_job_id={scheduler_job_id}, scheduler={run_scheduler}): {exc}"
                        )
                        errors += 1
                else:
                    rprint(
                        f"  [yellow]Warning:[/yellow] {r.id} has no molq job metadata — "
                        "updating workspace state only."
                    )

            r.cancel()
            rprint(f"  [green]OK[/green] Cancelled {r.id}")
            cancelled += 1
    finally:
        for submitor in submitor_cache.values():
            submitor.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    rprint(f"\n[green]Done.[/green] {cancelled} run(s) cancelled", end="")
    if errors:
        rprint(f", [yellow]{errors} scheduler error(s)[/yellow] (workspace state updated).")
    else:
        rprint(".")


@run_app.command("info")
def run_info(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    run_id: Annotated[str, typer.Argument(help="Run ID")],
    path: Annotated[
        Optional[Path], typer.Option("--path", "-p", help="Workspace path")
    ] = None,
) -> None:
    """Show run information."""
    ws = _get_workspace(path)

    project = ws.get_project(project_id)
    if not project:
        rprint(f"[red]Error:[/red] Project not found: {project_id}")
        raise typer.Exit(1)

    experiment = project.get_experiment(experiment_id)
    if not experiment:
        rprint(f"[red]Error:[/red] Experiment not found: {experiment_id}")
        raise typer.Exit(1)

    r = experiment.get_run(run_id)
    if not r:
        rprint(f"[red]Error:[/red] Run not found: {run_id}")
        raise typer.Exit(1)

    rprint(f"[bold]Run:[/bold] {r.id}")
    rprint(f"  Status: {r.status}")
    rprint(f"  Created: {r.metadata.created_at}")
    if r.metadata.finished_at:
        rprint(f"  Finished: {r.metadata.finished_at}")
    rprint(f"  Parameters: {json.dumps(r.parameters, indent=2)}")


# ============ Asset Commands ============

asset_app = typer.Typer(help="Asset management commands")
app.add_typer(asset_app, name="asset")


@asset_app.command("list")
def asset_list(
    path: Annotated[
        Optional[Path], typer.Option("--path", "-p", help="Workspace path")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Limit results")] = 50,
) -> None:
    """List workspace-level assets."""
    ws = _get_workspace(path)
    assets = ws.assets.list_assets()[:limit]

    if not assets:
        rprint("[yellow]No assets found[/yellow]")
        return

    table = Table(title="Workspace Assets")
    table.add_column("Asset ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Created")

    for asset in assets:
        table.add_row(
            asset.asset_id[:12] + "...",
            asset.name,
            asset.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


# ============ Helper Functions ============


def _get_workspace(path: Path | None = None) -> Workspace:
    """Get workspace from path or default config."""
    if path:
        return Workspace.from_path(path)
    # Default: try loading from cwd
    from molcfg import ConfigLoader, DictSource

    sources = [DictSource({"workspace_root": str(Path.cwd())})]
    config_file = Path.cwd() / "molexp.toml"
    if config_file.exists():
        from molcfg import TomlFileSource

        sources.append(TomlFileSource(str(config_file)))
    config = ConfigLoader(sources).load()
    return Workspace.from_config(config)


if __name__ == "__main__":
    app()

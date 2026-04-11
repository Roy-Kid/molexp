"""Command-line interface for molexp."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any, Callable, Optional

import typer
import uvicorn
from typer.core import TyperGroup
from rich import print as rprint
from rich.console import Console
from rich.table import Table

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


# ============ Run Sub-App ============


class _DefaultLocalGroup(TyperGroup):
    """Typer Group that defaults to ``local`` when the first arg isn't a
    known sub-command name."""

    def resolve_command(self, ctx: typer.Context, args: list[str]) -> tuple:  # type: ignore[override]
        if args and args[0] not in self.commands:
            args.insert(0, "local")
        return super().resolve_command(ctx, args)


run_cmd = typer.Typer(
    cls=_DefaultLocalGroup,
    help="Run or schedule a parameter sweep defined in a Python script.",
    no_args_is_help=True,
)
app.add_typer(run_cmd, name="run")


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
) -> None:
    """Iterate a parameter sweep and call *run_handler* for each run.

    This contains the shared project/experiment/run iteration logic used
    by both the built-in ``local`` sub-command and plugin-provided backends.
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
                rprint(f"  [cyan]>[/cyan] dispatched {exp_id}  seed={seed}")

            if resume:
                verb = "resumed"
            elif dry_run:
                verb = "completed in dry-run mode"
            else:
                verb = "completed"
            rprint(f"\n[green]OK[/green] {len(created_runs)} runs {verb}.")


# ── Built-in local sub-command ───────────────────────────────────────────────


@run_cmd.command(name="local", help="Execute runs locally (sequential).")
def run_local(
    script: Annotated[
        Path,
        typer.Argument(
            help="Python script with me.entry(project) call.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Execute workflows in dry-run mode. Tasks still run and can "
                "branch on ctx.dry_run; runs appear in the UI with a "
                "[dry-run] badge."
            ),
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help=(
                "Resume existing dry-run runs for real execution. "
                "Only runs with status 'dry_run' are executed; "
                "no new runs are created."
            ),
        ),
    ] = False,
    bg: Annotated[
        bool,
        typer.Option(
            "--bg",
            help="Run in background. Logs to <workspace>/molexp_bg_<pid>.log.",
        ),
    ] = False,
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root (overrides project config)."),
    ] = None,
) -> None:
    """Execute runs locally (sequential)."""
    if bg:
        _launch_bg(script=script, dry_run=dry_run, resume=resume, workspace=workspace)
        return

    if dry_run:
        label = "[yellow]dry-run[/yellow]"
    else:
        label = "[green]local[/green]"

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

    cmd = [sys.executable, "-m", "molexp.cli", "run", "local", str(script)]
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
        stdout=open(ws_root / f"molexp_bg.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    rprint(f"[green]OK[/green] Background PID {proc.pid}")
    rprint(f"  Log: {ws_root / 'molexp_bg.log'}")


# ── Auto-discover submit plugins ─────────────────────────────────────────────

try:
    from molexp.plugins.submit_molq import register_commands as _register_molq
    _register_molq(run_cmd)
except ImportError:
    pass  # molq not installed — only the local backend is available


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

"""Command-line interface for molexp."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
import uvicorn
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


# ============ Run Command ============


@app.command(
    context_settings={"allow_extra_args": False, "ignore_unknown_options": False},
)
def run(
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
    # ── Execution mode ────────────────────────────────────────────────
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
    slurm: Annotated[
        bool,
        typer.Option(
            "--slurm",
            help="Submit each run via ExecutionBackend plugin (SLURM).",
        ),
    ] = False,
    # ── SLURM resource options (only relevant with --slurm) ──────────
    partition: Annotated[
        str,
        typer.Option("--partition", "-p", help="SLURM partition.", rich_help_panel="SLURM Resources"),
    ] = "gpu",
    gpus: Annotated[
        int,
        typer.Option("--gpus", help="GPUs per job.", rich_help_panel="SLURM Resources"),
    ] = 1,
    gpu_type: Annotated[
        Optional[str],
        typer.Option("--gpu-type", help="GPU type constraint.", rich_help_panel="SLURM Resources"),
    ] = None,
    cpus: Annotated[
        int,
        typer.Option("--cpus", help="CPU cores per job.", rich_help_panel="SLURM Resources"),
    ] = 8,
    mem: Annotated[
        str,
        typer.Option("--mem", help="Memory per job.", rich_help_panel="SLURM Resources"),
    ] = "40G",
    time: Annotated[
        str,
        typer.Option("--time", "-t", help="Wall-clock time limit.", rich_help_panel="SLURM Resources"),
    ] = "12:00:00",
    account: Annotated[
        Optional[str],
        typer.Option("--account", "-A", help="SLURM account.", rich_help_panel="SLURM Resources"),
    ] = None,
    qos: Annotated[
        Optional[str],
        typer.Option("--qos", help="SLURM QOS.", rich_help_panel="SLURM Resources"),
    ] = None,
    cluster: Annotated[
        str,
        typer.Option("--cluster", help="molq cluster name.", rich_help_panel="SLURM Resources"),
    ] = "hpc",
    # ── Common options ────────────────────────────────────────────────
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root (overrides project config)."),
    ] = None,
) -> None:
    """Run or schedule a parameter sweep defined in a Python script.

    The script must call ``me.entry(project)`` at module level to register
    the project and its experiments.

    \b
    Examples:

    \b
      # Execute all runs in dry-run mode
      molexp run train.py --dry-run

    \b
      # Execute the sweep locally (sequential)
      molexp run train.py

    \b
      # Submit to SLURM via execution backend plugin
      molexp run train.py --slurm --partition gpu --gpus 1
    """
    from molexp.entry import load_projects

    # ── Validate flag combinations ──────────────────────────────────────
    if dry_run and slurm:
        rprint("[red]Error:[/red] --dry-run and --slurm are mutually exclusive.")
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
        # Resolve workspace root (CLI flag > project config)
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

            # Determine parameter combos
            param_iter: list[dict[str, Any]]
            if exp_spec.params is not None:
                param_iter = list(exp_spec.params)
            else:
                param_iter = [{}]

            total = len(param_iter) * exp_spec.n_replicas
            mode_label = (
                "[yellow]dry-run[/yellow]"
                if dry_run
                else ("[magenta]slurm[/magenta]" if slurm else "[green]local[/green]")
            )
            rprint(
                f"\n[bold]Experiment:[/bold] {exp_spec.name}"
                f"\n  Script:    {script}"
                f"\n  Workspace: {ws.root}"
                f"\n  Project:   {project_spec.name}"
                f"\n  Runs:      {len(param_iter)} combos x {exp_spec.n_replicas} replicas = {total}"
                f"\n  Mode:      {mode_label}"
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

                    # Idempotent: skip runs that already exist
                    existing = ws_exp.get_run(run_id)
                    if existing is not None:
                        status = existing.status
                        if status in ("succeeded", "running"):
                            rprint(f"  [dim]- {exp_id}  seed={seed} ({status}, skipped)[/dim]")
                            continue
                        # Re-run failed/pending/cancelled
                        mol_run = existing
                    else:
                        mol_run = ws_exp.create_run(
                            parameters=run_params, id=run_id,
                        )

                    created_runs.append((mol_run, params, seed))
                    icon = "[yellow]~[/yellow]" if dry_run else "[dim]o[/dim]"
                    rprint(f"  {icon} {exp_id}  seed={seed}")

            # ── Execute or submit ───────────────────────────────────────
            for mol_run, params, seed in created_runs:
                exp_id = _params_to_id(params) if params else exp_spec.name

                if slurm:
                    _submit_via_backend(
                        script=script,
                        mol_run=mol_run,
                        exp_spec=exp_spec,
                        project_spec=project_spec,
                        cluster=cluster,
                        resources={
                            "partition": partition,
                            "gpus": gpus,
                            "gpu_type": gpu_type,
                            "cpus": cpus,
                            "mem": mem,
                            "time": time,
                            "account": account,
                            "qos": qos,
                        },
                    )
                    rprint(f"  [magenta]>>[/magenta] queued  {exp_id}  seed={seed}")
                else:
                    verb = "dry-running" if dry_run else "running"
                    rprint(f"  [cyan]>[/cyan] {verb} {exp_id}  seed={seed}")
                    asyncio.run(workflow.execute(run=mol_run, dry_run=dry_run))

            verb = "submitted" if slurm else (
                "completed in dry-run mode" if dry_run else "completed"
            )
            rprint(f"\n[green]OK[/green] {len(created_runs)} runs {verb}.")


def _submit_via_backend(
    *,
    script: Path,
    mol_run: Any,
    exp_spec: Any,
    project_spec: Any,
    cluster: str,
    resources: dict[str, Any],
) -> None:
    """Submit a run through the ExecutionBackend plugin."""
    from molexp.plugins import Capability, registry
    from molexp.plugins.remote.backend import RunSubmission

    backend_cls = registry.get(Capability.EXECUTION_BACKEND)
    backend = backend_cls(cluster=cluster)
    backend.submit_run(
        RunSubmission(
            script=script,
            run_dir=mol_run.run_dir,
            run_id=mol_run.id,
            experiment_name=exp_spec.name,
            project_name=project_spec.name,
            resources=resources,
        )
    )


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
    dev: Annotated[
        bool,
        typer.Option("--dev", help="Development mode (API only, reload enabled)"),
    ] = False,
) -> None:
    """Start the MolExp Backend Server."""
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
    rprint(f"[cyan]->[/cyan] API Server at http://{host}:{port}")

    if dev:
        rprint("[yellow]Development Mode:[/yellow] Reload active. Serving API only.")
        uvicorn.run(
            "molexp.server.app:app",
            host=host,
            port=port,
            reload=True,
            log_level="info",
        )
    else:
        from molexp.server.app import create_app

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

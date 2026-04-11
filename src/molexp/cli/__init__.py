"""Command-line interface for molexp."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

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


# ============ Run Command ============
# molexp run <script>  [--dry-run]  [--slurm [SLURM options...]]
#
# SLURM submission is powered by the molq plugin (molq.Submitor).
# Pass resources directly as CLI flags — no JSON file required.


@app.command(
    context_settings={"allow_extra_args": False, "ignore_unknown_options": False},
)
def run(
    script: Annotated[
        Path,
        typer.Argument(
            help="Python script with EXPERIMENT and train() definitions.",
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
                "Register all runs in the workspace without executing them. "
                "Runs appear in the UI with a [dry-run] badge."
            ),
        ),
    ] = False,
    slurm: Annotated[
        bool,
        typer.Option(
            "--slurm",
            help="Submit each run as a SLURM job via molq (instead of local execution).",
        ),
    ] = False,
    # ── SLURM resource options (only relevant with --slurm) ──────────
    partition: Annotated[
        str,
        typer.Option("--partition", "-p", help="SLURM partition / queue name.", rich_help_panel="SLURM Resources"),
    ] = "gpu",
    gpus: Annotated[
        int,
        typer.Option("--gpus", help="Number of GPUs per job.", rich_help_panel="SLURM Resources"),
    ] = 1,
    gpu_type: Annotated[
        Optional[str],
        typer.Option("--gpu-type", help="GPU type constraint (e.g. 'a100', 'v100').", rich_help_panel="SLURM Resources"),
    ] = None,
    cpus: Annotated[
        int,
        typer.Option("--cpus", help="CPU cores per job.", rich_help_panel="SLURM Resources"),
    ] = 8,
    mem: Annotated[
        str,
        typer.Option("--mem", help="Memory per job (e.g. '40G', '80G').", rich_help_panel="SLURM Resources"),
    ] = "40G",
    time: Annotated[
        str,
        typer.Option("--time", "-t", help="Wall-clock time limit (e.g. '12:00:00', '24h').", rich_help_panel="SLURM Resources"),
    ] = "12:00:00",
    account: Annotated[
        Optional[str],
        typer.Option("--account", "-A", help="SLURM account / billing project.", rich_help_panel="SLURM Resources"),
    ] = None,
    qos: Annotated[
        Optional[str],
        typer.Option("--qos", help="SLURM QOS name.", rich_help_panel="SLURM Resources"),
    ] = None,
    cluster: Annotated[
        str,
        typer.Option("--cluster", help="molq cluster name (label for the SLURM target).", rich_help_panel="SLURM Resources"),
    ] = "hpc",
    # ── Common options ────────────────────────────────────────────────
    workspace: Annotated[
        Optional[Path],
        typer.Option("--workspace", "-w", help="Workspace root directory (overrides EXPERIMENT.workspace_root)."),
    ] = None,
    replicas: Annotated[
        Optional[int],
        typer.Option("--replicas", "-r", help="Override n_replicas from EXPERIMENT."),
    ] = None,
) -> None:
    """Run or schedule a parameter sweep defined in a Python script.

    The script must expose two names at module level:

    \b
      EXPERIMENT  — an ExperimentDef specifying the sweep
      train(ctx)  — callable invoked once per run with a RunContext

    \b
    Examples:

    \b
      # Register all 81 runs in the workspace (no execution)
      molexp run test_allegro_qm9.py --dry-run

    \b
      # Execute the sweep locally (sequential)
      molexp run test_allegro_qm9.py

    \b
      # Submit to SLURM A100 queue via molq (one job per run)
      molexp run test_allegro_qm9.py \\
          --slurm \\
          --partition gpu \\
          --gpus 1 --gpu-type a100 \\
          --mem 40G --time 12:00:00 \\
          --cpus 8
    """
    from molexp.runner import ExperimentDef, _params_to_id, _params_to_label

    # ── Validate flag combinations ──────────────────────────────────────
    if dry_run and slurm:
        rprint("[red]✗[/red] --dry-run and --slurm are mutually exclusive.")
        raise typer.Exit(1)

    # ── Load the script module ──────────────────────────────────────────
    spec_obj = importlib.util.spec_from_file_location("_molexp_script", script)
    if spec_obj is None or spec_obj.loader is None:
        rprint(f"[red]✗[/red] Cannot load script: {script}")
        raise typer.Exit(1)
    module = importlib.util.module_from_spec(spec_obj)
    try:
        spec_obj.loader.exec_module(module)  # type: ignore[union-attr]
    except SystemExit:
        pass  # silence __main__ guards
    except Exception as exc:
        rprint(f"[red]✗[/red] Error importing {script.name}: {exc}")
        raise typer.Exit(1)

    if not hasattr(module, "EXPERIMENT"):
        rprint(
            "[red]✗[/red] Script must define "
            "[bold]EXPERIMENT = ExperimentDef(...)[/bold] at module level."
        )
        raise typer.Exit(1)

    experiment_def: ExperimentDef = module.EXPERIMENT
    if replicas is not None:
        import dataclasses
        experiment_def = dataclasses.replace(experiment_def, n_replicas=replicas)

    # ── Workspace / project setup ───────────────────────────────────────
    ws_root = Path(workspace or experiment_def.workspace_root).resolve()
    ws = Workspace.from_path(ws_root)

    project = ws.get_project(experiment_def.project)
    if project is None:
        project = ws.create_project(name=experiment_def.project)
        rprint(f"[green]✓[/green] Created project: {experiment_def.project}")

    seeds = experiment_def.get_seeds()
    n_combos = len(experiment_def.param_space)
    total = n_combos * experiment_def.n_replicas

    mode_label = (
        "[yellow]dry-run[/yellow]"
        if dry_run
        else ("[magenta]slurm[/magenta]" if slurm else "[green]local[/green]")
    )
    rprint(
        f"\n[bold]Experiment:[/bold] {experiment_def.name}"
        f"\n  Script:    {script}"
        f"\n  Workspace: {ws.root}"
        f"\n  Project:   {experiment_def.project}"
        f"\n  Runs:      {n_combos} combos × {experiment_def.n_replicas} replicas = {total}"
        f"\n  Mode:      {mode_label}"
    )
    if slurm:
        rprint(
            f"\n  SLURM:     {cluster} | partition={partition}"
            f" | {gpus}×{gpu_type or 'gpu'} | {mem} | {time}"
            f"{' | account=' + account if account else ''}"
        )

    # ── Create experiments and runs ─────────────────────────────────────
    created_runs = []

    for params in experiment_def.param_space:
        exp_id = _params_to_id(params)
        exp_label = _params_to_label(params)

        experiment = project.get_experiment(exp_id)
        if experiment is None:
            experiment = project.create_experiment(
                name=exp_label,
                id=exp_id,
                workflow_source=str(script),
                parameter_space=dict(params),
            )

        for replica_idx, seed in enumerate(seeds):
            run_params = {**params, "seed": seed, "replica": replica_idx}
            mol_run = experiment.create_run(parameters=run_params)

            if dry_run:
                mol_run.metadata = mol_run.metadata.model_copy(
                    update={"dry_run": True, "labels": {"mode": "dry-run"}}
                )
                mol_run.save()
                rprint(f"  [dim]◎[/dim] {exp_id}  seed={seed}")
            else:
                created_runs.append((mol_run, params, seed))

    if dry_run:
        rprint(
            f"\n[green]✓[/green] Dry-run: [bold]{total}[/bold] runs registered (not executed)."
            f"\n  Inspect in UI: [cyan]molexp serve -w {ws.root}[/cyan]"
        )
        return

    # ── Execute or submit ───────────────────────────────────────────────
    for mol_run, params, seed in created_runs:
        exp_id = _params_to_id(params)
        run_dir = mol_run.run_dir

        if slurm:
            _submit_slurm_run(
                script=script,
                run_dir=run_dir,
                run_id=mol_run.id,
                experiment_name=experiment_def.name,
                cluster=cluster,
                partition=partition,
                gpus=gpus,
                gpu_type=gpu_type,
                cpus=cpus,
                mem=mem,
                time=time,
                account=account,
                qos=qos,
            )
            rprint(f"  [magenta]⬆[/magenta] queued  {exp_id}  seed={seed}")
        else:
            rprint(f"  [cyan]▶[/cyan] running {exp_id}  seed={seed}")
            result = subprocess.run(
                [sys.executable, str(script), "--run-dir", str(run_dir)],
                check=False,
            )
            if result.returncode != 0:
                rprint(f"  [red]✗[/red] exit {result.returncode}: {exp_id}  seed={seed}")

    verb = "submitted" if slurm else "completed"
    rprint(f"\n[green]✓[/green] {len(created_runs)} runs {verb}.")


def _submit_slurm_run(
    *,
    script: Path,
    run_dir: Path,
    run_id: str,
    experiment_name: str,
    cluster: str,
    partition: str,
    gpus: int,
    gpu_type: Optional[str],
    cpus: int,
    mem: str,
    time: str,
    account: Optional[str],
    qos: Optional[str],
) -> None:
    """Submit one run to SLURM using the molq plugin.

    molq's Submitor handles script materialisation, job tracking, and
    scheduler interaction. No JSON resource file needed.
    """
    from molq import (
        Duration,
        JobExecution,
        JobResources,
        JobScheduling,
        Memory,
        Submitor,
    )
    from molq.options import SlurmSchedulerOptions

    job_name = f"{experiment_name[:20]}-{run_id[:8]}"
    submitor = Submitor(
        cluster_name=cluster,
        scheduler="slurm",
        scheduler_options=SlurmSchedulerOptions(),
    )
    submitor.submit(
        argv=[sys.executable, str(script.resolve()), "--run-dir", str(run_dir)],
        resources=JobResources(
            cpu_count=cpus,
            memory=Memory.parse(mem),
            gpu_count=gpus,
            gpu_type=gpu_type,
            time_limit=Duration.parse(time),
        ),
        scheduling=JobScheduling(
            queue=partition,
            account=account,
            qos=qos,
        ),
        execution=JobExecution(
            job_name=job_name,
            cwd=str(run_dir),
            output_file=str(run_dir / "slurm_%j.out"),
            error_file=str(run_dir / "slurm_%j.err"),
        ),
        metadata={"run_id": run_id, "run_dir": str(run_dir)},
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
    """Start the MolExp Backend Server.

    Arguments:
        workdir: Workspace directory.
        dev: If set, enables hot-reload and disables static file serving (API Only).
             The frontend must be run separately (e.g., 'npm run dev').
    """
    os.chdir(workdir)

    rprint(f"[bold]Serving Workspace:[/bold] {workdir}")
    rprint(f"[cyan]→[/cyan] API Server at http://{host}:{port}")

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
    workspace = Workspace.from_path(workspace_path)

    rprint(f"[green]✓[/green] Initialized workspace at: {workspace.root}")
    rprint(f"  - Projects directory: {workspace.root / 'projects'}")
    rprint(f"  - Assets directory: {workspace.root / 'assets'}")


@app.command()
def info(
    path: Annotated[
        Optional[Path],
        typer.Option("--path", "-p", help="Workspace path"),
    ] = None,
) -> None:
    """Show workspace information."""
    workspace = _get_workspace(path)

    projects = workspace.list_projects()

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

            for run in runs:
                status = str(run.status).lower()
                if status in run_status_counts:
                    run_status_counts[status] += 1

    rprint(f"[bold]Workspace:[/bold] {workspace.root}")
    rprint(f"\n[bold]Statistics:[/bold]")
    rprint(f"  Projects: {len(projects)}")
    rprint(f"  Experiments: {total_experiments}")
    rprint(f"  Runs: {total_runs}")

    if total_runs > 0:
        rprint(f"\n[bold]Run Status:[/bold]")
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
    workspace = _get_workspace(path)

    try:
        project = workspace.create_project(name=name)
        rprint(f"[green]✓[/green] Created project: {project.id}")
        rprint(f"  Name: {project.name}")
    except Exception as e:
        rprint(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@project_app.command("list")
def project_list(
    path: Annotated[
        Optional[Path], typer.Option("--path", "-p", help="Workspace path")
    ] = None,
) -> None:
    """List all projects."""
    workspace = _get_workspace(path)
    projects = workspace.list_projects()

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
    workspace = _get_workspace(path)
    project = workspace.get_project(project_id)

    if not project:
        rprint(f"[red]✗[/red] Project not found: {project_id}")
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
    workspace = _get_workspace(path)

    try:
        project = workspace.get_project(project_id)
        if not project:
            rprint(f"[red]✗[/red] Project not found: {project_id}")
            raise typer.Exit(1)

        experiment = project.create_experiment(name=name)
        rprint(f"[green]✓[/green] Created experiment: {experiment.id}")
        rprint(f"  Name: {experiment.name}")
        rprint(f"  Project: {project_id}")
    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@experiment_app.command("list")
def experiment_list(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    path: Annotated[
        Optional[Path], typer.Option("--path", "-p", help="Workspace path")
    ] = None,
) -> None:
    """List all experiments in a project."""
    workspace = _get_workspace(path)
    project = workspace.get_project(project_id)

    if not project:
        rprint(f"[red]✗[/red] Project not found: {project_id}")
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
    workspace = _get_workspace(path)

    # Parse parameters
    parameters: dict = {}
    if params:
        params_path = Path(params)
        if params_path.exists():
            parameters = json.loads(params_path.read_text())
        else:
            try:
                parameters = json.loads(params)
            except json.JSONDecodeError:
                rprint(f"[red]✗[/red] Invalid JSON in parameters: {params}")
                raise typer.Exit(1)

    try:
        project = workspace.get_project(project_id)
        if not project:
            rprint(f"[red]✗[/red] Project not found: {project_id}")
            raise typer.Exit(1)

        experiment = project.get_experiment(experiment_id)
        if not experiment:
            rprint(f"[red]✗[/red] Experiment not found: {experiment_id}")
            raise typer.Exit(1)

        run = experiment.create_run(parameters=parameters)
        rprint(f"[green]✓[/green] Created run: {run.id}")
        rprint(f"  Project: {project_id}")
        rprint(f"  Experiment: {experiment_id}")
        rprint(f"  Status: {run.status}")
        if parameters:
            rprint(f"  Parameters: {json.dumps(parameters, indent=2)}")

    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[red]✗[/red] Error: {e}")
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
    workspace = _get_workspace(path)

    project = workspace.get_project(project_id)
    if not project:
        rprint(f"[red]✗[/red] Project not found: {project_id}")
        raise typer.Exit(1)

    experiment = project.get_experiment(experiment_id)
    if not experiment:
        rprint(f"[red]✗[/red] Experiment not found: {experiment_id}")
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

    for run in runs:
        status = str(run.status).lower()
        status_color = {
            "succeeded": "green",
            "failed": "red",
            "running": "yellow",
            "pending": "blue",
            "cancelled": "gray",
        }.get(status, "white")

        table.add_row(
            run.id,
            f"[{status_color}]{status}[/{status_color}]",
            run.metadata.created_at.strftime("%Y-%m-%d %H:%M:%S"),
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
    workspace = _get_workspace(path)

    project = workspace.get_project(project_id)
    if not project:
        rprint(f"[red]✗[/red] Project not found: {project_id}")
        raise typer.Exit(1)

    experiment = project.get_experiment(experiment_id)
    if not experiment:
        rprint(f"[red]✗[/red] Experiment not found: {experiment_id}")
        raise typer.Exit(1)

    run = experiment.get_run(run_id)
    if not run:
        rprint(f"[red]✗[/red] Run not found: {run_id}")
        raise typer.Exit(1)

    rprint(f"[bold]Run:[/bold] {run.id}")
    rprint(f"  Status: {run.status}")
    rprint(f"  Created: {run.metadata.created_at}")
    if run.metadata.finished_at:
        rprint(f"  Finished: {run.metadata.finished_at}")
    rprint(f"  Parameters: {json.dumps(run.parameters, indent=2)}")


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
    workspace = _get_workspace(path)
    assets = workspace.assets.list_assets()[:limit]

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
    """Get workspace from path or environment."""
    if path:
        return Workspace.from_path(path)
    return Workspace.from_env()


if __name__ == "__main__":
    app()

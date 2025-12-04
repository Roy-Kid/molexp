"""Command-line interface for molexp."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import print as rprint
from rich.console import Console
from molexp.workspace import Workspace

app = typer.Typer(
    name="molexp",
    help="Molecular experiment workflow management",
    no_args_is_help=True,
)

console = Console()


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Backend host address"),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Backend port"),
    ] = 8000,
    ui_port: Annotated[
        int,
        typer.Option("--ui-port", help="Frontend port"),
    ] = 5173,
    dev: Annotated[
        bool,
        typer.Option("--dev", help="Development mode with auto-reload"),
    ] = True,
) -> None:
    """Start both frontend and backend servers.
    
    This command starts:
    - Backend API server (FastAPI with uvicorn)
    - Frontend dev server (Rsbuild)
    
    Press Ctrl+C to stop both servers.
    """
    processes = []
    
    try:
        rprint("[bold]Starting molexp servers...[/bold]\n")
        
        # Start backend
        backend_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "molexp.api.server:app",
            "--host",
            host,
            "--port",
            str(port),
        ]
        
        if dev:
            backend_cmd.append("--reload")
        
        rprint(f"[cyan]→[/cyan] Starting backend on http://{host}:{port}")
        backend_process = subprocess.Popen(
            backend_cmd,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        processes.append(("backend", backend_process))
        
        # Give backend time to start and check for errors
        time.sleep(3)
        
        # Check if backend started successfully
        if backend_process.poll() is not None:
            # Backend died, show error
            stderr = backend_process.stderr.read() if backend_process.stderr else ""
            rprint(f"[red]✗[/red] Backend failed to start")
            if stderr:
                rprint(f"[red]Error:[/red]")
                print(stderr)
            if "Address already in use" in stderr or "address already in use" in stderr:
                rprint(f"\n[yellow]Port {port} is already in use. Try a different port:[/yellow]")
                rprint(f"  molexp serve --port 8001")
            raise typer.Exit(1)
        
        # Start frontend
        ui_dir = Path(__file__).parent.parent.parent.parent / "ui"
        
        if ui_dir.exists():
            if dev:
                rprint(f"[cyan]→[/cyan] Starting frontend on http://localhost:{ui_port}")
            
            # Check if node_modules exists
            if not (ui_dir / "node_modules").exists():
                rprint("[yellow]⚠[/yellow] Installing frontend dependencies...")
                npm_install = subprocess.run(
                    ["npm", "install"],
                    cwd=ui_dir,
                    capture_output=True,
                    text=True,
                )
                if npm_install.returncode != 0:
                    rprint("[red]✗[/red] Failed to install frontend dependencies")
                    rprint(npm_install.stderr)
                else:
                    rprint("[green]✓[/green] Frontend dependencies installed")
            
            frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=ui_dir,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env={**os.environ, "PORT": str(ui_port), "PUBLIC_USE_MOCK": "false"},
            )
            processes.append(("frontend", frontend_process))
        else:
            rprint(f"[yellow]⚠[/yellow] Frontend directory not found: {ui_dir}")
            rprint("[yellow]  Only backend will be started[/yellow]")
        
        rprint("\n[green]✓[/green] Servers started successfully!")
        if dev:
            rprint(f"\n[bold]API:[/bold] http://{host}:{port}")
            rprint(f"[bold]API Docs:[/bold] http://{host}:{port}/docs")
        
            if len(processes) > 1:
                rprint(f"[bold]UI:[/bold] http://localhost:{ui_port}")
        
        rprint("\n[bold]Press Ctrl+C to stop all servers[/bold]\n")
        
        # Monitor processes
        while True:
            time.sleep(1)
            
            # Check if any process died
            for name, proc in processes:
                if proc.poll() is not None:
                    stderr = proc.stderr.read() if proc.stderr else ""
                    rprint(f"\n[red]✗[/red] {name.capitalize()} server stopped unexpectedly")
                    if stderr:
                        rprint(f"[red]Error:[/red] {stderr[:500]}")
                    raise KeyboardInterrupt
    
    except KeyboardInterrupt:
        rprint("\n\n[bold]Stopping servers...[/bold]")
        
        # Stop all processes
        for name, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                rprint(f"[green]✓[/green] {name.capitalize()} server stopped")
            except subprocess.TimeoutExpired:
                proc.kill()
                rprint(f"[yellow]⚠[/yellow] {name.capitalize()} server force killed")
            except Exception as e:
                rprint(f"[red]✗[/red] Error stopping {name}: {e}")
        
        rprint("\n[green]✓[/green] All servers stopped")
    
    except Exception as e:
        rprint(f"\n[red]✗[/red] Error: {e}")
        
        # Cleanup on error
        for name, proc in processes:
            try:
                proc.terminate()
            except:
                pass
        
        raise typer.Exit(1)


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
    assets = workspace.list_assets()
    
    # Gather detailed statistics
    total_experiments = 0
    total_runs = 0
    run_status_counts = {"pending": 0, "running": 0, "succeeded": 0, "failed": 0, "cancelled": 0}
    
    for project in projects:
        experiments = workspace.list_experiments(project.project_id)
        total_experiments += len(experiments)
        
        for experiment in experiments:
            runs = workspace.list_runs(project.project_id, experiment.experiment_id)
            total_runs += len(runs)
            
            for run in runs:
                status = run.status.value.lower()
                if status in run_status_counts:
                    run_status_counts[status] += 1
    
    # Calculate total asset size
    total_asset_size = sum(asset.size_bytes for asset in assets)
    total_asset_size_mb = total_asset_size / (1024 * 1024)
    
    rprint(f"[bold]Workspace:[/bold] {workspace.root}")
    rprint(f"\n[bold]Statistics:[/bold]")
    rprint(f"  Projects: {len(projects)}")
    rprint(f"  Experiments: {total_experiments}")
    rprint(f"  Runs: {total_runs}")
    rprint(f"  Assets: {len(assets)} ({total_asset_size_mb:.2f} MB)")
    
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


@app.command()
def check(
    path: Annotated[
        Optional[Path],
        typer.Option("--path", "-p", help="Workspace path"),
    ] = None,
    fix: Annotated[
        bool,
        typer.Option("--fix", help="Attempt to fix issues automatically"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed validation results"),
    ] = False,
) -> None:
    """Validate workspace integrity."""
    workspace = _get_workspace(path)
    validator = WorkspaceValidator(workspace)
    
    rprint("[bold]Validating workspace...[/bold]\n")
    
    report = validator.validate(verbose=verbose)
    
    # Show statistics
    rprint(f"[bold]Statistics:[/bold]")
    rprint(f"  Projects: {report.total_projects}")
    rprint(f"  Experiments: {report.total_experiments}")
    rprint(f"  Runs: {report.total_runs}")
    rprint(f"  Assets: {report.total_assets}")
    if report.orphaned_assets > 0:
        rprint(f"  Orphaned Assets: [yellow]{report.orphaned_assets}[/yellow]")
    
    # Show issues
    if report.issues:
        rprint(f"\n[bold]Issues Found:[/bold]")
        
        for issue in report.issues:
            color = {"error": "red", "warning": "yellow", "info": "blue"}.get(
                issue.severity, "white"
            )
            icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(issue.severity, "•")
            
            rprint(f"  [{color}]{icon}[/{color}] {issue.message}")
            if verbose and issue.path:
                rprint(f"      Path: {issue.path}")
        
        # Attempt fixes if requested
        if fix:
            rprint(f"\n[bold]Attempting to fix issues...[/bold]")
            unfixed = validator.fix_issues(report.issues)
            fixed_count = len(report.issues) - len(unfixed)
            
            if fixed_count > 0:
                rprint(f"  [green]✓[/green] Fixed {fixed_count} issue(s)")
            if unfixed:
                rprint(f"  [yellow]⚠[/yellow] Could not fix {len(unfixed)} issue(s)")
    
    # Show summary
    rprint(f"\n{report.summary()}")
    
    if report.has_errors:
        raise typer.Exit(1)


# ============ Workflow Commands ============

workflow_app = typer.Typer(help="Workflow management commands")
app.add_typer(workflow_app, name="workflow")


@workflow_app.command("list")
def workflow_list(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed information"),
    ] = False,
) -> None:
    """List all registered workflows."""
    registry = get_workflow_registry()
    inspector = WorkflowInspector(registry)
    
    workflows = inspector.list_workflows()
    
    if not workflows:
        rprint("[yellow]No workflows registered[/yellow]")
        return
    
    table = Table(title="Registered Workflows")
    table.add_column("Workflow ID", style="cyan")
    table.add_column("Tasks", style="green")
    
    if verbose:
        table.add_column("Status")
    
    for wf in workflows:
        row = [wf["id"], str(wf["num_tasks"])]
        
        if verbose:
            if "error" in wf:
                row.append(f"[red]{wf['error']}[/red]")
            else:
                row.append("[green]OK[/green]")
        
        table.add_row(*row)
    
    console.print(table)


@workflow_app.command("info")
def workflow_info(
    workflow_id: Annotated[str, typer.Argument(help="Workflow ID")],
) -> None:
    """Show detailed workflow information."""
    registry = get_workflow_registry()
    inspector = WorkflowInspector(registry)
    
    try:
        info = inspector.get_workflow_info(workflow_id)
        tree = inspector.render_tree(workflow_id)
        
        rprint(f"[bold]Workflow:[/bold] {workflow_id}")
        rprint(f"  Tasks: {info['num_tasks']}")
        rprint(f"\n[bold]Task Graph:[/bold]")
        rprint(tree)
        
        if info["tasks"]:
            rprint(f"\n[bold]Task Details:[/bold]")
            for task in info["tasks"]:
                rprint(f"  • {task['name']} ({task['type']})")
                if task["dependencies"]:
                    rprint(f"    Dependencies: {', '.join(task['dependencies'])}")
    
    except ValueError as e:
        rprint(f"[red]✗[/red] {e}")
        raise typer.Exit(1)


@workflow_app.command("export")
def workflow_export(
    workflow_id: Annotated[str, typer.Argument(help="Workflow ID")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path (default: stdout)"),
    ] = None,
    pretty: Annotated[
        bool,
        typer.Option("--pretty", help="Pretty-print JSON"),
    ] = True,
) -> None:
    """Export workflow to JSON IR format."""
    registry = get_workflow_registry()
    inspector = WorkflowInspector(registry)
    
    try:
        ir = inspector.export_json_ir(workflow_id)
        
        json_str = json.dumps(ir, indent=2 if pretty else None)
        
        if output:
            output.write_text(json_str)
            rprint(f"[green]✓[/green] Exported workflow to: {output}")
        else:
            print(json_str)
    
    except ValueError as e:
        rprint(f"[red]✗[/red] {e}")
        raise typer.Exit(1)


# ============ Project Commands ============

project_app = typer.Typer(help="Project management commands")
app.add_typer(project_app, name="project")


@project_app.command("create")
def project_create(
    project_id: Annotated[str, typer.Argument(help="Project ID (slug)")],
    name: Annotated[str, typer.Option("--name", "-n", help="Project name")],
    description: Annotated[str, typer.Option("--desc", "-d", help="Description")] = "",
    owner: Annotated[str, typer.Option("--owner", "-o", help="Owner")] = "",
    tags: Annotated[Optional[str], typer.Option("--tags", "-t", help="Comma-separated tags")] = None,
    path: Annotated[Optional[Path], typer.Option("--path", "-p", help="Workspace path")] = None,
) -> None:
    """Create a new project."""
    workspace = _get_workspace(path)
    
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    
    try:
        project = workspace.create_project(
            project_id=project_id,
            name=name,
            description=description,
            owner=owner,
            tags=tag_list,
        )
        rprint(f"[green]✓[/green] Created project: {project.project_id}")
        rprint(f"  Name: {project.name}")
        rprint(f"  Path: {workspace.root / project.path}")
    except Exception as e:
        rprint(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@project_app.command("list")
def project_list(
    path: Annotated[Optional[Path], typer.Option("--path", "-p", help="Workspace path")] = None,
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
            project.project_id,
            project.name,
            project.owner,
            ", ".join(project.tags),
            project.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    
    console.print(table)


@project_app.command("info")
def project_info(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    path: Annotated[Optional[Path], typer.Option("--path", "-p", help="Workspace path")] = None,
) -> None:
    """Show project information."""
    workspace = _get_workspace(path)
    project = workspace.get_project(project_id)
    
    if not project:
        rprint(f"[red]✗[/red] Project not found: {project_id}")
        raise typer.Exit(1)
    
    rprint(f"[bold]Project:[/bold] {project.project_id}")
    rprint(f"  Name: {project.name}")
    rprint(f"  Description: {project.description}")
    rprint(f"  Owner: {project.owner}")
    rprint(f"  Tags: {', '.join(project.tags)}")
    rprint(f"  Created: {project.created_at}")
    
    experiments = workspace.list_experiments(project_id)
    rprint(f"  Experiments: {len(experiments)}")


# ============ Experiment Commands ============

experiment_app = typer.Typer(help="Experiment management commands")
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("create")
def experiment_create(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID (slug)")],
    name: Annotated[str, typer.Option("--name", "-n", help="Experiment name")],
    workflow: Annotated[str, typer.Option("--workflow", "-w", help="Workflow file path")],
    description: Annotated[str, typer.Option("--desc", "-d", help="Description")] = "",
    params: Annotated[Optional[str], typer.Option("--params", help="Parameters JSON")] = None,
    path: Annotated[Optional[Path], typer.Option("--path", "-p", help="Workspace path")] = None,
) -> None:
    """Create a new experiment."""
    workspace = _get_workspace(path)
    
    param_space = json.loads(params) if params else {}
    
    try:
        experiment = workspace.create_experiment(
            project_id=project_id,
            experiment_id=experiment_id,
            name=name,
            workflow_source=workflow,
            description=description,
            parameter_space=param_space,
        )
        rprint(f"[green]✓[/green] Created experiment: {experiment.experiment_id}")
        rprint(f"  Name: {experiment.name}")
        rprint(f"  Workflow: {experiment.workflow_template.source}")
        rprint(f"  Path: {workspace.root / experiment.path}")
    except Exception as e:
        rprint(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@experiment_app.command("list")
def experiment_list(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    path: Annotated[Optional[Path], typer.Option("--path", "-p", help="Workspace path")] = None,
) -> None:
    """List all experiments in a project."""
    workspace = _get_workspace(path)
    experiments = workspace.list_experiments(project_id)
    
    if not experiments:
        rprint(f"[yellow]No experiments found in project: {project_id}[/yellow]")
        return
    
    table = Table(title=f"Experiments in {project_id}")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Workflow")
    table.add_column("Created")
    
    for exp in experiments:
        table.add_row(
            exp.experiment_id,
            exp.name,
            exp.workflow_template.source,
            exp.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    
    console.print(table)


# ============ Run Commands ============

run_app = typer.Typer(help="Run management commands")
app.add_typer(run_app, name="run")


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
    parameters = {}
    if params:
        # Check if it's a file path
        params_path = Path(params)
        if params_path.exists():
            parameters = json.loads(params_path.read_text())
        else:
            # Try to parse as JSON string
            try:
                parameters = json.loads(params)
            except json.JSONDecodeError:
                rprint(f"[red]✗[/red] Invalid JSON in parameters: {params}")
                raise typer.Exit(1)
    
    try:
        # Get experiment to get workflow info
        experiment = workspace.get_experiment(project_id, experiment_id)
        if not experiment:
            rprint(f"[red]✗[/red] Experiment not found: {project_id}/{experiment_id}")
            raise typer.Exit(1)
        
        run = workspace.create_run(
            project_id=project_id,
            experiment_id=experiment_id,
            parameters=parameters,
            workflow_file=experiment.workflow_template.source,
            git_commit=experiment.workflow_template.git_commit,
        )
        
        rprint(f"[green]✓[/green] Created run: {run.run_id}")
        rprint(f"  Project: {project_id}")
        rprint(f"  Experiment: {experiment_id}")
        rprint(f"  Status: {run.status.value}")
        rprint(f"  Parameters: {json.dumps(parameters, indent=2)}")
    
    except Exception as e:
        rprint(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@run_app.command("list")
def run_list(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    path: Annotated[Optional[Path], typer.Option("--path", "-p", help="Workspace path")] = None,
) -> None:
    """List all runs in an experiment."""
    workspace = _get_workspace(path)
    runs = workspace.list_runs(project_id, experiment_id)
    
    if not runs:
        rprint(f"[yellow]No runs found in {project_id}/{experiment_id}[/yellow]")
        return
    
    table = Table(title=f"Runs in {project_id}/{experiment_id}")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Created")
    table.add_column("Duration")
    
    for run in runs:
        duration = ""
        if run.finished_at:
            delta = run.finished_at - run.created_at
            duration = f"{delta.total_seconds():.1f}s"
        
        status_color = {
            "succeeded": "green",
            "failed": "red",
            "running": "yellow",
            "pending": "blue",
            "cancelled": "gray",
        }.get(run.status.value, "white")
        
        table.add_row(
            run.run_id,
            f"[{status_color}]{run.status.value}[/{status_color}]",
            run.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            duration,
        )
    
    console.print(table)


@run_app.command("info")
def run_info(
    project_id: Annotated[str, typer.Argument(help="Project ID")],
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID")],
    run_id: Annotated[str, typer.Argument(help="Run ID")],
    path: Annotated[Optional[Path], typer.Option("--path", "-p", help="Workspace path")] = None,
) -> None:
    """Show run information."""
    workspace = _get_workspace(path)
    run = workspace.get_run(project_id, experiment_id, run_id)
    
    if not run:
        rprint(f"[red]✗[/red] Run not found: {run_id}")
        raise typer.Exit(1)
    
    rprint(f"[bold]Run:[/bold] {run.run_id}")
    rprint(f"  Status: {run.status.value}")
    rprint(f"  Created: {run.created_at}")
    rprint(f"  Finished: {run.finished_at or 'N/A'}")
    rprint(f"  Workflow: {run.workflow_snapshot.workflow_file}")
    rprint(f"  Parameters: {json.dumps(run.parameters, indent=2)}")
    
    # Show asset refs
    refs = workspace.get_asset_refs(project_id, experiment_id, run_id)
    if refs:
        rprint(f"\n[bold]Assets:[/bold]")
        rprint(f"  Inputs: {len(refs.inputs)}")
        rprint(f"  Outputs: {len(refs.outputs)}")


# ============ Asset Commands ============

asset_app = typer.Typer(help="Asset management commands")
app.add_typer(asset_app, name="asset")


@asset_app.command("list")
def asset_list(
    path: Annotated[Optional[Path], typer.Option("--path", "-p", help="Workspace path")] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Limit results")] = 50,
) -> None:
    """List all assets."""
    workspace = _get_workspace(path)
    assets = workspace.list_assets()[:limit]
    
    if not assets:
        rprint("[yellow]No assets found[/yellow]")
        return
    
    table = Table(title="Assets")
    table.add_column("Asset ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Format")
    table.add_column("Size")
    table.add_column("Created")
    
    for asset in assets:
        size_mb = asset.size_bytes / (1024 * 1024)
        table.add_row(
            asset.asset_id[:8] + "...",
            asset.type.value,
            asset.format,
            f"{size_mb:.2f} MB",
            asset.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    
    console.print(table)


@asset_app.command("info")
def asset_info(
    asset_id: Annotated[str, typer.Argument(help="Asset ID")],
    path: Annotated[Optional[Path], typer.Option("--path", "-p", help="Workspace path")] = None,
) -> None:
    """Show asset information."""
    workspace = _get_workspace(path)
    asset = workspace.get_asset(asset_id)
    
    if not asset:
        rprint(f"[red]✗[/red] Asset not found: {asset_id}")
        raise typer.Exit(1)
    
    rprint(f"[bold]Asset:[/bold] {asset.asset_id}")
    rprint(f"  Type: {asset.type.value}")
    rprint(f"  Format: {asset.format}")
    rprint(f"  Size: {asset.size_bytes / (1024 * 1024):.2f} MB")
    rprint(f"  Hash: {asset.content_hash}")
    rprint(f"  Created: {asset.created_at}")
    rprint(f"  Producer: {asset.producer_run_id or 'N/A'}")
    rprint(f"  Tags: {', '.join(asset.tags)}")
    rprint(f"  Files: {len(asset.files)}")


# ============ Helper Functions ============


def _get_workspace(path: Path | None = None) -> Workspace:
    """Get workspace from path or environment."""
    if path:
        return Workspace.from_path(path)
    return Workspace.from_env()


if __name__ == "__main__":
    app()

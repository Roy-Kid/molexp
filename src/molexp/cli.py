"""
Command Line Interface for MolExp project management.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich import print as rprint

from . import Project, ProjectConfig, Experiment
from .param import ParamSpace, CartesianSampler, RandomSampler

console = Console()


def error_exit(message: str, code: int = 1):
    """Print error message and exit."""
    console.print(f"[red]Error:[/red] {message}")
    sys.exit(code)


def success_print(message: str):
    """Print success message."""
    console.print(f"[green]✓[/green] {message}")


def info_print(message: str):
    """Print info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def warning_print(message: str):
    """Print warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_banner():
    """Print the welcome banner."""
    welcome = r"""
                    __              
   ____ ___  ____  / /__  _  ______ 
  / __ `__ \/ __ \/ / _ \| |/_/ __ \
 / / / / / / /_/ / /  __/>  </ /_/ /
/_/ /_/ /_/\____/_/\___/_/|_/ .___/ 
                           /_/      

    MolExp - Molecular Experiment Management System

    A powerful tool for managing molecular simulation experiments,
    parameter studies, and batch execution workflows.
    """
    console.print(welcome, style="bold green")


@click.group(invoke_without_command=True)
@click.version_option(version="1.0.0", prog_name="molexp")
@click.pass_context
def cli(ctx):
    """
    MolExp - Molecular Experiment Management System
    
    A powerful tool for managing molecular simulation experiments,
    parameter studies, and batch execution workflows.
    """
    # 只在用户执行 molexp 且没有任何参数时显示banner
    # 排除 --help 和 --version 的情况
    if ctx.invoked_subcommand is None and not ctx.resilient_parsing:
        import sys
        # 检查命令行参数，如果只有程序名或者包含help/version相关参数，就不显示banner
        args = sys.argv[1:]  # 排除程序名
        if not args or not any(arg in ['--help', '-h', '--version'] for arg in args):
            print_banner()


@cli.group()
def project():
    """Project management commands."""
    pass


@project.command("create")
@click.argument("name")
@click.option("--description", "-d", help="Project description")
@click.option("--author", "-a", help="Project author")
@click.option("--version", "-v", default="1.0.0", help="Project version")
@click.option("--tags", "-t", multiple=True, help="Project tags")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def create_project(name: str, description: str, author: str, version: str, tags: tuple, output: str):
    """Create a new project."""
    
    config = ProjectConfig(
        name=name,
        description=description or "",
        author=author or "",
        version=version,
        tags=list(tags)
    )
    
    proj = Project(name=name, config=config)
    
    # Build comprehensive panel content
    content_lines = []
    
    # Project creation confirmation
    content_lines.append("[bold green]✅ Project Created Successfully![/bold green]")
    content_lines.append("")
    
    # Basic Info Section
    content_lines.append("[bold cyan]📋 Project Details[/bold cyan]")
    content_lines.append(f"[yellow]Name:[/yellow] {config.name}")
    content_lines.append(f"[yellow]Description:[/yellow] {config.description or 'No description'}")
    content_lines.append(f"[yellow]Author:[/yellow] {config.author or 'Anonymous'}")
    content_lines.append(f"[yellow]Version:[/yellow] {config.version}")
    content_lines.append(f"[yellow]Created:[/yellow] {config.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    content_lines.append(f"[yellow]Tags:[/yellow] {', '.join(config.tags) if config.tags else 'None'}")
    
    # Next steps
    if output:
        content_lines.append("")
        content_lines.append("[bold magenta]📁 File Output[/bold magenta]")
        content_lines.append(f"[dim]Will be saved to:[/dim] [green]{output}[/green]")
    
    # Create unified panel
    panel_content = "\n".join(content_lines)
    panel = Panel(
        panel_content,
        title=f"[bold white]🧬 New Project: {config.name}[/bold white]",
        border_style="green",
        padding=(1, 2),
        expand=False
    )
    console.print(panel)
    
    # Save if output specified
    if output:
        output_path = Path(output)
        try:
            if output_path.suffix == '.json':
                proj.to_json(output_path)
                success_print(f"Project saved to {output_path}")
            else:
                # Default to YAML
                if not output_path.suffix:
                    output_path = output_path.with_suffix('.yaml')
                proj.to_yaml(output_path)
                success_print(f"Project saved to {output_path}")
        except Exception as e:
            error_exit(f"Failed to save project: {e}")


@project.command("info")
@click.argument("project_file", type=click.Path(exists=True))
def project_info(project_file: str):
    """Show project information."""
    
    project_path = Path(project_file)
    
    try:
        with console.status("[bold green]Loading project..."):
            if project_path.suffix == '.json':
                proj = Project.from_json(project_path)
            else:
                proj = Project.from_yaml(project_path)
        
        # Collect all information
        config = proj.config
        stats = proj.get_project_summary()
        
        # Build comprehensive panel content
        content_lines = []
        
        # Basic Info Section
        content_lines.append("[bold blue]📋 Project Details[/bold blue]")
        content_lines.append(f"[cyan]Name:[/cyan] {config.name}")
        content_lines.append(f"[cyan]Description:[/cyan] {config.description or 'No description'}")
        content_lines.append(f"[cyan]Author:[/cyan] {config.author or 'Unknown'}")
        content_lines.append(f"[cyan]Version:[/cyan] {config.version}")
        content_lines.append(f"[cyan]Created:[/cyan] {config.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        content_lines.append(f"[cyan]Tags:[/cyan] {', '.join(config.tags) if config.tags else 'None'}")
        
        # Statistics Section
        content_lines.append("")
        content_lines.append("[bold green]📊 Statistics[/bold green]")
        content_lines.append(f"[yellow]Experiments:[/yellow] {stats['experiment_count']}")
        content_lines.append(f"[yellow]Total Tasks:[/yellow] {stats['total_tasks']}")
        content_lines.append(f"[yellow]Shared Resources:[/yellow] {len(proj.shared_resources)}")
        
        # Resources Section (if any)
        if proj.shared_resources:
            content_lines.append("")
            content_lines.append("[bold magenta]🔧 Shared Resources[/bold magenta]")
            for name, value in proj.shared_resources.items():
                # Truncate long values for display
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                content_lines.append(f"[dim]•[/dim] [cyan]{name}:[/cyan] {value_str}")
        
        # Create unified panel
        panel_content = "\n".join(content_lines)
        panel = Panel(
            panel_content,
            title=f"[bold white]🧬 {config.name}[/bold white]",
            border_style="bright_blue",
            padding=(1, 2),
            expand=False
        )
        console.print(panel)
        
    except Exception as e:
        error_exit(f"Failed to load project: {e}")


@project.command("list")
@click.argument("project_file", type=click.Path(exists=True))
@click.option("--format", "-f", type=click.Choice(['table', 'tree', 'json', 'compact']), default='compact', help="Output format")
@click.option("--filter", "-F", help="Filter experiments by name pattern")
def list_experiments(project_file: str, format: str, filter: str):
    """List experiments in a project."""
    
    project_path = Path(project_file)
    
    try:
        with console.status("[bold green]Loading project..."):
            if project_path.suffix == '.json':
                proj = Project.from_json(project_path)
            else:
                proj = Project.from_yaml(project_path)
        
        experiments = list(proj.experiments.values())
        
        # Apply filter
        if filter:
            experiments = [exp for exp in experiments if filter.lower() in exp.name.lower()]
        
        if not experiments:
            warning_print("No experiments found matching the criteria")
            return
        
        if format == 'compact':
            # New compact format in a single panel
            content_lines = []
            
            # Header
            content_lines.append(f"[bold cyan]📊 Project Overview[/bold cyan]")
            content_lines.append(f"[yellow]Total Experiments:[/yellow] {len(experiments)}")
            if filter:
                content_lines.append(f"[yellow]Filtered by:[/yellow] '{filter}'")
            content_lines.append("")
            
            # Group experiments
            base_exps = []
            param_exps = []
            
            for exp in experiments:
                if 'parameters' in exp.metadata:
                    param_exps.append(exp)
                else:
                    base_exps.append(exp)
            
            # Base experiments
            if base_exps:
                content_lines.append("[bold green]🔬 Base Experiments[/bold green]")
                for exp in base_exps:
                    task_count = len(exp.task_pool.tasks) if exp.task_pool else 0
                    content_lines.append(f"[dim]•[/dim] [cyan]{exp.name}[/cyan] [dim]({task_count} tasks)[/dim]")
                content_lines.append("")
            
            # Parameter studies
            if param_exps:
                content_lines.append("[bold magenta]🧪 Parameter Studies[/bold magenta]")
                for exp in param_exps:
                    task_count = len(exp.task_pool.tasks) if exp.task_pool else 0
                    params = exp.metadata.get('parameters', {})
                    param_pairs = [f"{k}={v}" for k, v in params.items()]
                    param_str = ", ".join(param_pairs[:3])  # Show first 3 params
                    if len(param_pairs) > 3:
                        param_str += "..."
                    content_lines.append(f"[dim]•[/dim] [cyan]{exp.name}[/cyan] [dim]({task_count} tasks)[/dim]")
                    if param_str:
                        content_lines.append(f"  [dim]{param_str}[/dim]")
            
            # Create unified panel
            panel_content = "\n".join(content_lines)
            panel = Panel(
                panel_content,
                title=f"[bold white]🧬 {proj.config.name} - Experiments[/bold white]",
                border_style="bright_magenta",
                padding=(1, 2),
                expand=False
            )
            console.print(panel)
        
        elif format == 'table':
            table = Table(title=f"Experiments in {proj.config.name}")
            table.add_column("Name", style="cyan")
            table.add_column("Tasks", style="magenta")
            table.add_column("Parameters", style="green")
            table.add_column("Status", style="yellow")
            
            for exp in experiments:
                task_count = len(exp.task_pool.tasks) if exp.task_pool else 0
                params = exp.metadata.get('parameters', {})
                param_str = f"{len(params)} params" if params else "No params"
                
                # Simple status check
                status = "Ready" if task_count > 0 else "Empty"
                
                table.add_row(
                    exp.name,
                    str(task_count),
                    param_str,
                    status
                )
            
            console.print(table)
        
        elif format == 'tree':
            tree = Tree(f"[bold blue]{proj.config.name}[/bold blue]")
            
            # Group by experiment type
            base_exps = []
            param_exps = []
            
            for exp in experiments:
                if 'parameters' in exp.metadata:
                    param_exps.append(exp)
                else:
                    base_exps.append(exp)
            
            if base_exps:
                base_branch = tree.add("[bold green]Base Experiments[/bold green]")
                for exp in base_exps:
                    task_count = len(exp.task_pool.tasks) if exp.task_pool else 0
                    base_branch.add(f"{exp.name} ({task_count} tasks)")
            
            if param_exps:
                param_branch = tree.add("[bold yellow]Parameter Studies[/bold yellow]")
                for exp in param_exps:
                    task_count = len(exp.task_pool.tasks) if exp.task_pool else 0
                    params = exp.metadata.get('parameters', {})
                    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    param_branch.add(f"{exp.name} ({task_count} tasks) - {param_str}")
            
            console.print(tree)
        
        elif format == 'json':
            exp_data = []
            for exp in experiments:
                task_count = len(exp.task_pool.tasks) if exp.task_pool else 0
                exp_data.append({
                    'name': exp.name,
                    'task_count': task_count,
                    'parameters': exp.metadata.get('parameters', {}),
                    'metadata': exp.metadata
                })
            
            console.print(Syntax(json.dumps(exp_data, indent=2), "json"))
    
    except Exception as e:
        error_exit(f"Failed to load project: {e}")


@project.command("add-resource")
@click.argument("project_file", type=click.Path(exists=True))
@click.argument("name")
@click.argument("value")
@click.option("--save", "-s", is_flag=True, help="Save changes to file")
def add_shared_resource(project_file: str, name: str, value: str, save: bool):
    """Add a shared resource to the project."""
    
    project_path = Path(project_file)
    
    try:
        if project_path.suffix == '.json':
            proj = Project.from_json(project_path)
        else:
            proj = Project.from_yaml(project_path)
    except Exception as e:
        error_exit(f"Failed to load project: {e}")
    
    # Try to parse value as JSON, otherwise use as string
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value
    
    proj.add_shared_resource(name, parsed_value)
    success_print(f"Added shared resource: {name} = {parsed_value}")
    
    if save:
        if project_path.suffix == '.json':
            proj.to_json(project_path)
        else:
            proj.to_yaml(project_path)
        success_print(f"Project saved to {project_path}")


@project.command("execute")
@click.argument("project_file", type=click.Path(exists=True))
@click.option("--experiments", "-e", multiple=True, help="Specific experiments to execute")
@click.option("--all", "-a", is_flag=True, help="Execute all experiments")
@click.option("--dry-run", is_flag=True, help="Show what would be executed without running")
def execute_project(project_file: str, experiments: tuple, all: bool, dry_run: bool):
    """Execute experiments in a project."""
    
    if not experiments and not all:
        error_exit("Must specify either --experiments or --all")
    
    project_path = Path(project_file)
    
    try:
        with console.status("[bold green]Loading project..."):
            if project_path.suffix == '.json':
                proj = Project.from_json(project_path)
            else:
                proj = Project.from_yaml(project_path)
    except Exception as e:
        error_exit(f"Failed to load project: {e}")
    
    # Determine which experiments to run
    if all:
        exp_names = list(proj.experiments.keys())
    else:
        exp_names = list(experiments)
        # Validate experiment names
        for name in exp_names:
            if name not in proj.experiments:
                error_exit(f"Experiment '{name}' not found in project")
    
    if not exp_names:
        warning_print("No experiments to execute")
        return
    
    # Show execution plan
    table = Table(title="Execution Plan")
    table.add_column("Experiment", style="cyan")
    table.add_column("Tasks", style="magenta")
    table.add_column("Parameters", style="green")
    
    for name in exp_names:
        exp = proj.experiments[name]
        task_count = len(exp.task_pool.tasks) if exp.task_pool else 0
        params = exp.metadata.get('parameters', {})
        param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "None"
        
        table.add_row(name, str(task_count), param_str)
    
    console.print(table)
    
    if dry_run:
        info_print("Dry run - no experiments were actually executed")
        return
    
    # Confirm execution
    if not Confirm.ask(f"Execute {len(exp_names)} experiment(s)?"):
        info_print("Execution cancelled")
        return
    
    # Execute experiments
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing experiments...", total=len(exp_names))
            
            results = proj.batch_execute(exp_names)
            
            progress.advance(task, len(exp_names))
        
        # Show results
        success_count = sum(1 for r in results.values() if r['status'] == 'completed')
        failed_count = len(results) - success_count
        
        if failed_count == 0:
            success_print(f"All {success_count} experiments completed successfully!")
        else:
            console.print(f"[green]{success_count}[/green] completed, [red]{failed_count}[/red] failed")
            
            # Show failed experiments
            failed_exps = [name for name, result in results.items() if result['status'] == 'failed']
            if failed_exps:
                console.print("\n[red]Failed experiments:[/red]")
                for name in failed_exps:
                    console.print(f"  • {name}")
    
    except Exception as e:
        error_exit(f"Execution failed: {e}")


if __name__ == "__main__":
    cli()

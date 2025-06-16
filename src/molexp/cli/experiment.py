"""
Experiment management commands for MolExp CLI.
"""

import click
from .utils import console, error_exit, success_print, warning_print, info_print


@click.group()
def experiment():
    """Experiment management commands."""
    pass


@experiment.command("create")
@click.argument("name")
def create_experiment(name: str):
    """Create a new experiment."""
    # TODO: Implement experiment creation
    info_print(f"Creating experiment: {name}")
    warning_print("Experiment creation not yet implemented")


@experiment.command("info")
@click.argument("experiment_name")
def experiment_info(experiment_name: str):
    """Show experiment information."""
    # TODO: Implement experiment info display
    info_print(f"Showing info for experiment: {experiment_name}")
    warning_print("Experiment info not yet implemented")


@experiment.command("list")
def list_all_experiments():
    """List all experiments across projects."""
    # TODO: Implement experiment listing
    info_print("Listing all experiments")
    warning_print("Experiment listing not yet implemented")

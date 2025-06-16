"""
Task management commands for MolExp CLI.
"""

import click
from .utils import console, error_exit, success_print, warning_print, info_print


@click.group()
def task():
    """Task management commands."""
    pass


@task.command("create")
@click.argument("name")
@click.option("--type", "-t", type=click.Choice(['shell', 'local', 'remote']), default='shell', help="Task type")
def create_task(name: str, type: str):
    """Create a new task."""
    # TODO: Implement task creation
    info_print(f"Creating {type} task: {name}")
    warning_print("Task creation not yet implemented")


@task.command("info")
@click.argument("task_name")
def task_info(task_name: str):
    """Show task information."""
    # TODO: Implement task info display
    info_print(f"Showing info for task: {task_name}")
    warning_print("Task info not yet implemented")


@task.command("list")
def list_all_tasks():
    """List all tasks."""
    # TODO: Implement task listing
    info_print("Listing all tasks")
    warning_print("Task listing not yet implemented")


@task.command("execute")
@click.argument("task_name")
def execute_task(task_name: str):
    """Execute a specific task."""
    # TODO: Implement task execution
    info_print(f"Executing task: {task_name}")
    warning_print("Task execution not yet implemented")

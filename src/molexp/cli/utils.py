"""
Common utilities for CLI commands.
"""

import sys
from rich.console import Console

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

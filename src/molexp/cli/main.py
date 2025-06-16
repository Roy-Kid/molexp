"""
Main CLI entry point for MolExp.
"""

import click
from .utils import print_banner
from .project import project
from .experiment import experiment
from .task import task


@click.group(invoke_without_command=True)
@click.version_option(version="1.0.0", prog_name="molexp")
@click.pass_context
def cli(ctx):
    """
    MolExp - Molecular Experiment Management System
    
    A powerful tool for managing molecular simulation experiments,
    parameter studies, and batch execution workflows.
    """
    if ctx.invoked_subcommand is None and not ctx.params:
        print_banner()


# Register command groups
cli.add_command(project)
cli.add_command(experiment) 
cli.add_command(task)

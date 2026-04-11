"""molexp: Workflow and agent platform for research experiment management.

Core packages:
    - ``molexp.workspace``: File-system-backed experiment management
    - ``molexp.workflow``: DAG-based workflow definition and execution
    - ``molexp.plugins``: Optional capabilities (remote HPC, AI agent, ...)
"""

__version__ = "0.3.0"

# User-facing spec API
from molexp.entry import entry
from molexp.experiment import Experiment
from molexp.project import Project

# Workspace types used in user scripts
from molexp.workspace.param import GridSpace, ParamSpace, UniformSpace
from molexp.workspace.run import RunContext

__all__ = [
    "Project",
    "Experiment",
    "RunContext",
    "GridSpace",
    "UniformSpace",
    "ParamSpace",
    "entry",
]


# Lazy imports — heavy sub-packages are only loaded on first access.
def __getattr__(name: str):
    if name == "workspace":
        from molexp import workspace as _ws

        return _ws
    if name == "workflow":
        from molexp import workflow as _wf

        return _wf
    if name == "plugins":
        from molexp import plugins as _pl

        return _pl
    raise AttributeError(f"module 'molexp' has no attribute {name!r}")

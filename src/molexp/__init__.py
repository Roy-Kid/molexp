"""molexp: Workflow and agent platform for research experiment management.

Core packages:
    - ``molexp.workspace``: File-system-backed experiment management
    - ``molexp.workflow``: DAG-based workflow definition and execution
    - ``molexp.runner``: Script-based parameter sweep helpers
    - ``molexp.plugins``: Optional capabilities (remote HPC, AI agent, …)
"""

__version__ = "0.2.0"

# Eagerly export the lightweight runner API so training scripts can write:
#   from molexp import ExperimentDef, standalone_run
from molexp.runner import ExperimentDef, standalone_run

__all__ = ["ExperimentDef", "standalone_run"]


# Lazy imports — heavy sub-packages are only loaded on first access.
# This keeps ``import molexp`` fast and free of optional-dependency errors.


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

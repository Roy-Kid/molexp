"""molexp: Workflow and agent platform for research experiment management.

Core packages:
    - ``molexp.workspace``: File-system-backed experiment management
    - ``molexp.workflow``: DAG-based workflow definition and execution
    - ``molexp.plugins``: Optional capabilities (remote HPC, AI agent, ...)
"""

__version__ = "0.3.0"

import molcfg

from molexp._logger import Logger, get_logger
from molexp.entry import entry
from molexp.path import Path

# User-facing hierarchy (all from workspace — single source of truth)
from molexp.workspace.experiment import Experiment
from molexp.workspace.param import GridSpace, ParamSpace, UniformSpace
from molexp.workspace.project import Project
from molexp.workspace.run import Run, RunContext
from molexp.workspace.workspace import Workspace

#: Process-global, in-code molexp config — a live ``molcfg.Config``. The
#: sanctioned place to register runtime values (notably LLM API keys) in code,
#: never from environment variables. Mutate with molcfg-native syntax::
#:
#:     import molexp
#:     molexp.config["deepseek_api_key"] = "sk-..."
#:     molexp.config.get("deepseek_api_key")
#:
#: Distinct from :mod:`molexp.profile` — the file-based, per-run profile config.
config: molcfg.Config = molcfg.Config({})

__all__ = [
    "Experiment",
    "GridSpace",
    "Logger",
    "ParamSpace",
    "Path",
    "Project",
    "Run",
    "RunContext",
    "UniformSpace",
    "Workspace",
    "config",
    "entry",
    "get_logger",
]


# Lazy imports — heavy sub-packages are only loaded on first access.
def __getattr__(name: str):  # noqa: ANN202
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

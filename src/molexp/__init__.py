"""molexp: Workflow and agent platform for research experiment management.

Core packages:
    - ``molexp.workspace``: File-system-backed experiment management
    - ``molexp.workflow``: DAG-based workflow definition and execution
    - ``molexp.plugins``: Optional capabilities (remote HPC, AI agent, ...)

Workflow-layer conveniences are re-exported lazily at the top level —
``molexp.WorkflowCompiler``, ``molexp.TaskContext`` and
``molexp.WorkflowRuntime`` resolve on first attribute access (loading
``molexp.workflow``, and thus ``pydantic_graph``, only at that point;
plain ``import molexp`` stays light).
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
    "TaskContext",
    "UniformSpace",
    "WorkflowCompiler",
    "WorkflowRuntime",
    "Workspace",
    "config",
    "entry",
    "get_logger",
]

#: Workflow-layer attributes re-exported lazily at the top level. Resolving
#: any of them imports ``molexp.workflow`` (and transitively pydantic_graph)
#: on first access only — ``import molexp`` must stay light
#: (tests/test_workspace/test_import_guard.py enforces this).
_LAZY_WORKFLOW_ATTRS = ("WorkflowCompiler", "TaskContext", "WorkflowRuntime")


# Lazy imports — heavy sub-packages are only loaded on first access.
# NOTE: must use importlib, not ``from molexp import …`` — the latter re-enters
# this ``__getattr__`` via ``_handle_fromlist`` and recurses infinitely for
# ``from molexp import workflow``-style user imports.
def __getattr__(name: str):  # noqa: ANN202
    if name in ("workspace", "workflow", "plugins"):
        import importlib

        return importlib.import_module(f"molexp.{name}")
    if name in _LAZY_WORKFLOW_ATTRS:
        import importlib

        workflow = importlib.import_module("molexp.workflow")
        return getattr(workflow, name)
    raise AttributeError(f"module 'molexp' has no attribute {name!r}")

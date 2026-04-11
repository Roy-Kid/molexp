"""Entry point registry for CLI discovery.

User scripts call ``me.entry(project)`` at module level to register
a project for CLI execution.  The CLI calls ``load_projects(script)``
to import the script and retrieve all registered projects.

No source scanning, no magic attribute discovery — explicit registration.

Example (user script)::

    import molexp as me

    project = me.Project("my-project")
    # ... define experiments, bind workflows ...
    me.entry(project)

Example (CLI internal)::

    from molexp.entry import load_projects
    projects = load_projects(Path("train.py"))
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.project import Project

_registry: list[Project] = []


def entry(project: Project) -> None:
    """Register a project as a CLI entry point.

    When a script is imported by ``molexp run``, this call populates
    the global registry.  When the script is run directly
    (``python script.py``), nobody reads the registry — it is
    effectively a no-op.

    Args:
        project: A :class:`~molexp.Project` spec to register.
    """
    _registry.append(project)


def load_projects(script: Path) -> list[Project]:
    """Import a user script and return all registered projects.

    This is the **only** function the CLI calls.  It encapsulates
    clear → import → read in a single call.

    Args:
        script: Path to the Python script containing ``me.entry()`` calls.

    Returns:
        List of registered :class:`~molexp.Project` specs.

    Raises:
        RuntimeError: If the script cannot be loaded.
    """
    _registry.clear()
    _import_script(script)
    return list(_registry)


def clear_registry() -> None:
    """Clear the registry (for tests)."""
    _registry.clear()


def _import_script(script: Path) -> None:
    """Dynamically import a user script."""
    spec = importlib.util.spec_from_file_location("_molexp_script", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {script}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except SystemExit:
        pass  # silence __main__ guards

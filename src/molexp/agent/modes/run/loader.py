"""Load the LLM-authored workflow from a :class:`MaterializedWorkspaceHandoff`.

AuthorMode materializes an experiment workspace and hands RunMode a
:class:`~molexp.agent.modes.author.handoff.MaterializedWorkspaceHandoff`
naming the entrypoint: a dotted module path (``entrypoint_module``) and a
callable symbol (``entrypoint_symbol``) inside it. :func:`load_materialized_workflow`
puts ``source_root`` on ``sys.path``, imports the module, reads the
symbol, calls it if it is a factory, and asserts the resulting object is
a :class:`molexp.workflow.Workflow`.

This module imports the *public* :class:`molexp.workflow.Workflow` only —
no ``pydantic_graph``. The loaded :class:`Workflow` is driven purely
through its public ``execute`` / ``run_on`` API by :class:`RunExecutor`.

The import only ever happens *after* the ``approve_execution`` gate has
cleared — RunMode never calls this before the gate.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from molexp.workflow import Workflow

if TYPE_CHECKING:
    from molexp.agent.modes.author.handoff import MaterializedWorkspaceHandoff

__all__ = ["WorkflowLoadError", "load_materialized_workflow"]


class WorkflowLoadError(RuntimeError):
    """Raised when a materialized workflow entrypoint cannot be resolved.

    Covers a missing module, a missing symbol, or a symbol that does not
    resolve to a :class:`molexp.workflow.Workflow`.
    """


def load_materialized_workflow(handoff: MaterializedWorkspaceHandoff) -> Workflow:
    """Import the materialized workflow named by ``handoff`` and type-assert it.

    Steps:

    1. Prepend ``handoff.source_root`` to ``sys.path`` (idempotently).
    2. Import ``handoff.entrypoint_module``.
    3. Read ``handoff.entrypoint_symbol`` from the module.
    4. If the symbol is callable and not already a :class:`Workflow`,
       call it with no arguments (the factory convention — AuthorMode
       emits ``create_workflow``).
    5. Assert the resolved object is a :class:`molexp.workflow.Workflow`.

    Args:
        handoff: The :class:`MaterializedWorkspaceHandoff` from AuthorMode.

    Returns:
        The loaded :class:`Workflow`.

    Raises:
        WorkflowLoadError: when the module / symbol is missing or the
            resolved object is not a :class:`Workflow`.
    """
    _ensure_on_path(str(handoff.source_root))

    module_name = handoff.entrypoint_module
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise WorkflowLoadError(
            f"cannot import entrypoint module {module_name!r} "
            f"(source_root={handoff.source_root}): {exc}"
        ) from exc

    symbol_name = handoff.entrypoint_symbol
    if not hasattr(module, symbol_name):
        raise WorkflowLoadError(f"entrypoint module {module_name!r} has no symbol {symbol_name!r}")
    symbol = getattr(module, symbol_name)

    workflow = _resolve_workflow(symbol, module_name=module_name, symbol_name=symbol_name)
    if not isinstance(workflow, Workflow):
        raise WorkflowLoadError(
            f"entrypoint {module_name}:{symbol_name} resolved to "
            f"{type(workflow).__name__}, not a molexp.workflow.Workflow"
        )
    return workflow


def _resolve_workflow(symbol: object, *, module_name: str, symbol_name: str) -> object:
    """Resolve ``symbol`` to a workflow object — call factories, pass through.

    A symbol that is already a :class:`Workflow` is returned unchanged; a
    callable symbol is invoked once with no arguments (the factory
    convention). A non-callable, non-``Workflow`` symbol is returned
    as-is for the caller's type assertion to reject.
    """
    if isinstance(symbol, Workflow):
        return symbol
    if callable(symbol):
        factory = cast("Callable[[], object]", symbol)
        try:
            return factory()
        except Exception as exc:
            raise WorkflowLoadError(
                f"entrypoint factory {module_name}:{symbol_name} raised {type(exc).__name__}: {exc}"
            ) from exc
    return symbol


def _ensure_on_path(source_root: str) -> None:
    """Prepend ``source_root`` to ``sys.path`` if not already present."""
    if source_root and source_root not in sys.path:
        sys.path.insert(0, source_root)

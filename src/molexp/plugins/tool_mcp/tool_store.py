"""Tool resource store: file-backed user/workspace tool declarations.

Replaces the older ``tool_registry.py`` module. The store backs three
layers:

- **registrations** — populated at import time by the
  :func:`default_tool` decorator on package-shipped tools (e.g. the
  workspace-manipulation tools in ``_pydantic_ai/workspace_tools.py``)
- **user** — ``~/.molexp/tools.json`` (JSON-declared tools shared
  across every workspace)
- **workspace** — ``<root>/.tools.json`` (JSON-declared tools owned
  by one workspace)

The on-disk schema for a single tool entry mirrors :class:`ToolSpec`.
The ``invoker`` field is a discriminated union — Python tools use
:class:`PythonInvoker` (target by dotted module path), HTTP webhooks
use :class:`HttpInvoker` (URL + method + body template).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Union

from pydantic import BaseModel, Discriminator, Field

from .resources.base import ResourceSpec, Scope, TieredResourceStore, _now_iso

USER_TOOLS_FILENAME = "tools.json"
WORKSPACE_TOOLS_FILENAME = ".tools.json"
USER_HOME_DIR_NAME = ".molexp"
KIND_KEY = "tools"


# Process-level map from a :class:`PythonInvoker.target` (``module:name``)
# to the live callable. Populated by :func:`default_tool`. Distinct from
# :class:`ToolStore._registrations` because the spec is JSON-serialisable
# while the function reference is not. Catalog and admin layers consult
# this map to resolve a :class:`PythonInvoker` back to its implementation.
_PYTHON_TOOL_IMPLS: dict[str, Callable] = {}


def get_python_tool_impl(target: str) -> Callable | None:
    """Return the live callable backing a ``module:name`` Python invoker target.

    Args:
        target: Dotted ``module:name`` reference, identical to the
            value stored on :class:`PythonInvoker.target`.

    Returns:
        The registered async callable, or ``None`` if no Python tool
        has been registered against this target. The lookup is a plain
        dict read; safe for concurrent calls but the underlying map is
        mutated only at import time by :func:`default_tool`, so no lock
        is held.
    """
    return _PYTHON_TOOL_IMPLS.get(target)


def clear_python_tool_impls() -> None:
    """Drop every Python-tool implementation pointer (test helper).

    Empties :data:`_PYTHON_TOOL_IMPLS`; production code should not call
    this. Not thread-safe — tests invoke it in setup/teardown when no
    concurrent agent runs are active.
    """
    _PYTHON_TOOL_IMPLS.clear()


# ── Invoker variants ───────────────────────────────────────────────────────


class PythonInvoker(BaseModel):
    """Reference to a Python callable contributed by the package.

    ``target`` is a ``module:name`` dotted path. The catalog resolves it
    via :func:`importlib.import_module` at toolset-build time.
    Decorated tools registered via :func:`default_tool` always end up
    here.
    """

    kind: Literal["python"] = "python"
    target: str = Field(min_length=1, max_length=512)


class HttpInvoker(BaseModel):
    """Webhook invocation spec for user-declared tools.

    The catalog wraps this into an async function that issues the
    declared HTTP request when the agent calls the tool. ``headers``
    values may carry ``${SECRET:KEY}`` placeholders resolved against
    the workspace's MCP secret store at request time. ``body_template``
    receives the LLM-supplied arguments via simple ``{name}``
    substitution; an empty template means "send arguments as JSON body".
    """

    kind: Literal["http"] = "http"
    url: str = Field(min_length=1, max_length=4096)
    method: Literal["GET", "POST", "PUT", "DELETE"] = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    body_template: str = ""


ToolInvoker = Annotated[
    Union[PythonInvoker, HttpInvoker],
    Discriminator("kind"),
]


# ── Tool spec ──────────────────────────────────────────────────────────────


ToolCategory = Literal["workspace", "workflow", "chat", "control"]


class ToolSpec(ResourceSpec):
    """Single agent-callable tool declaration.

    Subclasses :class:`ResourceSpec` and adds tool-specific fields:

    - ``parameters_schema`` — JSON Schema for the tool's arguments. For
      Python invokers this can be empty (pydantic-ai introspects the
      live function); for HTTP invokers it documents the LLM's view.
    - ``requires_approval`` — when ``True``, the runtime gates calls
      through the approval policy (orthogonal to the policy's own
      ``require_approval_for`` glob list).
    - ``category`` — coarse grouping for UI display.
    - ``mutates`` — declares whether the tool has write side-effects.
      Plan-mode filtering reads this flag to hide mutators.
    - ``invoker`` — discriminated union; how to actually call the tool.
    """

    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    category: ToolCategory = "workspace"
    mutates: bool = False
    invoker: ToolInvoker


# ── Store ──────────────────────────────────────────────────────────────────


class ToolStore(TieredResourceStore[ToolSpec]):
    """Three-layer tool store: registrations + user file + workspace file.

    Constructor parameters mirror the user-home-injection pattern of
    :class:`SkillStore` for testing convenience: leaving
    ``user_home_dir`` as ``None`` defaults to ``~/.molexp/``.

    Thread-safety: CRUD operations inherit the parent's
    :class:`threading.Lock` around read-modify-write cycles, so calls
    from the FastAPI thread pool serialise per store instance.
    The class-level registrations list is populated only at import
    time by :func:`default_tool`; subsequent reads via
    :meth:`list_all` are safe without additional locking.
    """

    _registrations: ClassVar[list[ToolSpec]] = []

    def __init__(
        self,
        root: str | Path,
        user_home_dir: str | Path | None = None,
    ) -> None:
        """Bind the tool store to a workspace root and user-home dir.

        Args:
            root: Workspace root directory; the workspace-tier file
                lives at ``<root>/.tools.json``.
            user_home_dir: Override for the user-tier directory. When
                ``None`` (the default), resolves to ``~/.molexp/`` and
                the user-tier file is ``~/.molexp/tools.json``. Tests
                inject a temp directory here.
        """
        workspace_path = Path(root) / WORKSPACE_TOOLS_FILENAME
        if user_home_dir is None:
            user_home_dir = Path.home() / USER_HOME_DIR_NAME
        user_path = Path(user_home_dir) / USER_TOOLS_FILENAME
        super().__init__(
            user_path=user_path,
            workspace_path=workspace_path,
            spec_cls=ToolSpec,
            kind_key=KIND_KEY,
        )


# ── Decorator ──────────────────────────────────────────────────────────────


def default_tool(
    fn: Callable | None = None,
    *,
    category: ToolCategory = "workspace",
    mutates: bool = False,
    requires_approval: bool = False,
    name: str | None = None,
) -> Callable:
    """Register an async function as a default (package-shipped) tool.

    Routes through :class:`ToolStore` so the package's tools share the
    same listing/CRUD surface as user- and workspace-declared tools.
    The decorated function is returned unchanged so pydantic-ai can
    introspect its signature.

    Idempotent: re-importing the module replaces the prior registration
    rather than producing duplicates. Registration mutates two
    process-level structures (the ``ToolStore`` registrations list and
    :data:`_PYTHON_TOOL_IMPLS`) at import time only; readers do not
    hold a lock.

    Args:
        fn: Function being decorated. ``None`` when the decorator is
            applied with parentheses (e.g. ``@default_tool(...)``); in
            that case a decorator is returned for two-stage application.
        category: Coarse UI grouping; surfaced verbatim on
            :attr:`ToolSpec.category`.
        mutates: ``True`` if the tool has write side-effects. Plan-mode
            filtering hides mutators from the agent.
        requires_approval: When ``True``, every call is gated through
            the approval policy regardless of the policy's glob list.
        name: Override for the tool's id/name. Defaults to
            ``fn.__name__``.

    Returns:
        Either the unchanged function (when called without parentheses)
        or a decorator that itself returns the unchanged function. The
        side effect — registering the spec and storing the live impl
        pointer — happens during decoration.

    Example::

        @default_tool(category="workflow", mutates=True)
        async def submit_run(ctx, project_id: str, ...) -> dict:
            ...
    """

    def decorator(f: Callable) -> Callable:
        tool_name = name or f.__name__
        target = f"{f.__module__}:{f.__name__}"
        description = (inspect.getdoc(f) or "").strip()
        now = _now_iso()
        spec = ToolSpec(
            id=tool_name,
            name=tool_name,
            description=description,
            scope=Scope.USER,  # nominal — registrations carry no writable scope
            category=category,
            mutates=mutates,
            requires_approval=requires_approval,
            invoker=PythonInvoker(kind="python", target=target),
            created_at=now,
            updated_at=now,
        )
        ToolStore.register(spec)
        _PYTHON_TOOL_IMPLS[target] = f
        return f

    if fn is not None:
        return decorator(fn)
    return decorator


__all__ = [
    "HttpInvoker",
    "KIND_KEY",
    "PythonInvoker",
    "ToolCategory",
    "ToolInvoker",
    "ToolSpec",
    "ToolStore",
    "USER_HOME_DIR_NAME",
    "USER_TOOLS_FILENAME",
    "WORKSPACE_TOOLS_FILENAME",
    "clear_python_tool_impls",
    "default_tool",
    "get_python_tool_impl",
]

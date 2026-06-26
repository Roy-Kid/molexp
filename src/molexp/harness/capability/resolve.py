"""Resolve a capability ``callable_path`` to a real Python callable.

A :class:`~molexp.harness.schemas.capability.ToolCapability` carries a
``callable_path`` string naming the function the harness should invoke.
:func:`resolve_callable` turns that string into the callable, or raises the
typed :class:`~molexp.harness.errors.CapabilityResolutionError` — never a
silent fallback. The same resolver is used both as an in-process fail-fast
guard (before any subprocess is launched) and inside the materialized runner,
so the guard and the actual invocation resolve the identical symbol.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from molexp.harness.errors import CapabilityResolutionError

if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["resolve_callable"]


def resolve_callable(callable_path: str | None) -> Callable[..., object]:
    """Resolve a dotted ``callable_path`` to the callable it names.

    Two spellings are accepted: ``"module.path:attr"`` (preferred, an
    unambiguous module/attribute split) and ``"module.path.attr"`` (the last
    dotted segment is the attribute). The module is imported and the attribute
    fetched; the result must itself be callable.

    Args:
        callable_path: The capability's ``callable_path``. ``None`` or an
            empty/whitespace string is rejected.

    Returns:
        The resolved callable.

    Raises:
        CapabilityResolutionError: If the path is ``None``/empty, names an
            unimportable module, references a missing attribute, or resolves to
            a non-callable object. No fallback is attempted.
    """
    if not callable_path or not callable_path.strip():
        raise CapabilityResolutionError("callable_path is empty or None; nothing to resolve")

    if ":" in callable_path:
        module_name, _, attr = callable_path.partition(":")
    else:
        module_name, _, attr = callable_path.rpartition(".")
    if not module_name or not attr:
        raise CapabilityResolutionError(
            f"callable_path {callable_path!r} is not a 'module:attr' or 'module.attr' path"
        )

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # ImportError + any module-load-time error
        raise CapabilityResolutionError(
            f"could not import module {module_name!r} for callable_path {callable_path!r}: {exc!r}"
        ) from exc

    try:
        attribute = getattr(module, attr)
    except AttributeError as exc:
        raise CapabilityResolutionError(
            f"module {module_name!r} has no attribute {attr!r} (callable_path {callable_path!r})"
        ) from exc

    if not callable(attribute):
        raise CapabilityResolutionError(
            f"callable_path {callable_path!r} resolved to a non-callable {type(attribute).__name__}"
        )
    return attribute

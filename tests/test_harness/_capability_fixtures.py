"""Importable callable fixtures for the ``InvokeCapability`` stage tests.

This module is deliberately a *real, importable* submodule of the
``tests.test_harness`` package (which carries an ``__init__.py``) so that
:func:`molexp.harness.resolve_callable` and the materialized
``invoke_capability.py`` subprocess runner can resolve a callable from a
``"module.path:attr"`` reference end-to-end.

Two symbols are exported:

- :func:`echo` — a trivial callable used as the happy-path target. It echoes
  its keyword arguments so a test can assert the exact ``result.json`` the
  runner writes.
- :data:`NOT_CALLABLE` — a module attribute that *exists* but is not callable,
  exercising the resolver's "attribute is not callable" failure mode.
"""

from __future__ import annotations

from typing import Any

__all__ = ["NOT_CALLABLE", "echo"]


def echo(**kwargs: Any) -> dict[str, Any]:
    """Return a dict echoing the keyword arguments passed.

    Args:
        **kwargs: Arbitrary keyword arguments forwarded by the capability
            invocation runner (``the_callable(**parameters)``).

    Returns:
        A mapping ``{"echoed": <copy of kwargs>}`` so callers can assert the
        exact payload round-tripped through ``result.json``.
    """
    return {"echoed": dict(kwargs)}


NOT_CALLABLE: int = 42
"""A bound module attribute that exists but is not callable.

Used to drive :func:`molexp.harness.resolve_callable`'s
"attribute exists but is not callable" failure mode.
"""

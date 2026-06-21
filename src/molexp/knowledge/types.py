"""Generic concept-type registry — the sole content of ``molexp.knowledge``.

``molexp.knowledge`` owns **only** this mechanism: an open, forward-compatible
registry mapping a Concept's ``type`` string to its Python class. A storage
layer (``molexp.workspace``) uses it to reconstruct typed Concepts from disk;
upstream layers register their own types via ``@concept_type`` /
``register_concept_type`` without ``knowledge`` importing them. An unknown type
resolves to a caller-supplied default.

``Folder``, the concept hierarchy, ``Library`` and all storage live in
``molexp.workspace`` — knowledge is just the registry, so it stays a tiny,
dependency-free, cross-cutting utility with no duplicate of workspace's storage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

_REGISTRY: dict[str, type] = {}

_C = TypeVar("_C", bound=type)


def register_concept_type(type_str: str, cls: type) -> None:
    """Register *cls* as the class for Concept ``type`` *type_str*.

    Re-registering the *same* class is a no-op; registering a *different* class
    for an already-claimed type raises, to surface collisions.

    Raises:
        ValueError: if *type_str* is already bound to a different class.
    """
    existing = _REGISTRY.get(type_str)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"concept type {type_str!r} already registered to {existing.__name__!r}; "
            f"cannot re-register to {cls.__name__!r}"
        )
    _REGISTRY[type_str] = cls


def concept_type(type_str: str) -> Callable[[_C], _C]:
    """Class decorator registering the decorated Concept class as *type_str*."""

    def register(cls: _C) -> _C:
        register_concept_type(type_str, cls)
        return cls

    return register


def resolve_concept_type(type_str: str, default: type) -> type:
    """Return the class registered for *type_str*, or *default* if unknown."""
    return _REGISTRY.get(type_str, default)


__all__ = ["concept_type", "register_concept_type", "resolve_concept_type"]

"""Generic concept-type registry — the core of ``molexp.knowledge``.

An open, forward-compatible registry mapping a Concept's ``type`` string to its
Python class, so a storage layer (``molexp.workspace``) can reconstruct typed
Concepts from disk. Upstream layers register their own types via
``@concept_type`` / ``register_concept_type`` without ``knowledge`` importing
them; an unknown type resolves to a caller-supplied default.

The registry is **generic** — it stores plain classes and preserves the
caller's class type via a ``TypeVar`` — so it works for any storage layer's
Folder/Concept classes (knowledge owns no Folder of its own in the end state).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

_REGISTRY: dict[str, type] = {}


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


def concept_type[C: type](type_str: str) -> Callable[[C], C]:
    """Class decorator registering the decorated Concept class as *type_str*."""

    def register(cls: C) -> C:
        register_concept_type(type_str, cls)
        return cls

    return register


def resolve_concept_type[D: type](type_str: str, default: D) -> D:
    """Return the class registered for *type_str*, or *default* if unknown.

    The return type matches *default* (a type parameter), so a caller passing its
    own ``Folder`` base gets its own ``Folder`` subtype back.
    """
    return cast("D", _REGISTRY.get(type_str, default))


__all__ = ["concept_type", "register_concept_type", "resolve_concept_type"]

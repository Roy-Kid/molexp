"""Concept-type registry for ``molexp.knowledge``.

Maps a Concept's ``meta.yaml`` ``type`` string to its Python class so that
reading a bundle reconstructs *typed* Concepts (a ``project`` dir becomes a
:class:`~molexp.knowledge.Project`, not a bare ``Folder``). The registry is:

* **open** — upstream layers register their own Concept types
  (``register_concept_type`` / ``@concept_type``) without ``molexp.knowledge``
  ever importing them, keeping this the bottom layer of the DAG;
* **forward-compatible** — an unknown ``type`` resolves to a caller-supplied
  default (the base ``Folder``), so a bundle written by a newer version still
  walks rather than crashing.

This module imports nothing from ``.folder`` at runtime (the default class is
passed in by the caller), so it carries no import cycle. ``Folder`` is
referenced only for type-checking.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .folder import Folder

_REGISTRY: dict[str, type[Folder]] = {}

_C = TypeVar("_C", bound="type[Folder]")


def register_concept_type(type_str: str, cls: type[Folder]) -> None:
    """Register *cls* as the class for Concept ``type`` *type_str*.

    Re-registering the *same* class is a no-op; registering a *different*
    class for an already-claimed type raises, to surface collisions.

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


def resolve_concept_type(type_str: str, default: type[Folder]) -> type[Folder]:
    """Return the class registered for *type_str*, or *default* if unknown."""
    return _REGISTRY.get(type_str, default)

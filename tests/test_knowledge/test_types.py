"""Tests for the ``molexp.knowledge`` concept-type registry.

The registry maps a Concept's ``meta.yaml`` ``type`` string to its Python
class so a storage layer reconstructs typed Concepts (not bare bases). It is
*open* (upstream layers register their own types) and *forward-compatible*
(unknown types resolve to a supplied default). The registry is process-global,
so these tests use unique type strings to stay isolated.

The registry owns no storage class of its own: these tests use a tiny local
placeholder class to prove the registry works for any caller-supplied type.
"""

from __future__ import annotations

import pytest

import molexp.knowledge as knowledge
from molexp.knowledge.types import (
    concept_type,
    register_concept_type,
    resolve_concept_type,
)


class _Concept:
    """Local placeholder standing in for any caller's Concept/Folder class."""


def test_package_surface_is_registry_only() -> None:
    assert knowledge.__all__ == [
        "concept_type",
        "register_concept_type",
        "resolve_concept_type",
    ]
    for name in ("Folder", "Library", "ConceptMeta", "FileSystem", "Run", "Reference"):
        assert not hasattr(knowledge, name)


def test_register_and_resolve_round_trip() -> None:
    class CustomRT(_Concept):
        pass

    register_concept_type("test-custom-rt", CustomRT)
    assert resolve_concept_type("test-custom-rt", _Concept) is CustomRT


def test_decorator_registers_on_definition() -> None:
    @concept_type("test-custom-deco")
    class Decorated(_Concept):
        pass

    assert resolve_concept_type("test-custom-deco", _Concept) is Decorated


def test_unknown_type_resolves_to_default() -> None:
    assert resolve_concept_type("totally-unknown-xyz", _Concept) is _Concept


def test_reregister_same_class_is_noop() -> None:
    class Same(_Concept):
        pass

    register_concept_type("test-same", Same)
    register_concept_type("test-same", Same)  # idempotent — must not raise
    assert resolve_concept_type("test-same", _Concept) is Same


def test_reregister_different_class_raises() -> None:
    class First(_Concept):
        pass

    class Second(_Concept):
        pass

    register_concept_type("test-collide", First)
    with pytest.raises(ValueError):
        register_concept_type("test-collide", Second)

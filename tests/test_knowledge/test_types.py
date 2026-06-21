"""Tests for the ``molexp.knowledge`` concept-type registry.

The registry maps a Concept's ``meta.yaml`` ``type`` string to its Python
class so a bundle reconstructs typed Concepts (not bare ``Folder``s). It is
*open* (upstream layers register their own types) and *forward-compatible*
(unknown types resolve to a supplied default). The registry is process-global,
so these tests use unique type strings to stay isolated.
"""

from __future__ import annotations

import pytest

from molexp.knowledge import Folder
from molexp.knowledge.types import (
    concept_type,
    register_concept_type,
    resolve_concept_type,
)


def test_register_and_resolve_round_trip() -> None:
    class CustomRT(Folder):
        pass

    register_concept_type("test-custom-rt", CustomRT)
    assert resolve_concept_type("test-custom-rt", Folder) is CustomRT


def test_decorator_registers_on_definition() -> None:
    @concept_type("test-custom-deco")
    class Decorated(Folder):
        pass

    assert resolve_concept_type("test-custom-deco", Folder) is Decorated


def test_unknown_type_resolves_to_default() -> None:
    assert resolve_concept_type("totally-unknown-xyz", Folder) is Folder


def test_reregister_same_class_is_noop() -> None:
    class Same(Folder):
        pass

    register_concept_type("test-same", Same)
    register_concept_type("test-same", Same)  # idempotent — must not raise
    assert resolve_concept_type("test-same", Folder) is Same


def test_reregister_different_class_raises() -> None:
    class First(Folder):
        pass

    class Second(Folder):
        pass

    register_concept_type("test-collide", First)
    with pytest.raises(ValueError):
        register_concept_type("test-collide", Second)

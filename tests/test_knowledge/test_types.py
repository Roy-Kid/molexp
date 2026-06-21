"""Tests for the ``molexp.knowledge`` generic concept-type registry.

The registry maps a Concept's ``type`` string to its Python class so a storage
layer can reconstruct typed Concepts. It is *open* (callers register their own
types) and *forward-compatible* (unknown types resolve to a supplied default).
The registry is process-global, so these tests use unique type strings.
"""

from __future__ import annotations

import pytest

from molexp.knowledge import concept_type, register_concept_type, resolve_concept_type


class _Base:
    pass


def test_register_and_resolve_round_trip() -> None:
    class CustomRT:
        pass

    register_concept_type("test-custom-rt", CustomRT)
    assert resolve_concept_type("test-custom-rt", _Base) is CustomRT


def test_decorator_registers_on_definition() -> None:
    @concept_type("test-custom-deco")
    class Decorated:
        pass

    assert resolve_concept_type("test-custom-deco", _Base) is Decorated


def test_unknown_type_resolves_to_default() -> None:
    assert resolve_concept_type("totally-unknown-xyz", _Base) is _Base


def test_reregister_same_class_is_noop() -> None:
    class Same:
        pass

    register_concept_type("test-same", Same)
    register_concept_type("test-same", Same)  # idempotent — must not raise
    assert resolve_concept_type("test-same", _Base) is Same


def test_reregister_different_class_raises() -> None:
    class First:
        pass

    class Second:
        pass

    register_concept_type("test-collide", First)
    with pytest.raises(ValueError):
        register_concept_type("test-collide", Second)

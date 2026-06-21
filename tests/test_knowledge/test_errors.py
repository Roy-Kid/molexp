"""Tests for the typed error family of :mod:`molexp.knowledge`."""

from __future__ import annotations

from molexp.knowledge import ConceptExistsError, ConceptNotFoundError


def test_not_found_is_lookup_error_carrying_identifier() -> None:
    err = ConceptNotFoundError("projects/peo")
    assert isinstance(err, LookupError)
    assert "projects/peo" in str(err)


def test_exists_is_value_error_carrying_identifier() -> None:
    err = ConceptExistsError("projects/peo")
    assert isinstance(err, ValueError)
    assert "projects/peo" in str(err)

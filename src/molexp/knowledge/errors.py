"""Typed error family for the OKF ``molexp.knowledge`` layer.

Mirrors the workspace error taxonomy (``*NotFoundError`` →
:class:`LookupError`, ``*ExistsError`` → :class:`ValueError`). These are
wired onto the ``Folder`` base as the ``_not_found_error_cls`` /
``_exists_error_cls`` ClassVars in a later sub-spec (okf-01-03).
"""

from __future__ import annotations


class ConceptNotFoundError(LookupError):
    """No Concept exists at the requested path / identifier."""

    def __init__(self, identifier: object) -> None:
        self.identifier = identifier
        super().__init__(f"concept not found: {identifier}")


class ConceptExistsError(ValueError):
    """A Concept already exists at the requested path / identifier."""

    def __init__(self, identifier: object) -> None:
        self.identifier = identifier
        super().__init__(f"concept already exists: {identifier}")


__all__ = ["ConceptExistsError", "ConceptNotFoundError"]

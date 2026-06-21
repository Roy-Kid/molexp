"""``molexp.knowledge`` — the concept-type registry (and nothing else).

A tiny, dependency-free cross-cutting utility: an open, forward-compatible
registry mapping a Concept's ``type`` string to its Python class
(``@concept_type`` / ``register_concept_type`` / ``resolve_concept_type``).

Storage — ``Folder``, the ``Workspace/Project/Experiment/Run`` hierarchy,
``Library``, the filesystem seam — lives in ``molexp.workspace``. knowledge
deliberately does **not** re-implement any of it; the storage layer registers
its concept types here so a bundle can reconstruct typed Concepts from disk.
"""

from .types import concept_type, register_concept_type, resolve_concept_type

__all__ = [
    "concept_type",
    "register_concept_type",
    "resolve_concept_type",
]

"""``molexp.knowledge`` — the open concept-type registry.

A single-responsibility bottom layer: the generic, open registry that maps a
Concept's ``meta.yaml`` ``type`` string to its Python class
(``@concept_type`` / ``register_concept_type`` / ``resolve_concept_type``).
The OKF-native storage substrate lives in ``molexp.workspace``; that layer
uses this registry to reconstruct typed ``Folder`` subclasses from each
Concept's persisted ``type``. Upstream layers register their own types here
without ``knowledge`` importing them; an unknown type resolves to a
caller-supplied default.

Allowed imports for this layer: stdlib and pydantic only. It MUST NOT import
``molexp.workspace`` or any upstream layer — enforced by
``tests/test_knowledge/test_import_guard.py`` (an AST source scan).
"""

from .types import concept_type, register_concept_type, resolve_concept_type

__all__ = ["concept_type", "register_concept_type", "resolve_concept_type"]

"""Capability registry contract for ``molexp.harness``.

Two public symbols:

- :class:`CapabilityRegistry` — ``runtime_checkable`` Protocol every backend
  implements. The validator
  (:func:`molexp.harness.validators.bound_workflow.validate_bound_workflow`)
  programs against this Protocol; per-package adapters (molpy, molpack, …)
  ship their own concrete impls.
- :class:`InMemoryCapabilityRegistry` — concrete impl used by tests + by
  callers who want to assemble a registry by hand from a list of
  :class:`ToolCapability` instances.

Per-package adapters (MolPy / MolPack / MolExp / MolVis / MolQ from
``harness-goal.md`` §16 Phase 4) intentionally live in their *own*
packages, not in the harness — the harness defines the contract, each
package supplies its capability catalog.
"""

from __future__ import annotations

from molexp.harness.registry.capability_registry import CapabilityRegistry
from molexp.harness.registry.in_memory import InMemoryCapabilityRegistry

__all__ = ["CapabilityRegistry", "InMemoryCapabilityRegistry"]

"""Which catalog entries may be bound as plan capabilities.

molmcp indexes **everything** under the workspace and env (notes, tests, UI
mocks, private helpers, docs). Only a small subset is a legitimate
workflow-binding target. This module is the single gate:

* used by :mod:`molexp.mcp_capabilities` when mapping discovery hits, and
* used by :mod:`molexp.harness.prompts.capability_catalog` when rendering
  binder-facing catalogs.

Contract (molmcp README): a source symbol is **evidence**, not a capability.
We still accept science ``function`` / ``method`` / ``class`` symbols as
bindable *API surface* for plan codegen (the registry of ``executable=true``
manifests is often empty in local dev); we never accept notes, tests, UI,
docs sections, or non-science packages.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.harness.schemas import ToolCapability

__all__ = [
    "PLAN_SCIENCE_PACKAGE_ROOTS",
    "is_bindable_capability",
    "is_bindable_capability_id",
    "is_bindable_kind",
]

#: Package roots a plan may bind to (domain science + config/scheduler).
#: Built-in ``molexp.curation.*`` / ``molexp.lifecycle.*`` are registered
#: separately and pass the id check via the ``molexp.curation`` / ``lifecycle``
#: prefix rule — free-form ``molexp.harness.*`` search hits do **not**.
PLAN_SCIENCE_PACKAGE_ROOTS: frozenset[str] = frozenset(
    {
        "molpy",
        "molpack",
        "molcfg",
        "molq",
        "atomiverse",
    }
)

#: Explicitly never bindable as science capabilities (infra / noise).
_BLOCKED_PACKAGE_ROOTS: frozenset[str] = frozenset(
    {
        "mollog",
        "molmcp",
        "molhub",
        "molws",
        "molnex",
        "molrs",  # Rust core — not a Python agent surface
        "workspace",
        "tests",
        "ui",
        "mkdocstrings_handlers",
        "mkdocstrings",
    }
)

#: Node kinds that are evidence / docs / tests — never bind targets.
_BLOCKED_KINDS: frozenset[str] = frozenset(
    {
        "module",
        "package",
        "field",
        "test",
        "example",
        "section",
        "type_alias",
        "constant",
        "variable",
        "property",
        "html",
        "markdown",
        "doc",
        "note",
    }
)

#: Accept only these kinds as bindable API surface.
_ALLOWED_KINDS: frozenset[str] = frozenset(
    {
        "function",
        "method",
        "class",
        "",  # unknown kind: still subject to id hygiene
    }
)

#: Dotted Python qualname: ``pkg.mod.Name`` (at least one dot).
_PYTHON_QUALNAME_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$"
)

#: Path / notes / section-anchor pollution.
_JUNK_ID_MARKERS: tuple[str, ...] = (
    "::",
    ".claude",
    ".github",
    "/notes/",
    "notes/",
    "\\",
    " ",
    "<",
    ">",
    "`",
)


def is_bindable_kind(kind: str | None) -> bool:
    """Return True if the discovery ``kind`` may become a plan capability."""
    if kind is None:
        return True
    k = kind.strip().lower()
    if k in _BLOCKED_KINDS:
        return False
    return not (k and k not in _ALLOWED_KINDS)


def is_bindable_capability_id(cap_id: str) -> bool:
    """Return True if ``cap_id`` is a hygiene-clean, science-scoped qualname."""
    if not cap_id or not isinstance(cap_id, str):
        return False
    cid = cap_id.strip()
    if not cid or cid.startswith((".", "/")):
        return False
    if any(marker in cid for marker in _JUNK_ID_MARKERS):
        return False
    if not _PYTHON_QUALNAME_RE.match(cid):
        return False
    root = cid.split(".", 1)[0].lower()
    if root in _BLOCKED_PACKAGE_ROOTS:
        return False
    # molexp: only curation / lifecycle built-ins, never harness/server search hits.
    if root == "molexp":
        second = cid.split(".", 2)[1].lower() if cid.count(".") >= 1 else ""
        return second in {"curation", "lifecycle"}
    if root not in PLAN_SCIENCE_PACKAGE_ROOTS:
        return False
    parts = cid.split(".")
    leaf = parts[-1]
    if leaf.startswith("_"):
        return False
    # Dunder noise.
    if leaf.startswith("__") and leaf.endswith("__"):
        return False
    # Intermediate private modules (molpy.foo._internal.Bar) and CLI plumbing
    # are not plan-binding targets.
    if any(p.startswith("_") for p in parts[:-1]):
        return False
    return "cli" not in parts[1:3]


def is_bindable_capability(cap: ToolCapability) -> bool:
    """Full gate for a mapped :class:`ToolCapability`."""
    kind = cap.tags[0] if cap.tags else ""
    if not is_bindable_kind(kind):
        return False
    return is_bindable_capability_id(cap.id)

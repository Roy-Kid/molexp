"""Contract + drift-guard tests for the built-in curation capability catalog.

Feature under test (link 04, ``workspace-curation-toolset``): a new harness
subpackage ``molexp.harness.capabilities`` with a ``curation`` module exposing a
frozen ``CURATION_CAPABILITIES`` tuple and a ``curation_capabilities()``
accessor. Each entry is a :class:`molexp.harness.schemas.ToolCapability` that
binds one ``molexp.workspace.curation.*`` function.

These tests are intentionally *drift guards*: rather than asserting per-entry
literals, they iterate over the catalog and cross-check each entry against the
**live** signature of the function its ``callable_path`` resolves to. A renamed
function, a changed signature, or a mis-declared schema fails the build instead
of silently shipping a broken capability.

Written RED-first: the ``molexp.harness.capabilities`` package does not exist
yet, so collection fails with an ``ImportError`` until link 04 lands.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

from molexp.harness.capabilities import curation_capabilities

# ── shared constants ────────────────────────────────────────────────────────

#: Repo-relative root of the subpackage under AST scan (ac-006). ``parents[2]``
#: walks ``tests/test_harness/<file>`` up to the repository root.
CAPABILITIES_ROOT = (
    Path(__file__).resolve().parents[2] / "src" / "molexp" / "harness" / "capabilities"
)

#: Modules the built-in registration subpackage must never import: no app shell
#: (server / cli), no LLM/graph SDKs, and no MCP surface (these are *direct*
#: built-ins, not molmcp-grounded tools).
FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "molexp.server",
    "molexp.cli",
    "pydantic_ai",
    "pydantic_graph",
    "molmcp",
    "mcp",
)

#: The destructive (side-effecting) mutators: the three workspace reorg verbs
#: plus the outward-facing git push (workspace-git-projection-04). The
#: scan/query/report functions are read-only.
EXPECTED_DESTRUCTIVE_IDS = {
    "molexp.curation.move_run",
    "molexp.curation.rehome_asset",
    "molexp.curation.delete_folder",
    "molexp.curation.git_push",
}


# ── module-level helpers ─────────────────────────────────────────────────────


def _resolve_callable(callable_path: str) -> Callable[..., Any]:
    """Resolve a dotted ``module.attr`` path to its live callable.

    Splits on the **last** dot (these entries use the fully-dotted
    ``molexp.workspace.curation.<fn>`` form), imports the module, and returns
    the attribute.

    Args:
        callable_path: A fully-qualified ``module.attribute`` path.

    Returns:
        The resolved object (asserted callable by the caller).
    """
    module_name, _, attr = callable_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _signature_properties(fn: Callable[..., Any]) -> tuple[set[str], set[str], bool]:
    """Derive expected schema properties from a callable's live signature.

    Args:
        fn: The resolved callable to introspect.

    Returns:
        A ``(properties, required, has_var_keyword)`` triple. ``properties``
        excludes ``self`` / ``cls`` and ``*args`` / ``**kwargs``; ``required``
        is the subset of ``properties`` whose parameter has no default;
        ``has_var_keyword`` flags a ``**kwargs`` wildcard signature.
    """
    properties: set[str] = set()
    required: set[str] = set()
    has_var_keyword = False
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if name in {"self", "cls"}:
            continue
        properties.add(name)
        if param.default is inspect.Parameter.empty:
            required.add(name)
    return properties, required, has_var_keyword


def _imports_with_prefix(prefix: str, root: Path) -> list[tuple[Path, int, str]]:
    """Return ``(path, lineno, module)`` for every import matching ``prefix``.

    Mirrors the AST helper in ``tests/test_harness/test_import_guard.py``:
    matches both ``import <prefix>…`` and ``from <prefix>… import …`` (and any
    subpackage) wherever the statement sits — module top level or function
    body. A non-existent ``root`` simply yields no hits.

    Args:
        prefix: The forbidden module prefix to detect.
        root: Directory tree to scan for ``*.py`` files.

    Returns:
        One tuple per matching import statement.
    """
    hits: list[tuple[Path, int, str]] = []
    if not root.exists():
        return hits
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == prefix or alias.name.startswith(prefix + "."):
                        hits.append((py, node.lineno, alias.name))
                        break
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                if module and (module == prefix or module.startswith(prefix + ".")):
                    hits.append((py, node.lineno, module))
    return hits


# ── ac-001: accessor shape, freshness, immutability ──────────────────────────


class TestAccessorShape:
    """ac-001 — ``curation_capabilities()`` returns a fresh list each call."""

    def test_returns_a_fresh_list_each_call(self) -> None:
        first = curation_capabilities()
        second = curation_capabilities()
        assert first == second
        assert first is not second


# ── ac-002: id namespace + package contract ──────────────────────────────────


class TestIdContract:
    """ac-002 — ids are unique, ``molexp.curation.*`` namespaced, package ``molexp``."""

    def test_ids_are_unique(self) -> None:
        catalog = curation_capabilities()
        assert len({entry.id for entry in catalog}) == len(catalog)

    def test_every_id_is_curation_namespaced(self) -> None:
        for entry in curation_capabilities():
            assert entry.id.startswith("molexp.curation."), entry.id


# ── ac-004: side-effects contract, classified by tag ─────────────────────────


class TestSideEffectsContract:
    """ac-004 — read-only entries declare no side effects; destructive ones do."""

    def test_every_entry_is_classified_exactly_once(self) -> None:
        for entry in curation_capabilities():
            assert "curation" in entry.tags, entry.id
            classifiers = {"read-only", "destructive"} & set(entry.tags)
            assert len(classifiers) == 1, f"{entry.id} tags={entry.tags}"

    def test_read_only_entries_have_no_side_effects(self) -> None:
        for entry in curation_capabilities():
            if "read-only" in entry.tags:
                assert entry.side_effects == [], entry.id

    def test_destructive_entries_declare_concrete_tokens(self) -> None:
        for entry in curation_capabilities():
            if "destructive" in entry.tags:
                assert entry.side_effects, entry.id
                assert all(isinstance(token, str) and token for token in entry.side_effects), (
                    entry.id
                )

    def test_destructive_id_set_is_exactly_the_three_mutators(self) -> None:
        destructive_ids = {
            entry.id for entry in curation_capabilities() if "destructive" in entry.tags
        }
        assert destructive_ids == EXPECTED_DESTRUCTIVE_IDS


# ── ac-005: input_schema matches the live signature ──────────────────────────


class TestSchemaDriftGuard:
    """ac-005 (drift guard) — declared ``input_schema`` tracks the function signature."""

    def test_input_schema_properties_and_required_match_signature(self) -> None:
        for entry in curation_capabilities():
            assert entry.callable_path is not None, entry.id
            fn = _resolve_callable(entry.callable_path)
            expected_props, expected_required, has_var_keyword = _signature_properties(fn)

            if has_var_keyword:
                # Wildcard signature: the entry must NOT pin a properties block.
                assert "properties" not in entry.input_schema, entry.id
                continue

            declared_props = set(entry.input_schema["properties"].keys())
            assert declared_props == expected_props, (
                f"{entry.id}: declared properties {declared_props} != signature {expected_props}"
            )
            declared_required = set(entry.input_schema.get("required", []))
            assert declared_required == expected_required, (
                f"{entry.id}: declared required {declared_required} != "
                f"signature {expected_required}"
            )


# ── ac-006: import discipline (AST scan) ─────────────────────────────────────


class TestImportDiscipline:
    """ac-006 — the capabilities subpackage imports no forbidden modules."""

    def test_no_forbidden_imports_in_capabilities_subpackage(self) -> None:
        offenders: dict[str, list[str]] = {}
        for prefix in FORBIDDEN_IMPORT_PREFIXES:
            hits = _imports_with_prefix(prefix, CAPABILITIES_ROOT)
            if hits:
                offenders[prefix] = [
                    f"{path.relative_to(CAPABILITIES_ROOT)}:{lineno}: {module}"
                    for path, lineno, module in hits
                ]
        assert not offenders, (
            "molexp.harness.capabilities must not import server / cli / "
            "pydantic_ai / pydantic_graph / molmcp / mcp.\nOffenders:\n  "
            + "\n  ".join(
                f"[{prefix}] {hit}" for prefix, lines in offenders.items() for hit in lines
            )
        )

    def test_scan_detects_planted_violation(self, tmp_path: Path) -> None:
        """Negative control: the AST scan must catch a freshly-planted bad import."""
        fake = tmp_path / "tainted.py"
        fake.write_text(
            "def sneaky():\n    import molmcp\n    return molmcp\n",
            encoding="utf-8",
        )
        hits = _imports_with_prefix("molmcp", tmp_path)
        assert any(path == fake and module == "molmcp" for path, _, module in hits)

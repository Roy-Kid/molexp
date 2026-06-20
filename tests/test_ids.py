"""Unit tests for the cross-layer ``molexp.ids`` primitive module.

``molexp.ids`` holds the pure id / slug / content-hash helpers promoted
out of ``molexp.workspace.utils`` (okf-01-01) so the ``molexp.knowledge``
bottom layer can cite them without importing workspace.
"""

from __future__ import annotations

import ast
import uuid
from pathlib import Path

import molexp.ids as ids
from molexp.ids import (
    compute_content_hash,
    generate_asset_id,
    generate_id,
    slugify,
)


def test_module_all_lists_the_canonical_symbols() -> None:
    assert set(ids.__all__) >= {
        "slugify",
        "generate_id",
        "generate_asset_id",
        "compute_content_hash",
    }


def test_slugify_lowercases_hyphenates_collapses() -> None:
    assert slugify("Hello   World") == "hello-world"
    assert slugify("My_Cool  Project!!") == "my-cool-project"
    assert slugify("a---b") == "a-b"


def test_slugify_truncates_to_max_len() -> None:
    assert slugify("a" * 100, max_len=10) == "a" * 10


def test_generate_id_is_8_hex_chars() -> None:
    value = generate_id()
    assert len(value) == 8
    int(value, 16)  # parses as hex


def test_generate_asset_id_is_valid_uuid() -> None:
    value = generate_asset_id()
    assert str(uuid.UUID(value)) == value


def test_compute_content_hash_file_prefix_and_stability(tmp_path: Path) -> None:
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"identical bytes")
    f2.write_bytes(b"identical bytes")
    h1 = compute_content_hash(f1)
    h2 = compute_content_hash(f2)
    assert h1.startswith("sha256:")
    assert h1 == h2  # stable for identical bytes


def test_compute_content_hash_directory_is_order_invariant(tmp_path: Path) -> None:
    d = tmp_path / "tree"
    (d / "x").mkdir(parents=True)
    (d / "x" / "1.txt").write_bytes(b"one")
    (d / "2.txt").write_bytes(b"two")
    first = compute_content_hash(d)
    # Recompute — must be deterministic regardless of walk order.
    assert compute_content_hash(d) == first
    assert first.startswith("sha256:")


def test_source_imports_no_workspace_or_upstream_layer() -> None:
    """The primitive must not import workspace or any upstream layer.

    Asserted at the AST level on the module's own source — the
    enforceable layer-independence invariant (a runtime ``sys.modules``
    probe is confounded by the eager ``molexp/__init__.py``). Mirrors
    ``tests/test_workspace/test_import_guard.py``.
    """
    forbidden = (
        "molexp.workspace",
        "molexp.workflow",
        "molexp.agent",
        "molexp.harness",
        "molexp.server",
        "molexp.cli",
        "molexp.plugins",
        "molexp.sweep",
    )
    source = Path(ids.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.startswith(forbidden)]
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(forbidden):
            offenders.append(node.module)
    assert offenders == [], f"molexp.ids imports forbidden modules: {offenders}"

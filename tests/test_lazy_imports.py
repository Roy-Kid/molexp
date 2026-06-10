"""Regression tests for the lazy sub-package loader in ``molexp/__init__``.

``from molexp import workflow`` used to hit infinite recursion: the lazy
``__getattr__`` itself did ``from molexp import workflow``, which re-enters
``__getattr__`` through ``importlib._bootstrap._handle_fromlist``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_FORMS = [
    "from molexp import workflow",
    "from molexp import workspace",
    "from molexp import plugins",
    "import molexp; molexp.workflow",
    "import molexp.workflow",
]


@pytest.mark.parametrize("stmt", _FORMS)
def test_lazy_subpackage_import_forms(stmt: str) -> None:
    """Every natural import spelling must work (fresh interpreter each)."""
    proc = subprocess.run(
        [sys.executable, "-c", stmt],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"{stmt!r} failed:\n{proc.stderr[-2000:]}"


def test_unknown_attribute_raises_attribute_error() -> None:
    import molexp

    with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
        _ = molexp.nonexistent


def test_workflow_attrs_resolve_lazily() -> None:
    """Top-level workflow re-exports resolve, but only on first access.

    ``import molexp`` must stay light (no pydantic_graph); accessing
    ``molexp.WorkflowCompiler`` / ``TaskContext`` / ``WorkflowRuntime``
    loads ``molexp.workflow`` at that point.
    """
    code = (
        "import sys\n"
        "import molexp\n"
        "assert 'pydantic_graph' not in sys.modules, 'import molexp loaded pydantic_graph'\n"
        "assert 'molexp.workflow' not in sys.modules, 'import molexp loaded molexp.workflow'\n"
        "for attr in ('WorkflowCompiler', 'TaskContext', 'WorkflowRuntime'):\n"
        "    assert getattr(molexp, attr) is getattr(molexp.workflow, attr)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

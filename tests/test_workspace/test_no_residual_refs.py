"""Residual-reference guard for the workspace-slim-01 deletion spec.

Asserts the deleted unwired subsystems leave zero trace in the shipped
source tree: no symbol definitions, no imports, no ``__all__`` entries,
and no orphaned module files. Scans ``src/molexp/`` only — test files
referencing removed symbols are deleted separately.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import molexp

SRC = Path(molexp.__file__).resolve().parent  # .../src/molexp


def _py_files() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


# Distinctive symbols that must not appear anywhere under src/molexp/ after the cut.
DELETED_SYMBOLS = [
    "RunFingerprint",
    "_hash_payload",
    "_environment_signature",
    "_FINGERPRINT_HASH_HEX_LEN",
    "CheckpointState",
    "OutputAsset",
    "ExecutionStateAsset",
    "_register_channel",
    "resumed_step",
    "checkpoint_step",
    "last_step",
    # workspace-slim-02: the bare-pathlib container-index mechanism — the
    # entity *.json is the sole truth source, the catalog the derived index.
    "_rebuild_container_index",
    "_refresh_runs_index",
    "_refresh_executions_index",
    "_refresh_experiments_index",
    "_refresh_projects_index",
]

DELETED_FILES = [
    SRC / "workspace" / "checkpoint.py",
    SRC / "workspace" / "resume_policy.py",
    SRC / "workspace" / "assets" / "output.py",
    SRC / "workspace" / "assets" / "execution.py",
]


@pytest.mark.parametrize("symbol", DELETED_SYMBOLS)
def test_symbol_has_zero_residue(symbol: str) -> None:
    offenders = [
        str(p.relative_to(SRC)) for p in _py_files() if symbol in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"{symbol!r} still referenced in: {offenders}"


@pytest.mark.parametrize("path", DELETED_FILES)
def test_deleted_module_absent(path: Path) -> None:
    assert not path.exists(), f"{path} should have been deleted"


@pytest.mark.parametrize("name", ["RunFingerprint", "OutputAsset", "ExecutionStateAsset"])
def test_public_export_removed(name: str) -> None:
    mod = importlib.import_module("molexp.workspace")
    assert not hasattr(mod, name), f"molexp.workspace still exports {name}"
    assert name not in getattr(mod, "__all__", []), f"{name} still in molexp.workspace.__all__"

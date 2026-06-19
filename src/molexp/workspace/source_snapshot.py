"""Per-run source snapshot — capture the defining script's source bytes.

molexp records the entrypoint *path* (``RunMetadata.script``) and an AST *hash*
(``workflow_snapshot.code_hash``) on each run, but never the source *bytes*. If the
file is later edited or deleted, the exact code that produced a run is
unrecoverable. This module captures the entrypoint **plus its first-party
local-module import closure** (sibling ``.py`` modules in the entrypoint's
directory, transitively) into ``<run_dir>/source/`` and returns a manifest for
``RunMetadata.source_snapshot`` — so a Run is reproducible from its own directory,
independent of the live source tree.

Stdlib-only by design: the workspace layer must not import upstream molexp layers
(see the layer DAG in CLAUDE.md / ``test_import_guard``).

Scope (v1): first-party modules are flat siblings resolved as ``<root>/<name>.py``
where ``root`` is the entrypoint's directory. Package subdirectories and
namespace packages are out of scope; stdlib and third-party imports are ignored
(they have no sibling file). This matches the flat-module layout of the molexp
consumer scripts (e.g. ``phase1.py`` importing ``eval_df`` / ``experiment``).
"""

from __future__ import annotations

import ast
import hashlib
import shutil
from datetime import datetime
from pathlib import Path


def _local_import_closure(entrypoint: Path) -> list[Path]:
    """Entrypoint + transitively-imported first-party sibling ``.py`` modules.

    A module name ``m`` is first-party iff ``<root>/m.py`` exists (``root`` =
    the entrypoint's directory). Dotted imports contribute their first segment.
    Stdlib / third-party imports resolve to no sibling file and are skipped.
    Returns the entrypoint first, then the rest sorted by filename (deterministic).
    """
    entrypoint = entrypoint.resolve()
    root = entrypoint.parent
    seen: dict[Path, None] = {}
    stack = [entrypoint]
    while stack:
        current = stack.pop().resolve()
        if current in seen or not current.is_file():
            continue
        seen[current] = None
        try:
            tree = ast.parse(current.read_text(encoding="utf-8"), filename=str(current))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        for name in names:
            candidate = (root / f"{name}.py").resolve()
            if candidate.is_file() and candidate not in seen:
                stack.append(candidate)
    rest = sorted((p for p in seen if p != entrypoint), key=lambda p: p.name)
    return [entrypoint, *rest]


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def snapshot_sources(
    entrypoint: Path, run_dir: Path, *, now: datetime | None = None
) -> dict[str, object]:
    """Copy the entrypoint + its first-party import closure into ``run_dir/source/``.

    Args:
        entrypoint: The defining script (the file molexp re-imports to execute).
        run_dir: The Run's directory; sources land under ``run_dir/source/``.
        now: Capture timestamp (injectable for deterministic tests).

    Returns:
        A manifest suitable for ``RunMetadata.source_snapshot``::

            {
                "dir": "source",
                "entrypoint": "phase4.py",
                "files": [{"name": "phase4.py", "sha256": "sha256:..."}, ...],
                "captured_at": "2026-06-17T13:00:00",
            }

    Idempotent (re-snapshotting overwrites the copies). Best-effort: a file that
    cannot be read is skipped. Files are copied flat by basename; first-party
    module names are unique within one directory, so basenames do not collide.
    """
    entrypoint = Path(entrypoint).resolve()
    dest = Path(run_dir) / "source"
    dest.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, str]] = []
    for src in _local_import_closure(entrypoint):
        if not src.is_file():
            continue
        target = dest / src.name
        shutil.copy2(src, target)
        files.append({"name": src.name, "sha256": _sha256(target)})
    stamp = (now or datetime.now()).isoformat()
    return {"dir": "source", "entrypoint": entrypoint.name, "files": files, "captured_at": stamp}

"""Shared path confinement for ops implementations."""

from __future__ import annotations

from pathlib import Path


def safe_path(root: Path, path: str) -> Path:
    """Resolve *path* under *root*; allow absolute paths that stay inside root."""
    raw = Path(path)
    if ".." in raw.parts:
        raise ValueError(f"path {path!r} may not contain '..'")
    root_resolved = root.resolve()
    resolved = raw.resolve() if raw.is_absolute() else (root_resolved / raw).resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"path {path!r} escapes the workspace root {root_resolved}")
    return resolved

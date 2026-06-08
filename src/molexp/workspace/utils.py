"""Workspace utility functions for ID generation, slugification, and content hashing."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .._typing import JSONValue


def derive_run_id(params: Mapping[str, JSONValue], *, length: int = 16) -> str:
    """Derive a deterministic, content-addressed run id from a parameter dict.

    The id is a sha256 over the canonicalized parameters — keys sorted, each
    rendered ``k=repr(v)`` — so it is a pure function of the params and
    independent of dict insertion order. Identical params always map to the
    same id, which makes run materialization idempotent (see
    :meth:`Experiment.add_runs`). This is the single canonicalization shared by
    the workspace layer and ``cli._common.deterministic_run_id``.

    Args:
        params: The run's parameter mapping (JSON-serializable values).
        length: Number of leading hex characters to keep (default 16).

    Returns:
        A ``length``-character lowercase hex string.
    """
    raw = "|".join(f"{k}={v!r}" for k, v in sorted(params.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:length]


def generate_id() -> str:
    """Generate a unique 8-character hex run ID.

    Returns:
        8-character hex string, e.g., 'a3f2e8d9'
    """
    return uuid4().hex[:8]


def generate_asset_id() -> str:
    """Generate a unique asset ID using UUID.

    Returns:
        UUID string, e.g., 'a3f2e8d9-4b1c-4e5f-9a2b-1c3d4e5f6a7b'
    """
    return str(uuid4())


def derive_execution_id(run_id: str, exec_root: Path) -> str:
    """Return the execution_id for the next attempt of *run_id*.

    First attempt is ``exec-{run_id}``; retries are ``exec-{run_id}-2``,
    ``exec-{run_id}-3``, … The next suffix is ``max(existing suffix) + 1``
    (never ``len(existing)``): deleting a *middle* attempt would shrink the
    count and make ``len`` collide with a still-present higher id, whereas
    ``max + 1`` is always strictly greater than every live attempt, so it
    never reuses an existing id.

    Attempts are matched by *exact* name — ``exec-{run_id}`` itself (attempt
    1) or ``exec-{run_id}-<int>`` — so a run whose id is a string prefix of
    another run's id (``"ab"`` vs ``"abc"``) is not miscounted, and non-numeric
    suffixes are ignored.

    This is the single source of execution-id derivation; the workflow runtime
    (:func:`molexp.workflow.make_execution_id`) and the workspace
    ``ExecutionStore`` both delegate here.
    """
    base = f"exec-{run_id}"
    if not exec_root.exists():
        return base
    prefix = f"{base}-"
    highest = 0  # 0 → no attempt yet; `base` itself counts as attempt 1
    for entry in exec_root.iterdir():
        name = entry.name
        if name == base:
            highest = max(highest, 1)
        elif name.startswith(prefix):
            tail = name[len(prefix) :]
            if tail.isdigit():
                highest = max(highest, int(tail))
    if highest == 0:
        return base
    return f"{base}-{highest + 1}"


def slugify(text: str, max_len: int = 50) -> str:
    """Convert text to a valid slug.

    Args:
        text: Input text
        max_len: Maximum length

    Returns:
        Slugified string
    """
    # Convert to lowercase
    slug = text.lower()
    # Replace spaces and underscores with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)
    # Remove non-alphanumeric characters except hyphens
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    # Remove leading/trailing hyphens
    slug = slug.strip("-")
    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)
    # Truncate to max length
    return slug[:max_len]


def compute_content_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file's bytes or a directory tree.

    For files, returns the digest of the byte stream. For directories,
    walks every contained file in sorted-relative-path order and hashes
    ``relpath\\0bytes\\0`` per file, so the result is invariant to
    filesystem walk order but sensitive to filenames.

    Args:
        path: File or directory to hash.
        algorithm: Hash algorithm name (default: ``sha256``).

    Returns:
        Hash string with algorithm prefix, e.g.,
        ``"sha256:a3b4c5d6..."``.
    """
    hasher = hashlib.new(algorithm)

    if path.is_dir():
        for entry in sorted(path.rglob("*")):
            if entry.is_file():
                rel = entry.relative_to(path).as_posix().encode()
                hasher.update(rel + b"\0")
                with open(entry, "rb") as f:  # noqa: PTH123
                    while chunk := f.read(8192):
                        hasher.update(chunk)
                hasher.update(b"\0")
    else:
        with open(path, "rb") as f:  # noqa: PTH123
            while chunk := f.read(8192):
                hasher.update(chunk)

    return f"{algorithm}:{hasher.hexdigest()}"

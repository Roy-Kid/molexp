"""Layering invariant guard.

Spec: workflow-rectification (criterion `no-workspace-or-agent-imports`).

``src/molexp/workflow/`` MUST NOT import from ``molexp.workspace.*`` or
``molexp.agent.*``. This test scans every source file under the
workflow layer and rejects any occurrence of those import paths.
Cross-layer payloads flow through duck-typed ``run_context`` (opaque)
or ``Mapping[str, Any]`` config — never through a concrete type
import.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "workflow"

# Match real import statements only — not docstring references.
_IMPORT_PATTERNS = [
    re.compile(r"^\s*from\s+molexp\.workspace(\.|\s)"),
    re.compile(r"^\s*from\s+molexp\.agent(\.|\s)"),
    re.compile(r"^\s*import\s+molexp\.workspace(\.|\s|$)"),
    re.compile(r"^\s*import\s+molexp\.agent(\.|\s|$)"),
]


def _iter_workflow_py_files() -> list[Path]:
    return [p for p in WORKFLOW_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_workspace_or_agent_imports_in_workflow():
    offenders: list[str] = []
    for path in _iter_workflow_py_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            # Skip comment-only lines.
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for pat in _IMPORT_PATTERNS:
                if pat.search(line):
                    rel = path.relative_to(WORKFLOW_ROOT.parent.parent.parent)
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
                    break
    assert not offenders, (
        "src/molexp/workflow/ must not import from molexp.workspace.* or "
        "molexp.agent.*. Offenders:\n  " + "\n  ".join(offenders)
    )

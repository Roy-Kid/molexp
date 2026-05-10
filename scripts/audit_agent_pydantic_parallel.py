"""Audit agent/ subpackages for production import references.

For each candidate subpackage / file in ``src/molexp/agent/`` that is suspected
of being a parallel-to-pydantic-ai implementation (the targets enumerated by
spec ``agent-pydanticai-rectification``), count how many *production* files
under ``src/molexp/agent/`` import it. Self-references within the candidate
subpackage are excluded; ``tests/`` and ``examples/`` are outside scope.

A count of 0 means the candidate is deletable (Phase 1 evidence). A positive
count means production code still depends on it and Phase 1 must NOT delete it
without first migrating callers.

Usage::

    python scripts/audit_agent_pydantic_parallel.py        # table
    python scripts/audit_agent_pydantic_parallel.py --json # machine-readable
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_ROOT = REPO_ROOT / "src" / "molexp" / "agent"

# (display_name, module_prefix_to_count, exclusion_path)
CANDIDATES: tuple[tuple[str, str, Path], ...] = (
    ("tools", "molexp.agent.tools", AGENT_ROOT / "tools"),
    ("context", "molexp.agent.context", AGENT_ROOT / "context"),
    ("memory", "molexp.agent.memory", AGENT_ROOT / "memory"),
    ("recovery", "molexp.agent.recovery", AGENT_ROOT / "recovery"),
    ("skills", "molexp.agent.skills", AGENT_ROOT / "skills"),
    ("mcp.source", "molexp.agent.mcp.source", AGENT_ROOT / "mcp" / "source.py"),
    ("mcp.tool_store", "molexp.agent.mcp.tool_store", AGENT_ROOT / "mcp" / "tool_store.py"),
    ("mcp.probe", "molexp.agent.mcp.probe", AGENT_ROOT / "mcp" / "probe.py"),
)


def _imports_in(path: Path) -> Iterable[str]:
    """Yield every fully-qualified module imported by ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _is_under(path: Path, root: Path) -> bool:
    """True if ``path`` is ``root`` itself or lives inside ``root``."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def count_production_cites(prefix: str, dead_cluster: tuple[Path, ...]) -> int:
    """Count files under :data:`AGENT_ROOT` that import a module matching
    ``prefix``, excluding every file inside the dead-candidate closure.

    The dead-candidate closure is the union of every candidate's exclusion
    path (passed as ``dead_cluster``). Citations *between* dead candidates
    are circular noise (e.g. ``tools/`` imports from ``mcp/source.py`` which
    imports back from ``tools/``) and would mask the true production
    dependency. The closure exclusion strips those, leaving only the cites
    from genuine production files.
    """
    count = 0
    for p in AGENT_ROOT.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        if any(_is_under(p, dead) for dead in dead_cluster):
            continue
        for imp in _imports_in(p):
            if imp == prefix or imp.startswith(f"{prefix}."):
                count += 1
                break
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON only on stdout")
    args = ap.parse_args()

    if not AGENT_ROOT.exists():
        print(f"error: agent root not found at {AGENT_ROOT}", file=sys.stderr)
        return 2

    dead_cluster = tuple(exclude for _, _, exclude in CANDIDATES)
    counts: dict[str, int] = {
        name: count_production_cites(prefix, dead_cluster) for name, prefix, _ in CANDIDATES
    }

    if args.json:
        json.dump(counts, sys.stdout)
        return 0

    print(f"production-cite audit (agent_root={AGENT_ROOT.relative_to(REPO_ROOT)})")
    print("-" * 60)
    for name, count in counts.items():
        marker = "  (dead)" if count == 0 else ""
        print(f"  {name:<18}  {count}{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

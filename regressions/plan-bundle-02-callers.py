"""Public-API regression for plan-bundle-02-callers."""

from __future__ import annotations

import ast
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "molexp"
    hits: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "PlanOrchestrator":
                        hits.append(str(path))
            if isinstance(node, ast.Name) and node.id == "PlanOrchestrator":
                hits.append(str(path))
    assert not hits, f"PlanOrchestrator still referenced: {hits}"
    print("plan-bundle-02-callers: ok")


if __name__ == "__main__":
    main()

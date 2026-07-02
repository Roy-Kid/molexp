"""Every example script must be gated by the examples smoke test — or excused.

``tests/test_examples_smoke.py`` subprocess-runs the standalone examples and
drives the CLI-profile examples through ``molexp run``. This module closes the
gap that suite cannot see: an example added to ``examples/`` but never wired
into the smoke gate would otherwise rot silently. Each ``.py`` under
``examples/`` must be either

- covered — listed in the smoke test's standalone set or living inside one of
  its CLI-driven example directories, or
- excused — named in :data:`EXCUSED` with the reason it cannot run in CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"

# Examples that must NOT run in the smoke gate, with the reason.
EXCUSED: dict[str, str] = {
    "operations/scheduler_molq.py": "submits jobs through a live molq scheduler",
    "operations/server_lifecycle.py": "starts a real HTTP server (ports, daemons)",
    "harness/__init__.py": "package marker, not a runnable example",
}


def _load_smoke_module():
    path = REPO_ROOT / "tests" / "test_examples_smoke.py"
    spec = importlib.util.spec_from_file_location("_examples_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _covered_scripts() -> set[Path]:
    smoke = _load_smoke_module()
    covered = {Path(s) for s in smoke.STANDALONE_SCRIPTS}
    for rel_dir, _profile in smoke.CLI_PROFILE_EXAMPLES:
        covered.update((EXAMPLES / rel_dir).rglob("*.py"))
    return covered


def test_every_example_is_smoke_gated_or_excused() -> None:
    covered = _covered_scripts()
    problems: list[str] = []
    for script in sorted(EXAMPLES.rglob("*.py")):
        rel = script.relative_to(EXAMPLES).as_posix()
        if script in covered and rel in EXCUSED:
            problems.append(f"{rel}: excused but actually smoke-gated — drop the excuse")
        elif script not in covered and rel not in EXCUSED:
            problems.append(
                f"{rel}: not in tests/test_examples_smoke.py and not excused in "
                "tests/test_docs/test_examples_coverage.py"
            )
    assert not problems, "\n".join(problems)


def test_excused_examples_still_exist() -> None:
    stale = [rel for rel in EXCUSED if not (EXAMPLES / rel).exists()]
    assert not stale, f"EXCUSED entries point at deleted examples: {stale}"

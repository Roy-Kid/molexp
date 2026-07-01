"""Smoke-run every standalone example script — the docs/examples drift gate.

Each documented entry-path example under ``examples/getting_started/``,
``examples/workflow/``, and ``examples/workspace/`` must execute green with
the shipped API — and so must the **offline-first** agent/harness examples
(:data:`OFFLINE_LLM_SCRIPTS`), which run their scripted/canned LLM seams with
zero network and zero API keys. An API rename that reaches ``src/`` without
reaching the examples fails here, in CI, instead of failing for the first
new user.

Every script runs in a subprocess (its own interpreter, no shared module
state) with a temporary working directory; the examples themselves write
only into ``tempfile.mkdtemp()`` locations. The CLI-driven profile examples
(``04_cli_and_profiles`` and ``operations/run_profiles``) are copied into the
temp dir and driven through ``molexp run`` exactly as they document.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

# Per-script subprocess budget; the whole module must stay well under a minute.
SCRIPT_TIMEOUT_S = 60


def _scripts(subdir: str) -> list[Path]:
    """Standalone ``python <script>`` examples under ``examples/<subdir>/``."""
    return sorted((EXAMPLES / subdir).glob("*.py"))


# Offline-first agent/harness examples: scripted Router / canned AgentGateway
# seams replace the LLM, but validation, pytest, and workflow execution are
# real. An explicit file list (never a dir glob) keeps package markers like
# ``examples/harness/__init__.py`` out of the executed set.
OFFLINE_LLM_SCRIPTS = [
    EXAMPLES / "agent" / "chat_loop.py",
    EXAMPLES / "agent" / "interactive_loop.py",
    EXAMPLES / "harness" / "experiment_pipeline.py",
]

# Offline operations examples (no server, no molq, no network) — listed
# explicitly because ``operations/`` also holds network/scheduler-dependent
# scripts that must not run in the smoke gate.
OFFLINE_OPERATIONS_SCRIPTS = [
    EXAMPLES / "operations" / "curate.py",
]

STANDALONE_SCRIPTS = [
    *_scripts("getting_started"),
    *_scripts("workflow"),
    *_scripts("workspace"),
    *OFFLINE_LLM_SCRIPTS,
    *OFFLINE_OPERATIONS_SCRIPTS,
]


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_S,
    )


def _assert_exit_zero(proc: subprocess.CompletedProcess[str], label: str) -> None:
    assert proc.returncode == 0, (
        f"{label} exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


# "The example code itself is broken against the API" signatures. These never
# appear in an *intended* domain failure (which raises a clear RuntimeError /
# ValueError with a message, e.g. workflow_persistence's "first attempt boom").
# An API rename that leaves an example calling a removed attribute (the exact
# ``ctx.inputs`` drift) surfaces here even though the engine catches the task
# exception into ``status=failed`` and the process still exits 0.
_BROKEN_SIGNATURES = (
    "AttributeError",
    "ModuleNotFoundError",
    "ImportError",
    "NameError",
    "SyntaxError",
)


def _assert_no_api_drift(proc: subprocess.CompletedProcess[str], label: str) -> None:
    output = proc.stdout + proc.stderr
    hits = [sig for sig in _BROKEN_SIGNATURES if sig in output]
    assert not hits, (
        f"{label} emitted {hits} — the example is broken against the shipped API "
        f"(a task likely failed silently into status=failed).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "script",
    [pytest.param(s, id=str(s.relative_to(EXAMPLES))) for s in STANDALONE_SCRIPTS],
)
def test_example_script_runs_clean(script: Path, tmp_path: Path) -> None:
    """Every standalone example must exit 0 when run as ``python <script>``."""
    assert script.exists(), f"example script vanished: {script}"
    proc = _run([sys.executable, str(script)], cwd=tmp_path)
    label = str(script.relative_to(REPO_ROOT))
    _assert_exit_zero(proc, label)
    _assert_no_api_drift(proc, label)


# CLI-driven profile examples: each is copied into the temp dir (so the
# ``_workspace`` it creates never lands in the repo) and driven exactly as its
# docstring documents — ``molexp run train.py --profile <profile>`` reading
# profile fields as named parameters.
CLI_PROFILE_EXAMPLES = [
    ("getting_started/04_cli_and_profiles", "smoke"),
    ("operations/run_profiles", "smoke"),
]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("rel_dir", "profile"),
    [pytest.param(rel_dir, profile, id=rel_dir) for rel_dir, profile in CLI_PROFILE_EXAMPLES],
)
def test_cli_profile_example_runs_via_molexp_run(
    rel_dir: str, profile: str, tmp_path: Path
) -> None:
    """Each CLI-driven profile example must run green via the documented command.

    The drift check is load-bearing here: ``molexp run`` exits 0 (and prints
    ``OK … runs completed``) even when a task fails into ``status=failed``, so a
    removed-attribute break — the exact ``ctx.config`` / ``ctx.inputs`` drift —
    only surfaces as an ``AttributeError`` in the captured output, not the exit
    code.
    """
    src = EXAMPLES / rel_dir
    dst = tmp_path / Path(rel_dir).name
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("_workspace", "__pycache__"))

    proc = _run(
        [
            sys.executable,
            "-c",
            "from molexp.cli import app; app()",
            "run",
            "train.py",
            "--profile",
            profile,
        ],
        cwd=dst,
    )
    label = f"molexp run {rel_dir}/train.py --profile {profile}"
    _assert_exit_zero(proc, label)
    _assert_no_api_drift(proc, label)
    assert (dst / "_workspace" / "workspace.json").exists(), (
        "CLI run did not materialize the declared workspace"
    )

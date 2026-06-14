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
only into ``tempfile.mkdtemp()`` locations. The CLI-driven example
(``04_cli_and_profiles``) is copied into the temp dir and driven through
``molexp run`` exactly as its README documents.

Only examples that need a long-lived server or an external scheduler remain
skipped, with the reason pinned.
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

STANDALONE_SCRIPTS = [
    *_scripts("getting_started"),
    *_scripts("workflow"),
    *_scripts("workspace"),
    *OFFLINE_LLM_SCRIPTS,
]

# Examples that cannot run as offline smoke tests, with the reason pinned.
SKIPPED_SCRIPTS = [
    pytest.param(
        EXAMPLES / "operations" / "server_lifecycle.py",
        marks=pytest.mark.skip(reason="starts a long-lived uvicorn server"),
        id="operations/server_lifecycle.py",
    ),
    pytest.param(
        EXAMPLES / "operations" / "scheduler_molq.py",
        marks=pytest.mark.skip(reason="requires a molq scheduler backend"),
        id="operations/scheduler_molq.py",
    ),
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


@pytest.mark.integration
@pytest.mark.parametrize(
    "script",
    [
        *[pytest.param(s, id=str(s.relative_to(EXAMPLES))) for s in STANDALONE_SCRIPTS],
        *SKIPPED_SCRIPTS,
    ],
)
def test_example_script_runs_clean(script: Path, tmp_path: Path) -> None:
    """Every standalone example must exit 0 when run as ``python <script>``."""
    assert script.exists(), f"example script vanished: {script}"
    proc = _run([sys.executable, str(script)], cwd=tmp_path)
    _assert_exit_zero(proc, str(script.relative_to(REPO_ROOT)))


@pytest.mark.integration
def test_cli_and_profiles_example_runs_via_molexp_run(tmp_path: Path) -> None:
    """``04_cli_and_profiles`` is CLI-driven: verify the documented command.

    The directory is copied into the temp dir so the ``_workspace`` it
    creates never lands in the repo, then driven exactly as its docstring
    documents: ``molexp run train.py --profile smoke``.
    """
    src = EXAMPLES / "getting_started" / "04_cli_and_profiles"
    dst = tmp_path / "04_cli_and_profiles"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("_workspace", "__pycache__"))

    proc = _run(
        [
            sys.executable,
            "-c",
            "from molexp.cli import app; app()",
            "run",
            "train.py",
            "--profile",
            "smoke",
        ],
        cwd=dst,
    )
    _assert_exit_zero(proc, "molexp run 04_cli_and_profiles/train.py --profile smoke")
    assert (dst / "_workspace" / "workspace.json").exists(), (
        "CLI run did not materialize the declared workspace"
    )

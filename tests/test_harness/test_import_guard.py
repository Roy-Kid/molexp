"""Architectural direction guard for ``molexp.harness``.

``import molexp.harness`` MUST NOT pull ``pydantic_ai`` / ``pydantic_graph``
/ ``molexp.workflow`` into ``sys.modules``.

Post spec ``harness-as-mode-substrate-03a``: ``molexp.agent`` is **allowed**
— the new ``RouterBackedAgentGateway`` legally imports ``agent.router``
(the Protocol module, which is itself SDK-free per spec 01 ac-013). The
charter pivot lets harness sit above agent in the DAG; the only remaining
forbidden-edge invariant is that pydantic-ai SDKs and the workflow layer
must not load eagerly when the harness package is imported.

We run the check in a fresh subprocess so a stale ``sys.modules`` (from
another test that already imported the workflow layer) can't poison the
assertion.

Two kinds of guard live here:

1. **Runtime probes** (subprocess) — assert what actually lands in
   ``sys.modules`` when harness modules are imported.
2. **Static source scans** (AST, mirroring
   ``tests/test_workspace/test_import_guard.py``) — assert the import
   *statements* themselves. CLAUDE.md pins exactly one sanctioned
   ``harness → agent`` import target: ``molexp.agent.router``. A static
   violation (e.g. ``from molexp.agent.runner import AgentRunner`` inside
   a function body) would slip past the runtime probes; the AST scan
   catches it regardless of where the import hides.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_FORBIDDEN = ("pydantic_ai", "pydantic_graph", "molexp.workflow")

HARNESS_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "harness"

# CLAUDE.md: "The single sanctioned harness → agent import target is
# molexp.agent.router (the SDK-free Protocol module)."
SANCTIONED_AGENT_MODULE = "molexp.agent.router"

# Application-shell layers the harness must never import — statically,
# anywhere (top level, function bodies, TYPE_CHECKING blocks alike).
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "molexp.plugins",
    "molexp.server",
    "molexp.cli",
    "molexp.sweep",
)


class TestImportGuard:
    def test_import_molexp_harness_does_not_pull_forbidden_modules(self) -> None:
        probe = (
            "import sys, importlib;"
            "importlib.import_module('molexp.harness');"
            "loaded = [m for m in sys.modules if m in " + repr(list(_FORBIDDEN)) + "];"
            "print('LOADED:' + ','.join(loaded))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout.strip()
        assert output.startswith("LOADED:"), output
        loaded = [m for m in output.removeprefix("LOADED:").split(",") if m]
        assert loaded == [], f"forbidden modules imported transitively: {loaded}"

    def test_import_molexp_harness_mode_does_not_pull_forbidden_modules(self) -> None:
        """ac-008: ``import molexp.harness.mode`` must load ``molexp.workflow`` lazily.

        ``Mode.run`` imports ``molexp.workflow`` *inside the method body* (mirroring
        ``agent.AgentRunner.run()`` deferring ``pydantic_ai``). Merely importing the
        ``molexp.harness.mode`` module must therefore leave ``molexp.workflow`` —
        and the ``pydantic_ai`` / ``pydantic_graph`` SDKs it transitively loads —
        out of ``sys.modules``. Run in a fresh subprocess so a stale ``sys.modules``
        from another test cannot poison the assertion.
        """
        probe = (
            "import sys, importlib;"
            "importlib.import_module('molexp.harness.mode');"
            "loaded = [m for m in sys.modules if m in " + repr(list(_FORBIDDEN)) + "];"
            "print('LOADED:' + ','.join(loaded))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout.strip()
        assert output.startswith("LOADED:"), output
        loaded = [m for m in output.removeprefix("LOADED:").split(",") if m]
        assert loaded == [], f"forbidden modules imported transitively by harness.mode: {loaded}"

    @pytest.mark.parametrize(
        "module",
        [
            "molexp.harness.stages.validate_workflow_source",
            "molexp.harness.stages.generate_workflow_source",
        ],
    )
    def test_import_workflow_source_stage_modules_lazy(self, module: str) -> None:
        """ac-010: the codegen stage modules defer ``molexp.workflow``.

        ``ValidateWorkflowSource.run`` imports ``molexp.workflow`` *inside* the
        method body so the heavy ``pydantic_graph`` chain only loads when the
        source is actually compiled. Importing either stage module at top level
        must therefore leave ``molexp.workflow`` / ``pydantic_graph`` /
        ``pydantic_ai`` out of ``sys.modules``. Fresh subprocess so a stale
        ``sys.modules`` from another test cannot poison the assertion.
        """
        probe = (
            "import sys, importlib;"
            f"importlib.import_module({module!r});"
            "loaded = [m for m in sys.modules if m in " + repr(list(_FORBIDDEN)) + "];"
            "print('LOADED:' + ','.join(loaded))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        output = result.stdout.strip()
        assert output.startswith("LOADED:"), output
        loaded = [m for m in output.removeprefix("LOADED:").split(",") if m]
        assert loaded == [], f"forbidden modules imported transitively by {module}: {loaded}"

    # ───────────────────────────── static source scans ─────────────────────────────

    def test_harness_agent_imports_target_only_the_router_module(self) -> None:
        """Every ``molexp.agent.*`` import under harness/ is exactly ``molexp.agent.router``.

        ``from molexp.agent.router import …`` is the one sanctioned spelling.
        ``from molexp.agent import router`` is also a violation — it executes
        ``molexp/agent/__init__.py`` and drags the whole agent surface in.
        """
        hits = _imports_with_prefix("molexp.agent", HARNESS_ROOT)
        bad = [
            f"{path.relative_to(HARNESS_ROOT)}:{lineno}: {module}"
            for path, lineno, module in hits
            if module != SANCTIONED_AGENT_MODULE
        ]
        assert not bad, (
            "harness may import molexp.agent ONLY via the sanctioned "
            f"{SANCTIONED_AGENT_MODULE!r} Protocol module (CLAUDE.md layer DAG).\n"
            "Offenders:\n  " + "\n  ".join(bad)
        )

    def test_harness_forbids_application_layers_statically(self) -> None:
        """No plugins / server / cli / sweep import statements anywhere in harness/."""
        offenders: dict[str, list[str]] = {}
        for prefix in FORBIDDEN_PREFIXES:
            hits = _imports_with_prefix(prefix, HARNESS_ROOT)
            if hits:
                offenders[prefix] = _format_hits(hits)
        assert not offenders, (
            "molexp.harness must not import the application shell "
            "(plugins / server / cli / sweep).\nOffenders:\n  "
            + "\n  ".join(
                f"[{prefix}] {hit}" for prefix, lines in offenders.items() for hit in lines
            )
        )

    def test_static_scan_detects_planted_violations(self, tmp_path: Path) -> None:
        """Negative test: the AST scan must catch freshly-planted bad imports."""
        fake = tmp_path / "tainted.py"
        fake.write_text(
            "def sneaky():\n"
            "    from molexp.agent.runner import AgentRunner\n"
            "    import molexp.cli\n"
            "    return AgentRunner, molexp\n"
        )
        agent_hits = _imports_with_prefix("molexp.agent", tmp_path)
        cli_hits = _imports_with_prefix("molexp.cli", tmp_path)
        assert any(p == fake and m == "molexp.agent.runner" for p, _, m in agent_hits), (
            "guard failed to detect a planted molexp.agent.runner import inside a function body"
        )
        assert any(p == fake for p, _, _ in cli_hits), (
            "guard failed to detect a planted molexp.cli import"
        )


# ───────────────────────────── module-level helpers ─────────────────────────────


def _imports_with_prefix(prefix: str, root: Path) -> list[tuple[Path, int, str]]:
    """Return ``(path, lineno, module)`` for every import matching ``prefix``.

    Matches both ``import molexp.<prefix>…`` and ``from molexp.<prefix>…
    import …`` (and any subpackage), wherever the statement sits — module
    top level or inside a function body. Mirrors the workspace guard's
    AST approach so failure messages can quote the offender directly.
    """
    hits: list[tuple[Path, int, str]] = []
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == prefix or alias.name.startswith(prefix + "."):
                        hits.append((py, node.lineno, alias.name))
                        break
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                if module and (module == prefix or module.startswith(prefix + ".")):
                    hits.append((py, node.lineno, module))
    return hits


def _format_hits(hits: list[tuple[Path, int, str]]) -> list[str]:
    return [f"{path.relative_to(HARNESS_ROOT)}:{lineno}: {module}" for path, lineno, module in hits]

"""Layer firewall for ``molexp.services`` — the application-service layer.

``molexp.services`` is the shared backend the CLI and the server both
delegate to (one code path per user-facing operation). Dependency direction:

    cli ──┐
          ├──> services ──> harness / agent / workflow / workspace
 server ──┘

Services may import the domain layers below it (``harness`` / ``agent`` /
``workflow`` / ``workspace`` / ``knowledge`` and cross-layer primitives);
it MUST NOT import the application shells that sit above it —
``molexp.server``, ``molexp.cli``, ``molexp.plugins`` — statically,
anywhere (top level, function bodies, TYPE_CHECKING blocks alike).

AST source scan (mirrors ``tests/test_harness/test_import_guard.py``): a
runtime ``sys.modules`` probe cannot catch an import hidden inside a
function body, the scan catches it regardless of where it hides.
"""

from __future__ import annotations

import ast
from pathlib import Path

SERVICES_ROOT = Path(__file__).resolve().parents[2] / "src" / "molexp" / "services"

# Application shells above the services layer — never importable from it.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "molexp.server",
    "molexp.cli",
    "molexp.plugins",
    "molexp.sweep",
)


class TestServicesImportGuard:
    def test_services_package_exists(self) -> None:
        assert SERVICES_ROOT.is_dir(), SERVICES_ROOT

    def test_services_forbids_application_shells_statically(self) -> None:
        """No server / cli / plugins import statements anywhere in services/."""
        offenders: dict[str, list[str]] = {}
        for prefix in FORBIDDEN_PREFIXES:
            hits = _imports_with_prefix(prefix, SERVICES_ROOT)
            if hits:
                offenders[prefix] = [
                    f"{path.relative_to(SERVICES_ROOT)}:{lineno}: {module}"
                    for path, lineno, module in hits
                ]
        assert not offenders, (
            "molexp.services must not import the application shells "
            "(server / cli / plugins).\nOffenders:\n  "
            + "\n  ".join(
                f"[{prefix}] {hit}" for prefix, lines in offenders.items() for hit in lines
            )
        )

    def test_static_scan_detects_planted_violations(self, tmp_path: Path) -> None:
        """Negative test: the AST scan must catch freshly-planted bad imports."""
        fake = tmp_path / "tainted.py"
        fake.write_text(
            "def sneaky():\n"
            "    from molexp.server.app import create_app\n"
            "    import molexp.cli\n"
            "    return create_app, molexp\n"
        )
        server_hits = _imports_with_prefix("molexp.server", tmp_path)
        cli_hits = _imports_with_prefix("molexp.cli", tmp_path)
        assert any(p == fake and m == "molexp.server.app" for p, _, m in server_hits)
        assert any(p == fake for p, _, _ in cli_hits)


def _imports_with_prefix(prefix: str, root: Path) -> list[tuple[Path, int, str]]:
    """Return ``(path, lineno, module)`` for every import matching ``prefix``."""
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

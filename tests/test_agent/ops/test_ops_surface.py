"""Agent ops surface — chat vs full tool adapter + auto-discovery law."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from molexp.agent.execution_env import LocalExecutionEnv
from molexp.agent.ops import (
    ARCHIVE_TOOL_NAMES,
    BUILTIN_TOOL_NAMES,
    CHAT_TOOL_NAMES,
    FULL_TOOL_NAMES,
    OPS_TOOL_NAMES,
    build_ops_tools,
    build_session_context,
    builtin_tool_specs,
    render_discovery_catalog,
)
from molexp.agent.ops.preamble import CHAT_OPS_PREAMBLE, DEFAULT_OPS_PREAMBLE, FULL_OPS_PREAMBLE


class TestBuildOpsTools:
    def test_chat_surface_excludes_ensure_and_land(self, tmp_path: Path) -> None:
        ctx = build_session_context(
            workspace_root=tmp_path,
            execution_env=LocalExecutionEnv(scratch_dir=tmp_path / "scratch"),
            confine_code_to="agent/.scratch",
        )
        names = {t.__name__ for t in build_ops_tools(ctx, surface="chat")}
        assert names == CHAT_TOOL_NAMES
        assert "run_land" not in names
        assert "workspace_ensure" not in names
        assert "code_write" in names and "discover" in names

    def test_full_surface_exposes_archive_tools(self, tmp_path: Path) -> None:
        ctx = build_session_context(
            workspace_root=tmp_path,
            execution_env=LocalExecutionEnv(scratch_dir=tmp_path / "scratch"),
        )
        names = {t.__name__ for t in build_ops_tools(ctx, surface="full")}
        assert names == FULL_TOOL_NAMES == BUILTIN_TOOL_NAMES == OPS_TOOL_NAMES
        assert names >= ARCHIVE_TOOL_NAMES

    def test_chat_code_write_confines_to_scratch(self, tmp_path: Path) -> None:
        ctx = build_session_context(
            workspace_root=tmp_path,
            execution_env=LocalExecutionEnv(scratch_dir=tmp_path / "scratch"),
            confine_code_to="agent/.scratch",
        )
        tools = {t.__name__: t for t in build_ops_tools(ctx, surface="chat")}
        wrote = tools["code_write"]("pe_rg.py", "print(2)\n")
        assert wrote.startswith("wrote")
        assert "agent/.scratch" in wrote
        assert (tmp_path / "agent" / ".scratch" / "pe_rg.py").is_file()
        # Must not land under projects/
        assert not (tmp_path / "projects").exists()
        out = tools["code_run"](path="pe_rg.py")
        assert "exit_code=0" in out
        assert "2" in out
        assert "agent/.scratch" in out  # chat hint + confine messaging

    def test_chat_code_run_uses_scratch_cwd(self, tmp_path: Path) -> None:
        """Chat Mode must not exec with workspace root as cwd (avoids relative projects/)."""
        ctx = build_session_context(
            workspace_root=tmp_path,
            execution_env=LocalExecutionEnv(scratch_dir=tmp_path / "scratch"),
            confine_code_to="agent/.scratch",
        )
        tools = {t.__name__: t for t in build_ops_tools(ctx, surface="chat")}
        tools["code_write"](
            "cwd_probe.py",
            "import os\nfrom pathlib import Path\nprint(Path.cwd().resolve())\n",
        )
        out = tools["code_run"](path="cwd_probe.py")
        assert "exit_code=0" in out
        scratch = (tmp_path / "agent" / ".scratch").resolve()
        assert str(scratch) in out
        assert "Do not" in out or "add_project" in out

    def test_full_run_land_attaches_artifacts_and_settles_run(self, tmp_path: Path) -> None:
        ctx = build_session_context(
            workspace_root=tmp_path,
            execution_env=LocalExecutionEnv(scratch_dir=tmp_path / "scratch"),
        )
        tools = {t.__name__: t for t in build_ops_tools(ctx, surface="full")}
        assert tools["workspace_ensure"]("project", "p1").startswith("ok")
        assert tools["workspace_ensure"]("experiment", "e1", project="p1").startswith("ok")
        run_out = tools["workspace_ensure"](
            "run",
            "rg-scan",
            project="p1",
            experiment="e1",
            params_json='{"n": 10}',
        )
        assert run_out.startswith("ok kind=run")
        run_id = run_out.split("id=")[1].split()[0]
        (tmp_path / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
        (tmp_path / "analysis.py").write_text("print('ok')\n", encoding="utf-8")
        land = tools["run_land"](
            "p1",
            "e1",
            run_id,
            files="plot.png",
            sources="analysis.py",
            results_json='{"rg_mean": 1.23}',
        )
        assert land.startswith("ok")
        assert "status=succeeded" in land
        assert "plot.png" in land
        run_dir = tmp_path / "projects" / "p1" / "experiments" / "e1" / "runs" / f"run-{run_id}"
        assert (run_dir / "artifacts" / "plot.png").is_file()
        assert (run_dir / "source" / "analysis.py").is_file()
        assert not (run_dir / "metrics.mlp.jsonl").exists()
        assert "rg_mean" in land


class TestLandRecordTagging:
    """Landing tags record-shaped trees for plugins (molrec layout as tags only)."""

    def test_zarr_meta_attrs_tagged(self, tmp_path: Path) -> None:
        from molexp.agent.ops.land import _infer_tags, _looks_like_record_package

        root = tmp_path / "pkg"
        root.mkdir()
        (root / "zarr.json").write_text(
            json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}),
            encoding="utf-8",
        )
        meta = root / "meta"
        meta.mkdir()
        (meta / "zarr.json").write_text(
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {
                        "record_schema_version": 1,
                        "format_name": "molrec",
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / "status").mkdir()
        assert _looks_like_record_package(root)
        tags = _infer_tags("pkg", root)
        assert tags["molrec"] == "true"
        assert tags["molrec_layout"] == "zarr"
        assert tags["record_schema_version"] == "1"
        assert "status" in tags["molrec_sections"]

    def test_plain_artifact_not_tagged(self, tmp_path: Path) -> None:
        from molexp.agent.ops.land import _infer_tags

        d = tmp_path / "plots"
        d.mkdir()
        (d / "fig.png").write_bytes(b"x")
        tags = _infer_tags("plots", d)
        assert "molrec" not in tags


class TestAutoDiscoveryLaw:
    def test_chat_preamble_forbids_default_land(self) -> None:
        text = DEFAULT_OPS_PREAMBLE
        assert text == CHAT_OPS_PREAMBLE
        assert "Chat Mode" in text
        assert "no" in text.lower() and "run_land" in text
        assert "Plan" in text
        assert "agent/.scratch" in text
        banned = (
            "molexp_add_project",
            "molexp_materialize",
            "molmcp__",
            "write_file",
            "execute_python",
        )
        for token in banned:
            assert token not in text, f"preamble hard-codes {token!r}"

    def test_full_preamble_documents_land(self) -> None:
        assert "run_land" in FULL_OPS_PREAMBLE
        assert "MolRec" in FULL_OPS_PREAMBLE

    def test_catalog_lists_chat_builtins(self, tmp_path: Path) -> None:
        ctx = build_session_context(
            workspace_root=tmp_path,
            execution_env=LocalExecutionEnv(scratch_dir=tmp_path / "scratch"),
        )
        catalog = render_discovery_catalog(ctx, surface="chat")
        assert "surface=chat" in catalog
        assert "`code_write`" in catalog
        assert "`run_land`" not in catalog
        specs = builtin_tool_specs(surface="chat")
        assert all(s.source == "builtin" for s in specs)
        assert {s.name for s in specs} == CHAT_TOOL_NAMES

    def test_ops_package_ships_no_upstream_symbol_table(self) -> None:
        ops_dir = Path(__file__).resolve().parents[3] / "src" / "molexp" / "agent" / "ops"
        for path in ops_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    v = node.value
                    assert "molpy.compute" not in v
                    assert "molpack." not in v or "molpack" in path.name

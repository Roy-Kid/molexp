"""embed_plot / embed_structure + envelope parse."""

from __future__ import annotations

import json
from pathlib import Path

from molexp.agent.execution_env import LocalExecutionEnv
from molexp.agent.ops import CHAT_TOOL_NAMES, build_ops_tools, build_session_context
from molexp.agent.ops.embed import encode_embed_result, parse_tool_result_payload


def test_parse_embed_envelope() -> None:
    raw = encode_embed_result(
        summary="plot:Rg",
        artifacts=[{"kind": "plot", "title": "Rg", "payload": {"mark": "line"}}],
    )
    summary, ok, arts = parse_tool_result_payload(raw)
    assert ok and summary == "plot:Rg"
    assert arts[0]["kind"] == "plot"
    assert arts[0]["payload"]["mark"] == "line"


def test_chat_surface_includes_embed_tools(tmp_path: Path) -> None:
    ctx = build_session_context(
        workspace_root=tmp_path,
        execution_env=LocalExecutionEnv(scratch_dir=tmp_path / "scratch"),
        confine_code_to="agent/.scratch",
    )
    names = {t.__name__ for t in build_ops_tools(ctx, surface="chat")}
    assert "embed_plot" in names
    assert "embed_structure" in names
    assert names == CHAT_TOOL_NAMES


def test_embed_plot_and_structure_tools(tmp_path: Path) -> None:
    ctx = build_session_context(
        workspace_root=tmp_path,
        execution_env=LocalExecutionEnv(scratch_dir=tmp_path / "scratch"),
        confine_code_to="agent/.scratch",
    )
    tools = {t.__name__: t for t in build_ops_tools(ctx, surface="chat")}
    plot = tools["embed_plot"](
        "Rg vs N",
        json.dumps({"mark": "line", "data": {"values": [{"x": 1, "y": 2}]}}),
    )
    summary, ok, arts = parse_tool_result_payload(plot)
    assert ok and "plot" in summary
    assert arts[0]["kind"] == "plot"
    assert arts[0]["title"] == "Rg vs N"

    xyz = "2\n\nC 0 0 0\nC 1 0 0\n"
    st = tools["embed_structure"]("dimer", "xyz", content=xyz)
    _s2, ok2, a2 = parse_tool_result_payload(st)
    assert ok2 and a2[0]["kind"] == "structure"
    assert a2[0]["payload"]["content"].startswith("2")

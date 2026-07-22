"""Agent-task LLM call cache — write, project events, prune by age/count/bytes."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from molexp.harness.gateways.llm_trace import LlmCallTrace
from molexp.services.agent_task_llm_cache import (
    LLM_CACHE_SUBDIR,
    ensure_plan_session_started,
    make_llm_call_observer,
    prune_agent_task_llm_cache,
    record_llm_call,
)
from molexp.services.agent_task_store import read_agent_task_events


class TestRecordLlmCall:
    def test_writes_cache_file_and_slim_event(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        cache_id = record_llm_call(
            root,
            "plan-abc",
            agent_name="experiment_reporter",
            model="stub-model",
            prompt="hello prompt " * 20,
            raw='{"title":"ok"}',
            prompt_artifact_id="p1",
            raw_artifact_id="r1",
        )
        cache_path = root / "agent" / "_tasks" / "plan-abc" / LLM_CACHE_SUBDIR / f"{cache_id}.json"
        assert cache_path.is_file()
        body = json.loads(cache_path.read_text())
        assert body["prompt"].startswith("hello prompt")
        assert body["raw"] == '{"title":"ok"}'

        events = read_agent_task_events(root, "plan-abc")
        assert len(events) == 1
        assert events[0]["type"] == "llm_call"
        payload = events[0]["payload"]
        assert payload["agent_name"] == "experiment_reporter"
        assert payload["cache_id"] == cache_id
        assert payload["cache"] is True
        assert "hello prompt" in payload["prompt_preview"]
        assert payload["prompt_artifact_id"] == "p1"
        assert payload["raw_artifact_id"] == "r1"


class TestEnsurePlanSessionStarted:
    def test_emits_a_single_loop_started_event(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        ensure_plan_session_started(root, "plan-x", draft="do science")
        ensure_plan_session_started(root, "plan-x", draft="do science")
        events = read_agent_task_events(root, "plan-x")
        assert sum(1 for e in events if e["type"] == "loop_started") == 1


class TestMakeLlmCallObserver:
    def test_records_gateway_trace_with_turn_id(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        observe = make_llm_call_observer(root, "plan-obs", turn_id="t1")
        observe(
            LlmCallTrace(
                agent_name="a",
                model="m",
                prompt="p",
                raw="r",
                prompt_artifact_id="pa",
                raw_artifact_id="ra",
            )
        )
        events = read_agent_task_events(root, "plan-obs")
        assert events[-1]["type"] == "llm_call"
        assert events[-1]["payload"]["turn_id"] == "t1"


class TestPruneAgentTaskLlmCache:
    def test_drops_files_past_age_and_count_caps(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        task = "plan-prune"
        ids: list[str] = []
        for i in range(5):
            cid = record_llm_call(
                root,
                task,
                agent_name="a",
                model="m",
                prompt=f"prompt-{i}",
                raw=f"raw-{i}",
                prompt_artifact_id=f"p{i}",
                raw_artifact_id=f"r{i}",
            )
            ids.append(cid)

        cache_dir = root / "agent" / "_tasks" / task / LLM_CACHE_SUBDIR
        # Age the two oldest files past the retention window.
        old = time.time() - 10 * 86400
        for cid in ids[:2]:
            os.utime(cache_dir / f"{cid}.json", (old, old))

        stats = prune_agent_task_llm_cache(
            root,
            max_age_days=7,
            max_calls_per_task=2,
            max_bytes_per_workspace=10**9,
        )
        assert stats.removed_files >= 2
        remaining = sorted(p.stem for p in cache_dir.glob("*.json"))
        assert len(remaining) <= 2
        assert set(remaining).issubset(set(ids))

        # The projected llm_call events track the surviving cache files exactly.
        events = read_agent_task_events(root, task)
        event_ids = {e["payload"]["cache_id"] for e in events if e["type"] == "llm_call"}
        assert event_ids == set(remaining)

    def test_enforces_workspace_byte_budget(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        big = "x" * 2000
        for i in range(4):
            record_llm_call(
                root,
                f"plan-b{i}",
                agent_name="a",
                model="m",
                prompt=big,
                raw=big,
                prompt_artifact_id=f"p{i}",
                raw_artifact_id=f"r{i}",
            )
        stats = prune_agent_task_llm_cache(
            root,
            max_age_days=30,
            max_calls_per_task=100,
            max_bytes_per_workspace=5000,
        )
        assert stats.removed_files >= 1
        total = sum(
            path.stat().st_size for path in (root / "agent" / "_tasks").rglob("llm_cache/*.json")
        )
        assert total <= 5000

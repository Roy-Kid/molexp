"""``write_knowledge_item`` — the single sourced-KnowledgeItem writer (agent-record-export-01).

Owns the writer's own behavior: meta+body write, idempotent-on-name with an
emit-once event, the ``emit`` suppression flag, and the non-fatal emit contract.
The empty-sources invariant belongs to :class:`KnowledgeMeta` (test_knowledge_item).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from molexp.workspace import (
    KNOWLEDGE_ITEM_KIND,
    KnowledgeItem,
    SourceRef,
    read_workspace_events,
    write_knowledge_item,
)


class TestWriteKnowledgeItem:
    def test_creates_meta_body_and_emits_event_once(self, experiment: Any) -> None:
        item = write_knowledge_item(
            experiment,
            name="finding-demo",
            kind="Finding",
            sources=[SourceRef(kind="experiment", ref=experiment.id)],
            created_by="tester",
            body="# Finding\n\nhello\n",
            title="demo",
        )
        assert isinstance(item, KnowledgeItem)
        meta = item.read_knowledge_meta()
        assert meta.kind == "Finding"
        assert meta.created_by == "tester"
        assert item.body().startswith("# Finding")

        events = read_workspace_events(experiment.workspace.root, type="knowledge.created")
        assert len(events) == 1
        assert events[0].payload.get("type") == KNOWLEDGE_ITEM_KIND
        assert "finding-demo" in (events[0].refs[0] if events[0].refs else "")

    def test_repeat_write_is_idempotent_and_emits_no_second_event(self, experiment: Any) -> None:
        kwargs = {
            "name": "finding-demo",
            "kind": "Finding",
            "sources": [SourceRef(kind="experiment", ref=experiment.id)],
            "created_by": "tester",
            "body": "first\n",
        }
        write_knowledge_item(experiment, **kwargs)
        write_knowledge_item(experiment, **{**kwargs, "body": "second\n"})
        events = read_workspace_events(experiment.workspace.root, type="knowledge.created")
        assert len(events) == 1
        assert "second" in experiment.get_folder("finding-demo", cls=KnowledgeItem).body()

    def test_emit_false_suppresses_creation_event(self, experiment: Any) -> None:
        write_knowledge_item(
            experiment,
            name="quiet",
            kind="Observation",
            sources=[SourceRef(kind="experiment", ref=experiment.id)],
            created_by="tester",
            body="quiet\n",
            emit=False,
        )
        assert read_workspace_events(experiment.workspace.root, type="knowledge.created") == []

    def test_emit_failure_is_non_fatal(self, experiment: Any) -> None:
        with patch(
            "molexp.workspace.knowledge_write.emit_workspace_event",
            side_effect=RuntimeError("db locked"),
        ):
            item = write_knowledge_item(
                experiment,
                name="still-ok",
                kind="Finding",
                sources=[SourceRef(kind="experiment", ref=experiment.id)],
                created_by="tester",
                body="ok\n",
            )
        assert item.body().startswith("ok")

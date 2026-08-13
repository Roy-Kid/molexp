"""Tests for body-aware ``Bundle.search`` (vision-loop-04 retrieval substrate).

``Bundle.search`` is the one body-aware, snippet-returning retrieval verb: it
returns a frozen :class:`SearchResult` (``hits`` + ``truncated``) whose
:class:`SearchHit` rows pair the index entry with the matched fields and — for
a body match — the first matching ``index.md`` line trimmed to <=160 chars.
Body reads are lazy (only entries that survived the index-level filters and did
not already match on path/title/tag) and size-capped at 512 KiB; an oversized
body is skipped for body matching but the entry still matches on its index
fields (no silent drop). Index-level filters (type / tag kwargs, rebuild) are
owned by ``test_bundle_index.py``.
"""

from __future__ import annotations

from pathlib import Path

from molexp.workspace import Bundle, Note, Workspace
from molexp.workspace.folder import Folder

# The spec's body-read cap (mirrors agent/loops/interactive/tools.py).
MAX_BODY_SEARCH_BYTES = 512 * 1024

# Needles chosen to never collide with a concept dir name or H1 title unless a
# test plants them there deliberately.
BODY_NEEDLE = "octanol-partition-needle"
SCOPE_NEEDLE = "scoped-retrieval-needle"
LIMIT_NEEDLE = "limited-retrieval-needle"
SORT_NEEDLE = "sorted-retrieval-needle"


def _root(tmp_path: Path) -> Path:
    """An empty on-disk bundle root."""
    root = tmp_path / "bundle"
    root.mkdir()
    return root


def _note(bundle_root: Path, name: str, index_md: str, *, under: str | None = None) -> Note:
    """Materialize a ``Note`` Concept with *index_md* as its body.

    *under* is a bundle-relative plain organizational dir (no ``meta.json``),
    created on demand — the ``scope`` filter's subtree host.
    """
    host = bundle_root if under is None else bundle_root / under
    host.mkdir(parents=True, exist_ok=True)
    note = Note(name=name, root_path=str(host))
    note.materialize()  # meta.json concept identity
    note.write_index(index_md)
    return note


class TestSearch:
    """Body-aware retrieval over the Concept tree."""

    def test_body_only_needle_matches_with_body_field(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        _note(root, "findings", f"# Plain Title\n\nlead-in line\nthe {BODY_NEEDLE} sits here\n")
        _note(root, "unrelated", "# Another\n\nnothing to see\n")

        result = Bundle(root).search(BODY_NEEDLE)

        assert [hit.entry.path for hit in result.hits] == ["findings"]
        assert result.hits[0].matched_fields == ("body",)

    def test_body_match_snippet_is_first_matching_line(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        _note(
            root,
            "findings",
            (
                "# Plain Title\n\n"
                "lead-in line\n"
                f"the {BODY_NEEDLE} sits here\n"
                f"the {BODY_NEEDLE} also repeats later\n"
            ),
        )

        result = Bundle(root).search(BODY_NEEDLE)

        assert result.hits[0].snippet == f"the {BODY_NEEDLE} sits here"

    def test_body_match_snippet_trimmed_to_160_chars(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        long_line = f"{BODY_NEEDLE} " + "x" * 400
        _note(root, "long-line", f"# Plain Title\n\n{long_line}\n")

        result = Bundle(root).search(BODY_NEEDLE)

        snippet = result.hits[0].snippet
        assert snippet is not None
        assert len(snippet) <= 160
        assert BODY_NEEDLE in snippet

    def test_title_match_short_circuits_body_read(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        _note(root, "milestones", "# Solvation Milestones\n\nbody without the phrase\n")

        # lower-case needle vs H1 case — matching is case-insensitive; the title
        # match short-circuits, so "body" is NOT reported even though the H1 line
        # is part of index.md.
        result = Bundle(root).search("solvation milestones")

        assert [hit.entry.path for hit in result.hits] == ["milestones"]
        assert result.hits[0].matched_fields == ("title",)

    def test_body_needle_not_matched_when_include_body_false(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        _note(root, "findings", f"# Plain Title\n\nthe {BODY_NEEDLE} sits here\n")

        result = Bundle(root).search(BODY_NEEDLE, include_body=False)

        assert result.hits == ()
        assert result.truncated is False

    def test_tag_needle_reports_tag_field(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        root.mkdir()
        folder = Folder(name="untitled-doc", kind="bundle.concept", root_path=str(root))
        folder.materialize()
        folder.write_meta()
        # tags live in meta.json (idiom of test_bundle_index.test_filter_by_tag)
        import json

        (Path(folder.resolve()) / "meta.json").write_text(
            json.dumps(
                {
                    "type": "bundle.concept",
                    "id": "untitled-doc",
                    "tags": ["zwitterion"],
                },
                indent=2,
            )
            + "\n"
        )
        (Path(folder.resolve()) / "index.md").write_text("nothing relevant here\n")

        result = Bundle(root).search("zwitterion")

        assert [h.matched_fields for h in result.hits] == [("tag",)]
        assert result.hits[0].snippet is None

    def test_scope_filters_to_subtree(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        _note(root, "inside", f"# In\n\nthe {SCOPE_NEEDLE} in scope\n", under="projects/p")
        _note(root, "outside", f"# Out\n\nthe {SCOPE_NEEDLE} out of scope\n")

        result = Bundle(root).search(SCOPE_NEEDLE, scope="projects/p")

        assert [hit.entry.path for hit in result.hits] == ["projects/p/inside"]

    def test_scope_trailing_slash_matches_scope_root_itself(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path / "lab", name="lab")
        ws.materialize()
        ws.add_project("p")

        result = Bundle(ws.resolve()).search(scope="projects/p/")

        assert any(h.entry.path == "projects/p" for h in result.hits)

    def test_limit_undercount_sets_truncated(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        _note(root, "one", f"# One\n\nthe {LIMIT_NEEDLE} here\n")
        _note(root, "two", f"# Two\n\nthe {LIMIT_NEEDLE} here too\n")

        result = Bundle(root).search(LIMIT_NEEDLE, limit=1)

        assert len(result.hits) == 1
        assert result.hits[0].entry.path == "one"  # first in path-ascending order
        assert result.truncated is True

    def test_oversized_body_skipped_for_body_match(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        filler = "p" * (MAX_BODY_SEARCH_BYTES + 1)
        _note(root, "huge", f"# Plain Title\n\n{filler}\n{BODY_NEEDLE}\n")

        result = Bundle(root).search(BODY_NEEDLE)

        assert result.hits == ()

    def test_oversized_body_still_matches_on_title(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        filler = "p" * (MAX_BODY_SEARCH_BYTES + 1)
        _note(root, "huge-titled", f"# Contains {BODY_NEEDLE} up top\n\n{filler}\n")

        result = Bundle(root).search(BODY_NEEDLE)

        assert [hit.entry.path for hit in result.hits] == ["huge-titled"]
        assert result.hits[0].matched_fields == ("title",)

    def test_results_are_path_ascending_and_stable(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        # created deliberately out of path order
        _note(root, "mmm", f"# M\n\nthe {SORT_NEEDLE}\n")
        _note(root, "inner", f"# I\n\nthe {SORT_NEEDLE}\n", under="zzz")
        _note(root, "aaa", f"# A\n\nthe {SORT_NEEDLE}\n")

        b = Bundle(root)
        first = [hit.entry.path for hit in b.search(SORT_NEEDLE).hits]
        second = [hit.entry.path for hit in b.search(SORT_NEEDLE).hits]

        assert first == ["aaa", "mmm", "zzz/inner"]
        assert first == sorted(first)
        assert second == first

    def test_text_none_keeps_filter_only_behavior(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        _note(root, "aaa", "# A\n\nbody a\n")
        _note(root, "bbb", "# B\n\nbody b\n")

        result = Bundle(root).search()

        assert [hit.entry.path for hit in result.hits] == ["aaa", "bbb"]
        assert all(hit.snippet is None for hit in result.hits)
        assert all(hit.matched_fields == () for hit in result.hits)
        assert result.truncated is False

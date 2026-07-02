"""Execute every ```python block in ``docs/**/*.md`` as a pytest case.

The drift gate for the documentation: any example that no longer runs against
the shipped API fails here, in CI, instead of failing for the first reader
who pastes it. One test per block, named ``<page>:b<NN>``; blocks on the same
page share a namespace and working directory and run in page order (tutorial
pages build state progressively), so this module must not be sharded by
pytest-xdist or randomized.

Author conventions (full text in ``tests/test_docs/README.md``):

- ``# docs: skip — <reason>`` as the block's first line: never executed
  (pseudo-code, LLM-key/network-dependent snippets).
- ``# docs: xfail — <reason>``: executed, expected to fail until the named
  workflow lands (non-strict, so it flips green silently at integration).
- Top-level ``await`` is fine — blocks compile with
  ``PyCF_ALLOW_TOP_LEVEL_AWAIT`` and run under ``asyncio.run``.

Pages listed in :data:`PENDING_PAGES` are executed but expected to fail as a
whole (a rewrite by another workflow is in flight); remove the entry when the
rewrite lands.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import re
from pathlib import Path

import pytest

from .doc_blocks import DocBlock, collect_doc_blocks

# Pages whose blocks are expected to fail because a rewrite owned by another
# in-flight workflow has not landed yet. Key: page path relative to the repo
# root; value: the reason (must name the workflow to wait for).
PENDING_PAGES: dict[str, str] = {}

ALL_BLOCKS = collect_doc_blocks()


def _params() -> list[object]:
    params = []
    for block in ALL_BLOCKS:
        marks: list[pytest.MarkDecorator] = [pytest.mark.integration]
        if block.skip_reason is not None:
            marks.append(pytest.mark.skip(reason=f"docs marker: {block.skip_reason}"))
        elif block.page in PENDING_PAGES:
            marks.append(pytest.mark.xfail(reason=PENDING_PAGES[block.page], strict=False))
        elif block.xfail_reason is not None:
            marks.append(
                pytest.mark.xfail(reason=f"docs marker: {block.xfail_reason}", strict=False)
            )
        params.append(pytest.param(block, id=block.test_id, marks=marks))
    return params


class _PageState:
    """Shared namespace + working directory for one docs page."""

    def __init__(self, cwd: Path) -> None:
        self.namespace: dict[str, object] = {"__name__": "__docs__"}
        self.cwd = cwd
        self.failed_block: int | None = None


_PAGE_STATES: dict[str, _PageState] = {}


def _state_for(page: str, tmp_path_factory: pytest.TempPathFactory) -> _PageState:
    state = _PAGE_STATES.get(page)
    if state is None:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", page).strip("-")[-40:]
        state = _PageState(tmp_path_factory.mktemp(slug))
        _PAGE_STATES[page] = state
    return state


def test_collector_sees_docs() -> None:
    """The collector must never silently collect nothing."""
    assert ALL_BLOCKS, "no ```python blocks collected from docs/ — collector broken?"


@pytest.mark.parametrize("block", _params())
def test_docs_python_block(
    block: DocBlock,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state_for(block.page, tmp_path_factory)
    if state.failed_block is not None:
        pytest.skip(
            f"earlier block b{state.failed_block:02d} of {block.page} failed; "
            "this block shares its page namespace"
        )
    monkeypatch.chdir(state.cwd)
    code = compile(
        block.source,
        f"<{block.test_id} (docs line {block.lineno})>",
        "exec",
        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
    )
    try:
        if code.co_flags & inspect.CO_COROUTINE:
            # eval of an exec-compiled coroutine code object returns the coroutine.
            asyncio.run(eval(code, state.namespace))  # docs are first-party code
        else:
            exec(code, state.namespace)  # docs are first-party code
    except Exception:
        state.failed_block = block.index
        raise

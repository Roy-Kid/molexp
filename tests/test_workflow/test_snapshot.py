"""``snapshot._normalize_ast`` — AST-normalized code hashing (spec P0-2).

The content-addressed cache keys on a hash derived from AST-normalized task
source. Formatting/comments must be invisible to the hash, but decorators are
part of semantic identity — a behaviour-changing decorator (retry, units,
lru_cache, validation) MUST invalidate the hash, or the cache silently returns
stale/wrong results.
"""

from __future__ import annotations

from molexp.workflow.snapshot import _normalize_ast


class TestNormalizeAst:
    def test_comments_and_whitespace_are_ignored(self) -> None:
        """Pure formatting / comment differences hash identically."""
        plain = "def f(x):\n    return x + 1\n"
        noisy = "def f(x):\n    # explanatory comment\n    return x  +  1\n"
        assert _normalize_ast(plain) == _normalize_ast(noisy)

    def test_adding_a_decorator_changes_the_normalized_ast(self) -> None:
        """A decorated body must differ from the undecorated one (P0-2 bug)."""
        plain = "def f(x):\n    return x + 1\n"
        decorated = "@retry(3)\ndef f(x):\n    return x + 1\n"
        assert _normalize_ast(plain) != _normalize_ast(decorated)

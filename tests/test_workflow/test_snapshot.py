"""Tests for TaskSnapshot AST-normalized code hashing (spec P0-2).

The content-addressed cache keys on a code hash derived from the AST-normalized
task source. Formatting/comments must be invisible to the hash, but decorators
are part of semantic identity — a behaviour-changing decorator (retry, units,
lru_cache, validation) MUST invalidate the hash, or the cache silently returns
stale/wrong results.
"""

from molexp.workflow.snapshot import _normalize_ast


def test_comments_and_whitespace_are_ignored() -> None:
    """Pure formatting / comment differences hash identically."""
    plain = "def f(x):\n    return x + 1\n"
    noisy = "def f(x):\n    # explanatory comment\n    return x  +  1\n"
    assert _normalize_ast(plain) == _normalize_ast(noisy)


def test_adding_a_decorator_changes_the_normalized_ast() -> None:
    """A decorated body must differ from the undecorated one (P0-2 bug)."""
    plain = "def f(x):\n    return x + 1\n"
    decorated = "@retry(3)\ndef f(x):\n    return x + 1\n"
    assert _normalize_ast(plain) != _normalize_ast(decorated)


def test_changing_a_decorator_argument_changes_the_normalized_ast() -> None:
    """Different decorator behaviour (retry(3) vs retry(5)) must not collide."""
    three = "@retry(3)\ndef f(x):\n    return x\n"
    five = "@retry(5)\ndef f(x):\n    return x\n"
    assert _normalize_ast(three) != _normalize_ast(five)


def test_replacing_a_decorator_changes_the_normalized_ast() -> None:
    """Swapping one decorator for another must change the hash."""
    a = "@jit\ndef f(x):\n    return x\n"
    b = "@lru_cache\ndef f(x):\n    return x\n"
    assert _normalize_ast(a) != _normalize_ast(b)


def test_body_change_still_changes_the_normalized_ast() -> None:
    """Sanity: a real body change is detected (regression guard)."""
    a = "def f(x):\n    return x + 1\n"
    b = "def f(x):\n    return x + 2\n"
    assert _normalize_ast(a) != _normalize_ast(b)

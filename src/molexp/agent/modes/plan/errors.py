"""Domain exceptions for the PlanMode pipeline.

Currently exposes :class:`SkeletonCompileError` only;
``GenerateWorkflowSkeleton`` raises this when the Python source it
emits fails :func:`compile`. The exception is a
:class:`RuntimeError` subclass (the failure is a code-generation
defect, not a programmer-error), and the original
:class:`SyntaxError` is chained via ``__cause__`` so a future
repair-style task can introspect the parser diagnostic.
"""

from __future__ import annotations

__all__ = ["SkeletonCompileError"]


class SkeletonCompileError(RuntimeError):
    """Generated workflow skeleton failed :func:`compile` validation.

    Raised by ``GenerateWorkflowSkeleton`` when the LLM-emitted source
    cannot be parsed. The triggering :class:`SyntaxError` is chained
    through ``raise … from exc`` so callers can inspect
    ``error.__cause__`` to locate the offending line.

    Attributes:
        path: The would-be file path of the failed source. Useful for
            error messages even though no file was written.
    """

    def __init__(self, message: str, *, path: str = "") -> None:
        self.path = path
        super().__init__(message)

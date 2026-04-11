"""MolexpToolCatalog: builds a pydantic-ai toolset from molexp tools.

Combines:
1. Built-in Level 1/2/3 workspace tools
2. User-provided extra tools (molexp Tool instances or pydantic-ai tools)
3. ApprovalPolicy applied via pydantic-ai's approval_required() wrapper
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from ..policy import ApprovalPolicy
from ..tools import FunctionTool
from ..tools import Tool as MolexpTool
from ..types import ToolContext
from .deps import MolexpDeps
from .workspace_tools import get_all_builtin_tools


class MolexpToolCatalog:
    """Assembles the complete pydantic-ai toolset for a molexp agent session.

    Built-in tools cover Level 1 (workspace read) and Level 3 (create run).
    Level 2 workflow tools will be added in Phase 3.

    Args:
        extra_tools: User-provided additional tools (molexp.agent.Tool instances
                     or raw pydantic-ai tool functions)
        approval_policy: Policy controlling which tools need human approval
    """

    def __init__(
        self,
        extra_tools: list[Any] | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        self._extra_tools = extra_tools or []
        self._approval_policy = approval_policy or ApprovalPolicy()

    def build(self) -> AbstractToolset[MolexpDeps]:
        """Build and return the complete toolset.

        Returns:
            AbstractToolset ready to pass to pydantic-ai Agent
        """
        toolset: FunctionToolset[MolexpDeps] = FunctionToolset(
            tools=get_all_builtin_tools()
        )

        # Add user extra tools
        for tool in self._extra_tools:
            if hasattr(tool, "_tool_registration"):
                # @agent_tool decorated function → extract inner function
                registration: FunctionTool = tool._tool_registration
                wrapper = _make_pydantic_ai_wrapper(registration)
                toolset.add_function(wrapper)
            elif isinstance(tool, MolexpTool):
                # OOP Tool subclass instance
                wrapper = _make_pydantic_ai_wrapper_from_tool(tool)
                toolset.add_function(wrapper)
            elif callable(tool):
                # Already a pydantic-ai compatible function (takes RunContext first)
                toolset.add_function(tool)

        # Apply approval policy via pydantic-ai's built-in mechanism
        if self._approval_policy.require_approval_for:
            policy = self._approval_policy

            def approval_required_func(
                ctx: Any, tool_def: Any, tool_args: dict[str, Any]
            ) -> bool:
                return policy.needs_approval(tool_def.name)

            return toolset.approval_required(approval_required_func)

        return toolset


def _make_pydantic_ai_wrapper(registration: FunctionTool):
    """Wrap a molexp FunctionTool as a pydantic-ai tool function.

    The wrapper adapts `fn(ctx: ToolContext, **kwargs)` to the
    pydantic-ai `fn(ctx: RunContext[MolexpDeps], **kwargs)` signature.

    Note: Type annotations from the original function are preserved
    (minus the ToolContext first arg) so pydantic-ai can generate
    the correct JSON schema.
    """
    import inspect

    from pydantic_ai import RunContext

    original_fn = registration._fn
    sig = inspect.signature(original_fn)
    params = list(sig.parameters.values())

    # Remove first param (ToolContext ctx)
    _inner_params = [p for p in params if p.annotation is not ToolContext
                     and p.name not in ("ctx", "context")]

    async def wrapper(ctx: RunContext[MolexpDeps], **kwargs: Any) -> Any:
        tool_ctx = ToolContext(
            workspace=ctx.deps.workspace,
            run=ctx.deps.current_run,
            session=ctx.deps.session,
        )
        return await registration._fn(tool_ctx, **kwargs)

    wrapper.__name__ = registration.name.replace(".", "_")
    wrapper.__doc__ = original_fn.__doc__ or f"Tool: {registration.name}"

    # Copy annotations (skip ToolContext ctx arg)
    annotations = {
        k: v for k, v in getattr(original_fn, "__annotations__", {}).items()
        if k not in ("ctx", "context", "return")
    }
    annotations["ctx"] = RunContext[MolexpDeps]
    wrapper.__annotations__ = annotations

    return wrapper


def _make_pydantic_ai_wrapper_from_tool(tool: MolexpTool):
    """Wrap an OOP Tool instance as a pydantic-ai tool function."""
    from pydantic_ai import RunContext

    async def wrapper(ctx: RunContext[MolexpDeps], **kwargs: Any) -> Any:
        tool_ctx = ToolContext(
            workspace=ctx.deps.workspace,
            run=ctx.deps.current_run,
            session=ctx.deps.session,
        )
        return await tool.call(tool_ctx, **kwargs)

    wrapper.__name__ = type(tool).__name__
    wrapper.__doc__ = tool.__doc__ or f"Tool: {tool.name}"
    return wrapper

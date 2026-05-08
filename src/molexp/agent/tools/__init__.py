"""Tool layer.

Public exports: dataclasses + registry + dispatcher + policy. The
``native`` subpackage holds package-shipped tools
"""

from molexp.agent.tools.dispatcher import (
    ApprovalGate,
    AutoApproveGate,
    DenyAllGate,
    ToolDispatcher,
    ToolEventCallback,
)
from molexp.agent.tools.policy import (
    PERMISSIVE_POLICY,
    READ_ONLY_POLICY,
    ApprovalDecision,
    ToolPolicy,
)
from molexp.agent.tools.registry import (
    DuplicateToolError,
    ToolRegistry,
    get_native_spec,
    is_native_tool,
    native_tool,
)
from molexp.agent.tools.spec import (
    RegisteredTool,
    ToolCallable,
    ToolContext,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "PERMISSIVE_POLICY",
    "READ_ONLY_POLICY",
    "ApprovalDecision",
    "ApprovalGate",
    "AutoApproveGate",
    "DenyAllGate",
    "DuplicateToolError",
    "RegisteredTool",
    "ToolCallable",
    "ToolContext",
    "ToolDispatcher",
    "ToolEventCallback",
    "ToolPolicy",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "get_native_spec",
    "is_native_tool",
    "native_tool",
]

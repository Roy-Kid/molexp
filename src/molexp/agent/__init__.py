"""Model-agnostic agent harness (spec ``agent-harness-architecture``).

Public surface — the only names server routes, plugins, and external
callers should import:

- :class:`AgentService` — workspace-scoped facade.
- :class:`AgentSession` — per-session handle.
- :class:`Goal`, :class:`Message`, :class:`Usage` — core types.
- :class:`ToolSpec`, :class:`ToolResult`, :class:`ToolContext`,
  :func:`native_tool` — tool surface.
- :class:`ModelClient`, :class:`ModelRequest`, :class:`ModelResponse` —
  model boundary contract.

The package is stdlib-only. Importing this module must not pull in
``pydantic_ai``, MCP SDKs, HTTP clients, or any provider SDK; the
import-guard test in ``tests/agent/test_import_guard.py`` enforces
this rule.
"""

from molexp.agent.model import (
    ModelClient,
    ModelClientFactory,
    ModelConfig,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ProviderConfigValidator,
    ToolSchema,
)
from molexp.agent.model_registry import (
    UnknownProviderError,
    create_model_client,
    list_providers,
    register_model_provider,
)
from molexp.agent.orchestration.session import AgentSession
from molexp.agent.service import AgentService
from molexp.agent.tools.registry import (
    DuplicateToolError,
    ToolRegistry,
    native_tool,
)
from molexp.agent.tools.source import (
    ToolSource,
    UnknownToolSourceError,
    list_tool_sources,
    register_tool_source,
)
from molexp.agent.tools.spec import (
    ToolCallable,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from molexp.agent.types import (
    AgentFailure,
    AgentMode,
    ArtifactRef,
    FailureKind,
    Goal,
    Message,
    SessionStatus,
    Usage,
    WorkflowPreview,
)

__all__ = [
    "AgentFailure",
    "AgentMode",
    "AgentService",
    "AgentSession",
    "ArtifactRef",
    "DuplicateToolError",
    "FailureKind",
    "Goal",
    "Message",
    "ModelClient",
    "ModelClientFactory",
    "ModelConfig",
    "ModelEvent",
    "ModelRequest",
    "ModelResponse",
    "ModelToolCall",
    "ProviderConfigValidator",
    "SessionStatus",
    "ToolCallable",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
    "ToolSource",
    "ToolSpec",
    "UnknownProviderError",
    "UnknownToolSourceError",
    "Usage",
    "WorkflowPreview",
    "create_model_client",
    "list_providers",
    "list_tool_sources",
    "native_tool",
    "register_model_provider",
    "register_tool_source",
]

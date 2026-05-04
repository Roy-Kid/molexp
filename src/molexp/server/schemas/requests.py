"""Pydantic request models for MolExp API.

Aligned with workspace.models — field names match domain models.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Discriminator, Field

# ── Workspace ───────────────────────────────────────────────────────────────


class WorkspaceOpenRequest(BaseModel):
    path: str = Field(..., description="Absolute path to the workspace")
    create_if_missing: bool = Field(False, description="Create if missing")


# ── Project ─────────────────────────────────────────────────────────────────


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., description="Human-readable project name")
    description: str = Field("", description="Project description")
    owner: str = Field("", description="Project owner")
    tags: list[str] = Field(default_factory=list, description="Project tags")


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    owner: str | None = None
    tags: list[str] | None = None
    config: dict[str, Any] | None = None


# ── Experiment ──────────────────────────────────────────────────────────────


class ExperimentCreateRequest(BaseModel):
    name: str = Field(..., description="Human-readable experiment name")
    workflow_source: str | None = Field(None, description="Path to workflow file")
    description: str = Field("", description="Experiment description")
    parameter_space: dict[str, Any] = Field(
        default_factory=dict, description="Parameter space definition"
    )
    default_target: str | None = Field(
        default=None,
        alias="defaultTarget",
        description="Compute target name new runs should default to (must exist)",
    )

    model_config = {"populate_by_name": True}


# ── Run ─────────────────────────────────────────────────────────────────────


class RunCreateRequest(BaseModel):
    parameters: dict[str, Any] = Field(default_factory=dict, description="Run parameters")
    target: str | None = Field(
        default=None,
        description="Compute target name (must exist in workspace registry)",
    )


class RunStatusUpdateRequest(BaseModel):
    status: str = Field(..., description="New status value")


# ── Execution ───────────────────────────────────────────────────────────────


class ExecutionCreateRequest(BaseModel):
    project_id: str = Field(..., description="Target project ID")
    experiment_id: str = Field(..., description="Target experiment ID")
    parameters: dict[str, Any] = Field(default_factory=dict)
    workflow_json: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional workflow IR (matches schema/workflow.json). When "
            "provided and the experiment has no workflow bound yet, the "
            "server binds it and persists the IR to disk. Subsequent "
            "calls reuse the on-disk binding."
        ),
    )


# ── Asset ───────────────────────────────────────────────────────────────────


class AssetUpdateRequest(BaseModel):
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


# ── Agent ───────────────────────────────────────────────────────────────────


class GoalCreateRequest(BaseModel):
    description: str = Field(..., description="Natural language goal description")
    constraints: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)
    plan_mode: bool = Field(
        False,
        description=(
            "When true, the runtime registers only read-only tools and asks "
            "the agent to emit a structured plan instead of executing."
        ),
    )
    instructions_override: str | None = Field(
        None,
        description=(
            "Replace the layered system prompt for this single session. "
            "Workspace and skill addenda are bypassed; the molexp built-in "
            "preamble is also dropped."
        ),
    )
    skill_id: str | None = Field(
        None,
        description=(
            "When the goal originates from a slash command, the underlying "
            "skill id (informational; the route still resolves the skill's "
            "instructions server-side)."
        ),
    )


class ApprovalRespondRequest(BaseModel):
    request_id: str = Field(..., description="Approval request ID")
    approved: bool = Field(..., description="Whether to approve")


class PlanDecisionRequest(BaseModel):
    """User decision on a plan emitted by ``exit_plan_mode``.

    Pairs with :class:`PlanCreatedEvent` via ``request_id``. Approval
    flips the session out of plan mode so the agent can bind / run
    the (possibly user-edited) workflow IR. Rejection hands the
    feedback back to the agent so it can revise + call
    ``exit_plan_mode`` again.
    """

    request_id: str = Field(..., description="ID from the PlanCreatedEvent payload.")
    approved: bool = Field(
        ...,
        description=(
            "True to approve the plan. False to reject and keep the agent "
            "in plan mode for revision."
        ),
    )
    edited_plan: str | None = Field(
        None,
        description=(
            "Optional user edit of the plan markdown. When set, the agent "
            "sees this exact text as the post-approval starting point "
            "instead of its own draft."
        ),
    )
    edited_workflow_ir: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional user edit of the workflow IR. Replaces the agent's "
            "drafted IR on approval."
        ),
    )
    feedback: str = Field(
        "",
        description=(
            "Free-form rejection rationale. Surfaced to the agent so its "
            "next attempt addresses the user's concern."
        ),
    )


class ReviewDecisionRequest(BaseModel):
    """Approve or reject a persisted review item."""

    comment: str = Field("", description="Optional human resolution comment.")
    edited_plan: str | None = Field(
        None,
        description="Optional edited plan markdown when approving a plan review.",
    )
    edited_workflow_ir: dict[str, Any] | None = Field(
        None,
        description="Optional edited workflow IR when approving a plan review.",
    )


class UserMessageCreateRequest(BaseModel):
    """Mid-session chat message from the user to the agent."""

    content: str = Field(..., description="User's message")
    request_id: str | None = Field(
        None,
        description=(
            "Pending UserMessageRequestEvent id this message replies to "
            "(omit for an unsolicited follow-up)."
        ),
    )


# ── Skills (saved goal templates) ───────────────────────────────────────────


class SkillCreateRequest(BaseModel):
    name: str = Field(..., description="Display name")
    goal_template: str = Field(
        ...,
        description="Goal description, may contain {{param}} placeholders",
    )
    description: str = Field("", description="Long description")
    slash_name: str = Field(
        "",
        description=(
            "Optional slash command id (e.g. 'plot-energy'). When set, the "
            "skill is invokable from the chat input as /<slash_name>. "
            "Reserved names: plan, clear, model, help."
        ),
    )
    instructions: str = Field(
        "",
        description="System prompt addendum applied when this skill launches a session",
    )
    default_plan_mode: bool = Field(
        False,
        description="Sessions launched from this skill default to plan mode",
    )
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Optional fnmatch-style glob list narrowing the agent's tool "
            "surface for sessions launched from this skill. Empty = all "
            "tools that aren't denied. Example: ['list_*', 'mcp:python.*']."
        ),
    )
    denied_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Optional fnmatch-style glob list of tools to hide. Denial wins "
            "over allow on collision."
        ),
    )
    requires_exit_tool: str = Field(
        "",
        description=(
            "When set, names a builtin tool the agent MUST call to leave "
            "this skill's mode (e.g. 'exit_plan_mode' for plan mode)."
        ),
    )
    scope: str = Field(
        "workspace",
        description=(
            "Where to persist this skill: 'workspace' (default, "
            "<workspace>/.skills.json) or 'user' (~/.molexp/skills.json)."
        ),
    )


class SkillUpdateRequest(BaseModel):
    name: str | None = None
    goal_template: str | None = None
    description: str | None = None
    slash_name: str | None = None
    instructions: str | None = None
    default_plan_mode: bool | None = None
    constraints: list[str] | None = None
    success_criteria: list[str] | None = None
    tags: list[str] | None = None
    allowed_tools: list[str] | None = None
    denied_tools: list[str] | None = None
    requires_exit_tool: str | None = None


class SkillLaunchRequest(BaseModel):
    """Materialize a skill into a Goal and start a session."""

    parameters: dict[str, Any] = Field(default_factory=dict)
    plan_mode: bool | None = Field(
        None,
        description=(
            "Override the skill's ``default_plan_mode``. ``None`` (default) "
            "honors the skill's setting."
        ),
    )


class CommandParseRequest(BaseModel):
    """Parse a raw chat input that the user typed starting with '/'."""

    raw: str = Field(..., description="The raw chat text, including the leading '/'.")


# ── Agent provider config ───────────────────────────────────────────────────


class AgentProviderUpdateRequest(BaseModel):
    """Patch the workspace's LLM provider config.

    Any field left as ``None`` is preserved. Pass ``api_key=""`` to clear
    the stored key (e.g. when switching to env-var-only auth).
    """

    provider: str | None = Field(
        None,
        description="One of: 'anthropic', 'openai', 'google', 'openai-compatible'",
    )
    model: str | None = Field(None, description="Provider-specific model name")
    api_key: str | None = Field(None, description="Set to clear by passing empty string")
    base_url: str | None = Field(
        None, description="Optional override for proxy/self-hosted endpoints"
    )
    instructions: str | None = Field(
        None,
        description=(
            "Workspace-default system prompt addendum. Pass an empty string "
            "to clear; ``None`` leaves the existing value untouched."
        ),
    )


# ── MCP servers ─────────────────────────────────────────────────────────────


class McpStdioSpecRequest(BaseModel):
    """Local subprocess MCP server spec."""

    type: Literal["stdio"] = "stdio"
    command: str = Field(..., min_length=1, max_length=4096)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Values may contain ${SECRET:KEY} placeholders.",
    )


class McpOAuth2AuthRequest(BaseModel):
    """OAuth 2.0 (Authorization Code + PKCE) auth for an HTTP MCP server.

    The actual token exchange happens via the dedicated /oauth/* endpoints;
    this is just the *intent* persisted in the spec. Empty ``scopes`` means
    "let the IdP pick". ``clientId`` is optional and only set when the
    target IdP doesn't support Dynamic Client Registration.
    """

    type: Literal["oauth2"]
    scopes: list[str] = Field(default_factory=list)
    clientId: str | None = Field(
        default=None,
        max_length=512,
        description="Pre-registered client_id; leave null to use Dynamic Client Registration.",
    )


class McpHttpSpecRequest(BaseModel):
    """Remote HTTP MCP server spec.

    Two transports: ``http`` (streamable HTTP, Claude Code convention)
    and ``sse`` (legacy long-poll). Use ``http`` for any new server.
    """

    type: Literal["http", "sse"]
    url: str = Field(..., min_length=1, max_length=4096)
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Values may contain ${SECRET:KEY} placeholders.",
    )
    auth: McpOAuth2AuthRequest | None = Field(
        default=None,
        description=(
            "Optional structured auth. When set, the runtime drives the "
            "OAuth flow and ignores any 'Authorization' header here."
        ),
    )


class McpOAuthCallbackRequest(BaseModel):
    """OAuth callback payload posted by the SPA after the IdP bounces back.

    The SPA owns the redirect-URI route (``/oauth-callback``); it pulls
    ``code`` and ``state`` from the query string and forwards them here.
    """

    code: str = Field(..., min_length=1, max_length=4096)
    state: str | None = Field(default=None, max_length=4096)


McpSpecRequest = Annotated[
    Union[McpStdioSpecRequest, McpHttpSpecRequest],
    Discriminator("type"),
]


class McpServerUpsertRequest(BaseModel):
    """Create or replace an MCP server entry at the chosen scope."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description=(
            "Server name; lowercase letters, digits, underscore, hyphen; "
            "must start with a letter or digit."
        ),
    )
    scope: Literal["user", "workspace"] = Field(
        "workspace",
        description="VSCode-style scope. Workspace overrides User on name collision.",
    )
    spec: McpSpecRequest = Field(..., discriminator="type")


class McpSecretSetRequest(BaseModel):
    """Set or clear an MCP secret value at the chosen scope.

    The plaintext ``value`` is sent up only; the secret store never returns
    it via any GET endpoint. Pass an empty string to delete the key.
    """

    value: str = Field("", description="Plaintext value; empty deletes the key.")
    scope: Literal["user", "workspace"] = Field(
        "workspace",
        description="Where to write the secret. Workspace beats User on lookup.",
    )

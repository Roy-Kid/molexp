"""Pydantic models for :class:`molexp.plugins.gh.GitHubClient` I/O.

Inputs are passed as instances and serialized via ``model_dump(mode="json")``;
outputs are validated via ``model_validate(...)``. All field names are
GitHub's GraphQL camelCase mapped to Python snake_case via
``Field(alias=...)`` + ``populate_by_name=True``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _BaseModel(BaseModel):
    """Project-wide pydantic config: allow camelCase aliases on input,
    snake_case attribute access on output."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


# ── outputs ────────────────────────────────────────────────────────────────


class Issue(_BaseModel):
    """A GitHub issue surfaced through a Project V2 item.

    Attributes:
        node_id: Item ID on the Project (the GraphQL ``ProjectV2Item.id``).
            Used as the subject for status mutations.
        issue_node_id: GraphQL ID of the underlying Issue object.
        number: Repo-scoped issue number (e.g. 42).
        title: Issue title.
        url: Public issue URL.
        state: GitHub-side state (``"OPEN"`` / ``"CLOSED"``).
        body: Issue body markdown.
        repository: ``"owner/repo"`` full name.
        labels: List of label names attached to the issue.
    """

    node_id: str = Field(alias="itemId")
    issue_node_id: str = Field(alias="issueId")
    number: int
    title: str
    url: str
    state: str
    body: str = ""
    repository: str
    labels: list[str] = Field(default_factory=list)


class PullRequest(_BaseModel):
    """One pull request linked to (or proposed for) an issue.

    Attributes:
        number: Repo-scoped PR number.
        url: Public PR URL.
        state: ``"OPEN"`` / ``"CLOSED"`` / ``"MERGED"``.
        is_draft: Draft PR flag.
        mergeable: ``"MERGEABLE"`` / ``"CONFLICTING"`` / ``"UNKNOWN"``.
        head_ref: Source branch name (head ref).
    """

    number: int
    url: str
    state: str
    is_draft: bool = Field(alias="isDraft")
    mergeable: str
    head_ref: str = Field(alias="headRefName")


class CheckRun(_BaseModel):
    """One CI check or status context attached to a commit.

    Attributes:
        name: Check / context name.
        status: ``"QUEUED"`` / ``"IN_PROGRESS"`` / ``"COMPLETED"`` for
            CheckRun; mirrors ``state`` for StatusContext.
        conclusion: ``"SUCCESS"`` / ``"FAILURE"`` / etc., or ``None``
            while still running.
        details_url: Link to the detailed log / status page.
    """

    name: str
    status: str
    conclusion: str | None = None
    details_url: str | None = Field(default=None, alias="detailsUrl")


class Comment(_BaseModel):
    """A comment on an issue or PR.

    Attributes:
        node_id: GraphQL ID of the comment.
        url: Public URL of the comment anchor.
        body: Comment body markdown.
        created_at: ISO 8601 creation timestamp (None if the API did
            not include it).
        author: Author login, or None for anonymous / app accounts that
            don't surface a login.
    """

    node_id: str = Field(alias="id")
    url: str = ""
    body: str = ""
    created_at: datetime | None = Field(default=None, alias="createdAt")
    author: str | None = None


# ── inputs ─────────────────────────────────────────────────────────────────


class CreatePullRequestInput(_BaseModel):
    """Input for :meth:`GitHubClient.create_pull_request`."""

    repository_id: str = Field(alias="repositoryId")
    base_ref_name: str = Field(alias="baseRefName")
    head_ref_name: str = Field(alias="headRefName")
    title: str
    body: str = ""
    draft: bool = False


class AddIssueCommentInput(_BaseModel):
    """Input for :meth:`GitHubClient.add_issue_comment`.

    ``subject_id`` is the GraphQL node id of the issue / PR (or any
    commentable subject). For a Project V2 item, that is the
    ``Issue.id``, **not** the project item id.
    """

    subject_id: str = Field(alias="subjectId")
    body: str


class UpdateProjectFieldInput(_BaseModel):
    """Input for :meth:`GitHubClient.update_project_field`.

    ``value`` is forwarded as-is into the ``ProjectV2FieldValue`` input
    union; for a single-select field set ``value={"singleSelectOptionId": ...}``.
    """

    project_id: str = Field(alias="projectId")
    item_id: str = Field(alias="itemId")
    field_id: str = Field(alias="fieldId")
    value: dict[str, object]

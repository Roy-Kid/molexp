"""GitHub API plugin for molexp.

GitHub Project V2 issue tracking + PR / CI inspection over the GraphQL
API, with a REST fallback for the one thing GraphQL doesn't expose
(workflow run logs).

Public surface:

- :class:`GitHubClient` — async client with typed read / write methods.
- Pydantic models for requests + responses (e.g. :class:`Issue`,
  :class:`PullRequest`, :class:`Comment`, :class:`CreatePullRequestInput`,
  :class:`AddIssueCommentInput`, :class:`UpdateProjectFieldInput`).
- :class:`GitHubGraphQLError` — raised when the API returns ``errors``.

Capability: :data:`molexp.plugins.Capability.GH`. Available iff
``httpx`` is importable; the client itself fails late on auth /
network issues.
"""

from __future__ import annotations

from molexp.plugins.gh.client import GitHubClient, GitHubGraphQLError
from molexp.plugins.gh.models import (
    AddIssueCommentInput,
    CheckRun,
    Comment,
    CreatePullRequestInput,
    Issue,
    PullRequest,
    UpdateProjectFieldInput,
)

__all__ = [
    "AddIssueCommentInput",
    "CheckRun",
    "Comment",
    "CreatePullRequestInput",
    "GitHubClient",
    "GitHubGraphQLError",
    "Issue",
    "PullRequest",
    "UpdateProjectFieldInput",
]

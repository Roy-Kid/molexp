"""``GitHubClient`` — typed async GitHub API client.

GraphQL is the primary surface; REST is used only for the one thing
GraphQL does not expose (Actions workflow run logs). Both share a
single :class:`httpx.AsyncClient` and the same Bearer token.

Tests inject behavior via ``transport=httpx.MockTransport(...)`` so no
real network calls are made.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

from molexp.plugins.gh import queries
from molexp.plugins.gh.models import (
    AddIssueCommentInput,
    CheckRun,
    Comment,
    CreatePullRequestInput,
    Issue,
    PullRequest,
    UpdateProjectFieldInput,
)

DEFAULT_BASE_URL = "https://api.github.com"


class GitHubGraphQLError(RuntimeError):
    """Raised when a GraphQL response carries a non-empty ``errors`` array.

    Attributes:
        errors: The raw ``errors`` list from the GraphQL response.
    """

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        msg = "; ".join(e.get("message", str(e)) for e in errors) or "graphql error"
        super().__init__(msg)


class GitHubClient:
    """Async GitHub client with pydantic-typed methods.

    Use as an async context manager (``async with``) for automatic
    cleanup, or call ``await client.close()`` manually when done.

    Args:
        token: GitHub PAT or app token. Sent as ``Authorization: Bearer
            <token>`` on every request.
        base_url: API root (default ``https://api.github.com``).
        timeout: Per-request timeout, seconds.
        transport: Optional ``httpx.BaseTransport`` (mainly for tests
            via ``httpx.MockTransport``). When set, ``base_url`` and
            ``timeout`` still apply.
    """

    def __init__(
        self,
        token: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            transport=transport,
        )

    async def __aenter__(self) -> GitHubClient:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # ── primitives ────────────────────────────────────────────────────

    async def graphql(
        self,
        query: str,
        variables: BaseModel | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one GraphQL operation; return the ``data`` payload.

        Args:
            query: The GraphQL document.
            variables: Either a pydantic model (``model_dump(mode="json",
                by_alias=True)`` is invoked) or a plain dict, or None.

        Returns:
            The ``data`` key of the GraphQL response.

        Raises:
            GitHubGraphQLError: If the response includes an ``errors`` array.
            httpx.HTTPStatusError: On non-2xx HTTP status.
        """
        if isinstance(variables, BaseModel):
            payload_vars = variables.model_dump(mode="json", by_alias=True)
        elif variables is None:
            payload_vars = {}
        else:
            payload_vars = variables
        response = await self._client.post(
            "/graphql",
            json={"query": query, "variables": payload_vars},
        )
        response.raise_for_status()
        body = response.json()
        if body.get("errors"):
            raise GitHubGraphQLError(body["errors"])
        return body.get("data") or {}

    async def rest(
        self, method: str, path: str, **kwargs: Any
    ) -> httpx.Response:
        """Execute one REST request; return the raw httpx Response.

        Used as the escape hatch for endpoints GraphQL does not cover
        (currently: ``GET /repos/{o}/{r}/actions/runs/{id}/logs``).
        """
        return await self._client.request(method, path, **kwargs)

    # ── high-level reads ──────────────────────────────────────────────

    async def fetch_project_issues(
        self,
        *,
        owner: str,
        project_number: int,
        project_id: str | None = None,
    ) -> list[Issue]:
        """List active items on a GitHub Project V2 board as :class:`Issue`.

        Args:
            owner: Login that owns the project (org or user).
            project_number: Project number (e.g. 1 for the first board).
            project_id: GraphQL node id of the project. When ``None`` it
                is resolved on first use via ``owner`` + ``project_number``
                (extra round-trip). Passing it explicitly avoids that.
        """
        resolved_id = project_id or await self._resolve_project_id(
            owner=owner, project_number=project_number
        )
        data = await self.graphql(
            queries.FETCH_PROJECT_ISSUES,
            {"projectId": resolved_id, "cursor": None},
        )
        items = ((data.get("node") or {}).get("items") or {}).get("nodes") or []
        out: list[Issue] = []
        for item in items:
            content = item.get("content") or {}
            if not content:
                continue
            label_nodes = (content.get("labels") or {}).get("nodes") or []
            out.append(
                Issue(
                    itemId=item["id"],
                    issueId=content["id"],
                    number=content["number"],
                    title=content["title"],
                    url=content["url"],
                    state=content.get("state", "OPEN"),
                    body=content.get("body") or "",
                    repository=(content.get("repository") or {}).get("nameWithOwner", ""),
                    labels=[lbl["name"] for lbl in label_nodes if "name" in lbl],
                )
            )
        return out

    async def _resolve_project_id(self, *, owner: str, project_number: int) -> str:
        """Look up a Project V2 node id from owner + number.

        Tries the org route first; falls back to the user route if the
        org route returns a null node (i.e. ``owner`` is a user, not an
        org). Cached for the lifetime of the client.
        """
        if not hasattr(self, "_project_id_cache"):
            self._project_id_cache: dict[tuple[str, int], str] = {}
        key = (owner, project_number)
        if key in self._project_id_cache:
            return self._project_id_cache[key]
        for root in ("organization", "user"):
            data = await self.graphql(
                f"query Resolve($login: String!, $n: Int!) {{ {root}(login: $login) "
                "{ projectV2(number: $n) { id } } }",
                {"login": owner, "n": project_number},
            )
            project = ((data.get(root) or {}).get("projectV2") or {})
            if project.get("id"):
                self._project_id_cache[key] = project["id"]
                return project["id"]
        raise GitHubGraphQLError(
            [{"message": f"projectV2 not found for {owner}/{project_number}"}]
        )

    async def fetch_pr_for_issue(
        self, *, owner: str, repo: str, issue_number: int
    ) -> PullRequest | None:
        """Find the open PR linked to an issue, or the most recent if none open.

        Returns ``None`` when no PR is linked at all.
        """
        data = await self.graphql(
            queries.FETCH_PR_FOR_ISSUE,
            {"owner": owner, "repo": repo, "number": issue_number},
        )
        nodes = (
            ((data.get("repository") or {}).get("issue") or {})
            .get("closedByPullRequestsReferences", {})
            .get("nodes")
            or []
        )
        if not nodes:
            return None
        # Prefer an OPEN PR; fall back to first listed (GitHub orders most recent first).
        candidates = [n for n in nodes if n.get("state") == "OPEN"] or [nodes[0]]
        return PullRequest.model_validate(candidates[0])

    async def fetch_pr_status_checks(
        self, *, owner: str, repo: str, pr_number: int
    ) -> list[CheckRun]:
        """Return the latest CI checks attached to the PR's HEAD commit."""
        data = await self.graphql(
            queries.FETCH_PR_STATUS_CHECKS,
            {"owner": owner, "repo": repo, "number": pr_number},
        )
        commits = (
            ((data.get("repository") or {}).get("pullRequest") or {})
            .get("commits", {})
            .get("nodes")
            or []
        )
        if not commits:
            return []
        rollup = (commits[0].get("commit") or {}).get("statusCheckRollup") or {}
        contexts = (rollup.get("contexts") or {}).get("nodes") or []
        out: list[CheckRun] = []
        for ctx in contexts:
            if ctx.get("__typename") == "CheckRun":
                out.append(CheckRun.model_validate(ctx))
            else:
                # StatusContext shape → CheckRun shape
                out.append(
                    CheckRun(
                        name=ctx.get("context", "(status)"),
                        status="COMPLETED",
                        conclusion=str(ctx.get("state") or "").upper() or None,
                        detailsUrl=ctx.get("targetUrl"),
                    )
                )
        return out

    async def fetch_issue_comments(
        self, *, owner: str, repo: str, issue_number: int
    ) -> list[Comment]:
        """Return the most recent ~50 comments on an issue."""
        data = await self.graphql(
            queries.FETCH_ISSUE_COMMENTS,
            {"owner": owner, "repo": repo, "number": issue_number},
        )
        nodes = (
            ((data.get("repository") or {}).get("issue") or {})
            .get("comments", {})
            .get("nodes")
            or []
        )
        out: list[Comment] = []
        for n in nodes:
            author = (n.get("author") or {}).get("login")
            out.append(
                Comment(
                    id=n["id"],
                    url=n.get("url", ""),
                    body=n.get("body", ""),
                    createdAt=n.get("createdAt"),
                    author=author,
                )
            )
        return out

    async def fetch_workflow_run_failed_log(
        self, *, owner: str, repo: str, run_id: int
    ) -> str:
        """Fetch failed log text for a workflow run (REST — GraphQL has no equivalent)."""
        response = await self.rest(
            "GET", f"/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
        )
        response.raise_for_status()
        return response.text

    # ── high-level writes ─────────────────────────────────────────────

    async def create_pull_request(self, input: CreatePullRequestInput) -> PullRequest:
        """Open a PR; return the :class:`PullRequest` snapshot."""
        data = await self.graphql(queries.CREATE_PULL_REQUEST, input)
        pr = (data.get("createPullRequest") or {}).get("pullRequest") or {}
        return PullRequest.model_validate(pr)

    async def add_issue_comment(self, input: AddIssueCommentInput) -> Comment:
        """Post a comment; return the created :class:`Comment`."""
        data = await self.graphql(queries.ADD_ISSUE_COMMENT, input)
        node = (
            ((data.get("addComment") or {}).get("commentEdge") or {}).get("node") or {}
        )
        return Comment.model_validate(node)

    async def update_project_field(self, input: UpdateProjectFieldInput) -> dict[str, Any]:
        """Update a Project V2 item's field value; return the GraphQL data."""
        return await self.graphql(queries.UPDATE_PROJECT_FIELD, input)

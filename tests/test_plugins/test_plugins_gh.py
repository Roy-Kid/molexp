"""Tests for ``molexp.plugins.gh.GitHubClient``.

httpx is mocked via ``httpx.MockTransport`` (built-in, zero extra dep).
Each test installs a request handler that asserts the outbound shape
(URL, headers, body) and returns a canned response.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest


def _client_with(handler):
    """Build a ``GitHubClient`` whose underlying httpx uses ``handler``."""
    from molexp.plugins.gh import GitHubClient

    transport = httpx.MockTransport(handler)
    return GitHubClient(token="t-secret", transport=transport)


@pytest.mark.asyncio
async def test_graphql_bearer_and_errors():
    """POST goes to /graphql with Bearer auth; ``errors`` field surfaces."""
    from molexp.plugins.gh import GitHubGraphQLError

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"errors": [{"message": "bad"}], "data": None},
        )

    client = _client_with(handler)
    with pytest.raises(GitHubGraphQLError):
        await client.graphql("query { __typename }", variables={})
    await client.close()

    assert seen["url"].endswith("/graphql")
    assert seen["auth"] == "Bearer t-secret"
    assert seen["body"]["query"] == "query { __typename }"


@pytest.mark.asyncio
async def test_read_methods_typed():
    """Read methods return pydantic models, not raw dicts."""
    from molexp.plugins.gh import Issue, PullRequest

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "Resolve" in body["query"]:
            # project-id resolver: org-route returns the id directly.
            return httpx.Response(
                200,
                json={"data": {"organization": {"projectV2": {"id": "PROJ_1"}}}},
            )
        if "FetchProjectIssues" in body["query"]:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "node": {
                            "items": {
                                "nodes": [
                                    {
                                        "id": "ITEM_1",
                                        "content": {
                                            "id": "I_1",
                                            "number": 42,
                                            "title": "do it",
                                            "url": "https://x/42",
                                            "state": "OPEN",
                                            "body": "task body",
                                            "repository": {"nameWithOwner": "a/b"},
                                            "labels": {"nodes": []},
                                        },
                                    }
                                ],
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            }
                        }
                    }
                },
            )
        # PR-for-issue path
        return httpx.Response(
            200,
            json={
                "data": {
                    "repository": {
                        "issue": {
                            "closedByPullRequestsReferences": {
                                "nodes": [
                                    {
                                        "number": 7,
                                        "url": "https://x/7",
                                        "state": "OPEN",
                                        "isDraft": False,
                                        "mergeable": "MERGEABLE",
                                        "headRefName": "claude/issue-42",
                                    }
                                ]
                            }
                        }
                    }
                }
            },
        )

    client = _client_with(handler)

    issues = await client.fetch_project_issues(owner="acme", project_number=1)
    assert len(issues) == 1
    issue = issues[0]
    assert isinstance(issue, Issue)
    assert issue.number == 42
    assert issue.repository == "a/b"

    pr = await client.fetch_pr_for_issue(owner="acme", repo="b", issue_number=42)
    assert isinstance(pr, PullRequest)
    assert pr.number == 7
    assert pr.head_ref == "claude/issue-42"
    await client.close()


@pytest.mark.asyncio
async def test_write_mutations_typed():
    """create_pull_request / add_issue_comment accept pydantic Input models."""
    from molexp.plugins.gh import (
        AddIssueCommentInput,
        CreatePullRequestInput,
    )

    seen_methods: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_methods.append(body)
        if "createPullRequest" in body["query"]:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "createPullRequest": {
                            "pullRequest": {
                                "number": 9,
                                "url": "https://x/9",
                                "state": "OPEN",
                                "isDraft": False,
                                "mergeable": "UNKNOWN",
                                "headRefName": "claude/issue-1",
                            }
                        }
                    }
                },
            )
        if "addComment" in body["query"]:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "addComment": {
                            "commentEdge": {
                                "node": {"id": "C_1", "url": "https://x/c1", "body": "hi"}
                            }
                        }
                    }
                },
            )
        return httpx.Response(500, json={"errors": [{"message": "unexpected"}]})

    client = _client_with(handler)

    pr = await client.create_pull_request(
        CreatePullRequestInput(
            repository_id="REPO_1",
            base_ref_name="main",
            head_ref_name="claude/issue-1",
            title="t",
            body="b",
        )
    )
    assert pr.number == 9

    comment = await client.add_issue_comment(AddIssueCommentInput(subject_id="I_1", body="hi"))
    assert comment.body == "hi"
    await client.close()

    # variables are pydantic-dumped; mutation operations kept distinct.
    queries_seen = [b["query"] for b in seen_methods]
    assert any("CreatePullRequest" in q for q in queries_seen)
    assert any("AddIssueComment" in q for q in queries_seen)


@pytest.mark.asyncio
async def test_failed_log_via_rest():
    """workflow run failed log is fetched via REST (GraphQL doesn't expose it)."""

    def handler(request: httpx.Request) -> httpx.Response:
        # GraphQL endpoint = /graphql; logs endpoint is /repos/{o}/{r}/actions/runs/{id}/logs
        if request.url.path.endswith("/graphql"):
            return httpx.Response(500, json={"errors": [{"message": "wrong endpoint"}]})
        if "/actions/runs/" in request.url.path:
            assert request.headers["authorization"] == "Bearer t-secret"
            return httpx.Response(200, content=b"job 'build' failed: ENOENT\n")
        return httpx.Response(404)

    client = _client_with(handler)
    log = await client.fetch_workflow_run_failed_log(owner="acme", repo="b", run_id=123)
    assert "ENOENT" in log
    await client.close()


@pytest.mark.asyncio
async def test_async_context_manager():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"x": 1}})

    from molexp.plugins.gh import GitHubClient

    transport = httpx.MockTransport(handler)
    async with GitHubClient(token="t", transport=transport) as client:
        result = await client.graphql("query { x }", variables={})
    assert result == {"x": 1}

"""GraphQL operation strings for :class:`molexp.plugins.gh.GitHubClient`.

One constant per operation. Keeping the queries here (instead of inline
in client methods) makes them grep-able, reviewable, and easy to update
when GitHub's schema deprecates a field.

All queries fetch the minimum field set the molexp consumers actually
use; expand them deliberately when downstream code needs more.
"""

from __future__ import annotations

# ── reads ──────────────────────────────────────────────────────────────────

FETCH_PROJECT_ISSUES = """
query FetchProjectIssues($projectId: ID!, $cursor: String) {
  node(id: $projectId) {
    ... on ProjectV2 {
      items(first: 50, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          content {
            ... on Issue {
              id
              number
              title
              url
              state
              body
              repository { nameWithOwner }
              labels(first: 20) { nodes { name } }
            }
          }
        }
      }
    }
  }
}
"""

FETCH_PR_FOR_ISSUE = """
query FetchPRForIssue($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    issue(number: $number) {
      closedByPullRequestsReferences(first: 5, includeClosedPrs: true) {
        nodes {
          number
          url
          state
          isDraft
          mergeable
          headRefName
        }
      }
    }
  }
}
"""

FETCH_PR_STATUS_CHECKS = """
query FetchPRStatusChecks($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            statusCheckRollup {
              state
              contexts(first: 100) {
                nodes {
                  __typename
                  ... on CheckRun {
                    name
                    status
                    conclusion
                    detailsUrl
                  }
                  ... on StatusContext {
                    context
                    state
                    targetUrl
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

FETCH_ISSUE_COMMENTS = """
query FetchIssueComments($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    issue(number: $number) {
      comments(last: 50) {
        nodes {
          id
          url
          body
          createdAt
          author { login }
        }
      }
    }
  }
}
"""

# ── writes ─────────────────────────────────────────────────────────────────

CREATE_PULL_REQUEST = """
mutation CreatePullRequest(
  $repositoryId: ID!,
  $baseRefName: String!,
  $headRefName: String!,
  $title: String!,
  $body: String,
  $draft: Boolean
) {
  createPullRequest(input: {
    repositoryId: $repositoryId,
    baseRefName: $baseRefName,
    headRefName: $headRefName,
    title: $title,
    body: $body,
    draft: $draft
  }) {
    pullRequest {
      number
      url
      state
      isDraft
      mergeable
      headRefName
    }
  }
}
"""

ADD_ISSUE_COMMENT = """
mutation AddIssueComment($subjectId: ID!, $body: String!) {
  addComment(input: { subjectId: $subjectId, body: $body }) {
    commentEdge {
      node {
        id
        url
        body
      }
    }
  }
}
"""

UPDATE_PROJECT_FIELD = """
mutation UpdateProjectField(
  $projectId: ID!,
  $itemId: ID!,
  $fieldId: ID!,
  $value: ProjectV2FieldValue!
) {
  updateProjectV2ItemFieldValue(input: {
    projectId: $projectId,
    itemId: $itemId,
    fieldId: $fieldId,
    value: $value
  }) {
    projectV2Item { id }
  }
}
"""

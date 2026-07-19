"""Regression: OpenAPI approvals decision body accepts action + pack fields."""

from __future__ import annotations

from molexp.server.app import create_app
from molexp.server.routes.approvals import ApprovalDecisionRequest, PendingApprovalItem


def main() -> int:
    app = create_app()
    schema = app.openapi()
    props = schema["components"]["schemas"]["ApprovalDecisionRequest"]["properties"]
    assert "action" in props and "fieldValues" in props and "granted" in props
    item = schema["components"]["schemas"]["PendingApprovalItem"]["properties"]
    assert "packId" in item and "formDocument" in item
    # model_validator migration
    body = ApprovalDecisionRequest(requestId="x", granted=True)
    assert body.action == "approve"
    _ = PendingApprovalItem(
        taskKind="plan",
        taskId="t",
        runId="r",
        projectId="p",
        experimentId="e",
        requestId="req",
        intent="experiment_spec",
        reason="why",
        requestedAt="2026-07-19T00:00:00Z",
        packId=None,
        formDocument=None,
    )
    print("plan-step-audit-04-api: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

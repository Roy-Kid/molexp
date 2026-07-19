/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ApprovalDecisionRequest } from '../models/ApprovalDecisionRequest';
import type { ApprovalDecisionResponse } from '../models/ApprovalDecisionResponse';
import type { PendingApprovalsResponse } from '../models/PendingApprovalsResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class ApprovalsService {
    /**
     * List Pending Approvals
     * List every pending approval across suspended plan + curate tasks.
     * @returns PendingApprovalsResponse Successful Response
     * @throws ApiError
     */
    public static listPendingApprovalsApiApprovalsGet(): CancelablePromise<PendingApprovalsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/approvals',
        });
    }
    /**
     * Stream Approval Events
     * SSE: one ``changed`` event per suspend/decision — the UI refetch signal.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static streamApprovalEventsApiApprovalsEventsGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/approvals/events',
        });
    }
    /**
     * Decide Approval
     * Record a ReviewDecision-shaped answer and resume/reject the task.
     *
     * Plan tasks delegate to :func:`molexp.services.plan_runtime.decide_plan_review`.
     * Curate tasks keep the binary store path (no ReviewPack yet).
     * @param taskKind
     * @param taskId
     * @param requestBody
     * @returns ApprovalDecisionResponse Successful Response
     * @throws ApiError
     */
    public static decideApprovalApiApprovalsTaskKindTaskIdDecisionsPost(
        taskKind: 'plan' | 'curate',
        taskId: string,
        requestBody: ApprovalDecisionRequest,
    ): CancelablePromise<ApprovalDecisionResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/approvals/{task_kind}/{task_id}/decisions',
            path: {
                'task_kind': taskKind,
                'task_id': taskId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
}

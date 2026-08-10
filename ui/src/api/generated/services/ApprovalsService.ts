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
     *
     * Empty ``items`` is normal — the inbox only fills when a plan/curate task
     * is suspended waiting for an operator decision. Not a 404.
     * @param molexpSession
     * @returns PendingApprovalsResponse Successful Response
     * @throws ApiError
     */
    public static listPendingApprovalsApiApprovalsGet(
        molexpSession?: (string | null),
    ): CancelablePromise<PendingApprovalsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/approvals',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Stream Approval Events
     * SSE: one ``changed`` event per suspend/decision — the UI refetch signal.
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static streamApprovalEventsApiApprovalsEventsGet(
        molexpSession?: (string | null),
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/approvals/events',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
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
     * @param molexpSession
     * @returns ApprovalDecisionResponse Successful Response
     * @throws ApiError
     */
    public static decideApprovalApiApprovalsTaskKindTaskIdDecisionsPost(
        taskKind: 'plan' | 'curate',
        taskId: string,
        requestBody: ApprovalDecisionRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<ApprovalDecisionResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/approvals/{task_kind}/{task_id}/decisions',
            path: {
                'task_kind': taskKind,
                'task_id': taskId,
            },
            cookies: {
                'molexp_session': molexpSession,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
}

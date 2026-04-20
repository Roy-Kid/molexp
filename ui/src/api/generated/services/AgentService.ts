/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AgentSessionListResponse } from '../models/AgentSessionListResponse';
import type { AgentSessionResponse } from '../models/AgentSessionResponse';
import type { ApprovalRespondRequest } from '../models/ApprovalRespondRequest';
import type { GoalCreateRequest } from '../models/GoalCreateRequest';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class AgentService {
    /**
     * List Sessions
     * List all agent sessions.
     * @returns AgentSessionListResponse Successful Response
     * @throws ApiError
     */
    public static listSessionsApiAgentSessionsGet(): CancelablePromise<AgentSessionListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/sessions',
        });
    }
    /**
     * Create Session
     * Start a new agent session.
     * @param requestBody
     * @returns AgentSessionResponse Successful Response
     * @throws ApiError
     */
    public static createSessionApiAgentSessionsPost(
        requestBody: GoalCreateRequest,
    ): CancelablePromise<AgentSessionResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/sessions',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Session
     * Get a specific agent session.
     * @param sessionId
     * @returns AgentSessionResponse Successful Response
     * @throws ApiError
     */
    public static getSessionApiAgentSessionsSessionIdGet(
        sessionId: string,
    ): CancelablePromise<AgentSessionResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/sessions/{session_id}',
            path: {
                'session_id': sessionId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Stream Events
     * Stream agent session events via Server-Sent Events.
     * @param sessionId
     * @returns any Successful Response
     * @throws ApiError
     */
    public static streamEventsApiAgentSessionsSessionIdEventsGet(
        sessionId: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/sessions/{session_id}/events',
            path: {
                'session_id': sessionId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Respond Approval
     * Respond to a human-in-the-loop approval request.
     * @param sessionId
     * @param requestBody
     * @returns any Successful Response
     * @throws ApiError
     */
    public static respondApprovalApiAgentSessionsSessionIdApprovePost(
        sessionId: string,
        requestBody: ApprovalRespondRequest,
    ): CancelablePromise<Record<string, any>> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/sessions/{session_id}/approve',
            path: {
                'session_id': sessionId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
}

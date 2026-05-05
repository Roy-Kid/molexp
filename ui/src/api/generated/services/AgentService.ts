/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AgentSessionListResponse } from '../models/AgentSessionListResponse';
import type { AgentSessionResponse } from '../models/AgentSessionResponse';
import type { AgentSystemPromptResponse } from '../models/AgentSystemPromptResponse';
import type { ApprovalRespondRequest } from '../models/ApprovalRespondRequest';
import type { GoalCreateRequest } from '../models/GoalCreateRequest';
import type { MessageResponse } from '../models/MessageResponse';
import type { PlanDecisionRequest } from '../models/PlanDecisionRequest';
import type { SkillLaunchRequest } from '../models/SkillLaunchRequest';
import type { UserMessageCreateRequest } from '../models/UserMessageCreateRequest';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class AgentService {
    /**
     * List Sessions
     * List every agent session known to the workspace store.
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
     * Start a new agent session via :class:`AgentService`.
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
    /**
     * Respond Plan
     * Resolve a pending plan handoff.
     *
     * On approval the session flips out of PLAN mode and the runner
     * proceeds; on rejection the runner injects a synthetic user message
     * with the feedback (see ``render_reject_feedback``).
     * @param sessionId
     * @param requestBody
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static respondPlanApiAgentSessionsSessionIdPlanDecisionPost(
        sessionId: string,
        requestBody: PlanDecisionRequest,
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/sessions/{session_id}/plan-decision',
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
    /**
     * Post User Message
     * Deliver a chat message to a running session.
     *
     * Either resolves a pending :class:`UserMessageRequested` (when
     * ``request_id`` matches) or queues an unsolicited follow-up onto the
     * session inbox.
     * @param sessionId
     * @param requestBody
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static postUserMessageApiAgentSessionsSessionIdMessagesPost(
        sessionId: string,
        requestBody: UserMessageCreateRequest,
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/sessions/{session_id}/messages',
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
    /**
     * Launch Skill
     * Materialize a saved skill into a Goal and start a new session.
     * @param skillId
     * @param requestBody
     * @returns AgentSessionResponse Successful Response
     * @throws ApiError
     */
    public static launchSkillApiAgentSkillsSkillIdLaunchPost(
        skillId: string,
        requestBody: SkillLaunchRequest,
    ): CancelablePromise<AgentSessionResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/skills/{skill_id}/launch',
            path: {
                'skill_id': skillId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Session System Prompt
     * Return the layered system prompt for a session.
     * @param sessionId
     * @returns AgentSystemPromptResponse Successful Response
     * @throws ApiError
     */
    public static getSessionSystemPromptApiAgentSessionsSessionIdSystemPromptGet(
        sessionId: string,
    ): CancelablePromise<AgentSystemPromptResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/sessions/{session_id}/system-prompt',
            path: {
                'session_id': sessionId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}

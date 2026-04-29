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
import type { SkillLaunchRequest } from '../models/SkillLaunchRequest';
import type { UserMessageCreateRequest } from '../models/UserMessageCreateRequest';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class AgentService {
    /**
     * List Sessions
     * List all agent sessions — in-memory active + on-disk historical.
     *
     * Active sessions take precedence over their on-disk metadata so the
     * list reflects live status/stats. Historical sessions surviving a
     * restart appear with whatever final status was flushed at termination.
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
     * Get a specific agent session — falls back to disk for historicals.
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
    /**
     * Post User Message
     * Deliver a chat message from the user to a running agent session.
     *
     * Either resolves a pending ``UserMessageRequestEvent`` (when
     * ``request_id`` is supplied and matches) or queues an unsolicited
     * follow-up that re-prompts the agent with the user's content.
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
     *
     * The skill's ``instructions`` are threaded through to the runtime
     * automatically; ``plan_mode`` defaults to the skill's
     * ``default_plan_mode`` and may be overridden via the request body.
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
     * Execute Plan
     * Promote a finished plan-mode session into an executing follow-up.
     *
     * Inherits the original goal (description, constraints, success
     * criteria, ``skill_id``); flips ``plan_mode`` off; injects the prior
     * session's final answer (the plan) as ``instructions_override`` so the
     * new session can execute it without re-deriving it.
     * @param sessionId
     * @returns AgentSessionResponse Successful Response
     * @throws ApiError
     */
    public static executePlanApiAgentSessionsSessionIdExecutePlanPost(
        sessionId: string,
    ): CancelablePromise<AgentSessionResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/sessions/{session_id}/execute-plan',
            path: {
                'session_id': sessionId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Session System Prompt
     * Return the layered system prompt the session was started with.
     *
     * Live sessions report what the runtime actually composed (so live
     * edits to workspace instructions don't drift the displayed value);
     * historical (disk-only) sessions are re-composed from current
     * workspace + persisted goal fields, which is a best-effort reflection.
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

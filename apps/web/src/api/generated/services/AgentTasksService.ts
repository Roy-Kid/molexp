/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AgentSystemPromptResponse } from '../models/AgentSystemPromptResponse';
import type { AgentTaskListResponse } from '../models/AgentTaskListResponse';
import type { AgentTaskResponse } from '../models/AgentTaskResponse';
import type { ApprovalDecidedEvent } from '../models/ApprovalDecidedEvent';
import type { ApprovalRequestedEvent } from '../models/ApprovalRequestedEvent';
import type { ArtifactWrittenEvent } from '../models/ArtifactWrittenEvent';
import type { ClarificationRequiredEvent } from '../models/ClarificationRequiredEvent';
import type { CompactionPerformedEvent } from '../models/CompactionPerformedEvent';
import type { ErrorEvent } from '../models/ErrorEvent';
import type { GoalCreateRequest } from '../models/GoalCreateRequest';
import type { LoopCompletedEvent } from '../models/LoopCompletedEvent';
import type { LoopStartedEvent } from '../models/LoopStartedEvent';
import type { LoopSuspendedEvent } from '../models/LoopSuspendedEvent';
import type { MessageResponse } from '../models/MessageResponse';
import type { PlanEmittedEvent } from '../models/PlanEmittedEvent';
import type { PreflightFailedEvent } from '../models/PreflightFailedEvent';
import type { RepairProposedEvent } from '../models/RepairProposedEvent';
import type { StageCompletedEvent } from '../models/StageCompletedEvent';
import type { StageStartedEvent } from '../models/StageStartedEvent';
import type { ThinkingDeltaEvent } from '../models/ThinkingDeltaEvent';
import type { TokenDeltaEvent } from '../models/TokenDeltaEvent';
import type { ToolCallCompletedEvent } from '../models/ToolCallCompletedEvent';
import type { ToolCallStartedEvent } from '../models/ToolCallStartedEvent';
import type { UserMessageCreateRequest } from '../models/UserMessageCreateRequest';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class AgentTasksService {
    /**
     * List Agent Tasks
     * List active and historical agent tasks.
     * @param molexpSession
     * @returns AgentTaskListResponse Successful Response
     * @throws ApiError
     */
    public static listAgentTasksApiAgentTasksGet(
        molexpSession?: (string | null),
    ): CancelablePromise<AgentTaskListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent-tasks',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Agent Task
     * Create a user-facing agent task.
     *
     * The task is the stable conversation container. Each turn is dispatched to
     * either the interactive agent or the nine-stage Planning Agent.
     * @param requestBody
     * @param molexpSession
     * @returns AgentTaskResponse Successful Response
     * @throws ApiError
     */
    public static createAgentTaskApiAgentTasksPost(
        requestBody: GoalCreateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<AgentTaskResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent-tasks',
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
    /**
     * Delete Agent Task Route
     * Cancel any live turn, drop the runtime, and remove task metadata.
     * @param taskId
     * @param molexpSession
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteAgentTaskRouteApiAgentTasksTaskIdDelete(
        taskId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/agent-tasks/{task_id}',
            path: {
                'task_id': taskId,
            },
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Agent Task
     * Get a single agent task by task id.
     * @param taskId
     * @param molexpSession
     * @returns AgentTaskResponse Successful Response
     * @throws ApiError
     */
    public static getAgentTaskApiAgentTasksTaskIdGet(
        taskId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<AgentTaskResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent-tasks/{task_id}',
            path: {
                'task_id': taskId,
            },
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Cancel Agent Task
     * Stop the in-flight turn for this task (idempotent when already idle).
     *
     * Always succeeds when task metadata exists on disk — including zombie
     * ``running`` / ``waiting_approval`` rows after a server restart (no live
     * plan or chat runtime). Previously the chat cancel path 404'd when the
     * session registry was empty, leaving the UI without a Stop recovery.
     * @param taskId
     * @param molexpSession
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static cancelAgentTaskApiAgentTasksTaskIdCancelPost(
        taskId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent-tasks/{task_id}/cancel',
            path: {
                'task_id': taskId,
            },
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Stream Agent Task Events
     * Stream task activity events.
     *
     * Delegates to the existing session event stream until task events are
     * persisted independently.
     * @param taskId
     * @param molexpSession
     * @returns any Server-Sent Events stream; each `data:` frame is one AgentEvent (discriminated on `kind`), terminated by a `done` control frame.
     * @throws ApiError
     */
    public static streamAgentTaskEventsApiAgentTasksTaskIdEventsGet(
        taskId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<(LoopStartedEvent | StageStartedEvent | StageCompletedEvent | ArtifactWrittenEvent | ApprovalRequestedEvent | ApprovalDecidedEvent | PlanEmittedEvent | PreflightFailedEvent | RepairProposedEvent | ClarificationRequiredEvent | CompactionPerformedEvent | LoopCompletedEvent | LoopSuspendedEvent | ErrorEvent | ThinkingDeltaEvent | TokenDeltaEvent | ToolCallStartedEvent | ToolCallCompletedEvent)> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent-tasks/{task_id}/events',
            path: {
                'task_id': taskId,
            },
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Post Agent Task Message
     * Send a follow-up user message on an existing agent task.
     *
     * Continues the *same* runtime session (does not create a new task). A turn
     * that is genuinely live is rejected with 409; disk-only zombie
     * ``running`` / ``waiting_approval`` rows are reaped first so a frontend
     * refresh or server restart cannot trap the task forever.
     * @param taskId
     * @param requestBody
     * @param molexpSession
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static postAgentTaskMessageApiAgentTasksTaskIdMessagesPost(
        taskId: string,
        requestBody: UserMessageCreateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent-tasks/{task_id}/messages',
            path: {
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
    /**
     * Get Agent Task System Prompt
     * Return the composed system prompt for an agent task (inspector).
     *
     * Accepts either a task id or a runtime session id. Live surface replacement
     * for the retired ``GET /api/agent/sessions/{id}/system-prompt`` (which
     * 503s via the legacy agent catch-all).
     * @param taskId
     * @param molexpSession
     * @returns AgentSystemPromptResponse Successful Response
     * @throws ApiError
     */
    public static getAgentTaskSystemPromptApiAgentTasksTaskIdSystemPromptGet(
        taskId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<AgentSystemPromptResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent-tasks/{task_id}/system-prompt',
            path: {
                'task_id': taskId,
            },
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}

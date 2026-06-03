/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AgentTaskListResponse } from '../models/AgentTaskListResponse';
import type { AgentTaskResponse } from '../models/AgentTaskResponse';
import type { ApprovalDecidedEvent } from '../models/ApprovalDecidedEvent';
import type { ApprovalRequestedEvent } from '../models/ApprovalRequestedEvent';
import type { ArtifactWrittenEvent } from '../models/ArtifactWrittenEvent';
import type { ClarificationRequiredEvent } from '../models/ClarificationRequiredEvent';
import type { CompactionPerformedEvent } from '../models/CompactionPerformedEvent';
import type { ErrorEvent } from '../models/ErrorEvent';
import type { GoalCreateRequest } from '../models/GoalCreateRequest';
import type { MessageResponse } from '../models/MessageResponse';
import type { ModeCompletedEvent } from '../models/ModeCompletedEvent';
import type { ModeStartedEvent } from '../models/ModeStartedEvent';
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
     * @returns AgentTaskListResponse Successful Response
     * @throws ApiError
     */
    public static listAgentTasksApiAgentTasksGet(): CancelablePromise<AgentTaskListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent-tasks',
        });
    }
    /**
     * Create Agent Task
     * Create a user-facing agent task.
     *
     * Today this starts exactly one runtime session, but task identity is already
     * separate from the runtime session id.
     * @param requestBody
     * @returns AgentTaskResponse Successful Response
     * @throws ApiError
     */
    public static createAgentTaskApiAgentTasksPost(
        requestBody: GoalCreateRequest,
    ): CancelablePromise<AgentTaskResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent-tasks',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Agent Task
     * Get a single agent task by task id.
     * @param taskId
     * @returns AgentTaskResponse Successful Response
     * @throws ApiError
     */
    public static getAgentTaskApiAgentTasksTaskIdGet(
        taskId: string,
    ): CancelablePromise<AgentTaskResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent-tasks/{task_id}',
            path: {
                'task_id': taskId,
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
     * @returns any Server-Sent Events stream; each `data:` frame is one AgentEvent (discriminated on `kind`), terminated by a `done` control frame.
     * @throws ApiError
     */
    public static streamAgentTaskEventsApiAgentTasksTaskIdEventsGet(
        taskId: string,
    ): CancelablePromise<(ModeStartedEvent | StageStartedEvent | StageCompletedEvent | ArtifactWrittenEvent | ApprovalRequestedEvent | ApprovalDecidedEvent | PlanEmittedEvent | PreflightFailedEvent | RepairProposedEvent | ClarificationRequiredEvent | CompactionPerformedEvent | ModeCompletedEvent | ErrorEvent | ThinkingDeltaEvent | TokenDeltaEvent | ToolCallStartedEvent | ToolCallCompletedEvent)> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent-tasks/{task_id}/events',
            path: {
                'task_id': taskId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Post Agent Task Message
     * Send a user message to a running agent task.
     * @param taskId
     * @param requestBody
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static postAgentTaskMessageApiAgentTasksTaskIdMessagesPost(
        taskId: string,
        requestBody: UserMessageCreateRequest,
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent-tasks/{task_id}/messages',
            path: {
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

/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { SessionEventResponse } from './SessionEventResponse';
import type { SessionStatsResponse } from './SessionStatsResponse';
/**
 * User-facing task wrapper around one current runtime session.
 *
 * ``taskId`` is the product identifier the UI should route on; ``sessionId``
 * is the lower-level runtime handle used to continue the active execution.
 */
export type AgentTaskResponse = {
    activeMode?: AgentTaskResponse.activeMode;
    activePlanTaskId?: (string | null);
    activeTurnId?: (string | null);
    createdAt: string;
    events?: Array<SessionEventResponse>;
    experimentId?: (string | null);
    goal: string;
    planMode?: boolean;
    projectId?: (string | null);
    runId?: (string | null);
    sessionId: string;
    skillId?: (string | null);
    stats?: SessionStatsResponse;
    status: string;
    taskId: string;
    title: string;
    updatedAt?: (string | null);
};
export namespace AgentTaskResponse {
    export enum activeMode {
        CHAT = 'chat',
        PLAN = 'plan',
    }
}


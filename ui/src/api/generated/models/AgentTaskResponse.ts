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
    createdAt: string;
    events?: Array<SessionEventResponse>;
    goal: string;
    planMode?: boolean;
    sessionId: string;
    skillId?: (string | null);
    stats?: SessionStatsResponse;
    status: string;
    taskId: string;
    title: string;
    updatedAt?: (string | null);
};


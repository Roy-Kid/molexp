/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { SessionEventResponse } from './SessionEventResponse';
/**
 * Agent session response.
 */
export type AgentSessionResponse = {
    sessionId: string;
    status: string;
    goalDescription: string;
    createdAt: string;
    events: Array<SessionEventResponse>;
};

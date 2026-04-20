/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { SessionEventResponse } from './SessionEventResponse';
export type AgentSessionResponse = {
    sessionId: string;
    status: string;
    goalDescription: string;
    createdAt: string;
    events?: Array<SessionEventResponse>;
};


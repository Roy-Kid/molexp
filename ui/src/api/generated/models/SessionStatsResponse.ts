/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type SessionStatsResponse = {
    inputTokens?: number;
    outputTokens?: number;
    cacheReadTokens?: number;
    cacheWriteTokens?: number;
    totalTokens?: number;
    requests?: number;
    toolCalls?: number;
    events?: number;
    startedAt?: (string | null);
    completedAt?: (string | null);
    durationSeconds?: (number | null);
};

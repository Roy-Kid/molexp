/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type SessionStatsResponse = {
    cacheReadTokens?: number;
    cacheWriteTokens?: number;
    completedAt?: (string | null);
    durationSeconds?: (number | null);
    events?: number;
    inputTokens?: number;
    outputTokens?: number;
    requests?: number;
    startedAt?: (string | null);
    toolCalls?: number;
    totalTokens?: number;
};


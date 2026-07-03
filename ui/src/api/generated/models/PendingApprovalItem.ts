/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One pending request awaiting an operator decision.
 */
export type PendingApprovalItem = {
    experimentId: string;
    intent: string;
    metadata?: Record<string, any>;
    projectId: string;
    reason: string;
    requestId: string;
    requestedAt: string;
    runId: string;
    taskId: string;
    taskKind: PendingApprovalItem.taskKind;
};
export namespace PendingApprovalItem {
    export enum taskKind {
        PLAN = 'plan',
        CURATE = 'curate',
    }
}


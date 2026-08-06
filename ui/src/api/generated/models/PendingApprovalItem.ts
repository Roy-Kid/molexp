/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One pending request awaiting an operator decision.
 *
 * ``taskId`` is the **single public handle** shared with the Agents hub
 * (agent-task id / plan conversation id). Decide routes resolve the live
 * runtime task by this same id — never a second parallel plan-* registry
 * id that the UI cannot navigate to.
 */
export type PendingApprovalItem = {
    experimentId: string;
    formDocument?: (Record<string, any> | null);
    intent: string;
    metadata?: Record<string, any>;
    packId?: (string | null);
    preview?: string;
    projectId: string;
    reason: string;
    requestId: string;
    requestedAt: string;
    runId: string;
    scope?: string;
    targetAgentId?: (string | null);
    taskId: string;
    taskKind: PendingApprovalItem.taskKind;
};
export namespace PendingApprovalItem {
    export enum taskKind {
        PLAN = 'plan',
        CURATE = 'curate',
    }
}


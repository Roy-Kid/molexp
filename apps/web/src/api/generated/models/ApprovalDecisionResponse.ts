/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Post-decision task summary.
 */
export type ApprovalDecisionResponse = {
    status: string;
    taskId: string;
    taskKind: ApprovalDecisionResponse.taskKind;
};
export namespace ApprovalDecisionResponse {
    export enum taskKind {
        PLAN = 'plan',
        CURATE = 'curate',
    }
}


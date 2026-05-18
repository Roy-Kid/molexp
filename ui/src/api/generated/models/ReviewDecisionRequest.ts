/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Approve or reject a persisted review item.
 */
export type ReviewDecisionRequest = {
    /**
     * Optional human resolution comment.
     */
    comment?: string;
    /**
     * Optional edited plan markdown when approving a plan review.
     */
    edited_plan?: (string | null);
    /**
     * Optional edited workflow IR when approving a plan review.
     */
    edited_workflow_ir?: (Record<string, any> | null);
};

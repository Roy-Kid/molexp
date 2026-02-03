/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Request for execution plan.
 */
export type ExecutionPlanRequest = {
    /**
     * Workflow definition as JSON string
     */
    workflow_json: string;
    /**
     * Target node IDs (defaults to workflow targets)
     */
    targets?: (Array<string> | null);
};


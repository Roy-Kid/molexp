/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Request to create a new execution in a specific context.
 */
export type ExecutionCreateRequest = {
    /**
     * Serialized workflow graph
     */
    workflow_json: Record<string, any>;
    /**
     * Target Project ID
     */
    project_id: string;
    /**
     * Target Experiment ID
     */
    experiment_id: string;
    /**
     * Execution parameters
     */
    parameters?: Record<string, any>;
};


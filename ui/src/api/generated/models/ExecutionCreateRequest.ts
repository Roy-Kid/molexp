/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type ExecutionCreateRequest = {
    /**
     * Serialized workflow graph
     */
    workflow_json: Record<string, any>;
    /**
     * Target project ID
     */
    project_id: string;
    /**
     * Target experiment ID
     */
    experiment_id: string;
    parameters?: Record<string, any>;
};


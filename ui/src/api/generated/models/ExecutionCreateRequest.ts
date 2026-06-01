/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type ExecutionCreateRequest = {
    /**
     * Target project ID
     */
    project_id: string;
    /**
     * Target experiment ID
     */
    experiment_id: string;
    parameters?: Record<string, any>;
    /**
     * Optional workflow IR (matches schema/workflow.json). When provided and the experiment has no workflow bound yet, the server binds it and persists the IR to disk. Subsequent calls reuse the on-disk binding.
     */
    workflow_json?: (Record<string, any> | null);
};


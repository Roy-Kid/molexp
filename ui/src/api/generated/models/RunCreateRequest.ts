/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Request to create a run.
 */
export type RunCreateRequest = {
    /**
     * Run parameters
     */
    parameters?: Record<string, any>;
    /**
     * Workflow file path
     */
    workflow_file: string;
    /**
     * Git commit hash
     */
    git_commit?: (string | null);
};


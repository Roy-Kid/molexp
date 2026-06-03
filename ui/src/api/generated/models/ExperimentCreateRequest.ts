/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type ExperimentCreateRequest = {
    /**
     * Compute target name new runs should default to (must exist)
     */
    defaultTarget?: (string | null);
    /**
     * Experiment description
     */
    description?: string;
    /**
     * Human-readable experiment name
     */
    name: string;
    /**
     * Parameter space definition
     */
    parameter_space?: Record<string, any>;
    /**
     * Path to workflow file
     */
    workflow_source?: (string | null);
};


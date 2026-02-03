/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Request to create an experiment.
 */
export type ExperimentCreateRequest = {
    /**
     * Unique experiment identifier (slug)
     */
    id: string;
    /**
     * Human-readable experiment name
     */
    name: string;
    /**
     * Path to workflow file
     */
    workflow_source: string;
    /**
     * Experiment description
     */
    description?: string;
    /**
     * Parameter space definition
     */
    parameter_space?: Record<string, any>;
};


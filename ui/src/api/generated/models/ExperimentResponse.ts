/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { RunSummary } from './RunSummary';
/**
 * Experiment response model.
 */
export type ExperimentResponse = {
    /**
     * ISO 8601 creation timestamp
     */
    created: string;
    id: string;
    experimentId: string;
    projectId: string;
    name: string;
    description?: string;
    /**
     * Workflow source path
     */
    workflow: string;
    workflowType?: (string | null);
    gitCommit?: (string | null);
    parameterSpace?: Record<string, any>;
    defaultInputs?: Array<Record<string, any>>;
    runCount?: (number | null);
    runs?: Array<RunSummary>;
};


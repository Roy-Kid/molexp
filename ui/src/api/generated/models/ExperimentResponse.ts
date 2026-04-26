/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { RunSummary } from './RunSummary';
export type ExperimentResponse = {
    id: string;
    projectId: string;
    name: string;
    description?: string;
    workflow?: (string | null);
    workflowType?: (string | null);
    gitCommit?: (string | null);
    parameterSpace?: Record<string, any>;
    created: string;
    runCount?: (number | null);
    runs?: Array<RunSummary>;
};

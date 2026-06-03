/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { RunSummary } from './RunSummary';
export type ExperimentResponse = {
    created: string;
    defaultTarget?: (string | null);
    description?: string;
    gitCommit?: (string | null);
    id: string;
    name: string;
    parameterSpace?: Record<string, any>;
    projectId: string;
    runCount?: (number | null);
    runs?: Array<RunSummary>;
    workflow?: (string | null);
    workflowType?: (string | null);
};


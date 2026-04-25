/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One run row in the experiment comparison matrix.
 */
export type ComparisonRunRow = {
    runId: string;
    status: string;
    parameters?: Record<string, any>;
    metrics?: Record<string, any>;
    durationSec?: (number | null);
    created: string;
    finished?: (string | null);
    error?: (Record<string, string> | null);
};


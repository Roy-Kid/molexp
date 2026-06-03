/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One run row in the experiment comparison matrix.
 */
export type ComparisonRunRow = {
    created: string;
    durationSec?: (number | null);
    error?: (Record<string, string> | null);
    finished?: (string | null);
    metrics?: Record<string, any>;
    parameters?: Record<string, any>;
    runId: string;
    status: string;
};


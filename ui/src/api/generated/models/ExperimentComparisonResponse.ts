/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ComparisonRunRow } from './ComparisonRunRow';
/**
 * Sweep matrix: parameter columns x run rows + metric columns.
 */
export type ExperimentComparisonResponse = {
    experimentId: string;
    projectId: string;
    paramKeys?: Array<string>;
    metricKeys?: Array<string>;
    runs?: Array<ComparisonRunRow>;
};


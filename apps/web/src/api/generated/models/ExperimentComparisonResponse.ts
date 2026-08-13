/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ComparisonRunRow } from './ComparisonRunRow';
/**
 * Comparison matrix: parameter columns x run rows + metric columns.
 */
export type ExperimentComparisonResponse = {
    experimentId: string;
    metricKeys?: Array<string>;
    paramKeys?: Array<string>;
    projectId: string;
    runs?: Array<ComparisonRunRow>;
};


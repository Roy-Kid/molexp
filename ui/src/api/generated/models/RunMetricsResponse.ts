/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { MetricSeriesResponse } from './MetricSeriesResponse';
/**
 * Run-local metrics query response.
 */
export type RunMetricsResponse = {
    nextLine?: number;
    records?: Array<Record<string, any>>;
    series?: Array<MetricSeriesResponse>;
    parseErrors?: number;
};


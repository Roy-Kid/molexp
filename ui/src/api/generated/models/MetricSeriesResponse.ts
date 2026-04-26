/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Summary for one metric series in a run-local metrics query.
 */
export type MetricSeriesResponse = {
    key: string;
    type: string;
    count: number;
    latestStep?: (number | null);
    latestTimestamp?: (string | null);
    latestValue?: null;
};

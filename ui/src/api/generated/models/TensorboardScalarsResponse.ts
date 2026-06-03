/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { TensorboardScalarSeries } from './TensorboardScalarSeries';
/**
 * Parsed scalars across every tfevents logdir found under a run.
 */
export type TensorboardScalarsResponse = {
    logdirs?: Array<string>;
    runDir: string;
    runId: string;
    series?: Array<TensorboardScalarSeries>;
};


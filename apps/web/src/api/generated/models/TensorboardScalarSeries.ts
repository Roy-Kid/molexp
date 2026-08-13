/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { TensorboardScalarPoint } from './TensorboardScalarPoint';
/**
 * All scalar samples for a single tag inside one logdir.
 */
export type TensorboardScalarSeries = {
    logdir: string;
    points?: Array<TensorboardScalarPoint>;
    tag: string;
};


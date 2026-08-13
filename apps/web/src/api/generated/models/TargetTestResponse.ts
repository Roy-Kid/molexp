/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { TargetTestCheck } from './TargetTestCheck';
/**
 * Response for ``POST /api/targets/{name}/test``.
 */
export type TargetTestResponse = {
    checks: Array<TargetTestCheck>;
    error?: (string | null);
    name: string;
    ok: boolean;
};


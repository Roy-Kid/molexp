/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { TargetTestCheck } from './TargetTestCheck';
/**
 * Response for ``POST /api/targets/{name}/test``.
 */
export type TargetTestResponse = {
    name: string;
    ok: boolean;
    checks: Array<TargetTestCheck>;
    error?: (string | null);
};


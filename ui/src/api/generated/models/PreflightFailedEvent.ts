/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Emitted when a plan's preflight checks fail.
 *
 * ``failed_checks`` holds the names of the failing ``PlanCheck``\ s.
 */
export type PreflightFailedEvent = {
    failed_checks: Array<string>;
    kind?: string;
    timestamp?: string;
};


/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Emitted when a loop produces a plan graph.
 *
 * Carries a lightweight reference (``plan_id`` / ``step_count``)
 * rather than the whole ``PlanGraph`` so the event stream stays cheap
 * to serialize; consumers re-load the graph from disk by ``plan_id``.
 */
export type PlanEmittedEvent = {
    kind?: string;
    plan_id: string;
    step_count?: number;
    timestamp?: string;
};


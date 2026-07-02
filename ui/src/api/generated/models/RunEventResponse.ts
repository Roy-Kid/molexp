/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { JSONValue } from './JSONValue';
/**
 * One workspace-timeline event related to a run (read side of the spine).
 */
export type RunEventResponse = {
    actor: string;
    created_at: string;
    id: string;
    payload: Record<string, JSONValue>;
    refs: Array<string>;
    seq: number;
    type: string;
};


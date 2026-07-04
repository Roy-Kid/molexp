/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { JSONValue } from './JSONValue';
/**
 * One workspace-timeline event (read side of the event spine).
 *
 * The ONE wire shape for spine reads — the per-run route
 * (``GET /runs/{run_id}/events``) aliases this model, so the two surfaces
 * can never drift (vision-loop-12).
 */
export type WorkspaceEventResponse = {
    actor: string;
    created_at: string;
    id: string;
    payload: Record<string, JSONValue>;
    refs: Array<string>;
    seq: number;
    type: string;
};


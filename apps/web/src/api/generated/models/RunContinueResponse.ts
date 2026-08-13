/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Result of continuing a run in place — ``resume`` or ``rerun``.
 *
 * Both verbs act on the same ``runId`` (no clone, no new run). ``executionId``
 * is the execution the action targeted: the reopened one for ``resume``, the
 * freshly-derived ``exec-{run_id}-N`` for ``rerun``.
 */
export type RunContinueResponse = {
    executionId: string;
    experimentId: string;
    projectId: string;
    runId: string;
    status: string;
};


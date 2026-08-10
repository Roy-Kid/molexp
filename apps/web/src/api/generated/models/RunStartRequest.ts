/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Body for the ``run`` (start) verb on a pending run.
 *
 * A pending run is target-less (the create+dispatch contract dispatches a
 * targeted run immediately), so Start must supply the compute target to
 * execute on. ``None`` falls back to any target already recorded on the run;
 * when neither resolves, the route 422s (target-less runs are started with
 * ``molexp run`` on the host).
 */
export type RunStartRequest = {
    /**
     * Run inputs to apply before starting (the workflow's root inputs). None keeps the run's existing parameters; a pending run has not been hashed yet, so editing inputs here is safe.
     */
    parameters?: (Record<string, any> | null);
    /**
     * Compute target name to start the run on (must exist in the workspace registry)
     */
    target?: (string | null);
};


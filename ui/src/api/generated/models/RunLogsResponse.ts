/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Per-execution stdout/stderr for a run.
 *
 * `execution_id` is the attempt these logs belong to; the server
 * defaults to the most recent attempt when no specific execution is
 * requested.  Each value is the full content of
 * `executions/<execution_id>/{stdout,stderr}.log` (or `null` if the
 * file is absent — e.g. local executions skip stdout capture).
 */
export type RunLogsResponse = {
    execution_id?: (string | null);
    stdout?: (string | null);
    stderr?: (string | null);
};


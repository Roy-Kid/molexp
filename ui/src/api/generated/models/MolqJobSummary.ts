/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type MolqJobSummary = {
    target: string;
    jobId: string;
    schedulerJobId?: (string | null);
    clusterName?: (string | null);
    scheduler?: (string | null);
    name?: (string | null);
    state: string;
    submittedAt?: (string | null);
    startedAt?: (string | null);
    finishedAt?: (string | null);
    exitCode?: (number | null);
    durationSeconds?: (number | null);
    cwd?: (string | null);
};

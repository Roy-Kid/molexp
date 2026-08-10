/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type MolqJobSummary = {
    clusterName?: (string | null);
    cwd?: (string | null);
    durationSeconds?: (number | null);
    exitCode?: (number | null);
    finishedAt?: (string | null);
    jobId: string;
    name?: (string | null);
    scheduler?: (string | null);
    schedulerJobId?: (string | null);
    startedAt?: (string | null);
    state: string;
    submittedAt?: (string | null);
    target: string;
};


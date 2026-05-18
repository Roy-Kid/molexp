/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { MolqJobTransition } from './MolqJobTransition';
export type MolqJobDetailResponse = {
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
    failureReason?: (string | null);
    metadata?: Record<string, string>;
    commandDisplay?: (string | null);
    transitions?: Array<MolqJobTransition>;
    upstreamTotal?: number;
    upstreamSatisfied?: number;
    downstreamTotal?: number;
};

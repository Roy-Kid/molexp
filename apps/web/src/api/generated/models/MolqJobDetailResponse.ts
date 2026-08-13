/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { MolqJobTransition } from './MolqJobTransition';
export type MolqJobDetailResponse = {
    clusterName?: (string | null);
    commandDisplay?: (string | null);
    cwd?: (string | null);
    downstreamTotal?: number;
    durationSeconds?: (number | null);
    exitCode?: (number | null);
    failureReason?: (string | null);
    finishedAt?: (string | null);
    jobId: string;
    metadata?: Record<string, string>;
    name?: (string | null);
    scheduler?: (string | null);
    schedulerJobId?: (string | null);
    startedAt?: (string | null);
    state: string;
    submittedAt?: (string | null);
    target: string;
    transitions?: Array<MolqJobTransition>;
    upstreamSatisfied?: number;
    upstreamTotal?: number;
};


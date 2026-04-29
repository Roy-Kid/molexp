/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { MolqJobSummary } from './MolqJobSummary';
import type { MolqQueueStats } from './MolqQueueStats';
export type MolqJobsResponse = {
    jobs: Array<MolqJobSummary>;
    stats: MolqQueueStats;
    total: number;
};


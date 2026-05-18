/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type MolqTargetSummary = {
    name: string;
    scheduler: string;
    clusterName?: (string | null);
    jobsDir?: (string | null);
    healthy: boolean;
    healthReason?: (string | null);
    activeJobs: number;
};

/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Wire form for a :class:`ComputeTarget`.
 */
export type TargetResponse = {
    defaultResources?: Record<string, any>;
    defaultScheduling?: Record<string, any>;
    host?: (string | null);
    identityFile?: (string | null);
    isRemote: boolean;
    name: string;
    port?: (number | null);
    scheduler: TargetResponse.scheduler;
    scratchRoot: string;
    sshOpts?: Array<string>;
};
export namespace TargetResponse {
    export enum scheduler {
        LOCAL = 'local',
        SLURM = 'slurm',
        PBS = 'pbs',
        LSF = 'lsf',
    }
}


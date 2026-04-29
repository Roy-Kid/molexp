/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Wire form for a :class:`ComputeTarget`.
 */
export type TargetResponse = {
    name: string;
    scratchRoot: string;
    scheduler: TargetResponse.scheduler;
    host?: (string | null);
    port?: (number | null);
    identityFile?: (string | null);
    sshOpts?: Array<string>;
    isRemote: boolean;
    defaultResources?: Record<string, any>;
    defaultScheduling?: Record<string, any>;
};
export namespace TargetResponse {
    export enum scheduler {
        SHELL = 'shell',
        SLURM = 'slurm',
        PBS = 'pbs',
        LSF = 'lsf',
    }
}


/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Payload for ``POST /api/targets``.
 */
export type TargetCreateRequest = {
    /**
     * Unique target name within the workspace
     */
    name: string;
    /**
     * Absolute scratch root on the target's filesystem
     */
    scratchRoot: string;
    /**
     * Dispatch axis
     */
    scheduler?: TargetCreateRequest.scheduler;
    /**
     * user@host for SSH; omit for local
     */
    host?: (string | null);
    /**
     * SSH port
     */
    port?: (number | null);
    /**
     * Path to SSH identity file
     */
    identityFile?: (string | null);
    /**
     * Extra ssh argv tokens
     */
    sshOpts?: Array<string>;
};
export namespace TargetCreateRequest {
    /**
     * Dispatch axis
     */
    export enum scheduler {
        SHELL = 'shell',
        SLURM = 'slurm',
        PBS = 'pbs',
        LSF = 'lsf',
    }
}


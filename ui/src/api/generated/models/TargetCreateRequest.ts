/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Payload for ``POST /api/targets``.
 */
export type TargetCreateRequest = {
    /**
     * user@host for SSH; omit for local
     */
    host?: (string | null);
    /**
     * Path to SSH identity file
     */
    identityFile?: (string | null);
    /**
     * Unique target name within the workspace
     */
    name: string;
    /**
     * SSH port
     */
    port?: (number | null);
    /**
     * Dispatch axis
     */
    scheduler?: TargetCreateRequest.scheduler;
    /**
     * Absolute scratch root on the target's filesystem
     */
    scratchRoot: string;
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
        LOCAL = 'local',
        SLURM = 'slurm',
        PBS = 'pbs',
        LSF = 'lsf',
    }
}


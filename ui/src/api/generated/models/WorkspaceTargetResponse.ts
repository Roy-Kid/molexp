/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Wire form for a :class:`WorkspaceTarget`.
 */
export type WorkspaceTargetResponse = {
    cache_dir?: (string | null);
    cache_ttl_seconds?: number;
    host: string;
    identity_file?: (string | null);
    name: string;
    port?: (number | null);
    root_path: string;
    ssh_opts?: Array<string>;
};


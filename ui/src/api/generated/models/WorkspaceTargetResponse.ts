/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Wire form for a :class:`WorkspaceTarget`.
 */
export type WorkspaceTargetResponse = {
    name: string;
    host: string;
    root_path: string;
    port?: (number | null);
    identity_file?: (string | null);
    ssh_opts?: Array<string>;
    cache_dir?: (string | null);
    cache_ttl_seconds?: number;
};


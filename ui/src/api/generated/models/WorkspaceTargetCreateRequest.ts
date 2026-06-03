/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Payload for ``POST /api/workspace/targets``.
 */
export type WorkspaceTargetCreateRequest = {
    /**
     * Override local mirror root (defaults to ~/.molexp/remote_cache/<name>)
     */
    cache_dir?: (string | null);
    /**
     * Freshness window for cached file entries; 0 disables the fast path
     */
    cache_ttl_seconds?: number;
    /**
     * ``user@host`` or bare hostname for SSH
     */
    host: string;
    /**
     * Absolute path to an SSH identity file
     */
    identity_file?: (string | null);
    /**
     * Unique slug-shaped identifier
     */
    name: string;
    /**
     * SSH port
     */
    port?: (number | null);
    /**
     * Absolute POSIX path on the remote host
     */
    root_path: string;
    /**
     * Extra ``ssh`` argv tokens
     */
    ssh_opts?: Array<string>;
};


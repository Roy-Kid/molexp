/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One workspace the server is hosting.
 */
export type ServedWorkspaceResponse = {
    /**
     * True for the workspace the flat routes / active tree address
     */
    active?: boolean;
    /**
     * True for an SSH-backed remote workspace
     */
    isRemote: boolean;
    /**
     * Stable switch handle, unique per server process
     */
    key: string;
    /**
     * Human-facing label (path or user@host:/path)
     */
    label: string;
    /**
     * Absolute local root, null when remote
     */
    path?: (string | null);
    /**
     * True when a remote workspace's transport could not be reached
     */
    unreachable?: boolean;
};


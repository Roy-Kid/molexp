/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Local-workspace branch of ``POST /api/workspace/open``.
 */
export type WorkspaceOpenLocalRequest = {
    /**
     * Discriminator
     */
    kind?: string;
    /**
     * Absolute path to the workspace
     */
    path: string;
    /**
     * Create if missing
     */
    create_if_missing?: boolean;
};


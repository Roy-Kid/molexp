/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Remote-workspace branch of ``POST /api/workspace/open``.
 *
 * The descriptor must already be registered via
 * ``POST /api/workspace/targets``; auto-creation of remote roots is
 * out of scope for this endpoint (returns 404 if the descriptor or
 * its remote ``root_path`` is missing).
 */
export type WorkspaceOpenRemoteRequest = {
    /**
     * Discriminator
     */
    kind: string;
    /**
     * Registered workspace-target name
     */
    name: string;
};


/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type HealthResponse = {
    /**
     * True when the server process has auth enabled (UI should gate on login).
     */
    auth_required?: boolean;
    capabilities?: Record<string, boolean>;
    status: string;
    workspace_available: boolean;
};


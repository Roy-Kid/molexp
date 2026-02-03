/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Project response model.
 */
export type ProjectResponse = {
    /**
     * ISO 8601 creation timestamp
     */
    created: string;
    id: string;
    projectId: string;
    name: string;
    description?: string;
    owner?: string;
    tags?: Array<string>;
    config?: Record<string, any>;
    experimentCount?: (number | null);
};


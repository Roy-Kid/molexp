/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Request to create a project.
 */
export type ProjectCreateRequest = {
    /**
     * Unique project identifier (slug)
     */
    id: string;
    /**
     * Human-readable project name
     */
    name: string;
    /**
     * Project description
     */
    description?: string;
    /**
     * Project owner
     */
    owner?: string;
    /**
     * Project tags
     */
    tags?: Array<string>;
};


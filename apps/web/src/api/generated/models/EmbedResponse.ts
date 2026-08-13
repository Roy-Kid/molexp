/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Echo of the written embed edge.
 */
export type EmbedResponse = {
    role: EmbedResponse.role;
    srcPath: string;
    target: string;
};
export namespace EmbedResponse {
    export enum role {
        DERIVED_FROM = 'derived_from',
        CITES = 'cites',
        SUPERSEDES = 'supersedes',
        RECORDS = 'records',
        REFERENCES = 'references',
    }
}


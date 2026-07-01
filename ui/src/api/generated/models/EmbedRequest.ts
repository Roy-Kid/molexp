/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Embed a live workspace entity into a document as one typed provenance edge.
 */
export type EmbedRequest = {
    role?: ('derived_from' | 'cites' | 'supersedes' | 'records' | 'references' | null);
    target: string;
    target_kind: EmbedRequest.target_kind;
    text?: (string | null);
};
export namespace EmbedRequest {
    export enum target_kind {
        RUN = 'run',
        ASSET = 'asset',
        EXPERIMENT = 'experiment',
        REFERENCE = 'reference',
    }
}


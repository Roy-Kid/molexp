/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { KnowledgeListResponse } from '../models/KnowledgeListResponse';
import type { NoteDetailResponse } from '../models/NoteDetailResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class KnowledgeService {
    /**
     * List Knowledge
     * List every Note + ReferenceConcept in the active workspace's bundle.
     * @returns KnowledgeListResponse Successful Response
     * @throws ApiError
     */
    public static listKnowledgeApiKnowledgeGet(): CancelablePromise<KnowledgeListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/knowledge',
        });
    }
    /**
     * Get Note
     * Return one note's full body (its ``index.md``) + its outgoing links.
     * @param path The note Concept's bundle-relative path (its identity).
     * @returns NoteDetailResponse Successful Response
     * @throws ApiError
     */
    public static getNoteApiKnowledgeNoteGet(
        path: string,
    ): CancelablePromise<NoteDetailResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/knowledge/note',
            query: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}

/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { BacklinksResponse } from '../models/BacklinksResponse';
import type { DocBodyUpdate } from '../models/DocBodyUpdate';
import type { DocCreateRequest } from '../models/DocCreateRequest';
import type { DocMoveRequest } from '../models/DocMoveRequest';
import type { EmbedRequest } from '../models/EmbedRequest';
import type { EmbedResponse } from '../models/EmbedResponse';
import type { KnowledgeListResponse } from '../models/KnowledgeListResponse';
import type { MessageResponse } from '../models/MessageResponse';
import type { NoteDetailResponse } from '../models/NoteDetailResponse';
import type { NoteSummary } from '../models/NoteSummary';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class KnowledgeService {
    /**
     * List Knowledge
     * List every Note + ReferenceConcept in the active workspace's bundle.
     *
     * Optional ``tag`` / ``status`` query params AND-narrow the note list (both
     * read from the 05 :class:`~molexp.workspace.note_meta.NoteMeta` fields).
     * @param tag Only notes carrying this tag.
     * @param status Only notes with this lifecycle status.
     * @returns KnowledgeListResponse Successful Response
     * @throws ApiError
     */
    public static listKnowledgeApiKnowledgeGet(
        tag?: (string | null),
        status?: (string | null),
    ): CancelablePromise<KnowledgeListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/knowledge',
            query: {
                'tag': tag,
                'status': status,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Backlinks
     * Return every Concept linking at *path* — delegates to ``Bundle.backlinks``.
     * @param path The target Concept's bundle-relative path (its identity).
     * @returns BacklinksResponse Successful Response
     * @throws ApiError
     */
    public static getBacklinksApiKnowledgeBacklinksGet(
        path: string,
    ): CancelablePromise<BacklinksResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/knowledge/backlinks',
            query: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Delete Doc
     * Delete a note (its directory subtree) — delegates to ``Bundle.delete_note``.
     * @param path The note Concept's bundle-relative path (its identity).
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteDocApiKnowledgeDocDelete(
        path: string,
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/knowledge/doc',
            query: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Move Doc
     * Rename and/or reparent a note — delegates to ``Bundle.rename_note`` / ``move_note``.
     * @param path The note Concept's bundle-relative path (its identity).
     * @param requestBody
     * @returns NoteSummary Successful Response
     * @throws ApiError
     */
    public static moveDocApiKnowledgeDocPatch(
        path: string,
        requestBody: DocMoveRequest,
    ): CancelablePromise<NoteSummary> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/knowledge/doc',
            query: {
                'path': path,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Doc
     * Create a :class:`Note` document — delegates to ``Bundle.create_note``.
     * @param requestBody
     * @returns NoteSummary Successful Response
     * @throws ApiError
     */
    public static createDocApiKnowledgeDocPost(
        requestBody: DocCreateRequest,
    ): CancelablePromise<NoteSummary> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/knowledge/doc',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Edit Doc
     * Rewrite a note's body (its ``index.md``) — delegates to ``Note.set_body``.
     * @param path The note Concept's bundle-relative path (its identity).
     * @param requestBody
     * @returns NoteDetailResponse Successful Response
     * @throws ApiError
     */
    public static editDocApiKnowledgeDocPut(
        path: string,
        requestBody: DocBodyUpdate,
    ): CancelablePromise<NoteDetailResponse> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/knowledge/doc',
            query: {
                'path': path,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Embed Doc
     * Embed a live entity into a document — delegates to ``Bundle.embed``.
     *
     * Resolves the source ``Note`` (404 on miss / non-note) and the target entity
     * (``run`` / ``experiment`` / ``asset`` / ``reference``; 404 on miss), then
     * writes ONE typed provenance edge via ``Bundle.embed`` — the same verb the CLI
     * uses, so the edge-writing logic is never re-built at the HTTP boundary.
     * @param path The source note Concept's bundle-relative path.
     * @param requestBody
     * @returns EmbedResponse Successful Response
     * @throws ApiError
     */
    public static embedDocApiKnowledgeDocEmbedPost(
        path: string,
        requestBody: EmbedRequest,
    ): CancelablePromise<EmbedResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/knowledge/doc/embed',
            query: {
                'path': path,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Export Doc
     * Export a note as portable Markdown — delegates to ``Bundle.export_markdown``.
     * @param path The note Concept's bundle-relative path (its identity).
     * @returns any Successful Response
     * @throws ApiError
     */
    public static exportDocApiKnowledgeDocExportGet(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/knowledge/doc/export',
            query: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Note
     * Return one note's full body (its ``index.md``) + its outgoing links + cards.
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

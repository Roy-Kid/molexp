/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { BacklinksResponse } from '../models/BacklinksResponse';
import type { DocBodyUpdate } from '../models/DocBodyUpdate';
import type { DocCreateRequest } from '../models/DocCreateRequest';
import type { DocMetaUpdate } from '../models/DocMetaUpdate';
import type { DocMoveRequest } from '../models/DocMoveRequest';
import type { EmbedRequest } from '../models/EmbedRequest';
import type { EmbedResponse } from '../models/EmbedResponse';
import type { EntityBacklinksResponse } from '../models/EntityBacklinksResponse';
import type { KnowledgeListResponse } from '../models/KnowledgeListResponse';
import type { KnowledgeSearchResponse } from '../models/KnowledgeSearchResponse';
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
     * @param molexpSession
     * @returns KnowledgeListResponse Successful Response
     * @throws ApiError
     */
    public static listKnowledgeApiKnowledgeGet(
        tag?: (string | null),
        status?: (string | null),
        molexpSession?: (string | null),
    ): CancelablePromise<KnowledgeListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/knowledge',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns BacklinksResponse Successful Response
     * @throws ApiError
     */
    public static getBacklinksApiKnowledgeBacklinksGet(
        path: string,
        molexpSession?: (string | null),
    ): CancelablePromise<BacklinksResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/knowledge/backlinks',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteDocApiKnowledgeDocDelete(
        path: string,
        molexpSession?: (string | null),
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/knowledge/doc',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns NoteSummary Successful Response
     * @throws ApiError
     */
    public static moveDocApiKnowledgeDocPatch(
        path: string,
        requestBody: DocMoveRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<NoteSummary> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/knowledge/doc',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns NoteSummary Successful Response
     * @throws ApiError
     */
    public static createDocApiKnowledgeDocPost(
        requestBody: DocCreateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<NoteSummary> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/knowledge/doc',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns NoteDetailResponse Successful Response
     * @throws ApiError
     */
    public static editDocApiKnowledgeDocPut(
        path: string,
        requestBody: DocBodyUpdate,
        molexpSession?: (string | null),
    ): CancelablePromise<NoteDetailResponse> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/knowledge/doc',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns EmbedResponse Successful Response
     * @throws ApiError
     */
    public static embedDocApiKnowledgeDocEmbedPost(
        path: string,
        requestBody: EmbedRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<EmbedResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/knowledge/doc/embed',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static exportDocApiKnowledgeDocExportGet(
        path: string,
        molexpSession?: (string | null),
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/knowledge/doc/export',
            cookies: {
                'molexp_session': molexpSession,
            },
            query: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Update Doc Meta
     * Update a note's tags/status — delegates to ``Note.set_tags`` / ``Note.set_status``.
     *
     * Each field is applied only when present (``None`` = leave untouched), so a
     * partial update preserves the sibling field. The write logic is never re-built
     * here: the same ``Note`` verbs the CLI uses own it (the Python==UI invariant).
     * @param path The note Concept's bundle-relative path (its identity).
     * @param requestBody
     * @param molexpSession
     * @returns NoteSummary Successful Response
     * @throws ApiError
     */
    public static updateDocMetaApiKnowledgeDocMetaPatch(
        path: string,
        requestBody: DocMetaUpdate,
        molexpSession?: (string | null),
    ): CancelablePromise<NoteSummary> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/knowledge/doc/meta',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * Entity Backlinks
     * Knowledge documents citing one entity — a thin ``Bundle.backlinks`` read.
     *
     * Pure derived read (no reverse index persisted): resolves the entity
     * Folder, then asks the bundle which Concepts' ``index.md`` edges point at
     * it. 404 on an unresolvable entity — never an empty-list fallback for a
     * bad ref (no-fallback law).
     * @param kind Entity kind.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param molexpSession
     * @returns EntityBacklinksResponse Successful Response
     * @throws ApiError
     */
    public static entityBacklinksApiKnowledgeEntityBacklinksGet(
        kind: 'run' | 'experiment',
        projectId: string,
        experimentId: string,
        runId?: (string | null),
        molexpSession?: (string | null),
    ): CancelablePromise<EntityBacklinksResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/knowledge/entity-backlinks',
            cookies: {
                'molexp_session': molexpSession,
            },
            query: {
                'kind': kind,
                'projectId': projectId,
                'experimentId': experimentId,
                'runId': runId,
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
     * @param molexpSession
     * @returns NoteDetailResponse Successful Response
     * @throws ApiError
     */
    public static getNoteApiKnowledgeNoteGet(
        path: string,
        molexpSession?: (string | null),
    ): CancelablePromise<NoteDetailResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/knowledge/note',
            cookies: {
                'molexp_session': molexpSession,
            },
            query: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Search Knowledge
     * Search the workspace bundle — wraps the ONE ``Bundle.search`` verb.
     *
     * Pure exposure (vision-loop-08): all matching semantics (body reads, caps,
     * snippets, truncation) live in :meth:`molexp.workspace.Bundle.search`; this
     * route only projects its ``SearchResult`` onto the wire.
     * @param q Case-insensitive needle (path/title/tags/body).
     * @param type Exact Concept type filter.
     * @param tag Only concepts carrying this tag.
     * @param molexpSession
     * @returns KnowledgeSearchResponse Successful Response
     * @throws ApiError
     */
    public static searchKnowledgeApiKnowledgeSearchGet(
        q: string,
        type?: (string | null),
        tag?: (string | null),
        molexpSession?: (string | null),
    ): CancelablePromise<KnowledgeSearchResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/knowledge/search',
            cookies: {
                'molexp_session': molexpSession,
            },
            query: {
                'q': q,
                'type': type,
                'tag': tag,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}

/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { WorkflowDocumentRequest } from '../models/WorkflowDocumentRequest';
import type { WorkflowDocumentResponse } from '../models/WorkflowDocumentResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class WorkflowService {
    /**
     * Get Workflow Document
     * Return the persisted workflow IR document, or 404 if none stored.
     * @param projectId
     * @param experimentId
     * @param molexpSession
     * @returns WorkflowDocumentResponse Successful Response
     * @throws ApiError
     */
    public static getWorkflowDocumentApiProjectsProjectIdExperimentsExperimentIdWorkflowGet(
        projectId: string,
        experimentId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<WorkflowDocumentResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/workflow',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
            },
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Put Workflow Document
     * Validate, normalize, and persist an edited workflow IR document.
     * @param projectId
     * @param experimentId
     * @param requestBody
     * @param molexpSession
     * @returns WorkflowDocumentResponse Successful Response
     * @throws ApiError
     */
    public static putWorkflowDocumentApiProjectsProjectIdExperimentsExperimentIdWorkflowPut(
        projectId: string,
        experimentId: string,
        requestBody: WorkflowDocumentRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<WorkflowDocumentResponse> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/workflow',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
            },
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
     * Get Workflow Document
     * Return the persisted workflow IR document, or 404 if none stored.
     * @param projectId
     * @param experimentId
     * @param ws
     * @param molexpSession
     * @returns WorkflowDocumentResponse Successful Response
     * @throws ApiError
     */
    public static getWorkflowDocumentApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdWorkflowGet(
        projectId: string,
        experimentId: string,
        ws: string,
        molexpSession?: (string | null),
    ): CancelablePromise<WorkflowDocumentResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/workflow',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'ws': ws,
            },
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Put Workflow Document
     * Validate, normalize, and persist an edited workflow IR document.
     * @param projectId
     * @param experimentId
     * @param ws
     * @param requestBody
     * @param molexpSession
     * @returns WorkflowDocumentResponse Successful Response
     * @throws ApiError
     */
    public static putWorkflowDocumentApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdWorkflowPut(
        projectId: string,
        experimentId: string,
        ws: string,
        requestBody: WorkflowDocumentRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<WorkflowDocumentResponse> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/workflow',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'ws': ws,
            },
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
}

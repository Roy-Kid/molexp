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
     * @returns WorkflowDocumentResponse Successful Response
     * @throws ApiError
     */
    public static getWorkflowDocumentApiProjectsProjectIdExperimentsExperimentIdWorkflowGet(
        projectId: string,
        experimentId: string,
    ): CancelablePromise<WorkflowDocumentResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/workflow',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
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
     * @returns WorkflowDocumentResponse Successful Response
     * @throws ApiError
     */
    public static putWorkflowDocumentApiProjectsProjectIdExperimentsExperimentIdWorkflowPut(
        projectId: string,
        experimentId: string,
        requestBody: WorkflowDocumentRequest,
    ): CancelablePromise<WorkflowDocumentResponse> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/workflow',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
}

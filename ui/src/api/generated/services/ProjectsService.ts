/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { MessageResponse } from '../models/MessageResponse';
import type { ProjectCreateRequest } from '../models/ProjectCreateRequest';
import type { ProjectResponse } from '../models/ProjectResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class ProjectsService {
    /**
     * List Projects
     * List all projects.
     * @returns ProjectResponse Successful Response
     * @throws ApiError
     */
    public static listProjectsApiProjectsGet(): CancelablePromise<Array<ProjectResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects',
        });
    }
    /**
     * Create Project
     * Create a new project.
     * @param requestBody
     * @returns ProjectResponse Successful Response
     * @throws ApiError
     */
    public static createProjectApiProjectsPost(
        requestBody: ProjectCreateRequest,
    ): CancelablePromise<ProjectResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Project
     * Get project details.
     * @param id
     * @returns ProjectResponse Successful Response
     * @throws ApiError
     */
    public static getProjectApiProjectsIdGet(
        id: string,
    ): CancelablePromise<ProjectResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{id}',
            path: {
                'id': id,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Delete Project
     * Delete a project.
     * @param id
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteProjectApiProjectsIdDelete(
        id: string,
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/projects/{id}',
            path: {
                'id': id,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}

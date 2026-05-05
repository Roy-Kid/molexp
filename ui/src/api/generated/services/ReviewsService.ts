/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { MessageResponse } from '../models/MessageResponse';
import type { ReviewDecisionRequest } from '../models/ReviewDecisionRequest';
import type { ReviewItemResponse } from '../models/ReviewItemResponse';
import type { ReviewListResponse } from '../models/ReviewListResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class ReviewsService {
    /**
     * List Reviews
     * List persisted review items.
     * @param status
     * @param kind
     * @returns ReviewListResponse Successful Response
     * @throws ApiError
     */
    public static listReviewsApiReviewsGet(
        status?: (string | null),
        kind?: (string | null),
    ): CancelablePromise<ReviewListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/reviews',
            query: {
                'status': status,
                'kind': kind,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Review
     * Get one persisted review item.
     * @param reviewId
     * @returns ReviewItemResponse Successful Response
     * @throws ApiError
     */
    public static getReviewApiReviewsReviewIdGet(
        reviewId: string,
    ): CancelablePromise<ReviewItemResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/reviews/{review_id}',
            path: {
                'review_id': reviewId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Approve Review
     * Approve a review item and apply its target decision.
     * @param reviewId
     * @param requestBody
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static approveReviewApiReviewsReviewIdApprovePost(
        reviewId: string,
        requestBody: ReviewDecisionRequest,
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/reviews/{review_id}/approve',
            path: {
                'review_id': reviewId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Reject Review
     * Reject a review item and notify its target when possible.
     * @param reviewId
     * @param requestBody
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static rejectReviewApiReviewsReviewIdRejectPost(
        reviewId: string,
        requestBody: ReviewDecisionRequest,
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/reviews/{review_id}/reject',
            path: {
                'review_id': reviewId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
}

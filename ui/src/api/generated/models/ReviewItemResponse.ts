/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ReviewTargetRefResponse } from './ReviewTargetRefResponse';
export type ReviewItemResponse = {
    id: string;
    kind: string;
    title: string;
    description?: (string | null);
    riskLevel: string;
    status: string;
    targetRef: ReviewTargetRefResponse;
    createdAt: string;
    resolvedAt?: (string | null);
    resolutionComment?: (string | null);
    metadata?: Record<string, any>;
};


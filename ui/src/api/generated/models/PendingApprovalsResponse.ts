/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { PendingApprovalItem } from './PendingApprovalItem';
/**
 * The inbox: every pending request across both task kinds.
 */
export type PendingApprovalsResponse = {
    items: Array<PendingApprovalItem>;
    total: number;
};


/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Emitted with the verdict once an approval gate resolves.
 */
export type ApprovalDecidedEvent = {
    approved: boolean;
    gate: string;
    kind?: string;
    reason?: string;
    timestamp?: string;
};


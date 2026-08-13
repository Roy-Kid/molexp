/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Operator decision — ReviewDecision-shaped wire body.
 *
 * Preferred field is ``action`` (approve|reject|revise). ``granted`` remains
 * as a **deprecated** boolean alias for approve/reject only (migration for
 * older UI clients that only knew grant/deny).
 */
export type ApprovalDecisionRequest = {
    action?: ('approve' | 'reject' | 'revise' | null);
    edits?: (Record<string, any> | null);
    fieldValues?: Record<string, any>;
    granted?: (boolean | null);
    reason?: (string | null);
    requestId: string;
};


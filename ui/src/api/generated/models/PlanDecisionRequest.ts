/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * User decision on a plan emitted by ``exit_plan_mode``.
 *
 * Pairs with :class:`PlanCreatedEvent` via ``request_id``. Approval
 * flips the session out of plan mode so the agent can bind / run
 * the (possibly user-edited) workflow IR. Rejection hands the
 * feedback back to the agent so it can revise + call
 * ``exit_plan_mode`` again.
 */
export type PlanDecisionRequest = {
    /**
     * ID from the PlanCreatedEvent payload.
     */
    request_id: string;
    /**
     * True to approve the plan. False to reject and keep the agent in plan mode for revision.
     */
    approved: boolean;
    /**
     * Optional user edit of the plan markdown. When set, the agent sees this exact text as the post-approval starting point instead of its own draft.
     */
    edited_plan?: (string | null);
    /**
     * Optional user edit of the plan proposal (PlanProposal-shaped JSON). Replaces the agent's drafted proposal on approval.
     */
    edited_proposal?: (Record<string, any> | null);
    /**
     * Free-form rejection rationale. Surfaced to the agent so its next attempt addresses the user's concern.
     */
    feedback?: string;
};


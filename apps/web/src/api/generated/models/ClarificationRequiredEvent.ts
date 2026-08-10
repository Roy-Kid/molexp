/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Emitted when an intake stage cannot proceed without user clarification.
 *
 * PlanMode's ``ClarifyIntent`` stage yields this when the intent spec
 * carries unresolved ``MissingInfoItem``\ s with ``blocking=True``;
 * a registered :class:`~molexp.agent.repair.RepairPolicy`
 * routes the pipeline to the ``needs_clarification`` terminal state.
 *
 * Attributes:
 * questions: One-line concatenation of the blocking questions the
 * user must answer before planning can resume.
 */
export type ClarificationRequiredEvent = {
    kind?: string;
    questions: string;
    timestamp?: string;
};


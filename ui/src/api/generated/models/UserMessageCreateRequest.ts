/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Mid-session chat message from the user to the agent.
 */
export type UserMessageCreateRequest = {
    /**
     * User's message
     */
    content: string;
    /**
     * Pending UserMessageRequestEvent id this message replies to (omit for an unsolicited follow-up).
     */
    request_id?: (string | null);
};


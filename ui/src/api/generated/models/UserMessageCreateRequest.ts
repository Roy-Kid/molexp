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
     * Agent used for this turn; replies to a pending request keep its agent.
     */
    mode?: UserMessageCreateRequest.mode;
    /**
     * Pending UserMessageRequestEvent id this message replies to (omit for an unsolicited follow-up).
     */
    request_id?: (string | null);
};
export namespace UserMessageCreateRequest {
    /**
     * Agent used for this turn; replies to a pending request keep its agent.
     */
    export enum mode {
        CHAT = 'chat',
        PLAN = 'plan',
    }
}


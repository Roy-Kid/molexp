/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
// Manually re-shaped: tsc rejects a self-referential type alias (TS2456).
// The interface indirection makes the recursion legal while staying
// structurally identical to the generated form. Reapply this patch after
// each ``npm run generate:api`` until openapi-typescript-codegen emits
// recursive types correctly.
export interface JSONObject {
    [key: string]: JSONValue;
}
export type JSONArray = Array<JSONValue>;
export type JSONValue = string | number | boolean | JSONArray | JSONObject | null;


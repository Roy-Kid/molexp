/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * OpenAPI ``JSONValue`` — broken into interface forms so TypeScript does not
 * reject the recursive alias (``Record<string, JSONValue>`` self-reference).
 * ``scripts/patch-generated-api.mjs`` re-applies this after ``generate:api``.
 */
export type JSONValue = string | number | boolean | null | JSONArray | JSONObject;

export interface JSONObject {
    [key: string]: JSONValue;
}

export interface JSONArray extends Array<JSONValue> {}

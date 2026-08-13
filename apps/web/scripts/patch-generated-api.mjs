/**
 * Post-process openapi-typescript-codegen output.
 * - Fix circular JSONValue alias (TS2456).
 */
import { writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const jsonValuePath = join(root, "src/api/generated/models/JSONValue.ts");

writeFileSync(
  jsonValuePath,
  `/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * OpenAPI \`JSONValue\` — interface form avoids TS2456 circular alias.
 * Re-applied by ui/scripts/patch-generated-api.mjs after generate:api.
 */
export type JSONValue = string | number | boolean | null | JSONArray | JSONObject;

export interface JSONObject {
    [key: string]: JSONValue;
}

export interface JSONArray extends Array<JSONValue> {}
`,
);

console.log("patched", jsonValuePath);

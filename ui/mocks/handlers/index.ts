/**
 * Central export for all MSW mock handlers
 */

import { projectHandlers } from "./projects";
import { experimentHandlers } from "./experiments";
import { runHandlers } from "./runs";
import { assetHandlers } from "./assets";
import { workspaceHandlers } from "./workspace";
import { executionHandlers } from "./execution";
import { registryHandlers } from "./registry";

/**
 * All mock handlers combined
 */
export const handlers = [
    ...projectHandlers,
    ...experimentHandlers,
    ...runHandlers,
    ...assetHandlers,
    ...workspaceHandlers,
    ...executionHandlers,
    ...registryHandlers,
];

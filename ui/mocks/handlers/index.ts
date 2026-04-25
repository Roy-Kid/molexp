/**
 * Central export for all MSW mock handlers
 */

import { agentHandlers } from "./agent";
import { assetHandlers } from "./assets";
import { catalogHandlers } from "./catalog";
import { executionHandlers } from "./execution";
import { experimentHandlers } from "./experiments";
import { projectHandlers } from "./projects";
import { registryHandlers } from "./registry";
import { runHandlers } from "./runs";
import { workspaceHandlers } from "./workspace";

/**
 * All mock handlers combined
 */
export const handlers = [
    ...agentHandlers,
    ...assetHandlers,
    ...catalogHandlers,
    ...executionHandlers,
    ...experimentHandlers,
    ...projectHandlers,
    ...registryHandlers,
    ...runHandlers,
    ...workspaceHandlers,
];

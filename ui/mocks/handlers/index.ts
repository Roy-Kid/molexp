/**
 * Central export for all MSW mock handlers
 */

import { agentHandlers } from "./agent";
import { agentAdminHandlers } from "./agent_admin";
import { assetHandlers } from "./assets";
import { catalogHandlers } from "./catalog";
import { executionHandlers } from "./execution";
import { experimentHandlers } from "./experiments";
import { molqHandlers } from "./molq";
import { projectHandlers } from "./projects";
import { registryHandlers } from "./registry";
import { runHandlers } from "./runs";
import { workspaceHandlers } from "./workspace";

/**
 * All mock handlers combined
 */
export const handlers = [
    ...agentHandlers,
    ...agentAdminHandlers,
    ...assetHandlers,
    ...catalogHandlers,
    ...executionHandlers,
    ...experimentHandlers,
    ...molqHandlers,
    ...projectHandlers,
    ...registryHandlers,
    ...runHandlers,
    ...workspaceHandlers,
];

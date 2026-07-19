/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AgentToolListResponse } from '../models/AgentToolListResponse';
import type { McpServerListResponse } from '../models/McpServerListResponse';
import type { McpServerResponse } from '../models/McpServerResponse';
import type { McpServerUpsertRequest } from '../models/McpServerUpsertRequest';
import type { ProviderResponse } from '../models/ProviderResponse';
import type { ProviderTestResponse } from '../models/ProviderTestResponse';
import type { ProviderUpdateRequest } from '../models/ProviderUpdateRequest';
import type { SkillListResponse } from '../models/SkillListResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class AgentAdminService {
    /**
     * List Mcp Servers
     * Merged user + workspace MCP server entries (workspace shadows user).
     * @returns McpServerListResponse Successful Response
     * @throws ApiError
     */
    public static listMcpServersApiAgentMcpServersGet(): CancelablePromise<McpServerListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/mcp/servers',
        });
    }
    /**
     * Create Mcp Server
     * Upsert one MCP server entry at the requested scope.
     * @param requestBody
     * @returns McpServerResponse Successful Response
     * @throws ApiError
     */
    public static createMcpServerApiAgentMcpServersPost(
        requestBody: McpServerUpsertRequest,
    ): CancelablePromise<McpServerResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/mcp/servers',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Provider
     * Current provider settings from the operator config (keys masked).
     * @returns ProviderResponse Successful Response
     * @throws ApiError
     */
    public static getProviderApiAgentProviderGet(): CancelablePromise<ProviderResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/provider',
        });
    }
    /**
     * Update Provider
     * Persist submitted provider fields, then re-bridge the live process.
     *
     * The write goes through the shared :func:`set_operator_values` (same file,
     * same atomic writer as ``molexp config set``). The bridged
     * ``molexp.config`` keys this PUT changes are cleared before re-bridging so
     * the running server serves the new values immediately.
     * @param requestBody
     * @returns ProviderResponse Successful Response
     * @throws ApiError
     */
    public static updateProviderApiAgentProviderPut(
        requestBody: ProviderUpdateRequest,
    ): CancelablePromise<ProviderResponse> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/agent/provider',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Test Provider
     * Preflight the agent stack for the submitted (or stored) model.
     *
     * Constructor + credential validation only — no disk writes, no network,
     * no LLM call (the honest scope of a settings-page "test" that must never
     * spend tokens or mutate state). ``reply`` describes what was verified.
     * @param requestBody
     * @returns ProviderTestResponse Successful Response
     * @throws ApiError
     */
    public static testProviderApiAgentProviderTestPost(
        requestBody: ProviderUpdateRequest,
    ): CancelablePromise<ProviderTestResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/provider/test',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Skills
     * Return the configured skill catalog.
     *
     * Skill persistence is not wired into this admin service yet.  An empty
     * catalog is a valid state, so the read surface must not fall through to
     * the legacy agent 503 catch-all.
     * @returns SkillListResponse Successful Response
     * @throws ApiError
     */
    public static listSkillsApiAgentSkillsGet(): CancelablePromise<SkillListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/skills',
        });
    }
    /**
     * List Tools
     * Return tools grouped by their owning MCP server.
     *
     * Until runtime discovery is connected, expose the honest empty catalog
     * expected by Settings instead of reporting an unavailable service.
     * @returns AgentToolListResponse Successful Response
     * @throws ApiError
     */
    public static listToolsApiAgentToolsGet(): CancelablePromise<AgentToolListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/tools',
        });
    }
}

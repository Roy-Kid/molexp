/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { _CommandParseRequest } from '../models/_CommandParseRequest';
import type { AgentHealthResponse } from '../models/AgentHealthResponse';
import type { AgentToolListResponse } from '../models/AgentToolListResponse';
import type { CommandListResponse } from '../models/CommandListResponse';
import type { CommandParseResponse } from '../models/CommandParseResponse';
import type { KnowledgeSourcesResponse } from '../models/KnowledgeSourcesResponse';
import type { KnowledgeSourcesUpdateRequest } from '../models/KnowledgeSourcesUpdateRequest';
import type { McpSecretListResponse } from '../models/McpSecretListResponse';
import type { McpSecretPutRequest } from '../models/McpSecretPutRequest';
import type { McpServerListResponse } from '../models/McpServerListResponse';
import type { McpServerResponse } from '../models/McpServerResponse';
import type { McpServerTestResponse } from '../models/McpServerTestResponse';
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
     * List Admin Providers
     * Provider form registry for Settings (bootstrap schema; never 503).
     * @returns any Successful Response
     * @throws ApiError
     */
    public static listAdminProvidersApiAgentAdminProvidersGet(): CancelablePromise<Record<string, any>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/admin/providers',
        });
    }
    /**
     * List Commands
     * Slash-command catalog for the chat composer palette.
     *
     * Builtins always ship; skill-backed commands join when skill persistence
     * is wired (currently an empty skill catalog is valid).
     * @returns CommandListResponse Successful Response
     * @throws ApiError
     */
    public static listCommandsApiAgentCommandsGet(): CancelablePromise<CommandListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/commands',
        });
    }
    /**
     * Parse Command
     * Parse a raw slash line into a builtin / skill / error result.
     * @param requestBody
     * @returns CommandParseResponse Successful Response
     * @throws ApiError
     */
    public static parseCommandApiAgentCommandsParsePost(
        requestBody: _CommandParseRequest,
    ): CancelablePromise<CommandParseResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/commands/parse',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Agent Health
     * Agent readiness for the UI banner — always 200, never the 503 catch-all.
     *
     * ``ready=False`` is a normal configuration state (no model / no API key).
     * The legacy ``agent.router`` catch-all used to 503 unknown ``/api/agent*``
     * paths, which made the UI treat "missing route" as "stack unavailable"
     * and permanently stop probing. This endpoint exists so health is always a
     * real JSON readiness document.
     * @returns AgentHealthResponse Successful Response
     * @throws ApiError
     */
    public static agentHealthApiAgentHealthGet(): CancelablePromise<AgentHealthResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/health',
        });
    }
    /**
     * Get Knowledge Sources
     * Read package pin from the molmcp MCP server entry (``MOLMCP_SOURCES``).
     * @returns KnowledgeSourcesResponse Successful Response
     * @throws ApiError
     */
    public static getKnowledgeSourcesApiAgentKnowledgeSourcesGet(): CancelablePromise<KnowledgeSourcesResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/knowledge-sources',
        });
    }
    /**
     * Update Knowledge Sources
     * Write package pin onto the molmcp server's env (per-MCP, not global agent).
     * @param requestBody
     * @returns KnowledgeSourcesResponse Successful Response
     * @throws ApiError
     */
    public static updateKnowledgeSourcesApiAgentKnowledgeSourcesPut(
        requestBody: KnowledgeSourcesUpdateRequest,
    ): CancelablePromise<KnowledgeSourcesResponse> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/agent/knowledge-sources',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Mcp Secrets
     * List secret *keys* (never values) at the given scope.
     * @param scope
     * @returns McpSecretListResponse Successful Response
     * @throws ApiError
     */
    public static listMcpSecretsApiAgentMcpSecretsGet(
        scope: string = 'user',
    ): CancelablePromise<McpSecretListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/mcp/secrets',
            query: {
                'scope': scope,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Put Mcp Secret
     * Set or delete a secret value (empty value deletes).
     * @param key
     * @param requestBody
     * @returns any Successful Response
     * @throws ApiError
     */
    public static putMcpSecretApiAgentMcpSecretsKeyPut(
        key: string,
        requestBody: McpSecretPutRequest,
    ): CancelablePromise<Record<string, any>> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/agent/mcp/secrets/{key}',
            path: {
                'key': key,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
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
     * Delete Mcp Server
     * Delete one MCP server entry at the given scope.
     * @param name
     * @param scope
     * @returns void
     * @throws ApiError
     */
    public static deleteMcpServerApiAgentMcpServersNameDelete(
        name: string,
        scope: string = 'user',
    ): CancelablePromise<void> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/agent/mcp/servers/{name}',
            path: {
                'name': name,
            },
            query: {
                'scope': scope,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Replace Mcp Server
     * Replace one MCP server entry (name path must match body).
     * @param name
     * @param requestBody
     * @returns McpServerResponse Successful Response
     * @throws ApiError
     */
    public static replaceMcpServerApiAgentMcpServersNamePut(
        name: string,
        requestBody: McpServerUpsertRequest,
    ): CancelablePromise<McpServerResponse> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/agent/mcp/servers/{name}',
            path: {
                'name': name,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Test Mcp Server
     * Best-effort stdio/HTTP reachability probe (list_tools when possible).
     * @param name
     * @param scope
     * @returns McpServerTestResponse Successful Response
     * @throws ApiError
     */
    public static testMcpServerApiAgentMcpServersNameTestPost(
        name: string,
        scope: string = 'user',
    ): CancelablePromise<McpServerTestResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/mcp/servers/{name}/test',
            path: {
                'name': name,
            },
            query: {
                'scope': scope,
            },
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
     * Return agent tools: molexp **builtins** + MCP groups when discovered.
     *
     * Builtins (``workspace_ensure``, ``run_land``, ``code_write``, …) are
     * always present with ``source="builtin"``. MCP tools attach as
     * ``source="mcp:<server>"`` when runtime discovery is connected; until
     * then ``mcpGroups`` may be empty without hiding builtins.
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

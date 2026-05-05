/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { _SimpleMessage } from '../models/_SimpleMessage';
import type { AgentHealthResponse } from '../models/AgentHealthResponse';
import type { AgentProviderResponse } from '../models/AgentProviderResponse';
import type { AgentProviderTestResponse } from '../models/AgentProviderTestResponse';
import type { AgentProviderUpdateRequest } from '../models/AgentProviderUpdateRequest';
import type { AgentToolListResponse } from '../models/AgentToolListResponse';
import type { CommandListResponse } from '../models/CommandListResponse';
import type { CommandParseRequest } from '../models/CommandParseRequest';
import type { CommandParseResponse } from '../models/CommandParseResponse';
import type { CustomToolCreateRequest } from '../models/CustomToolCreateRequest';
import type { CustomToolListResponse } from '../models/CustomToolListResponse';
import type { CustomToolResponse } from '../models/CustomToolResponse';
import type { CustomToolUpdateRequest } from '../models/CustomToolUpdateRequest';
import type { MCPOAuthCallbackRequest } from '../models/MCPOAuthCallbackRequest';
import type { MCPOAuthStartResponse } from '../models/MCPOAuthStartResponse';
import type { MCPOAuthStatusResponse } from '../models/MCPOAuthStatusResponse';
import type { MCPSecretListResponse } from '../models/MCPSecretListResponse';
import type { MCPSecretSetRequest } from '../models/MCPSecretSetRequest';
import type { MCPServerListResponse } from '../models/MCPServerListResponse';
import type { MCPServerResponse } from '../models/MCPServerResponse';
import type { MCPServerTestResponse } from '../models/MCPServerTestResponse';
import type { MCPServerUpsertRequest } from '../models/MCPServerUpsertRequest';
import type { MessageResponse } from '../models/MessageResponse';
import type { SkillCreateRequest } from '../models/SkillCreateRequest';
import type { SkillListResponse } from '../models/SkillListResponse';
import type { SkillResponse } from '../models/SkillResponse';
import type { SkillUpdateRequest } from '../models/SkillUpdateRequest';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class AgentAdminService {
    /**
     * Get Mcp Servers
     * Return merged User+Workspace MCP servers, including shadowed entries.
     * @returns MCPServerListResponse Successful Response
     * @throws ApiError
     */
    public static getMcpServersApiAgentMcpServersGet(): CancelablePromise<MCPServerListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/mcp/servers',
        });
    }
    /**
     * Create Mcp Server
     * Create an MCP server entry at ``request.scope``.
     * @param requestBody
     * @returns MCPServerResponse Successful Response
     * @throws ApiError
     */
    public static createMcpServerApiAgentMcpServersPost(
        requestBody: MCPServerUpsertRequest,
    ): CancelablePromise<MCPServerResponse> {
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
     * Replace Mcp Server
     * Fully replace an MCP server entry. ``request.name`` must match the path.
     * @param name
     * @param requestBody
     * @returns MCPServerResponse Successful Response
     * @throws ApiError
     */
    public static replaceMcpServerApiAgentMcpServersNamePut(
        name: string,
        requestBody: MCPServerUpsertRequest,
    ): CancelablePromise<MCPServerResponse> {
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
     * Delete Mcp Server
     * @param name
     * @param scope Which scope to delete from.
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteMcpServerApiAgentMcpServersNameDelete(
        name: string,
        scope: 'user' | 'workspace' = 'workspace',
    ): CancelablePromise<MessageResponse> {
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
     * Test Mcp Server
     * Open a real connection to the server, list its tools, then disconnect.
     *
     * Bounded by a 10-second hard timeout. The test result never includes
     * secret values — only the resolved tool count + connection metrics.
     * @param name
     * @param scope Which scope's entry to probe.
     * @returns MCPServerTestResponse Successful Response
     * @throws ApiError
     */
    public static testMcpServerApiAgentMcpServersNameTestPost(
        name: string,
        scope: 'user' | 'workspace' = 'workspace',
    ): CancelablePromise<MCPServerTestResponse> {
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
     * Start Mcp Oauth
     * Begin the OAuth 2.0 + PKCE flow for an MCP server.
     *
     * Spawns a background task that drives the SDK's
     * ``OAuthClientProvider`` until it produces an authorize URL, then
     * returns that URL. The browser opens it; once the IdP bounces back to
     * the SPA, the SPA POSTs the code/state to ``/oauth/callback`` to
     * complete token exchange.
     *
     * Replaces any in-flight session for this server (clicking Connect
     * twice cancels the older flow rather than queueing).
     * @param name
     * @param scope
     * @returns MCPOAuthStartResponse Successful Response
     * @throws ApiError
     */
    public static startMcpOauthApiAgentMcpServersNameOauthStartPost(
        name: string,
        scope: 'user' | 'workspace' = 'workspace',
    ): CancelablePromise<MCPOAuthStartResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/mcp/servers/{name}/oauth/start',
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
     * Callback Mcp Oauth
     * Complete an in-flight OAuth flow with the IdP's callback payload.
     *
     * Fed by the SPA after the IdP redirects the browser back to
     * ``/oauth-callback``. We hand the ``(code, state)`` to the awaiting SDK
     * callback_handler, await flow completion, then return the new
     * connection status.
     * @param name
     * @param requestBody
     * @param scope
     * @returns MCPOAuthStatusResponse Successful Response
     * @throws ApiError
     */
    public static callbackMcpOauthApiAgentMcpServersNameOauthCallbackPost(
        name: string,
        requestBody: MCPOAuthCallbackRequest,
        scope: 'user' | 'workspace' = 'workspace',
    ): CancelablePromise<MCPOAuthStatusResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/mcp/servers/{name}/oauth/callback',
            path: {
                'name': name,
            },
            query: {
                'scope': scope,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Mcp Oauth Status
     * Whether the named server has a usable OAuth token on disk.
     * @param name
     * @param scope
     * @returns MCPOAuthStatusResponse Successful Response
     * @throws ApiError
     */
    public static getMcpOauthStatusApiAgentMcpServersNameOauthGet(
        name: string,
        scope: 'user' | 'workspace' = 'workspace',
    ): CancelablePromise<MCPOAuthStatusResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/mcp/servers/{name}/oauth',
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
     * Disconnect Mcp Oauth
     * Drop stored OAuth tokens for a server. Idempotent.
     *
     * Equivalent to "log out" — the spec stays in place, so a future Connect
     * walks the user through PKCE again. Any in-flight flow is cancelled.
     * @param name
     * @param scope
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static disconnectMcpOauthApiAgentMcpServersNameOauthDelete(
        name: string,
        scope: 'user' | 'workspace' = 'workspace',
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/agent/mcp/servers/{name}/oauth',
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
     * List Mcp Secrets
     * List secret keys at ``scope`` plus which servers reference them.
     *
     * Plaintext values are **never** returned. ``isSet`` is the only signal
     * of whether the value exists for a referenced key.
     * @param scope Which secret store to inspect.
     * @returns MCPSecretListResponse Successful Response
     * @throws ApiError
     */
    public static listMcpSecretsApiAgentMcpSecretsGet(
        scope: 'user' | 'workspace' = 'workspace',
    ): CancelablePromise<MCPSecretListResponse> {
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
     * Set Mcp Secret
     * Write a secret value at ``request.scope``. Empty value deletes the key.
     * @param key
     * @param requestBody
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static setMcpSecretApiAgentMcpSecretsKeyPut(
        key: string,
        requestBody: MCPSecretSetRequest,
    ): CancelablePromise<MessageResponse> {
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
     * Get Agent Tools
     * List tools the agent will see — native + each configured MCP server.
     *
     * Native tools always come back instantly. MCP servers are probed in
     * parallel; each server contributes (a) its tools (with ``source =
     * "mcp:<name>"`` so the UI can group) and (b) a row in ``mcpGroups``
     * capturing per-server connection status. A failed probe is not fatal —
     * its group surfaces ``ok=False`` + ``error`` so users can fix it
     * without losing the rest of the listing.
     * @returns AgentToolListResponse Successful Response
     * @throws ApiError
     */
    public static getAgentToolsApiAgentToolsGet(): CancelablePromise<AgentToolListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/tools',
        });
    }
    /**
     * List Skills
     * Return every skill across builtin + user-home + workspace tiers.
     *
     * Display order: builtin → user → workspace. Each entry carries its
     * ``scope`` and ``builtin`` flag so the UI can render shadowing
     * (a workspace skill with the same ``slash_name`` as a builtin sits
     * later in the list and the client highlights the override).
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
     * Create Skill
     * @param requestBody
     * @returns SkillResponse Successful Response
     * @throws ApiError
     */
    public static createSkillApiAgentSkillsPost(
        requestBody: SkillCreateRequest,
    ): CancelablePromise<SkillResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/skills',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Skill
     * @param skillId
     * @returns SkillResponse Successful Response
     * @throws ApiError
     */
    public static getSkillApiAgentSkillsSkillIdGet(
        skillId: string,
    ): CancelablePromise<SkillResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/skills/{skill_id}',
            path: {
                'skill_id': skillId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Update Skill
     * @param skillId
     * @param requestBody
     * @param scope Tier the skill belongs to. Builtin skills are immutable.
     * @returns SkillResponse Successful Response
     * @throws ApiError
     */
    public static updateSkillApiAgentSkillsSkillIdPatch(
        skillId: string,
        requestBody: SkillUpdateRequest,
        scope: 'user' | 'workspace' = 'workspace',
    ): CancelablePromise<SkillResponse> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/agent/skills/{skill_id}',
            path: {
                'skill_id': skillId,
            },
            query: {
                'scope': scope,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Delete Skill
     * @param skillId
     * @param scope Tier to delete from. Builtin skills cannot be deleted.
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteSkillApiAgentSkillsSkillIdDelete(
        skillId: string,
        scope: 'user' | 'workspace' = 'workspace',
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/agent/skills/{skill_id}',
            path: {
                'skill_id': skillId,
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
     * List Commands
     * Return all slash commands available to the chat input.
     *
     * Includes the four builtins plus every skill with a non-empty
     * ``slash_name``. Each entry carries enough metadata for the client to
     * render an autocomplete popover and validate arguments before
     * submitting.
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
     * Parse a raw chat input into a structured ``CommandParseResponse``.
     *
     * Mirrors :func:`molexp.plugins.agent_pydanticai.commands.parse`. Errors
     * surface as ``kind="error"`` with a UI-ready message — the route never
     * raises a 4xx for parser-level issues so the client can render the
     * message inline.
     * @param requestBody
     * @returns CommandParseResponse Successful Response
     * @throws ApiError
     */
    public static parseCommandApiAgentCommandsParsePost(
        requestBody: CommandParseRequest,
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
     * Get Provider
     * Return the workspace's LLM provider config (key redacted).
     * @returns AgentProviderResponse Successful Response
     * @throws ApiError
     */
    public static getProviderApiAgentProviderGet(): CancelablePromise<AgentProviderResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/provider',
        });
    }
    /**
     * Update Provider
     * Patch provider/model/api_key/base_url. Empty ``api_key`` clears the key.
     * @param requestBody
     * @returns AgentProviderResponse Successful Response
     * @throws ApiError
     */
    public static updateProviderApiAgentProviderPut(
        requestBody: AgentProviderUpdateRequest,
    ): CancelablePromise<AgentProviderResponse> {
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
     * Get Agent Health
     * Report whether the agent runtime is ready to start a new session.
     *
     * UI uses this to render a "configure provider" banner before the user
     * even tries to launch — much better UX than letting POST /sessions
     * return a structured 400 only after the goal is typed.
     * @returns AgentHealthResponse Successful Response
     * @throws ApiError
     */
    public static getAgentHealthApiAgentHealthGet(): CancelablePromise<AgentHealthResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/health',
        });
    }
    /**
     * Test Provider
     * Send a minimal probe to the configured provider — never persists.
     *
     * The request body is treated as an optional override over the stored
     * config so the UI can validate user input before saving.
     * @param requestBody
     * @returns AgentProviderTestResponse Successful Response
     * @throws ApiError
     */
    public static testProviderApiAgentProviderTestPost(
        requestBody: AgentProviderUpdateRequest,
    ): CancelablePromise<AgentProviderTestResponse> {
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
     * List Resources
     * @returns CustomToolListResponse Successful Response
     * @throws ApiError
     */
    public static listResourcesApiAgentToolsCustomGet(): CancelablePromise<CustomToolListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/tools/custom',
        });
    }
    /**
     * Create Resource
     * @param requestBody
     * @returns CustomToolResponse Successful Response
     * @throws ApiError
     */
    public static createResourceApiAgentToolsCustomPost(
        requestBody: CustomToolCreateRequest,
    ): CancelablePromise<CustomToolResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/tools/custom',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Resource
     * @param resourceId
     * @returns CustomToolResponse Successful Response
     * @throws ApiError
     */
    public static getResourceApiAgentToolsCustomResourceIdGet(
        resourceId: string,
    ): CancelablePromise<CustomToolResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/tools/custom/{resource_id}',
            path: {
                'resource_id': resourceId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Update Resource
     * @param resourceId
     * @param requestBody
     * @param scope Tier the entry belongs to.
     * @returns CustomToolResponse Successful Response
     * @throws ApiError
     */
    public static updateResourceApiAgentToolsCustomResourceIdPatch(
        resourceId: string,
        requestBody: CustomToolUpdateRequest,
        scope: 'user' | 'workspace' = 'workspace',
    ): CancelablePromise<CustomToolResponse> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/agent/tools/custom/{resource_id}',
            path: {
                'resource_id': resourceId,
            },
            query: {
                'scope': scope,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Delete Resource
     * @param resourceId
     * @param scope Tier to delete from.
     * @returns _SimpleMessage Successful Response
     * @throws ApiError
     */
    public static deleteResourceApiAgentToolsCustomResourceIdDelete(
        resourceId: string,
        scope: 'user' | 'workspace' = 'workspace',
    ): CancelablePromise<_SimpleMessage> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/agent/tools/custom/{resource_id}',
            path: {
                'resource_id': resourceId,
            },
            query: {
                'scope': scope,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}

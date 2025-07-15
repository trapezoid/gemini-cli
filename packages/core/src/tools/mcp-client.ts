/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import {
  SSEClientTransport,
  SSEClientTransportOptions,
} from '@modelcontextprotocol/sdk/client/sse.js';
import {
  StreamableHTTPClientTransport,
  StreamableHTTPClientTransportOptions,
} from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { parse } from 'shell-quote';
import { MCPServerConfig } from '../config/config.js';
import { DiscoveredMCPTool } from './mcp-tool.js';
import { FunctionDeclaration, Type, mcpToTool, Schema } from '@google/genai';
import { sanitizeParameters, ToolRegistry } from './tool-registry.js';

export const MCP_DEFAULT_TIMEOUT_MSEC = 10 * 60 * 1000; // default to 10 minutes

/**
 * Enum representing the connection status of an MCP server
 */
export enum MCPServerStatus {
  /** Server is disconnected or experiencing errors */
  DISCONNECTED = 'disconnected',
  /** Server is in the process of connecting */
  CONNECTING = 'connecting',
  /** Server is connected and ready to use */
  CONNECTED = 'connected',
}

/**
 * Enum representing the overall MCP discovery state
 */
export enum MCPDiscoveryState {
  /** Discovery has not started yet */
  NOT_STARTED = 'not_started',
  /** Discovery is currently in progress */
  IN_PROGRESS = 'in_progress',
  /** Discovery has completed (with or without errors) */
  COMPLETED = 'completed',
}

/**
 * Map to track the status of each MCP server within the core package
 */
const mcpServerStatusesInternal: Map<string, MCPServerStatus> = new Map();

/**
 * Track the overall MCP discovery state
 */
let mcpDiscoveryState: MCPDiscoveryState = MCPDiscoveryState.NOT_STARTED;

/**
 * Event listeners for MCP server status changes
 */
type StatusChangeListener = (
  serverName: string,
  status: MCPServerStatus,
) => void;
const statusChangeListeners: StatusChangeListener[] = [];

/**
 * Add a listener for MCP server status changes
 */
export function addMCPStatusChangeListener(
  listener: StatusChangeListener,
): void {
  statusChangeListeners.push(listener);
}

/**
 * Remove a listener for MCP server status changes
 */
export function removeMCPStatusChangeListener(
  listener: StatusChangeListener,
): void {
  const index = statusChangeListeners.indexOf(listener);
  if (index !== -1) {
    statusChangeListeners.splice(index, 1);
  }
}

/**
 * Update the status of an MCP server
 */
function updateMCPServerStatus(
  serverName: string,
  status: MCPServerStatus,
): void {
  mcpServerStatusesInternal.set(serverName, status);
  // Notify all listeners
  for (const listener of statusChangeListeners) {
    listener(serverName, status);
  }
}

/**
 * Get the current status of an MCP server
 */
export function getMCPServerStatus(serverName: string): MCPServerStatus {
  return (
    mcpServerStatusesInternal.get(serverName) || MCPServerStatus.DISCONNECTED
  );
}

/**
 * Get all MCP server statuses
 */
export function getAllMCPServerStatuses(): Map<string, MCPServerStatus> {
  return new Map(mcpServerStatusesInternal);
}

/**
 * Get the current MCP discovery state
 */
export function getMCPDiscoveryState(): MCPDiscoveryState {
  return mcpDiscoveryState;
}

/**
 * Discovers tools from all configured MCP servers and registers them with the tool registry.
 * It orchestrates the connection and discovery process for each server defined in the
 * configuration, as well as any server specified via a command-line argument.
 *
 * @param mcpServers A record of named MCP server configurations.
 * @param mcpServerCommand An optional command string for a dynamically specified MCP server.
 * @param toolRegistry The central registry where discovered tools will be registered.
 * @returns A promise that resolves when the discovery process has been attempted for all servers.
 */
export async function discoverMcpTools(
  mcpServers: Record<string, MCPServerConfig>,
  mcpServerCommand: string | undefined,
  toolRegistry: ToolRegistry,
  debugMode: boolean,
): Promise<void> {
  mcpDiscoveryState = MCPDiscoveryState.IN_PROGRESS;
  try {
    mcpServers = populateMcpServerCommand(mcpServers, mcpServerCommand);

    const discoveryPromises = Object.entries(mcpServers).map(
      ([mcpServerName, mcpServerConfig]) =>
        connectAndDiscover(
          mcpServerName,
          mcpServerConfig,
          toolRegistry,
          debugMode,
        ),
    );
    await Promise.all(discoveryPromises);
  } finally {
    mcpDiscoveryState = MCPDiscoveryState.COMPLETED;
  }
}

/** Visible for Testing */
export function populateMcpServerCommand(
  mcpServers: Record<string, MCPServerConfig>,
  mcpServerCommand: string | undefined,
): Record<string, MCPServerConfig> {
  if (mcpServerCommand) {
    const cmd = mcpServerCommand;
    const args = parse(cmd, process.env) as string[];
    if (args.some((arg) => typeof arg !== 'string')) {
      throw new Error('failed to parse mcpServerCommand: ' + cmd);
    }
    // use generic server name 'mcp'
    mcpServers['mcp'] = {
      command: args[0],
      args: args.slice(1),
    };
  }
  return mcpServers;
}

/**
 * Connects to an MCP server and discovers available tools, registering them with the tool registry.
 * This function handles the complete lifecycle of connecting to a server, discovering tools,
 * and cleaning up resources if no tools are found.
 *
 * @param mcpServerName The name identifier for this MCP server
 * @param mcpServerConfig Configuration object containing connection details
 * @param toolRegistry The registry to register discovered tools with
 * @returns Promise that resolves when discovery is complete
 */
export async function connectAndDiscover(
  mcpServerName: string,
  mcpServerConfig: MCPServerConfig,
  toolRegistry: ToolRegistry,
  debugMode: boolean,
): Promise<void> {
  updateMCPServerStatus(mcpServerName, MCPServerStatus.CONNECTING);

  try {
    const mcpClient = await connectToMcpServer(
      mcpServerName,
      mcpServerConfig,
      debugMode,
    );
    try {
      updateMCPServerStatus(mcpServerName, MCPServerStatus.CONNECTED);

      mcpClient.onerror = (error) => {
        console.error(`MCP ERROR (${mcpServerName}):`, error.toString());
        updateMCPServerStatus(mcpServerName, MCPServerStatus.DISCONNECTED);
      };

      const tools = await discoverTools(
        mcpServerName,
        mcpServerConfig,
        mcpClient,
      );
      for (const tool of tools) {
        toolRegistry.registerTool(tool);
      }
    } catch (error) {
      mcpClient.close();
      throw error;
    }
  } catch (error) {
    console.error(`Error connecting to MCP server '${mcpServerName}':`, error);
    updateMCPServerStatus(mcpServerName, MCPServerStatus.DISCONNECTED);
  }
}

/**
 * Discovers and sanitizes tools from a connected MCP client.
 * It retrieves function declarations from the client, filters out disabled tools,
 * generates valid names for them, and wraps them in `DiscoveredMCPTool` instances.
 *
 * @param mcpServerName The name of the MCP server.
 * @param mcpServerConfig The configuration for the MCP server.
 * @param mcpClient The active MCP client instance.
 * @returns A promise that resolves to an array of discovered and enabled tools.
 * @throws An error if no enabled tools are found or if the server provides invalid function declarations.
 */
export async function discoverTools(
  mcpServerName: string,
  mcpServerConfig: MCPServerConfig,
  mcpClient: Client,
): Promise<DiscoveredMCPTool[]> {
  try {
    const mcpCallableTool = mcpToTool(mcpClient);
    const tool = await mcpCallableTool.tool();

    if (!Array.isArray(tool.functionDeclarations)) {
      throw new Error(`Server did not return valid function declarations.`);
    }

    const discoveredTools: DiscoveredMCPTool[] = [];
    for (const funcDecl of tool.functionDeclarations) {
      if (!isEnabled(funcDecl, mcpServerName, mcpServerConfig)) {
        continue;
      }

      const toolNameForModel = generateValidName(funcDecl, mcpServerName);

      const parameters = processJsonSchema(
        (funcDecl.parametersJsonSchema ?? {
          type: 'object',
          properties: {},
        }) as Record<string, unknown>,
      );

      sanitizeParameters(parameters);

      discoveredTools.push(
        new DiscoveredMCPTool(
          mcpCallableTool,
          mcpServerName,
          toolNameForModel,
          funcDecl.description ?? '',
          parameters,
          funcDecl.name!,
          mcpServerConfig.timeout ?? MCP_DEFAULT_TIMEOUT_MSEC,
          mcpServerConfig.trust,
        ),
      );
    }
    if (discoveredTools.length === 0) {
      throw Error('No enabled tools found');
    }
    return discoveredTools;
  } catch (error) {
    throw new Error(`Error discovering tools: ${error}`);
  }
}

/**
 * Creates and connects an MCP client to a server based on the provided configuration.
 * It determines the appropriate transport (Stdio, SSE, or Streamable HTTP) and
 * establishes a connection. It also applies a patch to handle request timeouts.
 *
 * @param mcpServerName The name of the MCP server, used for logging and identification.
 * @param mcpServerConfig The configuration specifying how to connect to the server.
 * @returns A promise that resolves to a connected MCP `Client` instance.
 * @throws An error if the connection fails or the configuration is invalid.
 */
export async function connectToMcpServer(
  mcpServerName: string,
  mcpServerConfig: MCPServerConfig,
  debugMode: boolean,
): Promise<Client> {
  const mcpClient = new Client({
    name: 'gemini-cli-mcp-client',
    version: '0.0.1',
  });

  // patch Client.callTool to use request timeout as genai McpCallTool.callTool does not do it
  // TODO: remove this hack once GenAI SDK does callTool with request options
  if ('callTool' in mcpClient) {
    const origCallTool = mcpClient.callTool.bind(mcpClient);
    mcpClient.callTool = function (params, resultSchema, options) {
      return origCallTool(params, resultSchema, {
        ...options,
        timeout: mcpServerConfig.timeout ?? MCP_DEFAULT_TIMEOUT_MSEC,
      });
    };
  }

  try {
    const transport = createTransport(
      mcpServerName,
      mcpServerConfig,
      debugMode,
    );
    try {
      await mcpClient.connect(transport, {
        timeout: mcpServerConfig.timeout ?? MCP_DEFAULT_TIMEOUT_MSEC,
      });
      return mcpClient;
    } catch (error) {
      await transport.close();
      throw error;
    }
  } catch (error) {
    // Create a safe config object that excludes sensitive information
    const safeConfig = {
      command: mcpServerConfig.command,
      url: mcpServerConfig.url,
      httpUrl: mcpServerConfig.httpUrl,
      cwd: mcpServerConfig.cwd,
      timeout: mcpServerConfig.timeout,
      trust: mcpServerConfig.trust,
      // Exclude args, env, and headers which may contain sensitive data
    };

    let errorString =
      `failed to start or connect to MCP server '${mcpServerName}' ` +
      `${JSON.stringify(safeConfig)}; \n${error}`;
    if (process.env.SANDBOX) {
      errorString += `\nMake sure it is available in the sandbox`;
    }
    throw new Error(errorString);
  }
}

/** Visible for Testing */
export function createTransport(
  mcpServerName: string,
  mcpServerConfig: MCPServerConfig,
  debugMode: boolean,
): Transport {
  if (mcpServerConfig.httpUrl) {
    const transportOptions: StreamableHTTPClientTransportOptions = {};
    if (mcpServerConfig.headers) {
      transportOptions.requestInit = {
        headers: mcpServerConfig.headers,
      };
    }
    return new StreamableHTTPClientTransport(
      new URL(mcpServerConfig.httpUrl),
      transportOptions,
    );
  }

  if (mcpServerConfig.url) {
    const transportOptions: SSEClientTransportOptions = {};
    if (mcpServerConfig.headers) {
      transportOptions.requestInit = {
        headers: mcpServerConfig.headers,
      };
    }
    return new SSEClientTransport(
      new URL(mcpServerConfig.url),
      transportOptions,
    );
  }

  if (mcpServerConfig.command) {
    const transport = new StdioClientTransport({
      command: mcpServerConfig.command,
      args: mcpServerConfig.args || [],
      env: {
        ...process.env,
        ...(mcpServerConfig.env || {}),
      } as Record<string, string>,
      cwd: mcpServerConfig.cwd,
      stderr: 'pipe',
    });
    if (debugMode) {
      transport.stderr!.on('data', (data) => {
        const stderrStr = data.toString().trim();
        console.debug(`[DEBUG] [MCP STDERR (${mcpServerName})]: `, stderrStr);
      });
    }
    return transport;
  }

  throw new Error(
    `Invalid configuration: missing httpUrl (for Streamable HTTP), url (for SSE), and command (for stdio).`,
  );
}

/** Visible for testing */
export function generateValidName(
  funcDecl: FunctionDeclaration,
  mcpServerName: string,
) {
  // Replace invalid characters (based on 400 error message from Gemini API) with underscores
  let validToolname = funcDecl.name!.replace(/[^a-zA-Z0-9_.-]/g, '_');

  // Prepend MCP server name to avoid conflicts with other tools
  validToolname = mcpServerName + '__' + validToolname;

  // If longer than 63 characters, replace middle with '___'
  // (Gemini API says max length 64, but actual limit seems to be 63)
  if (validToolname.length > 63) {
    validToolname =
      validToolname.slice(0, 28) + '___' + validToolname.slice(-32);
  }
  return validToolname;
}

/** Visible for testing */
export function isEnabled(
  funcDecl: FunctionDeclaration,
  mcpServerName: string,
  mcpServerConfig: MCPServerConfig,
): boolean {
  if (!funcDecl.name) {
    console.warn(
      `Discovered a function declaration without a name from MCP server '${mcpServerName}'. Skipping.`,
    );
    return false;
  }
  const { includeTools, excludeTools } = mcpServerConfig;

  // excludeTools takes precedence over includeTools
  if (excludeTools && excludeTools.includes(funcDecl.name)) {
    return false;
  }

  return (
    !includeTools ||
    includeTools.some(
      (tool) => tool === funcDecl.name || tool.startsWith(`${funcDecl.name}(`),
    )
  );
}

//from: https://github.com/googleapis/js-genai/blob/v1.9.0/src/_transformers.ts#L293-L420
export function processJsonSchema(
  _jsonSchema: Schema | Record<string, unknown>,
): Schema {
  const genAISchema: Schema = {};
  const schemaFieldNames = ['items'];
  const listSchemaFieldNames = ['anyOf'];
  const dictSchemaFieldNames = ['properties'];

  if (_jsonSchema['type'] && _jsonSchema['anyOf']) {
    throw new Error('type and anyOf cannot be both populated.');
  }

  /*
  This is to handle the nullable array or object. The _jsonSchema will
  be in the format of {anyOf: [{type: 'null'}, {type: 'object'}]}. The
  logic is to check if anyOf has 2 elements and one of the element is null,
  if so, the anyOf field is unnecessary, so we need to get rid of the anyOf
  field and make the schema nullable. Then use the other element as the new
  _jsonSchema for processing. This is because the backend doesn't have a null
  type.
  This has to be checked before we process any other fields.
  For example:
    const objectNullable = z.object({
      nullableArray: z.array(z.string()).nullable(),
    });
  Will have the raw _jsonSchema as:
  {
    type: 'OBJECT',
    properties: {
        nullableArray: {
           anyOf: [
              {type: 'null'},
              {
                type: 'array',
                items: {type: 'string'},
              },
            ],
        }
    },
    required: [ 'nullableArray' ],
  }
  Will result in following schema compatible with Gemini API:
    {
      type: 'OBJECT',
      properties: {
         nullableArray: {
            nullable: true,
            type: 'ARRAY',
            items: {type: 'string'},
         }
      },
      required: [ 'nullableArray' ],
    }
  */
  const incomingAnyOf = _jsonSchema['anyOf'] as Array<Record<string, unknown>>;
  if (incomingAnyOf != null && incomingAnyOf.length === 2) {
    if (incomingAnyOf[0]!['type'] === 'null') {
      genAISchema['nullable'] = true;
      _jsonSchema = incomingAnyOf![1];
    } else if (incomingAnyOf[1]!['type'] === 'null') {
      genAISchema['nullable'] = true;
      _jsonSchema = incomingAnyOf![0];
    }
  }

  if (_jsonSchema['type'] instanceof Array) {
    flattenTypeArrayToAnyOf(_jsonSchema['type'], genAISchema);
  }

  for (const [fieldName, fieldValue] of Object.entries(_jsonSchema)) {
    // Skip if the fieldvalue is undefined or null.
    if (fieldValue === null || fieldValue === undefined) {
      continue;
    }

    if (fieldName === 'type') {
      if (fieldValue === 'null') {
        throw new Error(
          'type: null can not be the only possible type for the field.',
        );
      }
      if (fieldValue instanceof Array) {
        // we have already handled the type field with array of types in the
        // beginning of this function.
        continue;
      }
      genAISchema['type'] = Object.values(Type).includes(
        fieldValue.toUpperCase() as Type,
      )
        ? fieldValue.toUpperCase()
        : Type.TYPE_UNSPECIFIED;
    } else if (schemaFieldNames.includes(fieldName)) {
      (genAISchema as Record<string, unknown>)[fieldName] =
        processJsonSchema(fieldValue);
    } else if (listSchemaFieldNames.includes(fieldName)) {
      const listSchemaFieldValue: Schema[] = [];
      for (const item of fieldValue) {
        if (item['type'] === 'null') {
          genAISchema['nullable'] = true;
          continue;
        }
        listSchemaFieldValue.push(
          processJsonSchema(item as Record<string, unknown>),
        );
      }
      (genAISchema as Record<string, unknown>)[fieldName] =
        listSchemaFieldValue;
    } else if (dictSchemaFieldNames.includes(fieldName)) {
      const dictSchemaFieldValue: Record<string, Schema> = {};
      for (const [key, value] of Object.entries(
        fieldValue as Record<string, unknown>,
      )) {
        dictSchemaFieldValue[key] = processJsonSchema(
          value as Record<string, unknown>,
        );
      }
      (genAISchema as Record<string, unknown>)[fieldName] =
        dictSchemaFieldValue;
    } else {
      // additionalProperties is not included in JSONSchema, skipping it.
      if (fieldName === 'additionalProperties') {
        continue;
      }
      (genAISchema as Record<string, unknown>)[fieldName] = fieldValue;
    }
  }
  return genAISchema;
}

//from: https://github.com/googleapis/js-genai/blob/v1.9.0/src/_transformers.ts#L257-L291
/*
Transform the type field from an array of types to an array of anyOf fields.
Example:
  {type: ['STRING', 'NUMBER']}
will be transformed to
  {anyOf: [{type: 'STRING'}, {type: 'NUMBER'}]}
*/
function flattenTypeArrayToAnyOf(typeList: string[], resultingSchema: Schema) {
  if (typeList.includes('null')) {
    resultingSchema['nullable'] = true;
  }
  const listWithoutNull = typeList.filter((type) => type !== 'null');

  if (listWithoutNull.length === 1) {
    resultingSchema['type'] = Object.values(Type).includes(
      listWithoutNull[0].toUpperCase() as Type,
    )
      ? (listWithoutNull[0].toUpperCase() as Type)
      : Type.TYPE_UNSPECIFIED;
  } else {
    resultingSchema['anyOf'] = [];
    for (const i of listWithoutNull) {
      resultingSchema['anyOf'].push({
        type: Object.values(Type).includes(i.toUpperCase() as Type)
          ? (i.toUpperCase() as Type)
          : Type.TYPE_UNSPECIFIED,
      });
    }
  }
}

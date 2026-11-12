export enum ToolSource {
	BROWSER = 'browser',
	CUSTOM = 'custom',
	MCP = 'mcp',
	// [RDNA-FAST] the RDNA-FAST tool section, run in the browser (src/lib/rdna/rdna-tools.ts)
	RDNA = 'rdna',
	SERVER = 'server'
}

export enum ToolPermissionDecision {
	ALWAYS = 'always',
	ALWAYS_SERVER = 'always_server',
	DENY = 'deny',
	ONCE = 'once'
}

export enum ToolResponseField {
	ERROR = 'error',
	PLAIN_TEXT = 'plain_text_response'
}

/**
 * Entry types accepted by the `file_glob_search` tool's `type` parameter.
 * Mirrors the server-side validation in server-tools.cpp.
 */
export enum GlobSearchType {
	ALL = 'all',
	DIR = 'dir',
	FILE = 'file'
}

/**
 * Wire-format identifiers for server and browser tools. The string
 * value matches what the model emits in tool call names, so comparing
 * against `BuiltInTool.SERVER_READ_FILE` is equivalent to comparing
 * against the raw `'read_file'` literal - the enum just keeps the two in
 * lock-step and gives TypeScript a single source of truth for autocomplete
 * / rename support.
 *
 * The `SERVER_` / `BROWSER_` prefixes mirror the tool's primary source
 * (llama-server vs llama-ui). `get_info` is the exception: it is served by
 * the server, but llama-ui falls back to a browser implementation when the
 * server does not provide it, so it can surface under both categories in
 * the UI while keeping a single wire name.
 */
export enum BuiltInTool {
	BROWSER_GET_DATETIME = 'get_datetime',
	BROWSER_READ_MEDIA = 'read_media',
	BROWSER_RUN_JAVASCRIPT = 'run_javascript',
	BROWSER_SEARCH_CHATS = 'search_chats',
	// [RDNA-FAST] Keenable web search and page reading, src/lib/rdna/web-tools.ts
	BROWSER_WEB_FETCH = 'web_fetch',
	BROWSER_WEB_SEARCH = 'web_search',
	// [RDNA-FAST] the RDNA-FAST tool section, src/lib/rdna/rdna-tools.ts
	RDNA_DOWNLOAD_MODEL = 'download_model',
	RDNA_RECENT_CONVERSATIONS = 'recent_conversations',
	RDNA_ROUTER_CONFIG = 'router_config',
	RDNA_SAVE_PRESET = 'save_preset',
	RDNA_SERVER_INFO = 'server_info',
	SERVER_EDIT_FILE = 'edit_file',
	SERVER_EXEC_SHELL_COMMAND = 'exec_shell_command',
	SERVER_FILE_GLOB_SEARCH = 'file_glob_search',
	SERVER_GET_INFO = 'get_info',
	SERVER_GREP_SEARCH = 'grep_search',
	SERVER_READ_FILE = 'read_file',
	SERVER_WRITE_FILE = 'write_file'
}

/** Wire values of the `search_chats` `message_selection` parameter */
export enum SearchChatsMessageSelection {
	EARLIEST = 'earliest',
	LATEST = 'latest',
	MATCHING = 'matching'
}

/** Wire values of the `search_chats` `order_timestamp` parameter */
export enum SearchChatsOrder {
	ASC = 'asc',
	DESC = 'desc'
}

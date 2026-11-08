/**
 * RDNA-FAST chat tools (owner, 2026-11-08), their own "RDNA-FAST" section in the chat's tools, on by default:
 *   recent_conversations  this browser's chats: find them by words, or read one's last messages
 *   download_model        a GGUF from Hugging Face through the router, unless it is there or on its way already
 *   system_info           GPU, VRAM, CPU, RAM, the server build and the models it runs
 *   save_preset           the flags and env vars a model runs with, kept by the router ([TAG_MODEL_PRESET])
 *   router_config         what the router runs: each model's status, preset (its flags) and saved recipe
 * Descriptions and answers stay short: a local model reads every tool schema and result at its prefill speed.
 */
import { apiUrl, modelQuery, rdnaGet } from './api';
import { isClassifier } from './model-info';
import { BuiltInTool, JsonSchemaType, ServerModelStatus, ToolCallType } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import type {
	ApiModelDataEntry,
	DatabaseConversation,
	OpenAIToolDefinition,
	ToolExecutionResult
} from '$lib/types';
import { getAuthHeaders } from '$lib/utils/api-headers';

const RDNA_TOOLS = new Set<string>([
	BuiltInTool.RDNA_RECENT_CONVERSATIONS,
	BuiltInTool.RDNA_DOWNLOAD_MODEL,
	BuiltInTool.RDNA_SYSTEM_INFO,
	BuiltInTool.RDNA_SAVE_PRESET,
	BuiltInTool.RDNA_ROUTER_CONFIG
]);

/** The section's name in the chat's tools. */
export const RDNA_TOOL_GROUP_LABEL = 'RDNA-FAST';
export const RDNA_TOOL_GROUP_KEY = 'rdna-fast';

const CONVERSATIONS_DEFAULT = 5;
const CONVERSATIONS_MAX = 20;
/** Chats searched for words in their messages, the most recent first (titles: all of them). */
const CONVERSATIONS_SCANNED = 200;
const SNIPPET_CHARS = 160;
/** One chat read back: its last messages, each cut here. */
const READ_MESSAGES = 6;
const READ_MESSAGE_CHARS = 600;

export function isRdnaTool(name: string): boolean {
	return RDNA_TOOLS.has(name);
}

function fn(
	name: string,
	description: string,
	properties: Record<string, unknown>,
	required: string[] = []
): OpenAIToolDefinition {
	return {
		function: {
			description,
			name,
			parameters: { properties, required, type: JsonSchemaType.OBJECT }
		},
		type: ToolCallType.FUNCTION
	} as OpenAIToolDefinition;
}

export function buildRdnaToolDefinitions(): OpenAIToolDefinition[] {
	return [
		fn(
			BuiltInTool.RDNA_RECENT_CONVERSATIONS,
			'Find earlier chats by words in titles and messages; no query = the latest. With id: read that chat.',
			{
				id: { type: JsonSchemaType.STRING },
				limit: {
					description: `default ${CONVERSATIONS_DEFAULT}, max ${CONVERSATIONS_MAX}`,
					type: JsonSchemaType.NUMBER
				},
				query: { type: JsonSchemaType.STRING }
			}
		),
		fn(
			BuiltInTool.RDNA_DOWNLOAD_MODEL,
			'Download a GGUF from Hugging Face, e.g. "unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M". Reports it if it is there already.',
			{ model: { type: JsonSchemaType.STRING } },
			['model']
		),
		fn(
			BuiltInTool.RDNA_SYSTEM_INFO,
			'This machine: GPU, VRAM, CPU, RAM, server build, models running.',
			{}
		),
		fn(
			BuiltInTool.RDNA_SAVE_PRESET,
			"Save a model's whole recipe: llama-server flags and env vars (replaces its earlier one). Used from its next load; reload=true applies it now.",
			{
				args: {
					description: 'flags, e.g. ["-ngl","999","-fa","on"]',
					items: { type: JsonSchemaType.STRING },
					type: JsonSchemaType.ARRAY
				},
				env: {
					additionalProperties: { type: JsonSchemaType.STRING },
					description: 'e.g. {"GGML_HIP_FA_GQA6":"1"}',
					type: JsonSchemaType.OBJECT
				},
				model: {
					description: 'the model id, e.g. "org/repo-GGUF:Q4_K_M"',
					type: JsonSchemaType.STRING
				},
				reload: { type: JsonSchemaType.BOOLEAN }
			},
			['model']
		),
		fn(
			BuiltInTool.RDNA_ROUTER_CONFIG,
			"The router's model config: flags and saved recipe of the running models, or of one model.",
			{ model: { type: JsonSchemaType.STRING } }
		)
	];
}

function ok(content: string): ToolExecutionResult {
	return { content, isError: false };
}

function fail(content: string): ToolExecutionResult {
	return { content, isError: true };
}

function clip(text: string, max: number): string {
	const t = text.replace(/\s+/g, ' ').trim();

	return t.length > max ? `${t.slice(0, max - 1)}…` : t;
}

function when(ts: number | undefined): string {
	if (!ts) return '';

	const d = new Date(ts);
	const pad = (n: number) => String(n).padStart(2, '0');

	return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function gb(bytes: number | undefined): string {
	return bytes ? `${(bytes / 1e9).toFixed(1)} GB` : '?';
}

/** Words of a query, lower case, two letters or more */
function terms(query: string): string[] {
	return query
		.toLowerCase()
		.split(/[^\p{L}\p{N}_.-]+/u)
		.filter((t) => t.length >= 2);
}

/** Text around the first term found, or null */
function snippet(text: string, words: string[]): string | null {
	const lower = text.toLowerCase();

	let at = -1;

	for (const w of words) {
		const i = lower.indexOf(w);

		if (i >= 0 && (at < 0 || i < at)) at = i;
	}

	if (at < 0) return null;

	const start = Math.max(0, at - SNIPPET_CHARS / 2);

	return `${start > 0 ? '…' : ''}${clip(text.slice(start, start + SNIPPET_CHARS), SNIPPET_CHARS)}`;
}

function conversationLine(i: number, c: DatabaseConversation): string {
	return `${i + 1}. ${c.name || 'Untitled'} (${when(c.lastModified)}, id ${c.id})`;
}

async function recentConversations(args: Record<string, unknown>): Promise<ToolExecutionResult> {
	const id = typeof args.id === 'string' ? args.id.trim() : '';
	const query = typeof args.query === 'string' ? args.query.trim() : '';
	const limit = Math.min(
		CONVERSATIONS_MAX,
		Math.max(1, Math.round(Number(args.limit) || CONVERSATIONS_DEFAULT))
	);
	const all = await DatabaseService.getAllConversations(); // newest first

	if (id) {
		const conv = all.find((c) => c.id === id) ?? all.find((c) => c.id.startsWith(id));

		if (!conv) return fail(`No chat with id ${id}.`);

		const messages = (await DatabaseService.getConversationMessages(conv.id)).filter(
			(m) => (m.role === 'user' || m.role === 'assistant') && m.content?.trim()
		);
		const last = messages.slice(-READ_MESSAGES);

		return ok(
			[
				`${conv.name || 'Untitled'} (${when(conv.lastModified)}), last ${last.length} of ${messages.length} messages:`,
				...last.map((m) => `${m.role}: ${clip(m.content, READ_MESSAGE_CHARS)}`)
			].join('\n')
		);
	}

	if (!query) {
		if (all.length === 0) return ok('No chats yet.');

		return ok(
			all
				.slice(0, limit)
				.map((c, i) => conversationLine(i, c))
				.join('\n')
		);
	}

	const words = terms(query);

	if (words.length === 0) return fail('The query has no words to search for.');

	// titles first (every chat), then the messages of the most recent chats
	const hits: { conv: DatabaseConversation; score: number; text: string | null }[] = [];

	for (const [rank, conv] of all.entries()) {
		const title = (conv.name || '').toLowerCase();

		let score = words.filter((w) => title.includes(w)).length * 3;
		let text: string | null = null;

		if (rank < CONVERSATIONS_SCANNED) {
			for (const m of await DatabaseService.getConversationMessages(conv.id)) {
				const s = m.content ? snippet(m.content, words) : null;

				if (!s) continue;

				score += words.filter((w) => m.content.toLowerCase().includes(w)).length;
				text ??= s;
			}
		}

		if (score > 0) hits.push({ conv, score, text });
	}

	if (hits.length === 0) return ok(`No chat mentions "${query}".`);

	// the best matches, the newer first among equals (the list is newest first already)
	hits.sort((a, b) => b.score - a.score);

	return ok(
		hits
			.slice(0, limit)
			.map((h, i) => `${conversationLine(i, h.conv)}${h.text ? `\n   ${h.text}` : ''}`)
			.join('\n')
	);
}

const NOT_ON_DISK = new Set<string>([
	ServerModelStatus.DOWNLOADING,
	ServerModelStatus.INCOMPLETE,
	ServerModelStatus.DOWNLOAD_FAILED
]);

async function downloadModel(args: Record<string, unknown>): Promise<ToolExecutionResult> {
	const model = typeof args.model === 'string' ? args.model.trim() : '';

	if (!serverStore.isRouterMode) return fail('Downloads need the server in router mode.');

	if (!/^[^/\s]+\/[^/\s:]+(:[^\s]+)?$/.test(model)) {
		return fail(
			'Give the model as "org/repo" or "org/repo:QUANT", e.g. "unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M".'
		);
	}

	await modelsStore.fetch();

	const key = model.toLowerCase();
	const found = modelsStore.routerModels.find((m) => m.id.toLowerCase() === key);
	const status = found?.status?.value;

	if (found && status && !NOT_ON_DISK.has(status)) {
		return ok(`Already downloaded: ${found.id} (${status}).`);
	}

	const progress = () => modelsStore.status.getDownloadProgress(model);
	const running = progress();

	if (running) {
		return ok(
			`Already downloading ${model}: ${gb(running.downloadedBytes)} of ${gb(running.totalBytes)}.`
		);
	}

	await modelsStore.status.downloadModel(model);

	const started = progress();

	if (!started) {
		return fail(`The server did not start the download of ${model}: check the repo and quant.`);
	}

	return ok(
		`Download started: ${model}${started.totalBytes ? ` (${gb(started.totalBytes)})` : ''}. It continues in the background; the model id for save_preset is "${model}".`
	);
}

interface PropsDevice {
	name?: string;
	description?: string;
	type?: string;
	memory_total?: number;
	memory_free?: number;
}

interface PropsAnswer {
	build_info?: string;
	devices?: PropsDevice[];
}

function modelArgs(m: ApiModelDataEntry): string[] | undefined {
	return m.status?.args;
}

async function systemInfo(signal?: AbortSignal): Promise<ToolExecutionResult> {
	await modelsStore.fetch();

	const router = serverStore.isRouterMode;
	const models = modelsStore.routerModels;
	const running = models.filter((m) => m.status?.value === ServerModelStatus.LOADED);
	// a chat model's /props names every device of the machine (a CPU-only classifier only the CPU)
	const probe = running.find((m) => !isClassifier(modelArgs(m))) ?? running[0];
	const res = await rdnaGet<PropsAnswer>('/props', modelQuery(probe?.id ?? null, router), signal);
	const lines: string[] = [];

	for (const d of res.data?.devices ?? []) {
		const cpu = (d.type ?? '').toLowerCase() === 'cpu' || d.name === 'CPU';
		const memory = `${gb(d.memory_total)} ${cpu ? 'RAM' : 'VRAM'}, ${gb(d.memory_free)} free`;

		lines.push(`${cpu ? 'CPU' : 'GPU'}: ${d.description || d.name} (${d.name}): ${memory}`);
	}

	if (lines.length === 0) {
		lines.push(
			running.length === 0
				? 'GPU, CPU and memory: unknown until a model is loaded.'
				: 'GPU, CPU and memory: the server does not report them.'
		);
	}

	if (res.data?.build_info)
		lines.push(`Server: RDNA-FAST ${res.data.build_info}${router ? ', router mode' : ''}`);

	if (router) {
		lines.push(
			`Running: ${running.length ? running.map((m) => m.id).join(', ') : 'none'}`,
			`Downloaded: ${models.filter((m) => !NOT_ON_DISK.has(m.status?.value ?? '')).length} models`
		);
	}

	return ok(lines.join('\n'));
}

async function savePreset(
	args: Record<string, unknown>,
	signal?: AbortSignal
): Promise<ToolExecutionResult> {
	const model = typeof args.model === 'string' ? args.model.trim() : '';

	if (!serverStore.isRouterMode) return fail('Presets need the server in router mode.');

	if (!model) return fail('Name the model (its id, e.g. "org/repo-GGUF:Q4_K_M").');

	const flags = Array.isArray(args.args) ? args.args.map((a) => String(a)) : [];
	const env: Record<string, string> = {};

	if (args.env && typeof args.env === 'object' && !Array.isArray(args.env)) {
		for (const [k, v] of Object.entries(args.env as Record<string, unknown>))
			env[k] = String(v ?? '');
	}

	try {
		const res = await fetch(apiUrl('/models/preset'), {
			body: JSON.stringify({ args: flags, env, model, reload: args.reload === true }),
			headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' },
			method: 'POST',
			signal
		});
		const body = (await res.json().catch(() => null)) as {
			applied?: string;
			error?: { message?: string } | string;
		} | null;

		if (!res.ok) {
			const msg = typeof body?.error === 'string' ? body.error : body?.error?.message;

			return fail(`Not saved: ${msg || `the server answered ${res.status}`}.`);
		}

		const envText = Object.entries(env)
			.map(([k, v]) => `${k}=${v}`)
			.join(' ');

		return ok(
			`Saved for ${model}: ${flags.join(' ') || '(no flags)'}${envText ? ` | env ${envText}` : ''}. Applies ${body?.applied ?? 'on its next load'}.`
		);
	} catch (e) {
		return fail(`Not saved: ${e instanceof Error ? e.message : String(e)}.`);
	}
}

interface RouterModelRaw {
	id: string;
	status?: { value?: string; args?: string[]; preset?: string };
	rdna_preset?: { args?: string[]; env?: Record<string, string> };
}

/** Router-managed or noisy args left out of a model's flags (host, port, alias, tags, the API key) */
const ROUTER_ARGS = new Set([
	'--host',
	'--port',
	'--alias',
	'-a',
	'--tags',
	'--api-key',
	'--api-key-file'
]);
const PRESET_TEXT_MAX = 1500;

function flagsOf(args: readonly string[] | undefined): string {
	const out: string[] = [];

	for (let i = 0; i < (args?.length ?? 0); i++) {
		const a = args![i];

		if (ROUTER_ARGS.has(a)) {
			if (i + 1 < args!.length && !args![i + 1].startsWith('-')) i++;

			continue;
		}

		out.push(a);
	}

	// the executable's path is not a flag
	if (out.length && !out[0].startsWith('-')) out.shift();

	return out.join(' ');
}

function presetText(ini: string): string {
	const body = ini
		.split(/\r?\n/)
		.map((l) => l.trim())
		.filter((l) => l && !l.startsWith(';') && !l.startsWith('#'))
		.join('\n');

	return body.length > PRESET_TEXT_MAX ? `${body.slice(0, PRESET_TEXT_MAX)}\n…` : body;
}

async function routerConfig(
	args: Record<string, unknown>,
	signal?: AbortSignal
): Promise<ToolExecutionResult> {
	if (!serverStore.isRouterMode) return fail('router_config needs the server in router mode.');

	const res = await rdnaGet<{ data?: RouterModelRaw[] }>('/v1/models', undefined, signal);

	if (!res.ok) return fail(`The server answered ${res.status || 'nothing'}.`);

	const all = res.data?.data ?? [];
	const want = typeof args.model === 'string' ? args.model.trim().toLowerCase() : '';
	const running = (m: RouterModelRaw) =>
		m.status?.value === ServerModelStatus.LOADED || m.status?.value === ServerModelStatus.LOADING;
	const picked = want
		? (all.filter((m) => m.id.toLowerCase() === want).length
				? all.filter((m) => m.id.toLowerCase() === want)
				: all.filter((m) => m.id.toLowerCase().includes(want))
			).slice(0, 3)
		: all.filter(running);

	if (picked.length === 0) {
		return ok(
			want
				? `No model matches "${args.model}". Models: ${all.map((m) => m.id).join(', ')}`
				: `No model is running. Models: ${all.map((m) => m.id).join(', ')}`
		);
	}

	const blocks = picked.map((m) => {
		const lines = [`${m.id} (${m.status?.value ?? 'unknown'})`];

		if (m.status?.preset) lines.push(`preset:\n${presetText(m.status.preset)}`);
		else if (m.status?.args?.length) lines.push(`flags: ${flagsOf(m.status.args)}`);

		if (m.rdna_preset) {
			const env = Object.entries(m.rdna_preset.env ?? {})
				.map(([k, v]) => `${k}=${v}`)
				.join(' ');

			lines.push(
				`saved recipe: ${(m.rdna_preset.args ?? []).join(' ') || '(no flags)'}${env ? ` | env ${env}` : ''}`
			);
		}

		return lines.join('\n');
	});

	if (!want) {
		const others = all.filter((m) => !running(m)).map((m) => m.id);

		if (others.length)
			blocks.push(
				`not running: ${others.slice(0, 15).join(', ')}${others.length > 15 ? ', …' : ''}`
			);
	}

	return ok(blocks.join('\n\n'));
}

export async function executeRdnaTool(
	name: string,
	args: Record<string, unknown>,
	signal?: AbortSignal
): Promise<ToolExecutionResult> {
	try {
		switch (name) {
			case BuiltInTool.RDNA_RECENT_CONVERSATIONS:
				return await recentConversations(args);
			case BuiltInTool.RDNA_DOWNLOAD_MODEL:
				return await downloadModel(args);
			case BuiltInTool.RDNA_SYSTEM_INFO:
				return await systemInfo(signal);
			case BuiltInTool.RDNA_SAVE_PRESET:
				return await savePreset(args, signal);
			case BuiltInTool.RDNA_ROUTER_CONFIG:
				return await routerConfig(args, signal);
			default:
				return fail(`Unknown tool ${name}.`);
		}
	} catch (e) {
		return fail(e instanceof Error ? e.message : String(e));
	}
}

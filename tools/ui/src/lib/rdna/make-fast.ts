/**
 * RDNA-FAST guided chats (owner, 2026-11-08): "Make it RDNA-FAST!" in Discover Models and "Help me go RDNA-FASTer"
 * on the status page. Each asks the model that runs to work through the RDNA-FAST tools (server_info, router_config,
 * download_model, save_preset) and the project's docs on GitHub (web_fetch). The prompts are short numbered steps: the
 * model reads them at its prefill speed, and the steps keep a small model on track.
 */
import { isClassifier } from './model-info';
import { ServerModelStatus } from '$lib/enums';
import { chatStore, conversationsStore, modelsStore } from '$lib/stores';

const DOCS = 'https://raw.githubusercontent.com/serrynaimo/RDNA-FAST/main';
const DOCS_STEP = `Read the RDNA-FAST docs with web_fetch: ${DOCS}/README.md, then the docs pages it points to that fit (${DOCS}/docs/FLAGS.md; ${DOCS}/docs/expert-handoff.md for a MoE model bigger than the GPU).`;

/** The Discover button's prompt for one Hugging Face repo, with the quants its files offer */
export function makeItFastPrompt(repo: string, quants: string[]): string {
	return [
		`Make ${repo} RDNA-FAST on this machine.`,
		'1. server_info: OS, GPU, VRAM, RAM.',
		`2. ${DOCS_STEP}`,
		`3. Pick the quant of ${repo} that runs fastest here at good quality${quants.length ? ` (it has ${quants.join(', ')})` : ''}.`,
		`4. download_model "${repo}:<QUANT>" (it says so if the model is there already).`,
		'5. save_preset for that model id: the flags and env vars the docs recommend for maximum speed on this GPU.',
		'Then tell me in a few lines which quant and why, and the recipe.'
	].join('\n');
}

/** The status page's "Help me go RDNA-FASTer": can the model run faster with another recipe? */
export function fasterPrompt(modelId: string): string {
	return [
		`Can ${modelId} run RDNA-FASTer?`,
		`1. router_config for "${modelId}": its flags. server_info: OS, GPU, VRAM, RAM.`,
		`2. ${DOCS_STEP}`,
		"3. Compare the model's flags and env vars with the docs' recipe for this GPU and list what would make it faster.",
		'4. If a change clearly helps, save_preset the improved recipe for that model (it applies on its next load).',
		'Keep the answer short.'
	].join('\n');
}

/** GGUF quant tags in a repo's file names (Q4_K_M, IQ4_XS, MXFP4_MOE, BF16, ...), split files once, no mmproj */
export function quantsOfFiles(files: readonly string[]): string[] {
	const out: string[] = [];

	for (const f of files) {
		const name = f.split('/').pop() ?? f;

		if (!/\.gguf$/i.test(name) || /mmproj/i.test(name)) continue;

		const m = name
			.replace(/-\d{5}-of-\d{5}\.gguf$/i, '.gguf')
			.match(/[-_.]((?:UD-)?(?:I?Q\d[A-Z0-9_]*|BF16|F16|F32|MXFP4[A-Z0-9_]*))\.gguf$/i);
		const q = m?.[1]?.toUpperCase();

		if (q && !out.includes(q)) out.push(q);
	}

	return out;
}

/** The model that answers RDNA-FAST chats: the first loaded router model that is not a classifier */
export function primaryChatModel(): string | null {
	const loaded = modelsStore.routerModels.filter(
		(m) => m.status?.value === ServerModelStatus.LOADED && !isClassifier(m.status?.args)
	);

	return loaded[0]?.id ?? null;
}

/**
 * A new chat with the prompt, answered by the primary model; the window switches to it. Sent as from an empty chat:
 * with no conversation active, sendMessage creates one, navigates to it and names it from the prompt the way the title
 * settings say (its first line, or the model's title), not "Chat <date>"
 */
export async function startRdnaChat(prompt: string): Promise<void> {
	conversationsStore.clearActiveConversation();

	const id = primaryChatModel();
	const model = id ? modelsStore.findModelByName(id) : null;

	if (model) {
		try {
			await modelsStore.selectModelById(model.id);
		} catch {
			// the chat's own model answers then
		}
	}

	await chatStore.sendMessage(prompt);
}

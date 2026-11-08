/**
 * The chat's Display settings that name models (Settings > Display), applied on the status page as in the chat's
 * model picker (owner, 2026-11-08): raw model names, quantization, organization name, model tags (here the runtime
 * and classifier tags) and grouping by family (the tabs of one family side by side, the overflow list under family
 * headings). The speed card and the bench chart keep their names: they are the images people share.
 */
import type { ModelView } from './status-store.svelte';
import { SETTINGS_KEYS } from '$lib/constants';
import { settingsStore } from '$lib/stores';
import { groupModelFamilies } from '$lib/utils/model-families';

export interface RdnaDisplay {
	/** the router's model id instead of the file's NAME */
	raw: boolean;
	quant: boolean;
	org: boolean;
	/** the runtime and classifier tags */
	tags: boolean;
	families: boolean;
}

/** The settings as they stand (reactive inside $derived) */
export function rdnaDisplay(): RdnaDisplay {
	const c = settingsStore.config;

	return {
		families: (c[SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY] as boolean | undefined) ?? true,
		org: (c[SETTINGS_KEYS.SHOW_MODEL_ORG_NAME] as boolean | undefined) ?? true,
		quant: (c[SETTINGS_KEYS.SHOW_MODEL_QUANTIZATION] as boolean | undefined) ?? true,
		raw: (c[SETTINGS_KEYS.SHOW_RAW_MODEL_NAMES] as boolean | undefined) ?? false,
		tags: (c[SETTINGS_KEYS.SHOW_MODEL_TAGS] as boolean | undefined) ?? true
	};
}

/** A model's name as the settings show it: the router id when raw, else its NAME (with the quant where two
 * tabs would read the same, unless the quant is hidden) */
export function modelDisplayName(m: ModelView, d: RdnaDisplay): string {
	if (d.raw) return m.id;

	return d.quant ? m.label : m.naming?.name || m.label;
}

/** The models in tab order: one group, or one per family in the order the families first appear */
export function modelGroups(
	models: ModelView[],
	d: RdnaDisplay
): { key: string; models: ModelView[] }[] {
	if (!d.families) return [{ key: '', models }];

	return groupModelFamilies(models, (m) => m.naming?.name || m.id).map((f) => ({
		key: f.label,
		models: f.entries
	}));
}

/** A Hugging Face repo id with or without its organization */
export function repoDisplay(repo: string, d: RdnaDisplay): string {
	return d.org ? repo : repo.split('/').pop() || repo;
}

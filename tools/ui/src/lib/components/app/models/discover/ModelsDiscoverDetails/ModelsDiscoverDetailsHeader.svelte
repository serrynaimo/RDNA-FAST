<script lang="ts">
	import ModelCapabilityIcons from '../../ModelCapabilityIcons.svelte';
	import ModelOrgAvatar from '../../ModelOrgAvatar.svelte';
	import ModelsDiscoverDetailsHfHubStats from './ModelsDiscoverDetailsHfHubStats.svelte';
	import ModelsDiscoverDetailsMetadata from './ModelsDiscoverDetailsMetadata.svelte';
	import { ExternalLink } from '@lucide/svelte';
	import { base } from '$app/paths';
	import { Button } from '$lib/components/ui/button';
	import { makeItFastPrompt, quantsOfFiles, startRdnaChat } from '$lib/rdna/make-fast';
	import { HuggingFaceService } from '$lib/services';
	import { uiStore } from '$lib/stores';
	import type { HfModelDetailInfo, HfModelGguf } from '$lib/types/huggingface';
	import type { ModelModalities } from '$lib/types/models';
	import { orgOf } from '$lib/utils';

	interface Props {
		modelId: string;
		details: HfModelDetailInfo;
		gguf?: HfModelGguf;
		baseModels: string[];
		licenseTag: string | null;
		modalities: ModelModalities;
		hasTools: boolean;
		hasReasoning: boolean;
	}

	let {
		baseModels,
		details,
		gguf,
		hasReasoning,
		hasTools,
		licenseTag,
		modalities,
		modelId
	}: Props = $props();

	// Avatar shows the base model's org (e.g. the Qwen logo for a ggml-org GGUF)
	// with the quant org as a corner badge when they differ.
	let repoOrg = $derived(orgOf(details.id) || orgOf(modelId));
	let baseOrg = $derived(orgOf(baseModels[0]));
	let avatarOrg = $derived(baseOrg || repoOrg);
	let quantOrg = $derived(baseOrg && baseOrg !== repoOrg ? repoOrg : undefined);

	// [RDNA-FAST] "Make it RDNA-FAST!": a new chat in which the model that runs picks the quant, downloads it and saves
	// its recipe (src/lib/rdna/make-fast.ts)
	function makeItFast(): void {
		const repo = (details.id || modelId).split(':')[0];
		const quants = quantsOfFiles((details.siblings ?? []).map((s) => s.rfilename));

		uiStore.discoverModelsOpen = false;
		uiStore.manageModelsOpen = false;
		void startRdnaChat(makeItFastPrompt(repo, quants));
	}
</script>

<header class="space-y-3">
	<div class="flex flex-wrap items-start justify-between gap-3">
		<div class="flex min-w-0 items-center gap-2">
			<ModelOrgAvatar
				org={avatarOrg}
				{quantOrg}
				quantPositionClass="-bottom-1.5 -right-1.5"
				quantSize="h-6 w-6"
				size="h-12 w-12"
			/>

			<div class="min-w-0">
				<div class="flex items-center gap-2 @max-lg:flex-col @max-lg:items-start">
					<h1 class="truncate text-lg font-semibold @max-lg:w-full">{details.id ?? modelId}</h1>

					<ModelCapabilityIcons
						gapClass="gap-2"
						iconSize="h-4 w-4"
						{modalities}
						supportsThinking={hasReasoning}
						supportsToolUse={hasTools}
					/>
				</div>

				{#if baseModels.length}
					<div class="flex items-center gap-1">
						<span class="truncate text-xs text-muted-foreground">{baseModels.join(', ')}</span>

						<a
							aria-label="View base model on HuggingFace"
							class="shrink-0 text-muted-foreground transition-colors hover:text-foreground"
							href={HuggingFaceService.getModelUrl(baseModels[0])}
							rel="noopener noreferrer"
							target="_blank"
						>
							<ExternalLink class="h-3 w-3" />
						</a>
					</div>
				{/if}
			</div>
		</div>

		<div class="flex flex-col items-end gap-2">
			<Button
				class="h-auto bg-[#ed1c24] px-2.5 py-1.5 text-xs font-bold text-white hover:bg-[#c9141b]"
				onclick={makeItFast}
				size="sm"
			>
				Make it RDNA-FAST!
			</Button>

			<Button
				class="h-auto gap-1.5 px-2.5 py-1.5 text-xs"
				href={HuggingFaceService.getModelUrl(modelId)}
				rel="noopener noreferrer"
				size="sm"
				target="_blank"
				variant="tertiary"
			>
				<img alt="" class="h-3.5 w-3.5" src="{base}/recommended-mcp/huggingface.ico" />

				View on Hugging Face

				<ExternalLink class="size-3.5" />
			</Button>
		</div>
	</div>

	<ModelsDiscoverDetailsHfHubStats {details} />

	<ModelsDiscoverDetailsMetadata {details} {gguf} {licenseTag} {modelId} />
</header>

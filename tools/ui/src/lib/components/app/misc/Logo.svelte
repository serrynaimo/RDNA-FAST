<script>
	// [RDNA-FAST] the RF mark instead of the llama.cpp logo (owner, 2026-11-11); src/lib/assets/logo.svg is the app
	// icon tile the favicons and PWA icons are generated from
	import RdnaMark from '$lib/components/app/rdna/RdnaMark.svelte';
	let { class: className = '', style = '' } = $props();
</script>

<div class={className} {style}>
	<!-- fills a sized box (the sidebar's h-4.5 w-4.5), or takes --size where a caller sets it -->
	<RdnaMark style="width: var(--size, 100%); height: var(--size, 100%);" />
</div>

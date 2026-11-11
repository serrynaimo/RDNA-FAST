<script lang="ts">
	import SidebarNavigationActions from './SidebarNavigationActions.svelte';
	import RdnaSidebarStatus from '$lib/components/app/rdna/RdnaSidebarStatus.svelte';
	import { SIDEBAR_FOOTER_ITEMS } from '$lib/constants';
	import type { SidebarAction } from '$lib/enums';

	interface Props {
		onAction?: (action: SidebarAction) => void;
		/** [RDNA-FAST] the live server status card, last in the group (the open sidebar only) */
		showStatus?: boolean;
	}

	let { onAction, showStatus = false }: Props = $props();
</script>

<!-- the footer: managing the app around the chats, a group inset in the sidebar, its rows
     rounded concentric with it (see the nested surfaces in app.css); [RDNA-FAST] the server
     status card is its last item (owner, 2026-11-11) -->
<SidebarNavigationActions
	class="mx-2 mb-2 rounded-(--surface-group-radius) border border-border/20 bg-background/55 p-(--surface-group-padding) shadow-sm dark:bg-foreground/5"
	items={SIDEBAR_FOOTER_ITEMS}
	{onAction}
>
	{#if showStatus}
		<RdnaSidebarStatus github={false} inGroup />
	{/if}
</SidebarNavigationActions>

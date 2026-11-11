// A hidden desktop sidebar keeps its full width in the layout, floated under the page's
// left edge, so whatever it still lets the pointer hit there steals the page's clicks
// (#31971). Hiding it by inherited `visibility` alone is not enough: an open section's
// CollapsibleRegion sets `visibility: visible` again, and bits-ui's context-menu triggers
// put `pointer-events: auto` inline on the rows.

import '../../src/app.css';
import SidebarNavigationHarness from './components/SidebarNavigationHarness.svelte';
import { SETTINGS_KEYS } from '$lib/constants';
import { settingsStore, uiStore } from '$lib/stores';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const GRID_STEP_PX = 8;

/** Elements of the sidebar the pointer lands on over its box, the expand toggle aside. */
function hitTargetsInSidebar(aside: HTMLElement): Element[] {
	const toggle = aside.querySelector('[aria-label="Expand navigation"]');
	const box = aside.getBoundingClientRect();
	const hits = new Set<Element>();

	for (let x = box.left + 1; x < box.right; x += GRID_STEP_PX) {
		for (let y = box.top + 1; y < box.bottom; y += GRID_STEP_PX) {
			const el = document.elementFromPoint(x, y);

			if (el && aside.contains(el) && !toggle?.contains(el)) hits.add(el);
		}
	}

	return [...hits];
}

describe('hidden sidebar', () => {
	let wasExpanded: boolean;
	let wasAlwaysShown: boolean;

	beforeEach(async () => {
		await page.viewport(1280, 900);
		wasExpanded = uiStore.isSidebarExpanded;
		// [RDNA-FAST] the sidebar stays open on desktop by default here (settings.constants.ts); this is about the hidden one
		wasAlwaysShown = settingsStore.config[SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP] as boolean;
		settingsStore.updateConfig(SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP, false);
	});

	afterEach(() => {
		uiStore.isSidebarExpanded = wasExpanded;
		settingsStore.updateConfig(SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP, wasAlwaysShown);
	});

	it('lets clicks over its strip through to the page', async () => {
		uiStore.isSidebarExpanded = false;
		render(SidebarNavigationHarness);

		const aside = await vi.waitFor(() => {
			const el = document.querySelector<HTMLElement>('aside');

			if (!el) throw new Error('sidebar not rendered');

			return el;
		});

		// an open section is what used to bring the rows back to the pointer
		await vi.waitFor(() =>
			expect(aside.querySelector('.collapsible-region-content')).not.toBeNull()
		);
		// its width settles over a transition; the strip's area is what counts
		await vi.waitFor(() => expect(aside.getBoundingClientRect().width).toBeGreaterThan(48));

		expect(hitTargetsInSidebar(aside).map((el) => el.outerHTML.slice(0, 120))).toEqual([]);
	});
});

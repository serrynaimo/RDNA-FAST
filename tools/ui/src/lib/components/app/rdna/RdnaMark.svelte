<script lang="ts">
	/**
	 * RDNA-FAST's RF mark: the wordmark's R (silver) with its F (red) dropped and slid into the R's leg, Exo 2 Black
	 * Italic outlines (no font needed), the wordmark's gradients and outlines, optically centred in a square. Fills its
	 * box; the colours are the wordmark's tokens (rdna.css), so the R follows the theme. The outline is drawn as a
	 * stroked copy under each fill (no paint-order, for every renderer).
	 */
	import '$lib/rdna/rdna.css';
	import { RDNA_NAME } from '$lib/rdna/constants';

	interface Props {
		class?: string;
		style?: string;
	}

	let { class: className = '', style = '' }: Props = $props();

	const uid = $props.id();
</script>

<svg
	aria-label={RDNA_NAME}
	class="rf-mark {className}"
	role="img"
	{style}
	viewBox="-128.2 -958.7 1313.3 1313.3"
>
	<defs>
		<linearGradient id="{uid}-silver" x1="0" x2="0" y1="0" y2="1">
			<stop offset="0" style="stop-color: var(--rf-wm-silver-0);" />
			<stop offset="0.45" style="stop-color: var(--rf-wm-silver-1);" />
			<stop offset="0.55" style="stop-color: var(--rf-wm-silver-2);" />
			<stop offset="1" style="stop-color: var(--rf-wm-silver-3);" />
		</linearGradient>

		<linearGradient id="{uid}-red" x1="0" x2="0" y1="0" y2="1">
			<stop offset="0" style="stop-color: var(--rf-wm-red-0);" />
			<stop offset="0.55" style="stop-color: var(--rf-wm-red-1);" />
			<stop offset="1" style="stop-color: var(--rf-wm-red-2);" />
		</linearGradient>

		<path
			id="{uid}-r"
			d="M387 -700Q483 -700 547 -680Q611 -660 639 -612.5Q667 -565 656 -482Q651 -443 635.5 -409.5Q620 -376 588 -354Q556 -332 500 -326L499 -319Q516 -317 535.5 -306.5Q555 -296 570.5 -273Q586 -250 590 -208L616 0H392L374 -209Q373 -238 365 -245Q357 -252 334 -252Q257 -253 209 -253Q161 -253 128.5 -254Q96 -255 65 -257L81 -405H355Q385 -405 401.5 -411Q418 -417 426.5 -434.5Q435 -452 440 -484Q444 -508 440 -522Q436 -536 423 -541.5Q410 -547 385 -547Q278 -547 206.5 -545.5Q135 -544 110 -541L108 -690Q153 -695 192.5 -697Q232 -699 278 -699.5Q324 -700 387 -700ZM320 -690 223 0H11L108 -690Z"
		/>

		<path
			id="{uid}-f"
			d="M261 -694Q350 -694 442 -692Q534 -690 603 -679L579 -528H349Q322 -528 310 -519.5Q298 -511 295 -488L227 0H9L85 -541Q95 -612 143 -653Q191 -694 261 -694ZM85 -411H525L507 -267H64Z"
			transform="translate(514 140)"
		/>
	</defs>

	<g stroke-linejoin="round" stroke-width="62">
		<use href="#{uid}-r" fill="none" style="stroke: var(--rf-wm-silver-stroke);" />
		<use href="#{uid}-r" fill="url(#{uid}-silver)" />
		<use href="#{uid}-f" fill="none" style="stroke: var(--rf-wm-red-stroke);" />
		<use href="#{uid}-f" fill="url(#{uid}-red)" />
	</g>
</svg>

<style>
	.rf-mark {
		display: block;
		flex-shrink: 0;
	}
</style>

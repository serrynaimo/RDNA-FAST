# Expert handoff: MoE models bigger than the card

Mixture-of-experts models such as Qwen3.6-35B-A3B, Qwen3.8-Flash-Next or Gemma 4 26B-A4B use only a few of their
experts for each token. Expert handoff keeps all experts in system RAM and the most used of them in a VRAM cache. For
every token and layer, the GPU computes the experts it holds while the CPU computes the others from RAM at the same
time. The cache follows the conversation: experts that keep being used move into VRAM. It works similar to Strata.

## Quick start

```sh
llama-server -m model.gguf --expert-cache auto
```

That is all most models need. The cache takes the VRAM the rest of the model leaves free, the experts it doesn't hold
are copied into RAM, and every CPU core but one helps compute them. The settings below get more out of a particular
machine and model.

## Results

RX 7900 XT 20 GB, Ryzen 9 7900 (12 cores), 32 GB DDR5, Windows 11, ROCm 10.1, through `llama-server`.

**Qwen3.8-Flash-Next Q2_0** (Swift 1.5 Flash-Next, 66.5 GB): 36% of its experts fit in VRAM (28% next to the MTP
draft). With 24-25 GiB of RAM free, the RAM copy held all 20 GiB of the others (21 of 23 GiB next to the draft); what
doesn't fit there is read from the SSD ([Streaming from the SSD](#streaming-from-the-ssd)). Sampling at temperature 1,
2026-11-11:

| | Default | With MTP drafting |
| --- | ---: | ---: |
| Decode, first answer after a load | **43.9 t/s** | 34.8 t/s |
| Decode, short answers | 48.4-48.6 t/s | **49.9-55.2 t/s** |
| Decode, copying a 1,200-token text | 55.2 t/s | **76.8 t/s** |
| Decode, after a 4k-token prompt | 39.7 t/s | **41.2 t/s** |
| 700-token prompt | **2.2 s** | 2.6 s |
| 3.8k-token prompt | **429 t/s** | 416 t/s |
| 15k-token prompt | **452-455 t/s** | 427-435 t/s |

The same model in an agent session, on the build of 2026-11-10: the two-agent session of the
[README's speed test](../README.md#speed) (an orchestrator that grows from 4k to 41k tokens, a 10k-token worker), with
MTP drafting, the better of two sessions:

| | |
| --- | ---: |
| Cold 4k-token prompt | 326 t/s |
| 4.3k new tokens at 17k depth | 368 t/s |
| 4.0k new tokens at 41k depth | 364 t/s |
| Decode at 4k / 17k / 41k depth | 35.1 / 37.4 / 37.2 t/s |
| Code edit that copies the prompt | 53.3 t/s |

**Qwen3.6-35B-A3B UD-Q4_K_S** (20 GB of weights, more than the card holds next to its context), greedy, 2026-11-10:

| | Decode | 4k-token prompt |
| --- | ---: | ---: |
| `--n-cpu-moe 8` (8 layers' experts on the CPU) | 68-73 t/s | 3656 t/s |
| Expert handoff, `--expert-cache auto` | **107-108 t/s** | **4718 t/s** |

The cache holds 95% of the experts and serves 99.9% of the ones the model routes to.

**Gemma 4 26B-A4B Q4_0** with only half or a quarter of its experts in VRAM, as on a smaller card (whole in 20 GB:
106 t/s), greedy, 2026-11-10:

| Experts in VRAM, cache starts from | Cache stays as it started | Cache follows the conversation |
| --- | ---: | ---: |
| Half, an even share of each layer | 60-61 t/s | **69-82 t/s** |
| Half, a saved profile | 73-86 t/s | 70-83 t/s |
| A quarter, a saved profile | 58-64 t/s | **61-68 t/s** |

**Quality.** The CPU computes its experts in a different order and precision than the GPU would, so greedy answers
match the all-GPU answers for a while and then diverge into different but coherent text. Against the model fully in
VRAM, Gemma 4 with half its experts on the CPU has a KL divergence of about 0.08, the same as `--n-cpu-moe`.

## Settings for the best speed

| Flag | Default | What it does |
| --- | --- | --- |
| `--expert-cache N\|auto` | off | Turns expert handoff on with a VRAM cache of `N` experts (in units of the model's largest expert), or `auto` for the free VRAM less `--vram-reserve-mib`. |
| `--moe-cache-mib N` | off | The same with a cache of `N` MiB. |
| `--vram-reserve-mib N` | 1024 | VRAM that `auto` leaves free beyond the compute buffers and a draft model (MTP, DFlash and the like), which are counted first. |
| `--expert-profile FILE` | none | Starts the cache with the experts this file ranks most used ([Profiles](#profiles)). |
| `--expert-profile-save FILE` | none | Writes the routing counts of this run (plus those of `--expert-profile`) to `FILE` every 30 s of use and at exit. |
| `--ram-keep-gib N` | 4 | RAM left free for everything else. The experts the VRAM cache misses are copied into the rest, the most used first. |
| `--resident-budget-gib N` | all that fits | At most `N` GiB for that RAM copy. |
| `--expert-copy DIR` | none | Reads the experts outside VRAM and the RAM copy from copies of the model files in `DIR` (a folder or a drive), made there at load when missing or out of date ([Copy the model](#copy-the-model-for-the-reads)). |
| `--pool-workers N` | one per physical core but one | CPU threads that compute the experts the cache misses. |
| `--adapt-every N` | 4 | Every `N` decode steps, the most used experts outside the cache replace the least used ones in it (0 keeps the cache as it started). |
| `--adapt-swaps N` | 96 | The most experts swapped per round. |
| `--adapt-decay F` | 0.7 | How fast old use is forgotten: usage is multiplied by `F` after each round. |

- **RAM decides the speed.** The CPU computes its experts fastest from the RAM copy, and every expert that doesn't fit
  there is read from the model file, from the SSD if the system has pushed it out. Give the model all the RAM you can:
  close other programs, and lower `--ram-keep-gib` (say to 2) on a machine that runs nothing else. Flash-Next on a
  20 GB card has about 20 GB of experts outside VRAM, so it runs best with 40 GB of RAM or more. With 21 GB of RAM
  free for it, `--ram-keep-gib 2` held 18 instead of 16 GiB of them: the 700-token prompt took 3.3-3.6 instead of
  4.5 s and short answers ran about 15% faster, other answers as fast as with the default (2026-11-12), leaving the
  rest of the system about 2 GiB.
- **Save a profile for your work.** Without one, the cache starts with an even share of every layer's experts and the
  first answers are slower until it has learned. See [Profiles](#profiles).
- **Drafting: MTP, not n-gram.** The MTP draft (`-md mtp-Qwen3.8-Flash-Next-Q4_0.gguf --spec-type draft-mtp
  --spec-draft-n-max 3 --spec-draft-p-min 0.5`) makes copied text about 40% faster on the 20 GB card and short answers
  up to 14% faster. It takes VRAM that the expert cache then lacks, so long prompts are about 5% slower (see Results);
  on a card with VRAM to spare that cost goes away. N-gram drafting (`--spec-type ngram-*`) halved the speed of copied
  text: every draft it checks costs the CPU more than it saves.
- **Keep the default batch size.** `-ub 4096` makes the compute buffers larger, which leaves fewer experts in VRAM: on
  the 20 GB card prompts and answers both got slower.
- **Workers.** Lower `--pool-workers` only if other CPU-heavy programs share the machine.
- **If you choose the quant:** experts in Q2_0, Q4_0, Q8_0, IQ4_NL, MXFP4, IQ4_XS, Q4_K, Q5_K or Q6_K make prompts
  faster than IQ2, IQ3, Q2_K or Q3_K ones: the CPU computes them up to twice as fast when several tokens share an
  expert. Nothing to set; the server uses this wherever the CPU supports it (AMD Zen 4 and newer, recent Intel Xeons).
- **Qwen3.8-Flash-Next:** add `-ot per_layer_token_embd=CPU`. Its 28.8 GB per-layer token embedding table stays on the
  SSD and is read a few rows at a time; keep `--lazy-mode` at its default (`off` would load the table into RAM).
- **Placement.** `--cpu-moe` / `--n-cpu-moe N` still decide which layers keep their experts in RAM, and expert handoff
  splits exactly those. Without them, it splits all of them.
- **Several GPUs.** With the layer split (the default), each GPU caches the experts of its own layers in its free VRAM,
  and one set of CPU threads serves them all.

## Profiles

A profile lists how often the model routed to each of its experts, layer by layer. At start, the cache fills with the
experts it ranks highest. Without one, a fresh start serves far fewer experts from VRAM until the cache has learned:
Flash-Next on a 20 GB card served 34% of its routed experts from VRAM on the first answer, and 63-73% once the cache
had followed a few answers.

### Making a profile

1. Start the model with a file to write, plus your usual flags:

   ```sh
   llama-server -m model.gguf --expert-cache auto --expert-profile-save my-profile.txt
   ```

2. Use it for the work the profile is meant for: the agent or coding sessions, chats and languages you expect. The
   counts come from the tokens the model writes and from short prompts; long prompts aren't counted. A few thousand
   written tokens give a usable ranking, and more is better.
3. The file is written every 30 s of use and at exit. A server that sits idle writes nothing, so after the last
   answer, stop the server normally before you copy the file.

To keep refining one profile, pass the same file to both flags. Each run then adds its counts to the ones it loaded:

```sh
llama-server -m model.gguf --expert-cache auto --expert-profile my-profile.txt --expert-profile-save my-profile.txt
```

The counts only ever add up, so after weeks of one kind of work, a new kind barely moves the ranking. Start a new file
when the work changes.

### Using a profile someone else made

Pass it to `--expert-profile` alone. The file is then only read, never changed.

[models/expert-profiles](../models/expert-profiles) holds the profiles we ship:

| File | Model | Made from |
| --- | --- | --- |
| `Swift-1.5-Flash-Next.txt` | Swift 1.5 Flash-Next (Qwen3.8-Flash-Next), made with its Q2_0 pack | About 2,400 written tokens of everyday agent use, 2026-11-07 |

```sh
llama-server -m Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --expert-cache auto \
  -ot per_layer_token_embd=CPU --expert-profile models/expert-profiles/Swift-1.5-Flash-Next.txt
```

- **Hardware doesn't matter.** A profile holds no sizes. Each machine fills as many of the top-ranked experts as fit
  its own VRAM, so the same file works on a 16 GB card, a 32 GB card or several GPUs.
- **The model does.** Expert numbers mean nothing in another model, and a fine-tune may route differently from its
  base. Profiles aren't checked against the model when they load, so use one only with the model it was made for. Its
  first line names the architecture and the number of layers.
- **Other quants of the same model should work.** The routing depends on the model, not on how its experts are
  stored, so the experts a quant picks most often are almost always the same. This hasn't been measured yet.
- **The workload matters most.** A profile from coding sessions starts a chat with experts that suit code. The cache
  still follows the conversation either way, so a poor match only costs the first answers.

### The file

Plain text: a comment line, then one line per layer with the layer number, its number of experts, and one count per
expert:

```text
# expert handoff routing counts, arch qwen4exp, 48 layers
layer 0 512 1843 0 27 ...
layer 1 512 95 412 3 ...
```

To merge profiles from different machines or kinds of work, add their counts layer by layer and expert by expert.
Scale a file first if its share of the mix should be smaller than its raw counts.

## How it works

- **Answers.** For every layer, the GPU computes the experts in its cache while CPU threads compute the others from
  RAM, and the GPU adds their sum. Layers whose experts are all in VRAM never involve the CPU.
- **The caches follow the conversation.** Every few steps, the most used experts outside VRAM replace the least used
  ones inside, and the RAM copy swaps in the experts the CPU keeps reading from the model file. Experts that move into
  VRAM give their RAM slots to the ones that left it, so a long prompt finds nearly every expert in VRAM or RAM.
- **Prompts.** Short prompts take the same path when the CPU is sooner. Longer ones stream every expert the VRAM cache
  misses to the GPU while earlier layers compute, from the RAM copy where it holds them.

## Streaming from the SSD

A model can be bigger than VRAM and RAM together: Flash-Next Q2_0 is 66.5 GB, run here on a 20 GB card with 32 GB of
RAM. The experts live in three places, fastest first:

1. **The VRAM cache:** the most used experts, chosen by the profile and then by the conversation.
2. **The RAM copy:** the next most used ones, copied from the model file once at load (up to eight readers in
   parallel, Flash-Next's 20.2 GiB in about 20 s) into RAM that is locked, so the system never pages it out.
3. **The model file:** everything else stays memory-mapped. The system reads it from the SSD when the CPU or an upload
   touches it and keeps it in its file cache while RAM allows.

What streams from the SSD, and when:

- **Answers.** The CPU computes an expert outside the RAM copy straight from the mapped file. Once read it usually
  stays in the file cache, and the RAM copy swaps in the experts the CPU keeps going back to.
- **Prompts.** Every batch of a long prompt sends the experts the VRAM cache misses to the GPU. Sixteen copy threads
  read them from the RAM copy, or else from the file with many SSD reads in flight, into pinned buffers that go to the
  GPU while earlier layers compute; a 512 MiB ring in VRAM lets them run more than a layer ahead. When Flash-Next's
  experts had to come from the file, its first 700-token prompt after a load took 8.2 s this way (12.6 s with four
  threads); from the whole RAM copy it took 2.2 s.
- **Per-layer token embeddings.** Flash-Next's 28.8 GB table (`-ot per_layer_token_embd=CPU`) is never loaded. For each
  batch the server reads ahead the rows of the whole prompt, then reads exactly those rows with parallel reads in file
  order, each row once. Compared with letting the system fault the table in page by page, this took a 700-token prompt
  from 7.0 to 4.5 s and a 3.8k-token prompt from 197 to 328 t/s. It turns on by itself for tables over 4 GiB.

What helps:

- **An NVMe SSD, uncompressed.** The file is read in many small pieces at once. NTFS compression on the model drive
  capped reads at 143 MB/s on this machine.
- **RAM.** Every expert in the RAM copy is one the SSD doesn't have to deliver. Memory-mapped experts the system has
  pushed out of the process cost the CPU about 5x as long each (175 vs 35 µs on Flash-Next).
- **Patience after a load.** Experts outside the RAM copy come from the SSD until the file cache has them again:
  before the RAM copy, the first answers on Flash-Next came at 12.5 t/s. With every expert outside VRAM in the RAM
  copy, as in [Results](#results), the first answer after a load ran at 43.9 t/s.

### Copy the model for the reads

On Windows, every read of a file that a process has mapped takes about 10 ms longer, and the server maps the model.
`--expert-copy DIR` reads the experts outside VRAM and the RAM copy from copies of the model files in `DIR` instead,
which nobody maps: the prompt uploads and the CPU read them there through the file cache. A folder on another SSD
adds that drive's speed; one on the same drive works too.

The server makes the copies at load, before it maps the model: when a file is missing in `DIR` or differs in size or
modification time, it copies it there (66.5 GB for Flash-Next took 23 s on PCIe 4 to PCIe 3), as long as the drive
keeps 8 GiB free afterwards. Later loads use them as they are. To remove them, delete them.

Flash-Next with MTP on the 20 GB card, 21 GB of RAM free for it (RAM copy 16 of 22.8 GiB), 2026-11-12:

| | Mapped file | `--expert-copy` |
| --- | ---: | ---: |
| 700-token prompt | 4.5-4.6 s | **3.0-3.1 s** |
| 3.8k-token prompt | 310-385 t/s | **408-414 t/s** |
| 15k-token prompt | 411-421 t/s | 411-414 t/s |
| Decode, short answers | 32.4-37.3 t/s | **40.9-41.4 t/s** |
| Decode, copying a 1,200-token text | **70.5-73.0 t/s** | 56.1-60.2 t/s |
| Decode, after a 3.8k-token prompt | **40.2-40.6 t/s** | 33.2-33.7 t/s |

Longer answers after a prompt are slower: each token copies the experts it misses out of the file cache again, where
the mapping reads them in place. With all the experts outside VRAM in the RAM copy, the copy has nothing to add.

## Limits

- Several GPUs work with the layer split, not with `--split-mode tensor`. The multi-GPU path has not yet run on two
  GPUs.
- Experts with Hadamard-rotated weights, and GroveMoE, run without the handoff, as with `--n-cpu-moe`.
- A long prompt sends the experts outside VRAM to the GPU for every batch, so it is slower than with all experts in
  VRAM, and slower still when they have to come from the SSD.
- With the cache following the conversation, answers aren't bit-identical from run to run. `--adapt-every 0` keeps
  the cache fixed.
- Keep the experts memory-mapped (the default load mode).
- On Windows, loading takes about a minute longer: the first GPU work after the RAM copy is locked waits for the driver
  (the server's warmup takes it).

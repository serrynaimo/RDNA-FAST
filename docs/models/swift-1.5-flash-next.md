# Swift 1.5 Flash-Next

UkisAI's Swift 1.5 fine-tune of Qwen3.8-Flash-Next: a 125B mixture-of-experts model (512 experts, 10 used per token,
about 6B active) with an MTP draft head. Far more knowledge than the 27B for answers that need it, run on a 20 GB card
through [expert handoff](../expert-handoff.md): the most used experts in VRAM, the next ones in RAM, the rest read
from the SSD.

**Needs:** a 20 GB card, 32 GB of RAM (40 GB or more is faster) and the model on an NVMe SSD without NTFS
compression. Prompts and answers run at about half the 27B's speed.

## Files

- Swift 1.5 Flash-Next GSQ-RCO Q2_0, two parts, 66.5 GB:
  [part 1](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/main/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf),
  [part 2](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/main/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf).
  Keep both in one folder and pass the first.
- [`mtp-Qwen3.8-Flash-Next-Q4_0.gguf`](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF/blob/main/mtp-Qwen3.8-Flash-Next-Q4_0.gguf)
  (2.2 GB), the MTP draft.
- The expert profile we ship, [`models/expert-profiles/Swift-1.5-Flash-Next.txt`](../../models/expert-profiles/Swift-1.5-Flash-Next.txt):
  the cache starts with the experts everyday agent work uses most.

## Run

```
set "PATH=C:\TheRock\rocm-10.1\bin;%PATH%"
set GGML_CUDA_NO_PINNED=1
set LLAMA_MAX_QUEUED=3

llama-server -m Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf ^
  -md mtp-Qwen3.8-Flash-Next-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.5 ^
  -dev ROCm0 -ngl 999 -fa on --jinja --gpu-keepalive-ms 2000 ^
  --expert-cache auto --vram-reserve-mib 2048 -ot per_layer_token_embd=CPU ^
  --expert-profile models\expert-profiles\Swift-1.5-Flash-Next.txt ^
  -c 65536 -np 1 -b 2048 -ub 2048 --ctx-checkpoints 8 --checkpoint-min-step 2048 ^
  --cache-ram 0 --cache-disk D:\llama-cache --cache-disk-limit 65536 --cache-disk-checkpoints 4 ^
  --chat-template-kwargs "{\"reasoning_effort\":\"low\"}" ^
  --reasoning-budget 4096 --reasoning-budget-message "I have thought enough, so I will act on my plan now." ^
  --reasoning-preserve --temp 1 ^
  --alias qwen/qwen3.8-flash-next --host 0.0.0.0 --port 1234 --api-key-file api-keys.txt
```

A load takes about two minutes: the experts the VRAM cache misses are copied into locked RAM (about 20 s), and on
Windows the first GPU work after that waits for the driver. Every expert-handoff flag is in
[expert-handoff.md](../expert-handoff.md#settings-for-the-best-speed).

## Speed

RX 7900 XT 20 GB, Ryzen 9 7900, 32 GB DDR5, ROCm 10.1, the command above (with the MTP draft), November 2026.

| | |
| --- | ---: |
| 700-token prompt | 2.6 s |
| 3.8k / 15k-token prompt | 416 / 427-435 t/s |
| Answers, short | 49.9-55.2 t/s |
| Answers, copying a 1,200-token text | 76.8 t/s |
| Answers after a 4k-token prompt | 41.2 t/s |
| Two-agent session: prompts at 4k / 17k / 41k depth | 326 / 368 / 364 t/s |
| Two-agent session: answers at 4k / 17k / 41k depth | 35.1 / 37.4 / 37.2 t/s |

## What we learned

- **RAM decides the speed.** The CPU computes the experts the VRAM cache misses, fastest from the RAM copy. Close
  other programs; on a machine that runs nothing else, `--ram-keep-gib 2` holds 2 GiB more experts in RAM and made
  short answers about 15% faster. With 40 GB of RAM or more, all of them fit.
- **Start from a profile.** Without one, the first answers serve a third of the experts from VRAM until the cache
  has learned; with one, the cache starts where everyday agent work needs it. To make your own, see
  [Profiles](../expert-handoff.md#profiles).
- **MTP yes, n-gram no.** The MTP draft makes copied text about 40% faster and short answers up to 14% faster.
  N-gram drafting halves the speed of copied text here: every draft it checks costs the CPU more than it saves.
- **One slot, f16 cache, 64k.** Its sparse attention wants the f16 KV cache, and a second slot on the shared pool
  aborted, so requests queue. 64k of context leaves VRAM to the expert cache.
- **Keep the model memory-mapped** (no `--load-mode dio`): dio would copy all experts into RAM. The 28.8 GB per-layer
  token embedding table stays on the SSD (`-ot per_layer_token_embd=CPU`) and is read a few rows at a time.
- **`--expert-copy DIR`** reads the experts from an unmapped copy of the model: prompts and short answers got faster,
  long answers slower. Worth trying with the copy on a second SSD.
- **Keep `-ub 2048`.** A bigger batch makes the compute buffers larger and leaves fewer experts in VRAM.
- **Vision:** the repository has a `mmproj` for this model; we haven't tested it.

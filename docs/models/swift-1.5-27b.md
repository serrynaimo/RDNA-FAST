# Swift 1.5 27B

[UkisAI's Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b), a fine-tune of Qwen3.8-27B: a dense 27B
model with its own multi-token-prediction (MTP) head and vision. Our default for agent work: the fastest model here,
tested the longest, with the most reliable tool calls.

**Needs:** a 20 GB card (RX 7900 XT, RX 7900 XTX, Radeon PRO W7800 and up). The whole model and 96k tokens of context
stay in VRAM.

## Files

- [`Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF/blob/main/Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf)
  (15.5 GB). Against the model's Q8_0 it picks the same top token 93.5% of the time (mean KL divergence 0.021,
  wikitext-2).
- [`mmproj-Qwen3.8-27B-BF16.gguf`](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF/blob/main/mmproj-Qwen3.8-27B-BF16.gguf),
  the base model's vision projector.

Alternative: our ROCmFPX quant of the same model, `Swift-1.5-Qwen3.8-27B-ROCmFP4-FP6` (16.1 GB). Prompts run about 2%
faster, at 92.8% the same top token (KL divergence 0.026):
[recipe and tensor policy](../rocmfpx/swift-1.5-qwen3.8-27b-rocmfp4-fp6.md).

## Run

```
set "PATH=C:\TheRock\rocm-10.2\bin;%PATH%"
set GGML_CUDA_NO_PINNED=1
set LLAMA_MAX_QUEUED=3
set ROCMFPX_DRAFT_VOCAB=24576

llama-server -m Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf --mmproj mmproj-Qwen3.8-27B-BF16.gguf ^
  -dev ROCm0 -ngl 999 -fa on --jinja --load-mode dio --gpu-keepalive-ms 2000 ^
  -c 98304 -np 2 --kv-unified --kv-unified-reserve 12288 --no-cache-idle-slots ^
  -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 -b 2048 -ub 2048 --no-mmproj-offload ^
  --ctx-checkpoints 8 --checkpoint-min-step 2048 ^
  --cache-ram 0 --cache-disk D:\llama-cache --cache-disk-limit 65536 --cache-disk-checkpoints 4 ^
  --spec-type "draft-mtp,ngram-map-k4v" --spec-draft-n-max 4 --spec-draft-p-min 0.60 ^
  --chat-template-kwargs "{\"reasoning_effort\":\"low\"}" ^
  --reasoning-budget 4096 --reasoning-budget-message "I have thought enough, so I will act on my plan now." ^
  --reasoning-preserve --temp 1 ^
  --alias qwen/qwen3.8-27b --host 0.0.0.0 --port 1234 --api-key-file api-keys.txt
```

Every flag and variable is explained in [FLAGS.md](../FLAGS.md).

## Speed

RX 7900 XT 20 GB, ROCm 10.1, the command above. Details and the comparison with LM Studio: [README](../../README.md#speed).

| | |
| --- | ---: |
| Two-agent session (4k to 41k tokens, 14 requests) | 93 s (LM Studio: 130 s) |
| Prompts at 4k / 17k / 41k depth | 1,035 / 916 / 710 t/s |
| Answers at 4k / 17k / 41k depth | 70.1 / 66.7 / 65.3 t/s |
| Code edit that copies the prompt | 315 t/s |
| Answers, prose / code (peak) | 61.2 / 95.8 t/s |

## What we learned

- **IQ4_XS over K-quants.** On AMD cards the IQ quants are smaller than K-quants of similar quality and decode faster:
  IQ4_XS decodes about 40% faster than Q4_K_M and leaves room for the context.
- **MTP plus n-gram drafting.** The MTP head drafts every answer; where an answer repeats text from the prompt, as code
  edits do, n-gram drafting copies it in long runs (up to 5x faster). Keep `ROCMFPX_DRAFT_VOCAB=24576`: the draft
  ranks only the most used part of the vocabulary, which makes each draft step cheaper.
- **Reasoning effort low.** On 26 replayed chief-of-staff agent turns, `low` passed 25 and medium or xhigh did no
  better, at more thinking per reply (the template's default, xhigh, took three times as long on routine tasks).
  Don't switch thinking off: without it the model made real errors on simple tasks. The reasoning budget stops the
  rare reply that thinks for thousands of tokens.
- **Temperature 1.** Qwen3.8's answers get worse at greedy sampling; keep `--temp 1`.
- **Vision on the CPU.** `--no-mmproj-offload` keeps the projector off the card, which leaves about 1.6 GB of VRAM to
  the context. An image then takes 7-12 s to the first token instead of 3-5.5 s.
- **Two slots on one pool.** `-np 2 --kv-unified` lets a chat turn run beside a worker agent; the reserve makes a
  request wait instead of overflowing the shared cache.
- **Context.** 96k leaves room for a Windows desktop beside the model. With little else on the card, 128k
  (`-c 131072`) fits too; a context that spills into shared memory makes every prompt up to 2x slower.
- **`-ub 2048`** made prompts 3-4% faster than 1024 at the same VRAM.

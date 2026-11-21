# Inkling 2 Mini

[Thinking Machines' Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): a 26B mixture-of-experts
model (128 experts, 8 used per token) with vision. Its middle layers (6 to 18 of 24) run three times per token, so it
thinks with 48 layers' worth of compute from the weights of 24. Most of its attention looks at the last 1,024 tokens,
so a long context costs little memory.

**Needs:** a 20 GB card. The whole model and 128k tokens of context stay in VRAM.

## Files

- [`Inkling-2-Mini-IQ4_NL.gguf`](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/blob/main/Inkling-2-Mini-IQ4_NL.gguf)
  (17.6 GB) from ggml-org.
- [`mmproj-Inkling-2-Mini-Q8_0.gguf`](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/blob/main/mmproj-Inkling-2-Mini-Q8_0.gguf)
  (0.35 GB) for vision.

Use ggml-org's files: they carry the `inkling2_loop` architecture that llama.cpp released. Some early GGUFs of this
model (unsloth's among them) were made before that and don't load.

## Run

```
set "PATH=C:\TheRock\rocm-10.2\bin;%PATH%"
set GGML_CUDA_NO_PINNED=1
set LLAMA_MAX_QUEUED=3

llama-server -m Inkling-2-Mini-IQ4_NL.gguf --mmproj mmproj-Inkling-2-Mini-Q8_0.gguf --no-mmproj-offload ^
  -dev ROCm0 -ngl 999 -fa on --jinja --load-mode dio --gpu-keepalive-ms 2000 ^
  -c 131072 -np 2 --kv-unified --kv-unified-reserve 12288 --no-cache-idle-slots ^
  -ctk q4_0 -ctv q4_0 -b 2048 -ub 512 ^
  --ctx-checkpoints 8 --checkpoint-min-step 2048 ^
  --cache-ram 0 --cache-disk D:\llama-cache --cache-disk-limit 65536 --cache-disk-checkpoints 4 ^
  --reasoning-budget 4096 --reasoning-budget-message "I have thought enough, so I will act on my plan now." ^
  --alias inkling-2-mini --host 0.0.0.0 --port 1234 --api-key-file api-keys.txt
```

Sampling comes from the model file (Thinking Machines' defaults), so there is no `--temp` here.

## Speed

RX 7900 XT 20 GB, ROCm 10.1, the command above, November 2026.

| | |
| --- | ---: |
| 37.7k-token prompt | 591 t/s |
| Answers | 38-42 t/s |
| Answers after the 37.7k-token prompt | 37.1 t/s |
| VRAM | 18.6 GB, nothing spilled to shared memory |

## What we learned

- **128k needs RDNA-FAST's attention for long batches.** Inkling adds a learned bias per head to attention, by the
  distance between tokens. llama.cpp builds it for all 24 heads at once, which takes 51 GB at 128k tokens of context
  and 2048-token batches, and 13 GB at 64k and 1024 (which ran, 10 GB of it in system RAM, at 164 t/s on a long
  prompt). RDNA-FAST builds it one head at a time for long batches, with the same results, which makes 128k fit.
- **Keep `-ub 512` at 128k.** The batch size sets the size of that bias. Projected VRAM: 18.1 GB at `-ub 256`,
  18.5 GB at 512, 19.5 GB at 1024 (too close to the card's 20 GB beside a desktop).
- **No drafting.** The GGUF has no draft head, so every token is a full step: answers at about 40 t/s, against
  60-70 t/s for Swift 1.5 27B with MTP.
- **Tool calls work for plain requests.** Asked to "check the weather in Singapore", it calls the tool. Asked to say
  what it will do first and then call the tool, it writes the sentence and ends its turn without the call (0 of 3).
  Two things change that: `"tool_choice": "required"` makes it call every time (3 of 3, after the sentence), and a
  higher reasoning effort (`"chat_template_kwargs": {"reasoning_effort": "xhigh"}`) sometimes (1 of 3). An agent
  that narrates before it acts may stop early with this model.
- **Reasoning effort.** The chat template defaults to `high`; `none`, `minimal`, `low`, `medium` and `xhigh` are the
  other levels. We haven't measured what each does to agent work yet.
- **Vision on the CPU.** `--no-mmproj-offload` keeps the projector in RAM, which leaves the VRAM to the context.

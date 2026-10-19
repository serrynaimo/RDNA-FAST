# Qwen3.8-27B-pi ROCMFPX-MQ-Q4S

A 4-bit ROCmFP4 quantization of
[Qwen3.8-27B-pi](https://huggingface.co/bytkim/Qwen3.8-27B-pi) that promotes
the 61 most sensitive tensors to `Q6_0_ROCMFPX`. Qwen3.8-27B-pi is bytkim's
fine-tune of Qwen3.8-27B for tool-driven work in the
[Pi agent harness](https://github.com/earendil-works/pi): supervised training
on successful Pi sessions, then reinforcement learning on verified task outcomes
with a reward for reasoning economically at `low` and `medium` effort. Built
for the 20 GB RX 7900 XT: the promotions cost no decode speed, prefill speed or
VRAM.

Why this model, at `reasoning_effort: low`: on our tool-calling tasks it
finishes with 16% fewer output tokens and 9% less wall time than Swift 1.5, the
model this runtime used before, and it answered all 50 tasks of our set
correctly (see [Evidence](#evidence)). On pure reasoning it thinks a little
longer than Swift 1.5: faster where it can be, more careful where it has to be.

## Lane

- Public name: `Qwen3.8-27B-pi-ROCMFPX-MQ-Q4S`
- Policy: `Q4_0_ROCMFP4` bulk, 61 whole-tensor promotions to `Q6_0_ROCMFPX`
- Tensor policy file: [`qwen3.8-27b-mq-q4s.tensor-types.txt`](qwen3.8-27b-mq-q4s.tensor-types.txt)
- Tensor inventory: 360 F32, 445 `Q4_0_ROCMFP4`, 61 `Q6_0_ROCMFPX` (866 total)
- Size: 16,075,545,056 bytes (14.97 GiB), 4.70 BPW over the quantized tensors
- Loads only in ROCmFPX builds: the two tensor types do not exist in stock
  `llama.cpp`.

Promoted tensors:

| tensors | count | where |
| --- | ---: | --- |
| `output.weight` | 1 | |
| `blk.64.nextn.eh_proj.weight` | 1 | MTP head projection |
| `attn_k`, `attn_v`, `attn_output` | 3 x 17 | the 16 full-attention layers (3, 7, 11, ... 63) and the MTP block 64 |
| `ffn_down` | 8 | boundary layers 0, 1, 2, 3, 61, 62, 63, 64 |

Why this shape: on RDNA3 the K-quants decode about 60% slower than the ROCmFP4
kernel path, so a mixture that promotes to `Q5_K`/`Q6_K` (what the automatic
planner picks) pays for its quality in speed. `Q6_0_ROCMFPX` stays on the fast
path. The promoted set is the planner's own sensitivity ranking, re-expressed
in that type. The attention promotions are nearly free (GQA, 4 KV heads); the
eight `ffn_down` tensors carry most of the added size (0.33 GiB in total).

The policy was derived on another Qwen3.8-27B fine-tune and transfers
unchanged: tensor names and shapes are those of the `qwen35` 27B architecture,
and the pi file comes out within 192 bytes of the earlier Swift 1.5 one.

## Reproduction

Source: the BF16 GGUF (two shards) and the importance matrix
(`bytkim_Qwen3.8-27B-pi-imatrix.gguf`, 582 chunks) from
[`bartowski/bytkim_Qwen3.8-27B-pi-GGUF`](https://huggingface.co/bartowski/bytkim_Qwen3.8-27B-pi-GGUF).
The BF16 GGUF carries the MTP head (`nextn_predict_layers = 1`, the 15
`blk.64` tensors). Pass the first shard; the second is found automatically.
Never requantize from a low-bit file.

```bash
llama-quantize \
  --imatrix bytkim_Qwen3.8-27B-pi-imatrix.gguf \
  --tensor-type-file docs/rocmfpx/qwen3.8-27b-mq-q4s.tensor-types.txt \
  bytkim_Qwen3.8-27B-pi-bf16-00001-of-00002.gguf \
  Qwen3.8-27B-pi-ROCMFPX-MQ-Q4S.gguf \
  Q4_0_ROCMFP4 24
```

About 5 minutes on a 12-core CPU; the GPU is not used. Add `--dry-run` first:
it must report `quant size = 15320.35 MiB (4.70 BPW)` and convert nothing to
`q5_K`.

The policy file pins **every** quantized tensor, not only the promoted ones.
That is deliberate: left to itself the quantizer escalates a varying set of
`attn_qkv` and `ffn_gate` tensors to `Q5_K`, which silently puts them on the
slow kernel path.

The same file applies to Qwen3.8-27B itself and to other fine-tunes of it. Use
an importance matrix computed for the model you quantize.

For vision, pair the file with the f16 `mmproj` of the base Qwen3.8-27B model.
bartowski also publishes a projector converted from Qwen3.8-27B-pi; we did not
compare the two.

## Evidence

RX 7900 XT 20 GB, ROCm 7.2 HIP, Windows 11, 19 October 2026. Three models in
this recipe on the same server and the same day: Qwen3.8-27B-pi, Swift 1.5
Qwen3.8-27B (the previous model here) and Thinklite-27B High (another
reasoning-efficient fine-tune). 50 runs per model at `reasoning_effort: low`:
4 tool-calling agentic tasks x 3, 10 reasoning tasks x 2, 9 harder
python-verified tasks x 2, temperature 1, exact-match grading.

| | Qwen3.8-27B-pi | Swift 1.5 | Thinklite-27B High |
| --- | ---: | ---: | ---: |
| agentic: output tokens | 3,796 | 4,508 | 3,643 |
| agentic: wall s | 62.2 | 68.5 | 59.1 |
| agentic: correct | 12/12 | 12/12 | 12/12 |
| all 50: output tokens | 26,086 | 23,472 | 19,007 |
| all 50: wall s | 340 | 308 | 259 |
| all 50: correct | 50/50 | 47/50 | 47/50 |
| decode tok/s, median (agentic / reasoning / hard) | 82.9 / 80.6 / 81.6 | 83.9 / 82.7 / 81.1 | 84.7 / 77.1 / 78.9 |

Repeating a run moves the token total by a few percent and the score by one or
two answers. Thinklite is the fastest on this set; we chose pi because its
reinforcement learning rewarded completed tasks in an agent loop, while
trimming reasoning alone made an earlier candidate take more, smaller tool
calls on our real agent workloads.

Speed and memory are those of the recipe and the runtime, not of the model:
see the [speed tables in the README](../../README.md#speed), measured with
Swift 1.5 in this recipe.

## Notes

- If a variant of this recipe spills VRAM on your card, drop the eight
  `ffn_down` promotions first.
- A 24 GB RX 7900 XTX has room for more promotions or more context. Not
  measured.
- Qwen3.8-27B-pi is licensed under Apache-2.0, as is the base model.

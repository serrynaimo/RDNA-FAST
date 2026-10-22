# rocm_wmma_gemm (vendored)

A WMMA GEMM for AMD RDNA3 / RDNA3.5 GPUs by Adel Johar, MIT licensed (see [LICENSE](LICENSE), also in
[`licenses/LICENSE-rocm_wmma_gemm`](../../licenses/LICENSE-rocm_wmma_gemm)).

- Source: https://github.com/adelj88/rocm_wmma_gemm
- Commit: `ea3aa74` ("Overhaul tuning script, minor optimization, added float accumulator configs")

ROCmFPX uses it for prompt-sized products of its FP4/FP6 weight types on gfx1100 and gfx1151: the weight rows are
unpacked to f16 and multiplied with this GEMM (f16 inputs, f32 accumulation and output). See
`ggml/src/ggml-cuda/wmma-gemm.cu` ([TAG_WMMA_GEMM]) and `ggml/src/ggml-hip/wmma-gemm.cmake`, which builds one library
per GPU architecture from these files: `rocm_wmma_gemm_<arch>_accf32` next to the other binaries.

## What is included

Only what that library needs, unchanged unless listed below:

- `include/rocm_wmma_gemm/`: the kernel headers, without `kernel_loader.hpp` (a `dlopen` loader; ggml loads the
  library itself)
- `src/gemm.cpp`: configuration lookup, launch, and the C entry points
- `config/gemm_config_gfx1100_f32.json`, `config/gemm_config_gfx1151_f32.json`: the upstream tuned configurations for
  float accumulation
- `config/generate_config.py`: generates the kernel instances and lookup tables from a configuration at configure time

Not included: tests, benchmarks, the tuner, the f16-accumulation configurations and the documentation.

## Local changes

- `config/generate_config.py`: the generated `kernel_sources.cmake` lists its paths with forward slashes
  (`output_path.as_posix()`); with backslashes CMake on Windows reads `\U` in `C:\Users\...` as an escape and fails.
- `config/gemm_config_gfx1100_f32.json`: 18 entries added, none changed. They are the products Qwen3.8-27B runs at
  `-ub 1024` and `-ub 2048` in ROCmFPX (1,024 and 2,048 rows; A row major, B column major, C row major; f32
  accumulation), tuned on an RX 7900 XT with the library's own `tune.py` (budget 100 per shape, seed 42,
  22 October 2026), each at M = 1024 and M = 2048: Mx6400x5120, Mx4608x5120, Mx3840x5120, Mx5888x5120, Mx6144x5120,
  Mx5120x6144, Mx1792x17408, Mx1536x17408, Mx1024x5120.

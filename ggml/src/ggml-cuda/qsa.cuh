#pragma once

#include "common.cuh"

// Selected-attention for the Qwen3.8-Next / Qwen4 indexer path: the lightning indexer picks 4-key blocks per query, so
// attention only has to visit those (src[5] lists the selected cells, -1 for none). Every other flash-attention kernel
// ignores src[5], so the graph only builds such an op for a kernel ggml_cuda_qsa_caps() lists.

// [TAG_QSA_CAPS] the QSA kernels a device has for one attention shape (head size, query heads per KV head, K/V types)
enum : uint32_t {
    GGML_CUDA_QSA_DECODE = 1u << 0,   // 1..8 queries per op, reads the mask too (qsa-decode.cuh)
    GGML_CUDA_QSA_PROMPT = 1u << 1,   // 128+ queries, mask optional; f16 reads the packed layouts in src[6]/src[7] (qsa.cu)
};
uint32_t ggml_cuda_qsa_caps(int cc, int64_t head_size, int64_t gqa, ggml_type type_k, ggml_type type_v);

// prompt kernel (qsa3): false for anything it does not take, including the dense prefill case
bool ggml_cuda_flash_attn_ext_qsa_supported(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_qsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_qsa_decode_supported(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_qsa_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

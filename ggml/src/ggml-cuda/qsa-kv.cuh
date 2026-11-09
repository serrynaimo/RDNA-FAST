#pragma once
// K/V access shared by the QSA kernels (qsa.cu prompt kernel, qsa-decode*.cuh decode kernels): f16, and q8_0 / q4_0
// read straight from the KV cache. A quantized cache is dequantized in registers, so no f16 copy of the cache (at 128k
// a K+V copy is ~512 MB of compute buffer) is ever made. Head size is 256 and every wave owns 32 dims of it, which is
// exactly one q8_0 / q4_0 block: wave w reads block w of each key row.
#include "common.cuh"

typedef short    qsa_v16s __attribute__((ext_vector_type(16)));
typedef _Float16 qsa_h2   __attribute__((ext_vector_type(2)));

static constexpr __host__ __device__ bool qsa_kv_type_supported(const ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0;
}

// dwords of quants in one block (q8_0: 32 bytes, q4_0: 16 bytes); blocks are 34 / 18 bytes
template <ggml_type T> static constexpr int qsa_block_qdw    = T == GGML_TYPE_Q8_0 ? 8 : 4;
template <ggml_type T> static constexpr int qsa_block_bytes  = T == GGML_TYPE_Q8_0 ? (int) sizeof(block_q8_0) : (int) sizeof(block_q4_0);

// The raw dwords of one quantized block: a 4-byte aligned window of qdw+1 dwords. Rows are 4-byte aligned (the supported
// checks require it) and a block is 2 bytes off a 4-byte boundary exactly when w is odd (18*w, 34*w), so the window
// starts 2 bytes early then. It never leaves the 256-dim row (odd w >= 1 starts inside it, even w <= 6 ends 2 bytes past
// its block, before the row end).
template <ggml_type T>
struct qsa_block_raw {
    uint32_t a[qsa_block_qdw<T> + 1];
};

template <ggml_type T>
static __device__ __forceinline__ void qsa_load_block_raw(const char * row, const int w, qsa_block_raw<T> & raw) {
    const uint32_t * p = reinterpret_cast<const uint32_t *>(row + qsa_block_bytes<T> * w - 2 * (w & 1));
#pragma unroll
    for (int i = 0; i <= qsa_block_qdw<T>; ++i) { raw.a[i] = p[i]; }
}

// scale (f16 bits) and quants (qs[i] = quant bytes 4i..4i+3) of a raw block
template <ggml_type T>
static __device__ __forceinline__ void qsa_unpack_block(const qsa_block_raw<T> & raw, const int odd, uint32_t & d, uint32_t (&qs)[qsa_block_qdw<T>]) {
    if (odd) {
        d = raw.a[0] >> 16;
#pragma unroll
        for (int i = 0; i < qsa_block_qdw<T>; ++i) { qs[i] = raw.a[i + 1]; }
    } else {
        d = raw.a[0] & 0xFFFFu;
#pragma unroll
        for (int i = 0; i < qsa_block_qdw<T>; ++i) { qs[i] = (raw.a[i] >> 16) | (raw.a[i + 1] << 16); }
    }
}

// Two quants as an f16 pair times the scale pair d2, exact up to the one rounding of the product (the f16 cache rounding).
// q4_0: nibbles n0, n1 in bits [3:0] and [19:16] of x, everything else zero: 0x6400|n is the f16 1024+n, minus 1032 = n-8.
static __device__ __forceinline__ uint32_t qsa_q4_pair(const uint32_t x, const qsa_h2 d2) {
    const qsa_h2 off = {(_Float16) 1032.0f, (_Float16) 1032.0f};
    const qsa_h2 v = __builtin_bit_cast(qsa_h2, x | 0x64006400u) - off;
    return __builtin_bit_cast(uint32_t, v * d2);
}
// q8_0: int8 q0, q1 in bits [7:0] and [23:16] of x, everything else zero: q^0x80 = q+128 and 0x6400|(q+128) is 1024+q+128.
static __device__ __forceinline__ uint32_t qsa_q8_pair(const uint32_t x, const qsa_h2 d2) {
    const qsa_h2 off = {(_Float16) 1152.0f, (_Float16) 1152.0f};
    const qsa_h2 v = __builtin_bit_cast(qsa_h2, x ^ 0x64806480u) - off;
    return __builtin_bit_cast(uint32_t, v * d2);
}
// The key-row WMMA fragments of one wave: dims [32w, 32w+16) and [32w+16, 32w+32) of the row, element e = dim 16t+e.
// q4_0: dims 0..15 of the block are the low nibbles of qs[0..15], dims 16..31 the high nibbles; q8_0: qs[0..31].
template <ggml_type T>
static __device__ __forceinline__ void qsa_k_frags(const qsa_block_raw<T> & raw, const int odd, qsa_v16s & f0, qsa_v16s & f1) {
    uint32_t d, qs[qsa_block_qdw<T>];
    qsa_unpack_block<T>(raw, odd, d, qs);
    const qsa_h2 d2 = __builtin_bit_cast(qsa_h2, d | (d << 16));
    uint32_t e0[8], e1[8];
    if constexpr (T == GGML_TYPE_Q8_0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            e0[2*i + 0] = qsa_q8_pair(__builtin_amdgcn_perm(0u, qs[i],     0x0C010C00u), d2);
            e0[2*i + 1] = qsa_q8_pair(__builtin_amdgcn_perm(0u, qs[i],     0x0C030C02u), d2);
            e1[2*i + 0] = qsa_q8_pair(__builtin_amdgcn_perm(0u, qs[i + 4], 0x0C010C00u), d2);
            e1[2*i + 1] = qsa_q8_pair(__builtin_amdgcn_perm(0u, qs[i + 4], 0x0C030C02u), d2);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t lo = qs[i] & 0x0F0F0F0Fu, hi = (qs[i] >> 4) & 0x0F0F0F0Fu;
            e0[2*i + 0] = qsa_q4_pair(__builtin_amdgcn_perm(0u, lo, 0x0C010C00u), d2);
            e0[2*i + 1] = qsa_q4_pair(__builtin_amdgcn_perm(0u, lo, 0x0C030C02u), d2);
            e1[2*i + 0] = qsa_q4_pair(__builtin_amdgcn_perm(0u, hi, 0x0C010C00u), d2);
            e1[2*i + 1] = qsa_q4_pair(__builtin_amdgcn_perm(0u, hi, 0x0C030C02u), d2);
        }
    }
    f0 = __builtin_bit_cast(qsa_v16s, e0);
    f1 = __builtin_bit_cast(qsa_v16s, e1);
}

// V^T fragments through LDS. The WMMA A operand of O = V^T P wants, in lane r, dim 32w+16t+r of 16 keys; the cache holds
// a key's 32 dims together. So lane kk (0..15) of wave w stores block w of key kk (quants + scale) in the wave's own
// stage, and after a barrier every lane gathers its dim of the 16 keys: q4_0 dim j<16 is the low nibble of byte j, dim
// j+16 the high nibble of the same byte; q8_0 dim j is byte j. A stage is 16*qdw dwords of quants then 16 f16 scales.
template <ggml_type T> static constexpr int qsa_vstage_dwords = 16 * qsa_block_qdw<T> + 8;

template <ggml_type T>
static __device__ __forceinline__ void qsa_v_stage_store(uint32_t * stage, const int kk, const qsa_block_raw<T> & raw, const int odd) {
    uint32_t d, qs[qsa_block_qdw<T>];
    qsa_unpack_block<T>(raw, odd, d, qs);
#pragma unroll
    for (int i = 0; i < qsa_block_qdw<T>; ++i) { stage[kk * qsa_block_qdw<T> + i] = qs[i]; }
    reinterpret_cast<uint16_t *>(stage + 16 * qsa_block_qdw<T>)[kk] = (uint16_t) d;
}

template <ggml_type T>
static __device__ __forceinline__ void qsa_v_stage_load(const uint32_t * stage, const int r, qsa_v16s & f0, qsa_v16s & f1) {
    constexpr int qdw = qsa_block_qdw<T>;
    const int b = r & 3;
    // byte b of the low key -> bits [7:0], byte b of the high key -> bits [23:16]
    const uint32_t sel = 0x0C000C00u | ((uint32_t) (4 + b) << 16) | (uint32_t) b;
    uint32_t sc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { sc[i] = stage[16 * qdw + i]; }
    uint32_t e0[8], e1[8];
    if constexpr (T == GGML_TYPE_Q8_0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const qsa_h2 d2 = __builtin_bit_cast(qsa_h2, sc[i]);
            const uint32_t x0 = __builtin_amdgcn_perm(stage[(2*i + 1) * qdw + (r >> 2)],     stage[(2*i) * qdw + (r >> 2)],     sel);
            const uint32_t x1 = __builtin_amdgcn_perm(stage[(2*i + 1) * qdw + 4 + (r >> 2)], stage[(2*i) * qdw + 4 + (r >> 2)], sel);
            e0[i] = qsa_q8_pair(x0, d2);
            e1[i] = qsa_q8_pair(x1, d2);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const qsa_h2 d2 = __builtin_bit_cast(qsa_h2, sc[i]);
            const uint32_t x = __builtin_amdgcn_perm(stage[(2*i + 1) * qdw + (r >> 2)], stage[(2*i) * qdw + (r >> 2)], sel);
            e0[i] = qsa_q4_pair(x & 0x000F000Fu, d2);
            e1[i] = qsa_q4_pair((x >> 4) & 0x000F000Fu, d2);
        }
    }
    f0 = __builtin_bit_cast(qsa_v16s, e0);
    f1 = __builtin_bit_cast(qsa_v16s, e1);
}

// one element of a 256-dim row for the scalar decode kernel: lane l of a warp covers dims l + 32*c, i.e. byte l of
// block c (q8_0), or the low (l < 16) / high nibble of byte l%16 (q4_0)
template <ggml_type T>
static __device__ __forceinline__ float qsa_row_value(const char * row, const int c, const int lane) {
    if constexpr (T == GGML_TYPE_F16) {
        return __half2float(reinterpret_cast<const half *>(row)[lane + 32*c]);
    } else if constexpr (T == GGML_TYPE_Q8_0) {
        const block_q8_0 * b = reinterpret_cast<const block_q8_0 *>(row) + c;
        return (float) b->qs[lane] * __half2float(b->d);
    } else {
        const block_q4_0 * b = reinterpret_cast<const block_q4_0 *>(row) + c;
        const int q = (b->qs[lane & 15] >> (4 * (lane >> 4))) & 0xF;
        return (float) (q - 8) * __half2float(b->d);
    }
}

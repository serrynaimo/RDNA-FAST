#pragma once
#include "qsa-kv.cuh"

typedef short qd_half16 __attribute__((ext_vector_type(16)));
typedef float qd_float8 __attribute__((ext_vector_type(8)));

// gfx11 wave32 WMMA only: ggml_cuda_flash_attn_ext_qsa_decode_supported() admits RDNA3 and RDNA3.5 (gfx1100..1103,
// gfx1150..1153), so other AMD targets (RDNA4 has a different gfx12 WMMA encoding) compile a trap here that never runs.
template <typename V, typename A>
static __device__ __forceinline__ A qd_wmma_f16(const V a, const V b, const A c) {
#if defined(RDNA3)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    NO_DEVICE_CODE;
    return c;
#endif // defined(RDNA3)
}

// One workgroup per (KV head, query, split of 64 selected cells). The KV head's gqa query heads (up to 16) are the 16
// columns of the WMMA tiles; wave w owns dims [32w, 32w+32) of the 256, which for a quantized cache is block w of the
// row. Keys come 16 at a time from the query's selection (src[5]); -1, out-of-range and masked-out cells are skipped.
template <ggml_type T>
static __global__ __launch_bounds__(256) void qsa_decode_wmma_partial(
        const char * q, const char * k, const char * v, const char * mask, const char * ids,
        size_t q1, size_t q2, size_t k1, size_t k2, size_t v1, size_t v2, size_t m1, size_t i1,
        int nk, int ns, int nh, int gqa, int splits, float scale, float * partial) {
    constexpr bool quant = T != GGML_TYPE_F16;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, r = lane & 15, hi = lane >> 4;
    const int odd = __builtin_amdgcn_readfirstlane(w) & 1;
    const int kvh = blockIdx.x, query = blockIdx.y, split = blockIdx.z;
    const int * ir = (const int *) (ids + query*i1);
    const half * mr = mask ? (const half *) (mask + query*m1) : nullptr;
    __shared__ union {
        uint16_t query_stage[8][16][32];
        float4 scores[8][64];
    } shared;
    __shared__ __align__(16) uint16_t probabilities[16][16];
    __shared__ __align__(16) float alpha_s[16];
    __shared__ __align__(16) uint32_t vstage[quant ? 8 : 1][quant ? qsa_vstage_dwords<T> : 1];   // quantized V, per wave
    qd_half16 qf[2];
    const _Float16 hs = (_Float16) scale;
    for (int j = lane; j < 16*32; j += 32) {
        const int head = j/32, d = j%32;
        const float value = head < gqa ? *(const float *) (q + query*q1 + (kvh*gqa+head)*q2 + (w*32+d)*4) : 0.0f;
        shared.query_stage[w][head][d] = __builtin_bit_cast(uint16_t, (_Float16)value*hs);
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const uint16_t * src = &shared.query_stage[w][r][t*16];
        qf[t] = __builtin_bit_cast(qd_half16, (uint4[2]){*(const uint4 *)src, *(const uint4 *)(src+8)});
    }
    __syncthreads();
    qd_float8 output[2] = {};
    float maximum = -INFINITY, normalizer = 0.0f;
    const int end = min(ns, split*64+64);
    for (int start = split*64; start < end; start += 16) {
        int keys[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            int key = start+j < end ? ir[start+j] : -1;
            if (key < 0 || key >= nk || (mr && __half2float(mr[key]) == -INFINITY)) { key = -1; }
            keys[j] = key;
        }
        qd_float8 scores = {};
        if constexpr (quant) {
            // lane r (both halves) reads block w of key r: K for the fragments, V for the stage
            const int key = keys[r];
            qsa_block_raw<T> kraw, vraw;
            if (key >= 0) {
                qsa_load_block_raw<T>(k + key*k1 + kvh*k2, w, kraw);
                qsa_load_block_raw<T>(v + key*v1 + kvh*v2, w, vraw);
            } else {
#pragma unroll
                for (int i = 0; i <= qsa_block_qdw<T>; ++i) { kraw.a[i] = 0; vraw.a[i] = 0; }
            }
            qd_half16 kf0, kf1;
            qsa_k_frags<T>(kraw, odd, kf0, kf1);
            scores = qd_wmma_f16(kf0, qf[0], scores);
            scores = qd_wmma_f16(kf1, qf[1], scores);
            if (hi == 0) { qsa_v_stage_store<T>(vstage[w], r, vraw, odd); }
        } else {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                qd_half16 kf = {};
                const int key = keys[r];
                if (key >= 0 && key < nk) {
                    const uint16_t * src = (const uint16_t *) (k + key*k1 + kvh*k2) + 32*w+16*t;
                    kf = __builtin_bit_cast(qd_half16, (uint4[2]){*(const uint4 *)src, *(const uint4 *)(src+8)});
                }
                scores = qd_wmma_f16(kf, qf[t], scores);
            }
        }
        shared.scores[w][lane*2] = make_float4(scores[0],scores[1],scores[2],scores[3]);
        shared.scores[w][lane*2+1] = make_float4(scores[4],scores[5],scores[6],scores[7]);
        __syncthreads();
        if (w == 0) {
            float score[8] = {};
#pragma unroll
            for (int sw = 0; sw < 8; ++sw) {
                const float4 lo = shared.scores[sw][lane*2], high = shared.scores[sw][lane*2+1];
                score[0]+=lo.x;score[1]+=lo.y;score[2]+=lo.z;score[3]+=lo.w;
                score[4]+=high.x;score[5]+=high.y;score[6]+=high.z;score[7]+=high.w;
            }
            float local = -INFINITY;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int key = keys[2*e+hi];
                if (key < 0 || key >= nk) { score[e] = -INFINITY; }
                else if (mr) { score[e] += __half2float(mr[key]); }
                local = fmaxf(local,score[e]);
            }
            local = fmaxf(local,__shfl_xor(local,16));
            const float next = fmaxf(maximum,local);
            const float alpha = next == -INFINITY ? 1.0f : exp2f((maximum-next)*1.4426950408889634f);
            float sum = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float p = next == -INFINITY ? 0.0f : exp2f((score[e]-next)*1.4426950408889634f);
                sum += p;
                probabilities[r][2*e+hi] = __builtin_bit_cast(uint16_t,(_Float16)p);
            }
            sum += __shfl_xor(sum,16);
            normalizer = normalizer*alpha+sum;
            maximum = next;
            if (hi == 0) { alpha_s[r] = alpha; }
        }
        __syncthreads();
        const float alpha = alpha_s[r];
        const qd_half16 pf = __builtin_bit_cast(qd_half16, (uint4[2]){
            *(const uint4 *)&probabilities[r][0], *(const uint4 *)&probabilities[r][8]});
        if constexpr (quant) {
            qd_half16 vf[2];
            qsa_v_stage_load<T>(vstage[w], r, vf[0], vf[1]);
#pragma unroll
            for (int t = 0; t < 2; ++t) {
#pragma unroll
                for (int e = 0; e < 8; ++e) { output[t][e] *= alpha; }
                output[t] = qd_wmma_f16(vf[t],pf,output[t]);
            }
        } else {
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            uint16_t values[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int key = keys[j];
                values[j] = key >= 0 && key < nk ? *((const uint16_t *)(v+key*v1+kvh*v2)+32*w+16*t+r) : 0;
            }
            const qd_half16 vf = __builtin_bit_cast(qd_half16,values);
#pragma unroll
            for (int e = 0; e < 8; ++e) { output[t][e] *= alpha; }
            output[t] = qd_wmma_f16(vf,pf,output[t]);
        }
        }
        __syncthreads();
    }
    if (r < gqa) {
        float * dst = partial + ((query*nh+kvh*gqa+r)*splits+split)*258;
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { dst[32*w+16*t+2*e+hi] = output[t][e]; }
        if (w == 0 && hi == 0) { dst[256] = maximum; dst[257] = normalizer; }
    }
}

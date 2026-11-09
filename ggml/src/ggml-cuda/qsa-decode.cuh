#pragma once
#include "qsa-decode-wmma.cuh"

// scalar fallback: one workgroup per (query head, query, split of 128 selected cells), 4 warps over the cells, lane l
// covers dims l + 32*c. Used for more than 16 query heads per KV head and for f16 rows that are not 16-byte aligned.
template <ggml_type T>
static __global__ __launch_bounds__(128) void qsa_decode_partial(
        const char * q, const char * k, const char * v, const char * mask, const char * ids,
        size_t q1, size_t q2, size_t k1, size_t k2, size_t v1, size_t v2, size_t m1, size_t i1,
        int nk, int ns, int nh, int ratio, int splits, float scale, float * partial) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int head = blockIdx.x, query = blockIdx.y, split = blockIdx.z;
    const int kv_head = head / ratio;
    const float * qr = (const float *) (q + query*q1 + head*q2);
    const int * ir = (const int *) (ids + query*i1);
    const half * mr = mask ? (const half *) (mask + query*m1) : nullptr;
    float qv[8], acc[8] = {};
#pragma unroll
    for (int d = 0; d < 8; ++d) { qv[d] = qr[lane + d*32]; }
    float mx = -INFINITY, sum = 0.0f;
    const int begin = split*128, end = min(ns, begin + 128);
    for (int j = begin + warp; j < end; j += 4) {
        const int key = ir[j];
        if (key < 0 || key >= nk) { continue; }
        const float bias = mr ? __half2float(mr[key]) : 0.0f;
        if (bias == -INFINITY) { continue; }
        const char * kr = k + key*k1 + kv_head*k2;
        const char * vr = v + key*v1 + kv_head*v2;
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) { dot = fmaf(qv[d], qsa_row_value<T>(kr, d, lane), dot); }
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) { dot += __shfl_xor_sync(0xffffffff, dot, delta, 32); }
        const float score = dot*scale + bias;
        const float next = fmaxf(mx, score);
        const float alpha = expf(mx - next), weight = expf(score - next);
#pragma unroll
        for (int d = 0; d < 8; ++d) { acc[d] = acc[d]*alpha + weight*qsa_row_value<T>(vr, d, lane); }
        sum = sum*alpha + weight;
        mx = next;
    }
    __shared__ __align__(16) float values[4][256];
    __shared__ __align__(16) float maxima[4], sums[4];
#pragma unroll
    for (int d = 0; d < 8; ++d) { values[warp][lane + d*32] = acc[d]; }
    if (lane == 0) { maxima[warp] = mx; sums[warp] = sum; }
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; ++w) { maximum = fmaxf(maximum, maxima[w]); }
    float factors[4], total = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        factors[w] = sums[w] > 0.0f ? expf(maxima[w] - maximum) : 0.0f;
        total += factors[w]*sums[w];
    }
    float * dst = partial + ((query*nh + head)*splits + split)*258;
    for (int d = threadIdx.x; d < 256; d += 128) {
        float value = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) { value += values[w][d]*factors[w]; }
        dst[d] = value;
    }
    if (threadIdx.x == 0) { dst[256] = maximum; dst[257] = total; }
}

static __global__ __launch_bounds__(256) void qsa_decode_merge(
        const float * partial, float * out, int splits) {
    const int row = blockIdx.x, d = threadIdx.x;
    const float * src = partial + row*splits*258;
    float maximum = -INFINITY;
    for (int s = 0; s < splits; ++s) { maximum = fmaxf(maximum, src[s*258 + 256]); }
    float sum = 0.0f, value = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float count = src[s*258 + 257];
        const float factor = count > 0.0f ? expf(src[s*258 + 256] - maximum) : 0.0f;
        value += src[s*258 + d]*factor;
        sum += count*factor;
    }
    out[row*256 + d] = sum > 0.0f ? value/sum : 0.0f;
}

// [TAG_QSA_CAPS] any gfx11 (RDNA3 / RDNA3.5), head size 256, any GQA (WMMA up to 16 query heads per KV head, scalar
// beyond), K and V of one type: f16, or q8_0 / q4_0 read in place (rows 4-byte aligned)
bool ggml_cuda_flash_attn_ext_qsa_decode_supported(const int device, const ggml_tensor * dst) {
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * m = dst->src[3], * ids = dst->src[5];
    if (!q || !k || !v || !ids || dst->src[4] || dst->src[6] || dst->src[7]) { return false; }
    if (k->ne[2] < 1 || q->ne[2] % k->ne[2] ||
        !(ggml_cuda_qsa_caps(ggml_cuda_info().devices[device].cc, k->ne[0], q->ne[2]/k->ne[2], k->type, v->type) & GGML_CUDA_QSA_DECODE)) {
        return false;
    }
    float bias, cap;
    memcpy(&bias, (const char *) dst->op_params + 4, 4);
    memcpy(&cap, (const char *) dst->op_params + 8, 4);
    if (bias != 0.0f || cap != 0.0f || q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        q->ne[0] != 256 || k->ne[0] != 256 || v->ne[0] != 256 || q->ne[1] < 1 || q->ne[1] > 8 ||
        k->ne[1] < 1 || k->ne[1] > 262144 || v->ne[1] != k->ne[1] || v->ne[2] != k->ne[2] ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        q->nb[0] != 4 || k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type) || ids->nb[0] != 4 ||
        ids->ne[0] < 1 || ids->ne[0] > 2560 || ids->ne[1] < q->ne[1] || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        !ggml_is_contiguous(dst)) { return false; }
    if (ggml_is_quantized(k->type) && (k->nb[1] % 4 || k->nb[2] % 4 || v->nb[1] % 4 || v->nb[2] % 4 ||
            uintptr_t(k->data) % 4 || uintptr_t(v->data) % 4)) { return false; }
    return !m || (m->type == GGML_TYPE_F16 && m->nb[0] == 2 && m->ne[0] >= k->ne[1] &&
                  m->ne[1] >= q->ne[1] && m->ne[2] == 1 && m->ne[3] == 1);
}

template <ggml_type T>
static void qsa_decode_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * m = dst->src[3], * ids = dst->src[5];
    const int gqa = q->ne[2]/k->ne[2];
    // quantized rows are 4-byte aligned (admission); f16 fragments are 16-byte loads
    const bool wmma = gqa <= 16 && (T != GGML_TYPE_F16 ||
        (k->nb[1]%16 == 0 && k->nb[2]%16 == 0 && uintptr_t(k->data)%16 == 0));
    const int ns = ids->ne[0], nq = q->ne[1], nh = q->ne[2], step = wmma ? 64 : 128;
    const int splits = (ns + step - 1)/step;
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    ggml_cuda_pool_alloc<float> partial(ctx.pool(), size_t(nq)*nh*splits*258);
    if (wmma) {
    qsa_decode_wmma_partial<T><<<dim3(k->ne[2], nq, splits), 256, 0, ctx.stream()>>>(
        (const char *) q->data, (const char *) k->data, (const char *) v->data,
        m ? (const char *) m->data : nullptr, (const char *) ids->data,
        q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2], m ? m->nb[1] : 0, ids->nb[1],
        k->ne[1], ns, nh, gqa, splits, scale, partial.get());
    } else {
    qsa_decode_partial<T><<<dim3(nh, nq, splits), 128, 0, ctx.stream()>>>(
        (const char *) q->data, (const char *) k->data, (const char *) v->data,
        m ? (const char *) m->data : nullptr, (const char *) ids->data,
        q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2], m ? m->nb[1] : 0, ids->nb[1],
        k->ne[1], ns, nh, gqa, splits, scale, partial.get());
    }
    qsa_decode_merge<<<nq*nh, 256, 0, ctx.stream()>>>(partial.get(), (float *) dst->data, splits);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_ext_qsa_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[1]->type) {
        case GGML_TYPE_F16:  qsa_decode_launch<GGML_TYPE_F16>(ctx, dst);  break;
        case GGML_TYPE_Q8_0: qsa_decode_launch<GGML_TYPE_Q8_0>(ctx, dst); break;
        case GGML_TYPE_Q4_0: qsa_decode_launch<GGML_TYPE_Q4_0>(ctx, dst); break;
        default: GGML_ABORT("QSA decode: unsupported K/V type");
    }
}

#include "common.cuh"
#include "qsa.cuh"

#if defined(GGML_USE_HIP)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include "qsa-decode.cuh"

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f  __attribute__((ext_vector_type(8)));

#define QSA3_L2E 1.4426950408889634f

static __device__ __forceinline__ uint16_t qsa3_f2h(const float f) { return __builtin_bit_cast(uint16_t, (_Float16) f); }
static __device__ __forceinline__ float qsa3_h2f(const uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }

// ------------------------------------------------------------------------------------------------------------
// union builder, kernel A: per query row, check that the (validity-transformed) row is non-decreasing; if not,
// rank-sort it into srow[q] and set sflag[q]. Sentinel 0x7FFFFFFF for invalid (-1 / >= nk) entries.
#define QSA3_SENT 0x7FFFFFFF

// The QSA kernels use the gfx11 wave32 WMMA builtin; ggml_cuda_qsa_caps() admits RDNA3 and RDNA3.5 (gfx1100..1103,
// gfx1150..1153), so other AMD targets (RDNA4 has a different gfx12 WMMA encoding) compile a trap here that never runs.
template <typename V, typename A>
static __device__ __forceinline__ A qsa_wmma_f16(const V a, const V b, const A c) {
#if defined(RDNA3)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    NO_DEVICE_CODE;
    return c;
#endif // defined(RDNA3)
}

__global__ __launch_bounds__(256) void qsa3_rows_kernel(
        const int * __restrict__ ids, const size_t i1, const int ns, const int nk, int * __restrict__ srow, int * __restrict__ sflag) {
    extern __shared__ __align__(16) int ent[];
    __shared__ int unsorted;
    const int q = blockIdx.x, tid = threadIdx.x;
    const int * row = reinterpret_cast<const int *>(reinterpret_cast<const char *>(ids) + (size_t) q * i1);
    for (int j = tid; j < ns; j += 256) { const int key = row[j]; ent[j] = (key >= 0 && key < nk) ? key : QSA3_SENT; }
    if (tid == 0) { unsorted = 0; }
    __syncthreads();
    for (int j = tid; j + 1 < ns; j += 256) { if (ent[j] > ent[j+1]) { unsorted = 1; } }
    __syncthreads();
    if (unsorted) {
        int * dst = srow + (size_t) q * ns;
        for (int j = tid; j < ns; j += 256) {
            const int e = ent[j];
            int rank = 0;
            for (int k = 0; k < ns; ++k) { const int f = ent[k]; rank += (f < e) || (f == e && k < j); }
            dst[rank] = e;
        }
    }
    if (tid == 0) { sflag[q] = unsorted; }
}

// kernel B: one workgroup (128 lanes) per group of G queries (4 for GQA 12, 8 for GQA 6: either way 48 rows = 3 WMMA
// tiles). Lane = key sub-range; the range boundaries are the quantiles of the first non-empty row (block aligned), so
// the work per lane is balanced for the real (sorted, similar) rows. Each lane merges the G sorted rows inside its range:
// pass 1 counts, block-wide prefix sum, pass 2 writes. A union entry is a 4-key block and a mask of G queries x 4 cells
// (16 bits for G = 4, 32 bits for G = 8). A cell listed twice in a row counts once (it used to re-open its block when it
// followed a whole block, attending to that cell twice; top-k rows never repeat a cell, test rows do).
// STAGE copies the rows into LDS first (4 rows of up to 2560 keys = 40 KB). 8 rows of a 2048-token budget (2051 with the
// tail) need 64.1 KB, more than the 64 KB a gfx11 workgroup may allocate, so G = 8 reads them in place: ids rows that are
// already sorted (sflag 0) are validity-transformed on the fly by qsa3_get, and inside [0, cnt) such a row holds valid
// keys only (the transformed row is non-decreasing, so its sentinels are all at the end), which the merge loops read raw.
#define QSA3_MERGE_LANES 128
template <int G> using qsa3_mask_t = std::conditional_t<G == 4, uint16_t, uint32_t>;

static __device__ __forceinline__ int qsa3_get(const int * __restrict__ p, const int j, const int nk, const bool raw) {
    const int key = p[j];
    return raw ? ((key >= 0 && key < nk) ? key : QSA3_SENT) : key;
}
template <int G, bool STAGE>
__global__ __launch_bounds__(QSA3_MERGE_LANES) void qsa3_merge_kernel(
        const int * __restrict__ ids, const size_t i1, const int n_q, const int ns, const int nk,
        const int * __restrict__ srow, const int * __restrict__ sflag,
        uint16_t * __restrict__ ublk, qsa3_mask_t<G> * __restrict__ umask, int * __restrict__ ucount, const int cap) {
    static_assert(G == 4 || G == 8, "4 or 8 queries per group");
    extern __shared__ __align__(16) int rows_s[];
    __shared__ __align__(16) int scan_s[QSA3_MERGE_LANES];
    const int lane = threadIdx.x;
    const int g = blockIdx.x;
    const int * rp[G]; bool raw[G]; int cnt[G];
    if constexpr (STAGE) {
#pragma unroll
        for (int qi = 0; qi < G; ++qi) {
            const int q = G * g + qi;
            int * dst = rows_s + qi * ns;
            if (q < n_q) {
                const bool rw = sflag[q] == 0;
                const int * src = rw ? reinterpret_cast<const int *>(reinterpret_cast<const char *>(ids) + (size_t) q * i1) : srow + (size_t) q * ns;
                for (int j = lane; j < ns; j += QSA3_MERGE_LANES) { dst[j] = qsa3_get(src, j, nk, rw); }
            } else {
                for (int j = lane; j < ns; j += QSA3_MERGE_LANES) { dst[j] = QSA3_SENT; }
            }
        }
        __syncthreads();
#pragma unroll
        for (int qi = 0; qi < G; ++qi) {
            raw[qi] = false; rp[qi] = rows_s + qi * ns;
            int a = 0, b = ns;
            while (a < b) { const int mid = (a + b) >> 1; if (rp[qi][mid] < QSA3_SENT) { a = mid + 1; } else { b = mid; } }
            cnt[qi] = a;
        }
    } else {
#pragma unroll
        for (int qi = 0; qi < G; ++qi) {
            const int q = G * g + qi;
            if (q < n_q) {
                raw[qi] = sflag[q] == 0;
                rp[qi] = raw[qi] ? reinterpret_cast<const int *>(reinterpret_cast<const char *>(ids) + (size_t) q * i1) : srow + (size_t) q * ns;
                int a = 0, b = ns;
                while (a < b) { const int mid = (a + b) >> 1; if (qsa3_get(rp[qi], mid, nk, raw[qi]) < QSA3_SENT) { a = mid + 1; } else { b = mid; } }
                cnt[qi] = a;
            } else {
                raw[qi] = false; rp[qi] = srow; cnt[qi] = 0;
            }
        }
    }
    int kmax = -1, ref = -1;
#pragma unroll
    for (int qi = 0; qi < G; ++qi) {
        if (cnt[qi] > 0) { kmax = max(kmax, qsa3_get(rp[qi], cnt[qi] - 1, nk, raw[qi])); if (ref < 0) { ref = qi; } }
    }
    uint16_t * ob = ublk + (size_t) g * cap;
    qsa3_mask_t<G> * om = umask + (size_t) g * cap;
    if (kmax < 0) { if (lane == 0) { ucount[g] = 0; } return; }
    // lane range [lo, hi) in key space, block aligned, from the reference row's quantiles
    const int rc = cnt[ref];
    const int lo = lane == 0 ? 0 : (qsa3_get(rp[ref], (int) (((long long) rc * lane) / QSA3_MERGE_LANES), nk, raw[ref]) & ~3);
    const int hi = lane == QSA3_MERGE_LANES - 1 ? ((kmax + 4) & ~3)
                 : (qsa3_get(rp[ref], (int) (((long long) rc * (lane + 1)) / QSA3_MERGE_LANES), nk, raw[ref]) & ~3);
    int p[G], pe[G];
#pragma unroll
    for (int qi = 0; qi < G; ++qi) {
        int a = 0, b = cnt[qi];
        while (a < b) { const int mid = (a + b) >> 1; if (qsa3_get(rp[qi], mid, nk, raw[qi]) < lo) { a = mid + 1; } else { b = mid; } }
        p[qi] = a; b = cnt[qi];
        while (a < b) { const int mid = (a + b) >> 1; if (qsa3_get(rp[qi], mid, nk, raw[qi]) < hi) { a = mid + 1; } else { b = mid; } }
        pe[qi] = a;
    }
    // pass 1: count union blocks in range
    int n = 0;
    {
        int pp[G]; int head[G];
#pragma unroll
        for (int qi = 0; qi < G; ++qi) { pp[qi] = p[qi]; head[qi] = pp[qi] < pe[qi] ? qsa3_get(rp[qi], pp[qi], nk, raw[qi]) : QSA3_SENT; }
        while (true) {
            int b = QSA3_SENT;
#pragma unroll
            for (int qi = 0; qi < G; ++qi) { b = min(b, head[qi]); }
            if (b == QSA3_SENT) { break; }
            b >>= 2;
#pragma unroll
            for (int qi = 0; qi < G; ++qi) {
                if (head[qi] != QSA3_SENT && (head[qi] >> 2) == b) {
                    const int * rr = rp[qi];
                    int pn = pp[qi];
                    // a whole block in order takes the shortcut; the loop also eats a repeated cell after it
                    pn += (head[qi] & 3) == 0 && pn + 3 < pe[qi] && rr[pn+3] == head[qi] + 3 && rr[pn+1] == head[qi] + 1 && rr[pn+2] == head[qi] + 2 ? 4 : 1;
                    while (pn < pe[qi] && (rr[pn] >> 2) == b) { ++pn; }
                    pp[qi] = pn; head[qi] = pn < pe[qi] ? rr[pn] : QSA3_SENT;
                }
            }
            ++n;
        }
    }
    scan_s[lane] = n;
    __syncthreads();
    for (int off = 1; off < QSA3_MERGE_LANES; off <<= 1) {
        const int v = lane >= off ? scan_s[lane - off] : 0;
        __syncthreads();
        scan_s[lane] += v;
        __syncthreads();
    }
    const int total = scan_s[QSA3_MERGE_LANES - 1];
    int o = scan_s[lane] - n;
    {
        int head[G];
#pragma unroll
        for (int qi = 0; qi < G; ++qi) { head[qi] = p[qi] < pe[qi] ? qsa3_get(rp[qi], p[qi], nk, raw[qi]) : QSA3_SENT; }
        while (true) {
            int b = QSA3_SENT;
#pragma unroll
            for (int qi = 0; qi < G; ++qi) { b = min(b, head[qi]); }
            if (b == QSA3_SENT) { break; }
            b >>= 2;
            uint32_t mk = 0;
#pragma unroll
            for (int qi = 0; qi < G; ++qi) {
                if (head[qi] != QSA3_SENT && (head[qi] >> 2) == b) {
                    const int * rr = rp[qi];
                    int pn = p[qi];
                    if ((head[qi] & 3) == 0 && pn + 3 < pe[qi] && rr[pn+3] == head[qi] + 3 && rr[pn+1] == head[qi] + 1 && rr[pn+2] == head[qi] + 2) { mk |= 0xFu << (4*qi); pn += 4; }
                    else { mk |= 1u << (4*qi + (rr[pn] & 3)); ++pn; }
                    while (pn < pe[qi] && (rr[pn] >> 2) == b) { mk |= 1u << (4*qi + (rr[pn] & 3)); ++pn; }
                    p[qi] = pn; head[qi] = pn < pe[qi] ? rr[pn] : QSA3_SENT;
                }
            }
            ob[o] = (uint16_t) b; om[o] = (qsa3_mask_t<G>) mk; ++o;
        }
    }
    if (lane == QSA3_MERGE_LANES - 1) {
        int t = total;
        while (t & 3) { ob[t] = 0xFFFFu; om[t] = 0; ++t; }
        ucount[g] = t;
    }
}

struct qsa3_layout {
    size_t q1, q2, o1, o2, m1;
    size_t k1, k2, v1, v2;   // q8_0 / q4_0: the cache rows (cell stride, KV-head stride); unused for f16 (packed layouts)
    int nk, n_q, cap;
    float scale;
};

// One workgroup per (group of G queries, KV head): 48 rows (G queries x GQA heads) = 3 WMMA tiles, 8 waves, wave w owns
// dims [32w, 32w+32) of the 256 (for a quantized cache that is block w of the row). The union blocks of the group come 4
// per chunk (16 keys). f16 K/V come from the packed layouts (src[6], src[7]); q8_0 / q4_0 are read from the cache rows:
// K as raw blocks one chunk ahead (dequantized at use, so the prefetch still overlaps the softmax), V through the wave's
// LDS stage (qsa-kv.cuh), stored before the first barrier and gathered after the second.
template <int G, int GQA, ggml_type T>
__global__ __launch_bounds__(256) void qsa3_attn_kernel(
        const float * __restrict__ q, const char * __restrict__ kd, const char * __restrict__ vd,
        const uint16_t * __restrict__ mask, const uint16_t * __restrict__ ublk, const qsa3_mask_t<G> * __restrict__ umask,
        const int * __restrict__ ucount, float * __restrict__ out, const qsa3_layout s) {
    static_assert(G * GQA == 48, "G queries x GQA heads must fill 3 WMMA tiles of 16 rows");
    constexpr bool quant = T != GGML_TYPE_F16;
    constexpr int NM = G / 2;   // mask dwords per chunk of 4 union blocks
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, r = lane & 15, hi = lane >> 4;
    const int odd = __builtin_amdgcn_readfirstlane(w) & 1;
    const int g = blockIdx.x, kvh = blockIdx.y;
    __shared__ __align__(16) float4 part[3][8][64];
    __shared__ __align__(16) uint4  ptile[3][32];
    __shared__ __align__(16) float  alpha_s[48];
    __shared__ __align__(16) float  l_s[48];
    __shared__ __align__(16) uint32_t vstage[quant ? 8 : 1][quant ? qsa_vstage_dwords<T> : 1];

    const size_t nblk = (size_t) s.nk / 4;
    const uint16_t * pkg = reinterpret_cast<const uint16_t *>(kd) + (size_t) kvh * nblk * 1024;
    const uint16_t * pvg = reinterpret_cast<const uint16_t *>(vd) + (size_t) kvh * nblk * 1024;
    const char * krows = kd + (size_t) kvh * s.k2;
    const char * vrows = vd + (size_t) kvh * s.v2;
    const uint16_t * gblk = ublk + (size_t) g * s.cap;
    const qsa3_mask_t<G> * gmsk = umask + (size_t) g * s.cap;
    const int nchunks = ucount[g] >> 2;

    v16s qf[3][2];
    {
        uint16_t * stage = reinterpret_cast<uint16_t *>(&part[0][0][0]) + w * (48 * 32);   // 3 KB per wave, row stride 64 B
        const _Float16 hs = (_Float16) s.scale;
#pragma unroll
        for (int c = 0; c < 12; ++c) {
            const int row = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int qi = row / GQA, h = row - GQA*qi;
            const int query = G * g + qi;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (query < s.n_q) {
                v = *reinterpret_cast<const float4 *>(reinterpret_cast<const char *>(q) + (size_t) query * s.q1 + (size_t) (kvh * GQA + h) * s.q2 + (size_t) (32*w + dq) * 4);
            }
            const uint16_t h0 = __builtin_bit_cast(uint16_t, (_Float16) v.x * hs), h1 = __builtin_bit_cast(uint16_t, (_Float16) v.y * hs);
            const uint16_t h2 = __builtin_bit_cast(uint16_t, (_Float16) v.z * hs), h3 = __builtin_bit_cast(uint16_t, (_Float16) v.w * hs);
            *reinterpret_cast<uint2 *>(stage + row * 32 + dq) = make_uint2((uint32_t) h0 | ((uint32_t) h1 << 16), (uint32_t) h2 | ((uint32_t) h3 << 16));
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const uint16_t * src = stage + (16*i + r) * 32 + 16*t;
                qf[i][t] = __builtin_bit_cast(v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(src), *reinterpret_cast<const uint4 *>(src + 8)});
            }
        __syncthreads();
    }
    v8f O[3][2];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { O[i][t][e] = 0.f; }
    float m = -INFINITY, l = 0.f;                       // owner state (waves 0..2): row 16w + r
    const int own_row = 16*w + r, own_qi = own_row / GQA;
    const int own_query = G * g + own_qi;
    const uint16_t * maskq = (mask && w < 3 && own_query < s.n_q)
        ? reinterpret_cast<const uint16_t *>(reinterpret_cast<const char *>(mask) + (size_t) own_query * s.m1) : nullptr;

    auto load_desc = [&](const int c, uint32_t & b01, uint32_t & b23, uint32_t (&mm)[NM]) {
        const uint2 bb = *reinterpret_cast<const uint2 *>(gblk + 4*c);
        b01 = bb.x; b23 = bb.y;
        if constexpr (G == 4) {
            const uint2 x = *reinterpret_cast<const uint2 *>(gmsk + 4*c);
            mm[0] = x.x; mm[1] = x.y;
        } else {
            const uint4 x = *reinterpret_cast<const uint4 *>(gmsk + 4*c);
            mm[0] = x.x; mm[1] = x.y; mm[2] = x.z; mm[3] = x.w;
        }
    };
    // G queries x 4 cells of union block j of the chunk
    auto mask_of = [&](const uint32_t (&mm)[NM], const int j) -> uint32_t {
        if constexpr (G == 4) {
            return (mm[j >> 1] >> (16 * (j & 1))) & 0xFFFFu;
        } else {
            return mm[j];
        }
    };
    auto blk_of = [&](const uint32_t b01, const uint32_t b23, const int j) -> int {
        const uint32_t v = (j < 2 ? b01 : b23) >> (16 * (j & 1)) & 0xFFFFu;
        return v == 0xFFFFu ? 0 : (int) v;
    };
    auto load_k = [&](const uint32_t b01, const uint32_t b23, v16s * kf) {
        const int kb = blk_of(b01, b23, r >> 2);
        const uint16_t * krow = pkg + (size_t) kb * 1024 + (r & 3) * 16 + w * 128;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const uint4 lo  = *reinterpret_cast<const uint4 *>(krow + t*64);
            const uint4 hi4 = *reinterpret_cast<const uint4 *>(krow + t*64 + 8);
            kf[t] = __builtin_bit_cast(v16s, (uint4[2]){lo, hi4});
        }
    };
    // quantized: lane r (both halves) reads block w of key r of the chunk, K and V
    auto load_raw = [&](const uint32_t b01, const uint32_t b23, qsa_block_raw<T> & kraw, qsa_block_raw<T> & vraw) {
        const int key = 4 * blk_of(b01, b23, r >> 2) + (r & 3);
        qsa_load_block_raw<T>(krows + (size_t) key * s.k1, w, kraw);
        qsa_load_block_raw<T>(vrows + (size_t) key * s.v1, w, vraw);
    };

    uint32_t b01 = 0, b23 = 0, mm[NM] = {};
    v16s kf[2];
    qsa_block_raw<T> kraw, vraw;
    if (nchunks > 0) {
        load_desc(0, b01, b23, mm);
        if constexpr (quant) { load_raw(b01, b23, kraw, vraw); } else { load_k(b01, b23, kf); }
    }

    for (int c = 0; c < nchunks; ++c) {
        if constexpr (quant) { qsa_k_frags<T>(kraw, odd, kf[0], kf[1]); }
        v8f sc[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { sc[i][e] = 0.f; }
#pragma unroll
            for (int t = 0; t < 2; ++t) { sc[i] = qsa_wmma_f16(kf[t], qf[i][t], sc[i]); }
        }
        v16s vf[2];
        if constexpr (quant) {
            if (hi == 0) { qsa_v_stage_store<T>(vstage[w], r, vraw, odd); }
        } else {
            const int kb0 = blk_of(b01, b23, 0), kb1 = blk_of(b01, b23, 1), kb2 = blk_of(b01, b23, 2), kb3 = blk_of(b01, b23, 3);
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int d = 32*w + 16*t + r;
                const uint2 v0 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb0 * 1024 + d * 4);
                const uint2 v1 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb1 * 1024 + d * 4);
                const uint2 v2 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb2 * 1024 + d * 4);
                const uint2 v3 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb3 * 1024 + d * 4);
                vf[t] = __builtin_bit_cast(v16s, (uint2[4]){v0, v1, v2, v3});
            }
        }
        // --- prefetch next chunk's descriptor and K (f16 fragments, or raw K and V blocks) ---
        uint32_t nb01 = 0, nb23 = 0, nmm[NM] = {};
        v16s kfn[2];
        if (c + 1 < nchunks) {
            load_desc(c + 1, nb01, nb23, nmm);
            if constexpr (quant) { load_raw(nb01, nb23, kraw, vraw); } else { load_k(nb01, nb23, kfn); }
        } else if constexpr (!quant) {
            kfn[0] = kf[0]; kfn[1] = kf[1];
        }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            if (i != w) {
                part[i][w][lane*2+0] = make_float4(sc[i][0], sc[i][1], sc[i][2], sc[i][3]);
                part[i][w][lane*2+1] = make_float4(sc[i][4], sc[i][5], sc[i][6], sc[i][7]);
            }
        }
        __syncthreads();
        if (w < 3) {
            float scf[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) { scf[e] = 0.f; }
#pragma unroll
            for (int w2 = 0; w2 < 8; ++w2) {
                if (w2 == w) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) { scf[e] += sc[w][e]; }
                } else {
                    const float4 a = part[w][w2][lane*2+0], b = part[w][w2][lane*2+1];
                    scf[0] += a.x; scf[1] += a.y; scf[2] += a.z; scf[3] += a.w; scf[4] += b.x; scf[5] += b.y; scf[6] += b.z; scf[7] += b.w;
                }
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int j = e >> 1, cc = 2*(e & 1) + hi;
                const uint32_t mk = mask_of(mm, j);
                const bool kv = (mk >> (own_qi * 4 + cc)) & 1u;
                if (maskq && kv) { scf[e] += qsa3_h2f(maskq[4 * blk_of(b01, b23, j) + cc]); }
                if (!kv) { scf[e] = -INFINITY; }
            }
            float mloc = scf[0];
#pragma unroll
            for (int e = 1; e < 8; ++e) { mloc = fmaxf(mloc, scf[e]); }
            mloc = fmaxf(mloc, __shfl_xor(mloc, 16));
            const float mnew  = fmaxf(m, mloc);
            const float alpha = mnew == -INFINITY ? 1.f : exp2f((m - mnew) * QSA3_L2E);
            float p[8], ls = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) { p[e] = mnew == -INFINITY ? 0.f : exp2f((scf[e] - mnew) * QSA3_L2E); ls += p[e]; }
            ls += __shfl_xor(ls, 16);
            l = l * alpha + ls;
            m = mnew;
            uint32_t pp[4], po[4];
#pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) {
                pp[e2] = (uint32_t) qsa3_f2h(p[2*e2]) | ((uint32_t) qsa3_f2h(p[2*e2+1]) << 16);
                po[e2] = (uint32_t) __shfl_xor((int) pp[e2], 16);
            }
            uint16_t ph[16];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint16_t mine  = (e & 1) ? (uint16_t) (pp[e>>1] >> 16) : (uint16_t) (pp[e>>1] & 0xffffu);
                const uint16_t other = (e & 1) ? (uint16_t) (po[e>>1] >> 16) : (uint16_t) (po[e>>1] & 0xffffu);
                ph[2*e]   = hi ? other : mine;
                ph[2*e+1] = hi ? mine  : other;
            }
            if (hi == 0) {
                ptile[w][r*2+0] = make_uint4((uint32_t) ph[0] | ((uint32_t) ph[1] << 16), (uint32_t) ph[2] | ((uint32_t) ph[3] << 16),
                                             (uint32_t) ph[4] | ((uint32_t) ph[5] << 16), (uint32_t) ph[6] | ((uint32_t) ph[7] << 16));
                ptile[w][r*2+1] = make_uint4((uint32_t) ph[8] | ((uint32_t) ph[9] << 16), (uint32_t) ph[10] | ((uint32_t) ph[11] << 16),
                                             (uint32_t) ph[12] | ((uint32_t) ph[13] << 16), (uint32_t) ph[14] | ((uint32_t) ph[15] << 16));
                alpha_s[own_row] = alpha;
            }
        }
        __syncthreads();
        float al[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) { al[i] = alpha_s[16*i + r]; }
        if (__ballot(al[0] != 1.f || al[1] != 1.f || al[2] != 1.f)) {
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int t = 0; t < 2; ++t)
#pragma unroll
                    for (int e = 0; e < 8; ++e) { O[i][t][e] *= al[i]; }
        }
        if constexpr (quant) { qsa_v_stage_load<T>(vstage[w], r, vf[0], vf[1]); }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const v16s pf = __builtin_bit_cast(v16s, (uint4[2]){ptile[i][r*2+0], ptile[i][r*2+1]});
#pragma unroll
            for (int t = 0; t < 2; ++t) { O[i][t] = qsa_wmma_f16(vf[t], pf, O[i][t]); }
        }
        if constexpr (!quant) { kf[0] = kfn[0]; kf[1] = kfn[1]; }
        b01 = nb01; b23 = nb23;
#pragma unroll
        for (int i = 0; i < NM; ++i) { mm[i] = nmm[i]; }
    }

    if (w < 3 && hi == 0) { l_s[own_row] = l; }
    __syncthreads();
    // output: per tile, stage [16 rows x 32 dims] F32 per wave in LDS (2 KB), then coalesced 16-byte row stores
    float * ostage = reinterpret_cast<float *>(&part[0][0][0]) + w * (16 * 32);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float li = l_s[16*i + r];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { ostage[r * 32 + 16*t + 2*e + hi] = li > 0.f ? O[i][t][e] / li : 0.f; }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int rr = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int row = 16*i + rr, qi = row / GQA, h = row - GQA*qi;
            const int query = G * g + qi;
            if (query < s.n_q) {
                float * dst = reinterpret_cast<float *>(reinterpret_cast<char *>(out) + (size_t) query * s.o2 + (size_t) (kvh * GQA + h) * s.o1) + 32*w + dq;
                *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<const float4 *>(ostage + rr * 32 + dq);
            }
        }
        __syncthreads();
    }
}

// [TAG_QSA_CAPS] The QSA kernels for one attention shape on a device with compute capability cc. Both need gfx11 wave32
// WMMA (RDNA3 gfx1100..1103 and RDNA3.5 gfx1150..1153; RDNA4's gfx12 WMMA has another fragment layout), head size 256
// and K, V of one type: f16, or q8_0 / q4_0 read from the cache in place. Decode takes any GQA (WMMA up to 16 query heads
// per KV head, scalar beyond); the prompt kernel fills its 48 rows with 4 queries x 12 heads or 8 queries x 6 heads.
uint32_t ggml_cuda_qsa_caps(const int cc, const int64_t head_size, const int64_t gqa, const ggml_type type_k, const ggml_type type_v) {
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || head_size != 256 || gqa < 1 || type_k != type_v || !qsa_kv_type_supported(type_k)) {
        return 0;
    }
    uint32_t caps = GGML_CUDA_QSA_DECODE;
    if (gqa == 12 || gqa == 6) {
        caps |= GGML_CUDA_QSA_PROMPT;
    }
    return caps;
}

bool ggml_cuda_flash_attn_ext_qsa_supported(const int device, const ggml_tensor * dst) {
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * m = dst->src[3], * ids = dst->src[5];
    const auto * packed = dst->src[6], * pv = dst->src[7];
    if (!q || !k || !v || !ids || dst->src[4] || k->ne[2] < 1 || q->ne[2] % k->ne[2] ||
        !(ggml_cuda_qsa_caps(ggml_cuda_info().devices[device].cc, k->ne[0], q->ne[2]/k->ne[2], k->type, v->type) & GGML_CUDA_QSA_PROMPT)) {
        return false;
    }
    if (q->ne[1] < 128) { return false; }
    float bias, softcap; memcpy(&bias, (const char *) dst->op_params + 4, 4); memcpy(&softcap, (const char *) dst->op_params + 8, 4);
    if (bias != 0 || softcap != 0 || q->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || q->ne[0] != 256 || k->ne[0] != 256 || v->ne[0] != 256 ||
        q->ne[1] < 1 || q->ne[1] > INT_MAX || k->ne[1] < 1 || k->ne[1] > 262140 || k->ne[1] % 4 ||
        v->ne[2] != k->ne[2] || v->ne[1] != k->ne[1] ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || q->nb[0] != 4 ||
        k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type) ||
        ids->nb[0] != 4 || ids->ne[1] < q->ne[1] || ids->ne[2] != 1 || ids->ne[3] != 1 || ids->ne[0] < 1 || ids->ne[0] > 2560 ||
        ids->nb[1] % 4 || uintptr_t(ids->data) % 4 || q->nb[1] % 16 || q->nb[2] % 16 || uintptr_t(q->data) % 16 ||
        !ggml_is_contiguous(dst)) { return false; }
    if (k->type == GGML_TYPE_F16) {
        // f16 reads the packed layouts the graph builds next to the cache
        if (!packed || !pv) { return false; }
        if (packed->type != GGML_TYPE_F16 || !ggml_is_contiguous(packed) || packed->ne[0] != 16 || packed->ne[1] != 4 ||
            packed->ne[2] != 16 || packed->ne[3] != k->ne[1]/4*k->ne[2] || uintptr_t(packed->data) % 16) { return false; }
        if (pv->type != GGML_TYPE_F16 || !ggml_is_contiguous(pv) || pv->ne[0] != 4 || pv->ne[1] != 256 ||
            pv->ne[2] != k->ne[1]/4*k->ne[2] || pv->ne[3] != 1 || uintptr_t(pv->data) % 8) { return false; }
    } else {
        // q8_0 / q4_0 rows are read in place through 4-byte aligned windows; there is no packed copy
        if (packed || pv || k->nb[1] % 4 || k->nb[2] % 4 || v->nb[1] % 4 || v->nb[2] % 4 ||
            uintptr_t(k->data) % 4 || uintptr_t(v->data) % 4) { return false; }
    }
    return !m || (m->type == GGML_TYPE_F16 && m->nb[0] == 2 && m->ne[0] >= k->ne[1] && m->ne[1] >= q->ne[1] && m->ne[2] == 1 && m->ne[3] == 1);
}

template <int G, int GQA, ggml_type T>
static void qsa3_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * m = dst->src[3], * ids = dst->src[5];
    const auto * pk = dst->src[6], * pv = dst->src[7];
    float scale; memcpy(&scale, dst->op_params, 4);
    const int n_q = (int) q->ne[1], ns = (int) ids->ne[0], nk = (int) k->ne[1];
    const int ngroups = (n_q + G - 1) / G;
    const int cap = (G * ns + 3) & ~3;
    ggml_cuda_pool_alloc<uint16_t>       ublk(ctx.pool(), (size_t) ngroups * cap);
    ggml_cuda_pool_alloc<qsa3_mask_t<G>> umask(ctx.pool(), (size_t) ngroups * cap);
    ggml_cuda_pool_alloc<int>            ucount(ctx.pool(), (size_t) ngroups);
    ggml_cuda_pool_alloc<int>            srow(ctx.pool(), (size_t) n_q * ns);
    ggml_cuda_pool_alloc<int>            sflag(ctx.pool(), (size_t) n_q);
    {
        const ggml_cuda_kernel_launch_params launch(dim3(n_q), dim3(256), (size_t) ns * sizeof(int), ctx.stream());
        ggml_cuda_kernel_launch(qsa3_rows_kernel, launch, (const int *) ids->data, ids->nb[1], ns, nk, srow.get(), sflag.get());
        CUDA_CHECK(cudaGetLastError());
        constexpr bool stage = G == 4;
        const ggml_cuda_kernel_launch_params launch2(dim3(ngroups), dim3(QSA3_MERGE_LANES), stage ? (size_t) G * ns * sizeof(int) : (size_t) 0, ctx.stream());
        ggml_cuda_kernel_launch(qsa3_merge_kernel<G, stage>, launch2, (const int *) ids->data, ids->nb[1], n_q, ns, nk,
                                (const int *) srow.get(), (const int *) sflag.get(), ublk.get(), umask.get(), ucount.get(), cap);
        CUDA_CHECK(cudaGetLastError());
    }
    qsa3_layout layout{q->nb[1], q->nb[2], dst->nb[1], dst->nb[2], m ? m->nb[1] : 0,
                       k->nb[1], k->nb[2], v->nb[1], v->nb[2], nk, n_q, cap, scale};
    const char * kd = (const char *) (T == GGML_TYPE_F16 ? pk->data : k->data);
    const char * vd = (const char *) (T == GGML_TYPE_F16 ? pv->data : v->data);
    const ggml_cuda_kernel_launch_params launch(dim3(ngroups, k->ne[2]), dim3(256), 0, ctx.stream());
    ggml_cuda_kernel_launch(qsa3_attn_kernel<G, GQA, T>, launch, (const float *) q->data, kd, vd,
                            m ? (const uint16_t *) m->data : nullptr, (const uint16_t *) ublk.get(), (const qsa3_mask_t<G> *) umask.get(),
                            (const int *) ucount.get(), (float *) dst->data, layout);
    CUDA_CHECK(cudaGetLastError());
}

template <int G, int GQA>
static void qsa3_launch_type(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[1]->type) {
        case GGML_TYPE_F16:  qsa3_launch<G, GQA, GGML_TYPE_F16>(ctx, dst);  break;
        case GGML_TYPE_Q8_0: qsa3_launch<G, GQA, GGML_TYPE_Q8_0>(ctx, dst); break;
        case GGML_TYPE_Q4_0: qsa3_launch<G, GQA, GGML_TYPE_Q4_0>(ctx, dst); break;
        default: GGML_ABORT("QSA: unsupported K/V type");
    }
}

void ggml_cuda_flash_attn_ext_qsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int gqa = (int) (dst->src[0]->ne[2] / dst->src[1]->ne[2]);
    if (gqa == 12) {
        qsa3_launch_type<4, 12>(ctx, dst);
    } else {
        GGML_ASSERT(gqa == 6);
        qsa3_launch_type<8, 6>(ctx, dst);
    }
}

#else // !defined(GGML_USE_HIP)

// QSA kernels use AMD WMMA builtins; other backends report the op as unsupported here.
uint32_t ggml_cuda_qsa_caps(int, int64_t, int64_t, ggml_type, ggml_type) { return 0; }
bool ggml_cuda_flash_attn_ext_qsa_supported(int, const ggml_tensor *) { return false; }
void ggml_cuda_flash_attn_ext_qsa(ggml_backend_cuda_context &, ggml_tensor *) { GGML_ABORT("QSA attention is HIP-only"); }
bool ggml_cuda_flash_attn_ext_qsa_decode_supported(int, const ggml_tensor *) { return false; }
void ggml_cuda_flash_attn_ext_qsa_decode(ggml_backend_cuda_context &, ggml_tensor *) { GGML_ABORT("QSA attention is HIP-only"); }

#endif // defined(GGML_USE_HIP)

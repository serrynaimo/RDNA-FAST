// ROCmFPX [TAG_EXPERT_HANDOFF] expert handoff, the host side (protocol in ggml-expert-handoff.h):
//  - ggml_compute_forward_expert_handoff: the send and join ops when the graph itself runs on the CPU backend
//  - ggml_expert_handoff_host_*: the pool of threads that answers a mailbox by computing the routed experts the
//    device does not hold, in place from host memory, while the device computes the ones it holds

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-expert-handoff.h"
#include "ggml-impl.h"
#include "ops.h"
#include "simd-mappings.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_WIN32) // [TAG_EH_POOL2] the physical cores
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define EH_PAUSE() _mm_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define EH_PAUSE() __asm__ __volatile__("yield")
#else
#define EH_PAUSE() std::this_thread::yield()
#endif

// the mailbox fields are plain uint32_t shared with the device: access them as atomics
static inline uint32_t eh_load(const uint32_t * p) {
    return reinterpret_cast<const std::atomic<uint32_t> *>(p)->load(std::memory_order_acquire);
}
static inline void eh_store(uint32_t * p, uint32_t v) {
    reinterpret_cast<std::atomic<uint32_t> *>(p)->store(v, std::memory_order_release);
}

static inline int64_t eh_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static ggml_expert_handoff_dims eh_dims(const ggml_tensor * dst) {
    ggml_expert_handoff_dims d;
    d.max_tokens = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MAX_TOKENS);
    d.max_embd   = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MAX_EMBD);
    d.max_used   = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MAX_USED);
    return d;
}

static void * eh_ptr(const ggml_tensor * dst, int lo_i, int hi_i) {
    const uint64_t lo = (uint32_t) ggml_get_op_params_i32(dst, lo_i);
    const uint64_t hi = (uint32_t) ggml_get_op_params_i32(dst, hi_i);
    return (void *) (uintptr_t) (lo | (hi << 32));
}

static uint8_t * eh_mailbox(const ggml_tensor * dst) {
    return (uint8_t *) eh_ptr(dst, GGML_EXPERT_HANDOFF_P_PTR_LO, GGML_EXPERT_HANDOFF_P_PTR_HI);
}

//
// the ops on the CPU backend
//

void ggml_compute_forward_expert_handoff(const ggml_compute_params * params, ggml_tensor * dst) {
    if (params->ith != 0) {
        return;
    }

    uint8_t * mb = eh_mailbox(dst);
    const ggml_expert_handoff_dims d = eh_dims(dst);
    auto * hdr = (ggml_expert_handoff_header *) mb;
    const int32_t layer = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_LAYER);
    int32_t * flags = (int32_t *) eh_ptr(dst, GGML_EXPERT_HANDOFF_P_FLAGS_LO, GGML_EXPERT_HANDOFF_P_FLAGS_HI);

    if (ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MODE) == GGML_EXPERT_HANDOFF_SEND) {
        const ggml_tensor * x     = dst->src[0];
        const ggml_tensor * ids   = dst->src[1];
        const ggml_tensor * map   = dst->src[2];
        const ggml_tensor * w     = dst->src[3];
        int32_t * slots = (int32_t *) dst->data;

        const int64_t n_embd   = x->ne[0];
        const int64_t n_tokens = ggml_nrows(x);
        const int64_t n_used   = ids->ne[0];

        // the misses, and the routing counts
        const int32_t miss_slot = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_MISS_SLOT);
        int32_t * counts = (int32_t *) eh_ptr(dst, GGML_EXPERT_HANDOFF_P_COUNTS_LO, GGML_EXPERT_HANDOFF_P_COUNTS_HI);
        const int64_t stride = ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_COUNTS_STRIDE);
        const int64_t n_map = ggml_nelements(map);
        int32_t n_miss = 0;
        for (int64_t i = 0; i < n_used*n_tokens; ++i) {
            const int32_t e = ((const int32_t *) ids->data)[i];
            slots[i] = e >= 0 && e < n_map ? ((const int32_t *) map->data)[e] : miss_slot;
            n_miss += slots[i] == miss_slot;
            if (counts && e >= 0 && e < stride) {
                counts[layer*stride + e]++;
            }
        }
        flags[layer] = n_miss;
        if (n_miss == 0) {
            return;
        }

        float   * mx = (float   *) (mb + ggml_expert_handoff_off_x(d));
        int32_t * mi = (int32_t *) (mb + ggml_expert_handoff_off_ids(d));
        int32_t * ms = (int32_t *) (mb + ggml_expert_handoff_off_slots(d));
        float   * mw = (float   *) (mb + ggml_expert_handoff_off_weights(d));

        for (int64_t t = 0; t < n_tokens; ++t) {
            memcpy(mx + t*d.max_embd, (const float *) x->data + t*n_embd, n_embd*sizeof(float));
            memcpy(mi + t*d.max_used, (const int32_t *) ids->data   + t*n_used, n_used*sizeof(int32_t));
            memcpy(ms + t*d.max_used, slots + t*n_used, n_used*sizeof(int32_t));
            memcpy(mw + t*d.max_used, (const float   *) w->data     + t*n_used, n_used*sizeof(float));
        }
        hdr->layer    = layer;
        hdr->n_tokens = (int32_t) n_tokens;
        hdr->n_embd   = (int32_t) n_embd;
        hdr->n_used   = (int32_t) n_used;

        eh_store(&hdr->seq_send, eh_load(&hdr->seq_send) + 1);
        return;
    }

    // join
    const ggml_tensor * y = dst->src[0];
    const int64_t n_embd   = y->ne[0];
    const int64_t n_tokens = ggml_nrows(y);

    if (flags[layer] == 0) {
        // nothing was sent
        memcpy(dst->data, y->data, ggml_nbytes(y));
        return;
    }

    const uint32_t want = eh_load(&hdr->seq_send);
    const int64_t timeout_ns = (int64_t) ggml_get_op_params_i32(dst, GGML_EXPERT_HANDOFF_P_TIMEOUT_MS)*1000000;
    const int64_t t0 = eh_now_ns();
    bool ok = true;
    uint32_t done = 0;
    while ((done = eh_load(&hdr->seq_done)) != want) {
        if (eh_now_ns() - t0 > timeout_ns) {
            ok = false;
            break;
        }
        EH_PAUSE();
    }
    if (!ok) {
        // [TAG_EH_TIMEOUT_DIAG] what it waited for and what it saw, for check()
        eh_store(&hdr->to_want, want);
        eh_store(&hdr->to_done, done);
        eh_store((uint32_t *) &hdr->to_layer, (uint32_t) layer);
        eh_store(&hdr->n_timeouts, eh_load(&hdr->n_timeouts) + 1);
    }

    const float * my = (const float *) (mb + ggml_expert_handoff_off_y(d));
    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * src = (const float *) y->data + t*n_embd;
        float       * out = (float       *) dst->data + t*n_embd;
        for (int64_t i = 0; i < n_embd; ++i) {
            out[i] = src[i] + (ok ? my[t*d.max_embd + i] : 0.0f);
        }
    }
}

//
// the host pool
//

namespace {

// [TAG_EH_MULTI_DOT] one weight row against the rows of several tokens (the tokens of a batch that miss the same
// expert): each row is read once for all of them
constexpr int eh_multi_max = 8;

// s[k] = row . ys[k] for k < ny: the generic path, one vec_dot per token (the ys of the row type's vec_dot type)
void eh_dot_n(const ggml_type_traits_cpu * tr, int n, const uint8_t * row, const uint8_t * const * ys, int ny, float * s) {
    for (int k = 0; k < ny; ++k) {
        tr->vec_dot(n, s + k, 0, row, 0, ys[k], 0, 1);
    }
}

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
#define EH_INT8_FAST 1
// [TAG_EH_Q2_ACT] expert rows against int8 activations (AVX-512 VBMI/VNNI), for the expert types below. An activation
// is int8 in chunks of 32 values with an f32 scale per chunk, 64-byte aligned, and per 64 values, lane by lane (16
// lanes of 4 values), its chunk's scale d16. Every weight type is unpacked, 64 weights at a time, to bytes w and per
// lane a scale a, the offsets going in one of three ways (eh_mode): the 32-block types and Q6_K unpack to signed codes
// (weight = a * w) and take the activations plus 128 (as unsigned bytes), and per 64 weights and token a VNNI dot
// started at -128 times the lane's weight sum (once for all the tokens) is the exact integer dot. The super-block
// types with an offset per sub-block of 32 (the K-quants' mins, IQ4_XS's 128 times its scales: weight = a * w - m,
// w unsigned) take the activations as they are, and their 8 offsets times the chunk sums h32 (d * the chunk's int8
// sum) once per super-block. Q2_0 (one scale per 64 weights) takes them as they are too, its offset times the two
// chunk sums a scalar per block and token: at 2 bits a weight one token is not bound by memory, and the weight sum
// costs it 3-8% there. A token then costs per 64 weights one dot, a convert and two float ops, and loads only its 64
// bytes and d16 (a third load, a per-lane offset sum, cost 25% at 8 tokens); the unpack is shared by all the tokens
// that miss the expert. One or two tokens use 4 or 2 accumulators, as one would chain every block's FMA on the last.
// The scale per 32 activations makes the K-quants' and IQ4_XS's dots 2-3x closer to the exact one than ggml's Q8_K
// (one scale per 256).
struct eh_act {
    const uint8_t * q;   // the int8 values (plus 128 for the biased types)
    const float   * d16;
    const float   * h32; // the others: per 32 values d * (sum of the chunk's int8)
};

// x (n values, n % 64 == 0) into q (n bytes) and d16 (n/4 floats); biased: q holds the values plus 128, else h32 (n/32
// floats) gets the chunk sums
void eh_act_quant(const float * x, int n, uint8_t * q, float * d16, float * h32, bool biased) {
    const __m512i ones = _mm512_set1_epi8(1);
    const __m128i bias = _mm_set1_epi8(biased ? (char) 0x80 : 0);
    for (int b = 0; b < n/64; ++b) {
        float d[2];
        __m512i v[4];
        for (int h = 0; h < 2; ++h) {
            const __m512 x0 = _mm512_loadu_ps(x + 64*b + 32*h);
            const __m512 x1 = _mm512_loadu_ps(x + 64*b + 32*h + 16);
            const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(x0), _mm512_abs_ps(x1)));
            const __m512 inv = _mm512_set1_ps(amax > 0.0f ? 127.0f/amax : 0.0f);
            v[2*h]     = _mm512_cvtps_epi32(_mm512_mul_ps(x0, inv));
            v[2*h + 1] = _mm512_cvtps_epi32(_mm512_mul_ps(x1, inv));
            d[h] = amax/127.0f;
        }
        uint8_t * qb = q + 64*b;
        for (int i = 0; i < 4; ++i) {
            _mm_storeu_si128((__m128i *) (qb + 16*i), _mm_xor_si128(_mm512_cvtsepi32_epi8(v[i]), bias));
        }
        const __m512 dv = _mm512_mask_blend_ps((__mmask16) 0xFF00, _mm512_set1_ps(d[0]), _mm512_set1_ps(d[1]));
        _mm512_storeu_ps(d16 + 16*b, dv);
        if (!biased) {
            // the lane sums: a VNNI dot with ones adds each lane's 4 bytes
            const __m512i sum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, _mm512_load_si512((const void *) qb));
            const __m512 sv = _mm512_mul_ps(dv, _mm512_cvtepi32_ps(sum));
            h32[2*b]     = _mm512_mask_reduce_add_ps((__mmask16) 0x00FF, sv);
            h32[2*b + 1] = _mm512_mask_reduce_add_ps((__mmask16) 0xFF00, sv);
        }
    }
}

// two 16-byte nibble runs (two blocks of 32: low nibbles then high nibbles of each) -> 64 codes in value order, in the
// low 4 bits of each byte (the high 4 are left over: the lookups below take 16 entries repeated 4 times)
static inline __m512i eh_nib4(const uint8_t * p0, const uint8_t * p1) {
    const __m512i v = _mm512_mask_broadcast_i32x4(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) p0)),
                                                  (__mmask16) 0xFF00, _mm_loadu_si128((const __m128i *) p1));
    const long long s4 = 0x0004000400040004LL; // the high nibbles in the 2nd and 4th 16 bytes
    return _mm512_srlv_epi16(v, _mm512_set_epi64(s4, s4, 0, 0, s4, s4, 0, 0));
}

static inline __m512 eh_two(float s0, float s1) {
    return _mm512_mask_blend_ps((__mmask16) 0xFF00, _mm512_set1_ps(s0), _mm512_set1_ps(s1));
}

// lanes 0-7: v[i], lanes 8-15: v[i + 1] (v: 8 floats)
static inline __m512 eh_pair(__m256 v, int i) {
    return _mm512_permutexvar_ps(_mm512_add_epi32(_mm512_set1_epi32(i),
        _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)), _mm512_castps256_ps512(v));
}

// the 8 six-bit scales and mins of a Q4_K / Q5_K super-block (the steps of ggml's x86 K-quant dots)
static inline void eh_k4_scales(const uint8_t * q, float d, float dmin, __m256 & sc, __m256 & mn) {
    uint32_t utmp[4];
    memcpy(utmp, q, 12);
    const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    sc = _mm256_mul_ps(_mm256_set1_ps(d),    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) &utmp[0]))));
    mn = _mm256_mul_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) &utmp[2]))));
}

// 64-byte lookup tables for vpermb (the low 6 bits of the index): 16 codes repeated 4 times; IQ4_XS's plus 128
#define EH_LUT16(...) { __VA_ARGS__, __VA_ARGS__, __VA_ARGS__, __VA_ARGS__ }
alignas(64) static const int8_t  eh_lut_q4_0[64]   = EH_LUT16(-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
alignas(64) static const int8_t  eh_lut_iq4nl[64]  = EH_LUT16(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);
alignas(64) static const int8_t  eh_lut_mxfp4[64]  = EH_LUT16(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
alignas(64) static const uint8_t eh_lut_iq4nl_u[64] = EH_LUT16(1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241);
#undef EH_LUT16

// how a type's offsets go in (see above)
enum eh_mode {
    EH_BIASED, // signed codes against the activations plus 128
    EH_MINS,   // unsigned codes; the 8 offsets of the current super-block in c8, against the chunk sums
    EH_OFF64,  // unsigned codes, one scale per 64 weights; the block's offset in ak, against its two chunk sums
};

// the unpackers: block b of 64 weights of a row -> its bytes w and lane scales a
struct eh_up_q2_0 {
    static constexpr eh_mode kMode = EH_OFF64;
    static constexpr int     kQ    = 64;
    const uint8_t * row;
    float ak; // the block's offset: its scale times 1 (codes 0-3 for -1..2)
    void get(int b, __m512i & w, __m512 & a) {
        const uint8_t * blk = row + (size_t) b*18;
        uint16_t h;
        memcpy(&h, blk, 2);
        w = _mm512_and_si512(_mm512_multishift_epi64_epi8(_mm512_set1_epi64(0x0E0C0A0806040200LL),
                _mm512_cvtepu16_epi64(_mm_loadu_si128((const __m128i *) (blk + 2)))), _mm512_set1_epi8(3));
        ak = _cvtsh_ss(h);
        a = _mm512_set1_ps(ak);
    }
};

// the 32-block 4-bit types: two blocks of a byte scale (or two) and 16 bytes of nibbles each
template <int kBlk, int kHdr, const int8_t * kLut>
struct eh_up_nib32 {
    static constexpr eh_mode kMode = EH_BIASED;
    static constexpr int     kQ    = 64;
    const uint8_t * row;
    __m512i lut = _mm512_load_si512((const void *) kLut);
    void get(int b, __m512i & w, __m512 & a) {
        const uint8_t * b0 = row + (size_t) (2*b)*kBlk;
        const uint8_t * b1 = b0 + kBlk;
        w = _mm512_permutexvar_epi8(eh_nib4(b0 + kHdr, b1 + kHdr), lut);
        if constexpr (kHdr == 2) {
            uint16_t h0, h1;
            memcpy(&h0, b0, 2);
            memcpy(&h1, b1, 2);
            a = eh_two(_cvtsh_ss(h0), _cvtsh_ss(h1));
        } else {
            a = eh_two(GGML_E8M0_TO_FP32_HALF(b0[0]), GGML_E8M0_TO_FP32_HALF(b1[0]));
        }
    }
};
using eh_up_q4_0   = eh_up_nib32<18, 2, eh_lut_q4_0>;
using eh_up_iq4_nl = eh_up_nib32<18, 2, eh_lut_iq4nl>;
using eh_up_mxfp4  = eh_up_nib32<17, 1, eh_lut_mxfp4>;

struct eh_up_q8_0 {
    static constexpr eh_mode kMode = EH_BIASED;
    static constexpr int     kQ    = 64;
    const uint8_t * row;
    void get(int b, __m512i & w, __m512 & a) {
        const uint8_t * b0 = row + (size_t) (2*b)*34;
        const uint8_t * b1 = b0 + 34;
        uint16_t h0, h1;
        memcpy(&h0, b0, 2);
        memcpy(&h1, b1, 2);
        w = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) (b0 + 2))),
                               _mm256_loadu_si256((const __m256i *) (b1 + 2)), 1);
        a = eh_two(_cvtsh_ss(h0), _cvtsh_ss(h1));
    }
};

struct eh_up_iq4_xs {
    static constexpr eh_mode kMode = EH_MINS;
    static constexpr int     kQ    = 256;
    const uint8_t * row;
    __m512i lut = _mm512_load_si512((const void *) eh_lut_iq4nl_u);
    int    cur = -1;
    __m256 dl, c8; // the 8 scales, and 128 times them (the codes come plus 128)
    void get(int b, __m512i & w, __m512 & a) {
        const block_iq4_xs * x = (const block_iq4_xs *) row + b/4;
        const int j = b % 4;
        if (b/4 != cur) {
            cur = b/4;
            // the 8 six-bit scales: four bits each from scales_l, two from scales_h, spread into bytes
            uint32_t sl;
            memcpy(&sl, x->scales_l, 4);
            const uint64_t ls = _pdep_u64(sl, 0x0F0F0F0F0F0F0F0FULL) | (_pdep_u64(x->scales_h, 0x0303030303030303ULL) << 4);
            const __m256 v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) ls)));
            dl = _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x->d)), _mm256_sub_ps(v, _mm256_set1_ps(32.0f)));
            c8 = _mm256_mul_ps(dl, _mm256_set1_ps(128.0f));
        }
        w = _mm512_permutexvar_epi8(eh_nib4(x->qs + 32*j, x->qs + 32*j + 16), lut);
        a = eh_pair(dl, 2*j);
    }
};

// Q4_K / Q5_K: per 64 weights, 32 bytes of nibbles (low: the first 32 values, high: the next 32), two sub-block
// scales and mins
struct eh_up_q4_K {
    static constexpr eh_mode kMode = EH_MINS;
    static constexpr int     kQ    = 256;
    const uint8_t * row;
    int    cur = -1;
    __m256 a8, c8;
    void get(int b, __m512i & u, __m512 & a) {
        const block_q4_K * x = (const block_q4_K *) row + b/4;
        const int j = b % 4;
        if (b/4 != cur) {
            cur = b/4;
            ggml_fp16_t dm[2]; // d, then dmin (the union is named differently in C and C++)
            memcpy(dm, x, sizeof(dm));
            eh_k4_scales(x->scales, GGML_CPU_FP16_TO_FP32(dm[0]), GGML_CPU_FP16_TO_FP32(dm[1]), a8, c8);
        }
        const __m256i y = _mm256_loadu_si256((const __m256i *) (x->qs + 32*j));
        const __m256i mk = _mm256_set1_epi8(0x0F);
        u = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(y, mk)),
                               _mm256_and_si256(_mm256_srli_epi16(y, 4), mk), 1);
        a = eh_pair(a8, 2*j);
    }
};

struct eh_up_q5_K {
    static constexpr eh_mode kMode = EH_MINS;
    static constexpr int     kQ    = 256;
    const uint8_t * row;
    int    cur = -1;
    __m256 a8, c8;
    void get(int b, __m512i & u, __m512 & a) {
        const block_q5_K * x = (const block_q5_K *) row + b/4;
        const int j = b % 4;
        if (b/4 != cur) {
            cur = b/4;
            ggml_fp16_t dm[2]; // d, then dmin (the union is named differently in C and C++)
            memcpy(dm, x, sizeof(dm));
            eh_k4_scales(x->scales, GGML_CPU_FP16_TO_FP32(dm[0]), GGML_CPU_FP16_TO_FP32(dm[1]), a8, c8);
        }
        const __m256i y  = _mm256_loadu_si256((const __m256i *) (x->qs + 32*j));
        const __m256i qh = _mm256_loadu_si256((const __m256i *) x->qh);
        const __m256i mk = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
        // the fifth bit: bit 2j of qh[l] for the first 32 values, bit 2j + 1 for the next 32
        const __m256i b0 = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(2*j)), one);
        const __m256i b1 = _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(2*j + 1)), one);
        const __m256i lo = _mm256_or_si256(_mm256_and_si256(y, mk), _mm256_slli_epi16(b0, 4));
        const __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(y, 4), mk), _mm256_slli_epi16(b1, 4));
        u = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        a = eh_pair(a8, 2*j);
    }
};

// Q6_K (signed: the codes less 32): per 128 weights (half a super-block) ql 64 bytes, qh 32 bytes, 8 sub-block scales
// of 16 weights; the first 64 weights take the low nibbles of ql and qh bits 0-3, the next 64 the high nibbles and qh
// bits 4-7
struct eh_up_q6_K {
    static constexpr eh_mode kMode = EH_BIASED;
    static constexpr int     kQ    = 256;
    const uint8_t * row;
    int   cur = -1;
    float sd[16]; // the super-block's 16 scales times d
    void get(int b, __m512i & w, __m512 & a) {
        const block_q6_K * x = (const block_q6_K *) row + b/4;
        const int j = b % 4;
        const int n = j/2, p = j % 2;
        const uint8_t * ql = x->ql + 64*n;
        const uint8_t * qh = x->qh + 32*n;
        if (b/4 != cur) {
            cur = b/4;
            const __m512 d = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(x->d));
            _mm512_storeu_ps(sd, _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *) x->scales)))));
        }
        const __m256i L0 = _mm256_loadu_si256((const __m256i *) ql);
        const __m256i L1 = _mm256_loadu_si256((const __m256i *) (ql + 32));
        const __m256i H  = _mm256_loadu_si256((const __m256i *) qh);
        const __m256i mk = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(3);
        const __m128i sl = _mm_cvtsi32_si128(4*p), sh0 = _mm_cvtsi32_si128(4*p), sh1 = _mm_cvtsi32_si128(4*p + 2);
        const __m256i lo = _mm256_or_si256(_mm256_and_si256(_mm256_srl_epi16(L0, sl), mk),
                                           _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(H, sh0), m3), 4));
        const __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_srl_epi16(L1, sl), mk),
                                           _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(H, sh1), m3), 4));
        w = _mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1), _mm512_set1_epi8(32));
        // lanes of 4 weights: lanes 0-3 take scale 8n + 4p, 4-7 the next, ...
        const int s0 = 8*n + 4*p;
        a = _mm512_permutexvar_ps(_mm512_add_epi32(_mm512_set1_epi32(s0),
                _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)), _mm512_loadu_ps(sd));
    }
};

// out[t] = row . a[t] for NT tokens; n values per row (a multiple of UP::kQ)
template <int NT, typename UP>
void eh_rows(const uint8_t * row, int n, const eh_act * act, float * out) {
    constexpr int U = NT == 1 ? 4 : NT == 2 ? 2 : 1;
    // EH_MINS: a sub-block's offset weighs the sum of its 32 activations, so the offsets go in once per super-block:
    // its 8 offsets times the 8 chunk sums (as a vector per 64 weights they cost more than the dot: 4 tokens 407 ns per
    // row of 2560 for Q4_K against ggml's 229)
    __m512 acc[U][NT];
    __m256 accm[NT];
    float  accs[NT] = {};
    for (int k = 0; k < U; ++k) {
        for (int t = 0; t < NT; ++t) {
            acc[k][t] = _mm512_setzero_ps();
        }
    }
    for (int t = 0; t < NT; ++t) {
        accm[t] = _mm256_setzero_ps();
    }
    UP up;
    up.row = row;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i c128 = _mm512_set1_epi8((char) 0x80);
    // inlined by force: from 3-6 tokens up clang called it per block, with the accumulators on the stack (Q4_K at 4
    // tokens 342 ns per row of 2560 against 220 for ggml)
    auto block = [&](int b, int k) __attribute__((always_inline)) {
        __m512i w;
        __m512 a;
        up.get(b, w, a);
        __m512i init = zero;
        if constexpr (UP::kMode == EH_MINS) {
            if (b % 4 == 0) {
                for (int t = 0; t < NT; ++t) {
                    accm[t] = _mm256_fmadd_ps(up.c8, _mm256_loadu_ps(act[t].h32 + 2*b), accm[t]);
                }
            }
        } else if constexpr (UP::kMode == EH_BIASED) {
            // the activations come plus 128: each lane's dot starts at -128 times its weights' sum
            init = _mm512_sub_epi32(zero, _mm512_dpbusd_epi32(zero, c128, w));
        }
        for (int t = 0; t < NT; ++t) {
            const __m512i x = _mm512_load_si512((const void *) (act[t].q + (size_t) b*64));
            const __m512i dot = UP::kMode == EH_BIASED ? _mm512_dpbusd_epi32(init, x, w) : _mm512_dpbusd_epi32(zero, w, x);
            if constexpr (UP::kMode == EH_OFF64) {
                accs[t] += up.ak*(act[t].h32[2*b] + act[t].h32[2*b + 1]);
            }
            const __m512 d16 = _mm512_loadu_ps(act[t].d16 + 16*b);
            acc[k][t] = _mm512_fmadd_ps(a, _mm512_mul_ps(_mm512_cvtepi32_ps(dot), d16), acc[k][t]);
        }
    };
    const int nb = n/64;
    int b = 0;
    for (; b + U <= nb; b += U) {
        // unrolled by force too: as a loop (Q5_K, 1 token) the accumulators went through the stack
#pragma GCC unroll 4
        for (int k = 0; k < U; ++k) {
            block(b + k, k);
        }
    }
    for (; b < nb; ++b) {
        block(b, 0);
    }
    for (int t = 0; t < NT; ++t) {
        __m512 s = acc[0][t];
        for (int k = 1; k < U; ++k) {
            s = _mm512_add_ps(s, acc[k][t]);
        }
        float r = _mm512_reduce_add_ps(s) - accs[t];
        if constexpr (UP::kMode == EH_MINS) {
            __m128 m = _mm_add_ps(_mm256_castps256_ps128(accm[t]), _mm256_extractf128_ps(accm[t], 1));
            m = _mm_hadd_ps(m, m);
            m = _mm_hadd_ps(m, m);
            r -= _mm_cvtss_f32(m);
        }
        out[t] = r;
    }
}

template <typename UP>
void eh_rows_n(const uint8_t * row, int n, const eh_act * a, int nt, float * out) {
    switch (nt) {
        case 1: eh_rows<1, UP>(row, n, a, out); break;
        case 2: eh_rows<2, UP>(row, n, a, out); break;
        case 3: eh_rows<3, UP>(row, n, a, out); break;
        case 4: eh_rows<4, UP>(row, n, a, out); break;
        case 5: eh_rows<5, UP>(row, n, a, out); break;
        case 6: eh_rows<6, UP>(row, n, a, out); break;
        case 7: eh_rows<7, UP>(row, n, a, out); break;
        default: eh_rows<8, UP>(row, n, a, out); break;
    }
}

// calls f with a null pointer to the unpacker of type t; false for the types without an int8 path
template <typename F>
static bool eh_with_up(ggml_type t, F && f) {
    switch (t) {
        case GGML_TYPE_Q2_0:   f((eh_up_q2_0   *) nullptr); return true;
        case GGML_TYPE_Q4_0:   f((eh_up_q4_0   *) nullptr); return true;
        case GGML_TYPE_Q8_0:   f((eh_up_q8_0   *) nullptr); return true;
        case GGML_TYPE_IQ4_NL: f((eh_up_iq4_nl *) nullptr); return true;
        case GGML_TYPE_MXFP4:  f((eh_up_mxfp4  *) nullptr); return true;
        case GGML_TYPE_IQ4_XS: f((eh_up_iq4_xs *) nullptr); return true;
        case GGML_TYPE_Q4_K:   f((eh_up_q4_K   *) nullptr); return true;
        case GGML_TYPE_Q5_K:   f((eh_up_q5_K   *) nullptr); return true;
        case GGML_TYPE_Q6_K:   f((eh_up_q6_K   *) nullptr); return true;
        default:               return false;
    }
}

// rows of n values of type t have an int8 path (n a multiple of its unpack size)
bool eh_fast_ok(ggml_type t, int64_t n) {
    bool ok = false;
    eh_with_up(t, [&](auto up) { ok = n % std::remove_pointer_t<decltype(up)>::kQ == 0; });
    return ok;
}

// whether type t takes the activations plus 128 (EH_BIASED), for eh_act_quant
bool eh_fast_biased(ggml_type t) {
    bool biased = false;
    eh_with_up(t, [&](auto up) { biased = std::remove_pointer_t<decltype(up)>::kMode == EH_BIASED; });
    return biased;
}

void eh_fast_dot(ggml_type t, const uint8_t * row, int n, const eh_act * a, int nt, float * out) {
    if (!eh_with_up(t, [&](auto up) { eh_rows_n<std::remove_pointer_t<decltype(up)>>(row, n, a, nt, out); })) {
        GGML_ABORT("fatal error");
    }
}
#endif

struct eh_layer {
    bool    set = false;
    ggml_expert_handoff_layer l = {};
    int64_t n_embd   = 0;
    int64_t n_ff     = 0;
    int64_t n_expert = 0;
    bool    has_gate = true;
    int64_t gu_rows  = 0; // rows of the gate/up phase per expert: 2*n_ff, or n_ff without a gate
    ggml_type type_gu   = GGML_TYPE_COUNT; // gate/up weights
    ggml_type type_down = GGML_TYPE_COUNT;
    ggml_type vt_gu     = GGML_TYPE_COUNT; // their vec_dot types
    ggml_type vt_down   = GGML_TYPE_COUNT;
    std::vector<uint64_t> counts;
};

// one job: the misses of one layer
struct eh_miss {
    int32_t t;      // token
    int32_t e;      // expert
    float   w;      // routing weight
};

// a byte buffer with a 64-byte aligned start, grown as needed
struct eh_abuf {
    std::vector<uint8_t> v;
    uint8_t * p = nullptr;
    size_t cap = 0;
    uint8_t * get(size_t n) {
        if (n > cap) {
            v.assign(n + 64, 0);
            p = (uint8_t *) (((uintptr_t) v.data() + 63) & ~(uintptr_t) 63);
            cap = n;
        }
        return p;
    }
};

// [TAG_EH_POOL2] the physical cores of processor group 0 (Windows; 0 elsewhere: unknown)
int eh_n_cores() {
    int res = 0;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        return res;
    }
    std::vector<uint8_t> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        return res;
    }
    for (DWORD off = 0; off < len; ) {
        const auto * info = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *) (buf.data() + off);
        if (info->Relationship == RelationProcessorCore && info->Processor.GroupCount >= 1 &&
            info->Processor.GroupMask[0].Group == 0 && info->Processor.GroupMask[0].Mask != 0) {
            res++;
        }
        off += info->Size;
    }
#endif
    return res;
}

} // namespace

// [TAG_EH_POOL2] The pool. Worker 0 alone polls the mailbox; it reads a send, prepares the job and runs it in phases. A phase is published by a
// generation bump; every worker (worker 0 too) claims its tasks (about three per worker, ranges of rows or columns)
// with a CAS on one word holding the generation, the task count and the next task, and worker 0 waits for the phase's
// count of finished tasks before the next one. Phases: the inputs into the dot type (inline for a few tokens), gate
// and up rows with the activation applied in the same task, the activation into the down dot type (inline for a few
// misses), the down rows by output columns over all missed experts (each task writes its columns of y). The other
// workers spin on a cached word of their own, never on the mailbox (mapped into the device), and a worker that sleeps
// or is late only leaves its tasks to the others: no barrier waits for all of them (the pool wake races of 2026-11-07
// were barriers that waited for a sleeping worker). Before: 4 barriers per layer and every worker polled the mailbox;
// host time per layer was ~2-3x the standalone benchmark in the server (Flash-Next, 2026-11-08). One worker per physical
// core but one by default (11 on a 12-core Ryzen: 142-158 us per layer against 240 for 6); pinning them one per core
// measured the same and is not done.
struct ggml_expert_handoff_host {
    uint8_t * mb = nullptr;
    ggml_expert_handoff_dims dims = {};
    ggml_expert_handoff_header * hdr = nullptr;

    std::vector<eh_layer> layers;

    int n_workers = 1;
    std::vector<std::thread> threads;

    std::atomic<bool> stop{false};
    std::atomic<int64_t> spin_until{0};
    std::mutex mtx;
    std::condition_variable cv;
    uint64_t wake_gen = 0; // under mtx
    std::atomic<int> n_asleep{0};
    static constexpr int64_t active_ns = 1000*1000*1000; // in use: answered within the last second
    std::atomic<int64_t> t_active{0};

    // spin this long after the last work or wake; the context also calls idle() once a graph's results are in, so the
    // spinning threads do not take the CPU from whatever runs between graphs (an MTP draft's own CPU work)
    static constexpr int64_t linger_ns = 20*1000*1000;

    uint32_t seq_base = 0; // seq_send when the pool was made

    // measuring aids (environment, read when the pool is made; the answers are wrong with either):
    // ROCMFPX_EH_SIM_RESIDENT=N reads expert e's weights from expert e % N of the layer, so a model bigger than RAM
    //   runs with the host's compute and memory traffic of one whose experts all fit in RAM (no page faults);
    // ROCMFPX_EH_SIM_NOHOST=1 answers every send with zeros at once: the device's side alone
    int64_t sim_resident = 0;
    bool    sim_nohost   = false;
    int64_t de(int32_t e) const { return sim_resident > 0 ? e % sim_resident : e; }

    // [TAG_EH_POOL2] the phases
    enum { PH_XQ = 0, PH_GU = 1, PH_HQ = 2, PH_DN = 3, PH_FETCH = 4 };
    alignas(64) std::atomic<uint64_t> claim{0};   // generation (24 bits) << 40 | tasks (20) << 20 | next task (20)
    alignas(64) std::atomic<uint32_t> gen{0};     // the generation of the last published phase
    alignas(64) std::atomic<int32_t>  n_done{0};  // tasks of the current phase finished
    alignas(64) int     phase    = 0;
    int64_t             ph_units = 0;             // rows, columns, tokens or misses, split into the tasks' ranges
    int                 ph_tasks = 0;

    // the job, prepared by worker 0
    uint32_t seq = 0;
    const eh_layer * job_layer = nullptr;
    int32_t n_tokens = 0;
    bool fast_gu = false, fast_dn = false;       // [TAG_EH_Q2_ACT] the int8 path for gate/up, for down
    std::vector<eh_miss> misses;
    std::vector<int32_t> miss_tokens;      // the tokens with misses
    std::vector<int32_t> tok_slot;         // [n_tokens]: the token's index in miss_tokens, -1 without misses
    std::vector<int32_t> uniq;             // the missed experts, each once
    std::vector<int32_t> uniq_begin;       // per missed expert: its first entry in by_expert, then the end
    std::vector<int32_t> by_expert;        // miss indices grouped by expert (a batch's tokens share an expert's reads)
    std::vector<int32_t> uniq_of;          // [n_expert] scratch: index in uniq or -1 (reset after each job)
    std::vector<int32_t> cnt, fill;        // scratch of prepare()
    std::vector<uint8_t> xq;               // generic: [max_tokens] rows of the gate/up vec_dot type
    size_t xq_row = 0;
    eh_abuf xa_q;                          // int8 path: [n_tokens][n_embd] bytes, d16 ([n_tokens][n_embd/4]) and
    std::vector<float> xa_d, xa_h;         //   the chunk sums ([n_tokens][n_embd/32])
    std::vector<float> hf;                 // [n_miss][n_ff] the activations, f32
    std::vector<uint8_t> hq;               // generic: [n_miss] rows of the down vec_dot type
    size_t hq_row = 0;
    eh_abuf ha_q;                          // int8 path: [n_miss][n_ff] bytes, d16 and the chunk sums
    std::vector<float> ha_d, ha_h;
    std::vector<std::vector<float>> acc;   // [worker] the down task's sums: [miss token][column]

    // stats
    std::atomic<uint64_t> st_layers{0};
    std::atomic<uint64_t> st_routed{0};
    std::atomic<uint64_t> st_host{0};
    std::atomic<uint64_t> st_busy_ns{0};
    // [TAG_EH_TIMEOUT_DIAG] worker 0 writes these; get_stats may read a job's fields torn, which only blurs a diagnostic
    std::atomic<uint64_t> st_multi{0};
    std::atomic<uint64_t> st_busy_multi{0};
    std::atomic<uint64_t> st_skipped{0};
    std::atomic<uint64_t> st_max_ns{0};
    std::atomic<int32_t>  st_max_layer{-1};
    std::atomic<int32_t>  st_max_tokens{0};
    std::atomic<int32_t>  st_max_miss{0};
    std::atomic<int32_t>  st_max_uniq{0};
    std::atomic<uint64_t> st_max_step[4] = {};
    std::atomic<uint64_t> st_step[5] = {}; // [TAG_EH_LOCAL_SUM]
    std::atomic<uint64_t> st_gather{0};      // [TAG_EH_LOCAL_SUM]
    std::atomic<uint64_t> st_gap{0};
    std::atomic<uint64_t> st_n_gap{0};
    int64_t t_prev_end = 0;                  // worker 0: when the last answer was stored
    std::atomic<uint64_t> st_read_copy{0};   // [TAG_EH_RESIDENT]
    std::atomic<uint64_t> st_read_tensor{0};

    // [TAG_EH_FILE_READ] the job's missed experts outside the RAM copy, read from file copies into stg before the
    // gate/up phase (one read per expert and tensor, spread over the workers); stg_ptr[k][expert] is the staged start
    // of tensor kind k (0 gate, 1 up, 2 gate_up, 3 down), set for this job only
    struct eh_fetch {
        int      k;
        int64_t  ed;
        uint64_t aoff;   // the aligned read
        size_t   alen;
        size_t   dst;    // its place in stg
        size_t   skip;   // the expert's start in it
    };
    std::vector<eh_fetch> fetch;
    std::vector<uint8_t>  stg_v;
    uint8_t *             stg     = nullptr;
    size_t                stg_cap = 0;
    std::vector<const uint8_t *> stg_ptr[4];
    std::atomic<uint64_t> st_file_bytes{0};
    std::atomic<uint64_t> st_file_ns{0};
    std::atomic<uint64_t> st_file_fail{0};
    int32_t job_il = -1;
    int64_t t_seen = 0;

    void wake() {
        spin_until.store(eh_now_ns() + linger_ns, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mtx);
            wake_gen++;
        }
        cv.notify_all();
    }

    // worker 0: wait for a new send; false if the pool stops. Past the spin window it looks every millisecond (a
    // send can come long after the wake: the first kernels of a cold device load for hundreds of milliseconds)
    bool wait_send(uint32_t seen, uint32_t & s, uint64_t & my_wake) {
        for (uint32_t k = 0;; ++k) {
            s = eh_load(&hdr->seq_send);
            if (s != seen) {
                return true;
            }
            if (stop.load(std::memory_order_acquire)) {
                return false;
            }
            if ((k & 255) == 0 && eh_now_ns() > spin_until.load(std::memory_order_acquire)) {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait_for(lk, std::chrono::milliseconds(1), [&] { return stop.load() || wake_gen != my_wake; });
                my_wake = wake_gen;
                continue;
            }
            EH_PAUSE();
        }
    }

    // workers 1..: wait for a phase newer than seen; false if the pool stops. Past the spin window they sleep until
    // worker 0 publishes (it wakes the sleepers) or a wake; a pool in use sleeps a millisecond at most
    bool wait_gen(uint32_t seen, uint32_t & g, uint64_t & my_wake) {
        for (uint32_t k = 0;; ++k) {
            g = gen.load(std::memory_order_acquire);
            if (g != seen) {
                return true;
            }
            if (stop.load(std::memory_order_acquire)) {
                return false;
            }
            if ((k & 255) == 0 && eh_now_ns() > spin_until.load(std::memory_order_acquire)) {
                n_asleep.fetch_add(1, std::memory_order_seq_cst);
                {
                    std::unique_lock<std::mutex> lk(mtx);
                    auto ready = [&] { return stop.load() || wake_gen != my_wake || gen.load(std::memory_order_seq_cst) != seen; };
                    if (eh_now_ns() - t_active.load(std::memory_order_relaxed) < active_ns) {
                        cv.wait_for(lk, std::chrono::milliseconds(1), ready);
                    } else {
                        cv.wait(lk, ready);
                    }
                    my_wake = wake_gen;
                }
                n_asleep.fetch_sub(1, std::memory_order_seq_cst);
                continue;
            }
            EH_PAUSE();
        }
    }

    // worker 0: a phase of `units` for the workers, in about three tasks per worker
    uint32_t publish(int ph, int64_t units) {
        ph_tasks = (int) std::max<int64_t>(1, std::min<int64_t>(units, 3*(int64_t) n_workers));
        phase    = ph;
        ph_units = units;
        n_done.store(0, std::memory_order_relaxed);
        const uint32_t g = gen.load(std::memory_order_relaxed) + 1;
        claim.store(((uint64_t) (g & 0xFFFFFF) << 40) | ((uint64_t) ph_tasks << 20), std::memory_order_release);
        gen.store(g, std::memory_order_seq_cst);
        if (n_asleep.load(std::memory_order_seq_cst) > 0) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                wake_gen++;
            }
            cv.notify_all();
        }
        return g;
    }

    bool claim_task(uint32_t g, int & i, int & n) {
        uint64_t v = claim.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (v >> 40) != (g & 0xFFFFFF)) {
                return false;
            }
            n = (int) ((v >> 20) & 0xFFFFF);
            i = (int) (v & 0xFFFFF);
            if (i >= n) {
                return false;
            }
            if (claim.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                return true;
            }
        }
    }

    void run_tasks(int iw, uint32_t g) {
        int i = 0, n = 0;
        while (claim_task(g, i, n)) {
            const int64_t u0 = ph_units*i/n;
            const int64_t u1 = ph_units*(i + 1)/n;
            switch (phase) {
                case PH_XQ: task_xq(u0, u1); break;
                case PH_GU: task_gu(u0, u1); break;
                case PH_HQ: task_hq(u0, u1); break;
                case PH_DN: task_dn(iw, u0, u1); break;
                case PH_FETCH: task_fetch(u0, u1); break;
                default: break;
            }
            n_done.fetch_add(1, std::memory_order_release);
        }
    }

    // worker 0: a phase, done by all; small ones inline
    void run_phase(int ph, int64_t units, int64_t inline_max) {
        if (units <= inline_max) {
            phase    = ph;
            ph_units = units;
            switch (ph) {
                case PH_XQ: task_xq(0, units); break;
                case PH_HQ: task_hq(0, units); break;
                default: GGML_ABORT("fatal error");
            }
            return;
        }
        const uint32_t g = publish(ph, units);
        run_tasks(0, g);
        while (n_done.load(std::memory_order_acquire) < ph_tasks) {
            EH_PAUSE();
        }
    }

    // worker 0: read the send, list the misses
    void prepare(uint32_t s) {
        // [TAG_EH_TIMEOUT_DIAG] each send bumps seq_send by one: a bigger step means sends this pool never took up
        if (s - seq > 1) {
            st_skipped.fetch_add(s - seq - 1, std::memory_order_relaxed);
        }
        seq = s;
        t_seen = eh_now_ns();

        // [TAG_EH_FILE_READ] the last job's staged experts are gone
        for (const eh_fetch & f : fetch) {
            stg_ptr[f.k][f.ed] = nullptr;
        }
        fetch.clear();

        const int32_t il = hdr->layer;
        job_il = il;
        n_tokens         = hdr->n_tokens;
        const int32_t n_used = hdr->n_used;

        job_layer = (il >= 0 && il < (int32_t) layers.size() && layers[il].set) ? &layers[il] : nullptr;
        misses.clear();
        miss_tokens.clear();
        uniq.clear();
        if (job_layer == nullptr || hdr->n_embd != job_layer->n_embd || n_tokens > dims.max_tokens || n_used > dims.max_used) {
            if (job_layer == nullptr) {
                GGML_LOG_ERROR("%s: no experts registered for layer %d, answering zero\n", __func__, il);
            } else {
                GGML_LOG_ERROR("%s: layer %d: mailbox shape mismatch, answering zero\n", __func__, il);
            }
            job_layer = nullptr;
            return;
        }

        const int32_t * ids   = (const int32_t *) (mb + ggml_expert_handoff_off_ids(dims));
        const int32_t * slots = (const int32_t *) (mb + ggml_expert_handoff_off_slots(dims));
        const float   * w     = (const float   *) (mb + ggml_expert_handoff_off_weights(dims));
        eh_layer & L = layers[il];

        tok_slot.assign(n_tokens, -1);
        const int32_t n_count = L.l.n_used > 0 ? std::min(n_used, L.l.n_used) : n_used;
        for (int32_t t = 0; t < n_tokens; ++t) {
            bool any = false;
            for (int32_t k = 0; k < n_count; ++k) {
                const int32_t e = ids[t*dims.max_used + k];
                if (e < 0 || e >= L.n_expert) {
                    continue;
                }
                L.counts[e]++;
                if (slots[t*dims.max_used + k] == L.l.miss_slot) {
                    misses.push_back({ t, e, w[t*dims.max_used + k] });
                    any = true;
                }
            }
            if (any) {
                tok_slot[t] = (int32_t) miss_tokens.size();
                miss_tokens.push_back(t);
            }
        }

        // the misses by expert: each expert's rows are read once for all the tokens that miss it
        if ((int64_t) uniq_of.size() != L.n_expert) {
            uniq_of.assign(L.n_expert, -1);
        }
        cnt.clear();
        for (const eh_miss & ms : misses) {
            if (uniq_of[ms.e] < 0) {
                uniq_of[ms.e] = (int32_t) uniq.size();
                uniq.push_back(ms.e);
                cnt.push_back(0);
            }
            cnt[uniq_of[ms.e]]++;
        }
        uniq_begin.assign(uniq.size() + 1, 0);
        for (size_t u = 0; u < uniq.size(); ++u) {
            uniq_begin[u + 1] = uniq_begin[u] + cnt[u];
        }
        by_expert.assign(misses.size(), 0);
        fill.assign(uniq_begin.begin(), uniq_begin.end() - 1);
        for (size_t m = 0; m < misses.size(); ++m) {
            by_expert[fill[uniq_of[misses[m].e]]++] = (int32_t) m;
        }
        for (const int32_t e : uniq) {
            uniq_of[e] = -1;
        }

        // [TAG_EH_RESIDENT] where the distinct missed experts are read from
        {
            const void * const * src = L.l.down_src;
            uint64_t n_copy = 0;
            for (const int32_t e : uniq) {
                n_copy += src != nullptr && src[de(e)] != nullptr;
            }
            st_read_copy.fetch_add(n_copy, std::memory_order_relaxed);
            st_read_tensor.fetch_add(uniq.size() - n_copy, std::memory_order_relaxed);
        }

        // [TAG_EH_FILE_READ] the distinct missed experts outside the RAM copy, per tensor, read from the file copies
        {
            const void * const * srcs[4] = { L.l.gate_src, L.l.up_src, L.l.gate_up_src, L.l.down_src };
            const ggml_tensor *  ts[4]   = { L.l.gate, L.l.up, L.l.gate_up, L.l.down };
            size_t total = 0;
            for (const int32_t e : uniq) {
                const int64_t ed = de(e);
                for (int k = 0; k < 4; ++k) {
                    if (ts[k] == nullptr || L.l.file_read[k] == nullptr || (srcs[k] != nullptr && srcs[k][ed] != nullptr)) {
                        continue;
                    }
                    const size_t   a    = std::max<size_t>(1, L.l.file_align[k]);
                    const uint64_t off  = L.l.file_off[k] + (uint64_t) ed*ts[k]->nb[2];
                    const uint64_t aoff = off/a*a;
                    const uint64_t aend = (off + ts[k]->nb[2] + a - 1)/a*a;
                    const size_t   dst  = (total + std::max<size_t>(a, 64) - 1)/std::max<size_t>(a, 64)*std::max<size_t>(a, 64);
                    total = dst + (size_t) (aend - aoff);
                    fetch.push_back({ k, ed, aoff, (size_t) (aend - aoff), dst, (size_t) (off - aoff) });
                }
            }
            if (!fetch.empty()) {
                if (total > stg_cap) {
                    // 4 KiB aligned: unbuffered reads land on sector multiples
                    stg_v.assign(total + 4096, 0);
                    stg     = (uint8_t *) (((uintptr_t) stg_v.data() + 4095) & ~(uintptr_t) 4095);
                    stg_cap = total;
                }
                for (int k = 0; k < 4; ++k) {
                    if (ts[k] != nullptr && (int64_t) stg_ptr[k].size() != L.n_expert) {
                        stg_ptr[k].assign(L.n_expert, nullptr);
                    }
                }
            }
        }

        st_layers.fetch_add(1, std::memory_order_relaxed);
        st_routed.fetch_add((uint64_t) n_tokens*n_used, std::memory_order_relaxed);
        st_host.fetch_add(misses.size(), std::memory_order_relaxed);

        // scratch
        const eh_layer & J = *job_layer;
#if defined(EH_INT8_FAST)
        fast_gu = eh_fast_ok(J.type_gu,   J.n_embd);
        fast_dn = eh_fast_ok(J.type_down, J.n_ff);
#endif
        const size_t n_miss = misses.size();
        if (fast_gu) {
            xa_q.get((size_t) n_tokens*J.n_embd);
            if (xa_d.size() < (size_t) n_tokens*J.n_embd/4) {
                xa_d.resize((size_t) n_tokens*J.n_embd/4);
                xa_h.resize((size_t) n_tokens*J.n_embd/32);
            }
        } else {
            xq_row = ggml_row_size(J.vt_gu, J.n_embd);
            if (xq.size() < (size_t) n_tokens*xq_row) {
                xq.resize((size_t) n_tokens*xq_row);
            }
        }
        if (hf.size() < n_miss*J.n_ff) {
            hf.resize(n_miss*J.n_ff);
        }
        if (fast_dn) {
            ha_q.get(n_miss*J.n_ff);
            if (ha_d.size() < n_miss*J.n_ff/4) {
                ha_d.resize(n_miss*J.n_ff/4);
                ha_h.resize(n_miss*J.n_ff/32);
            }
        } else {
            hq_row = ggml_row_size(J.vt_down, J.n_ff);
            if (hq.size() < n_miss*hq_row) {
                hq.resize(n_miss*hq_row);
            }
        }
    }

    // [TAG_EH_RESIDENT] where expert e of t (kind k: 0 gate, 1 up, 2 gate_up, 3 down) starts: its copy if src has
    // one, else its staged read from a file copy [TAG_EH_FILE_READ], else in the tensor
    const uint8_t * expert_base(const ggml_tensor * t, const void * const * src, int32_t e, int k) const {
        const int64_t ed = de(e);
        if (src != nullptr && src[ed] != nullptr) {
            return (const uint8_t *) src[ed];
        }
        if (!stg_ptr[k].empty() && stg_ptr[k][ed] != nullptr) {
            return stg_ptr[k][ed];
        }
        return (const uint8_t *) t->data + ed*t->nb[2];
    }

    // [TAG_EH_FILE_READ] fetch units [u0, u1): each read into its place in stg; a short or failed read leaves the
    // expert to the tensor
    void task_fetch(int64_t u0, int64_t u1) {
        const eh_layer & J = *job_layer;
        const ggml_tensor * ts[4] = { J.l.gate, J.l.up, J.l.gate_up, J.l.down };
        uint64_t bytes = 0, fail = 0;
        for (int64_t i = u0; i < u1; ++i) {
            const eh_fetch & f = fetch[i];
            const int64_t n = J.l.file_read[f.k](J.l.file_ctx[f.k], stg + f.dst, f.aoff, f.alen);
            if (n >= (int64_t) (f.skip + ts[f.k]->nb[2])) {
                stg_ptr[f.k][f.ed] = stg + f.dst + f.skip;
                bytes += (uint64_t) n;
            } else {
                fail++;
            }
        }
        st_file_bytes.fetch_add(bytes, std::memory_order_relaxed);
        st_file_fail.fetch_add(fail, std::memory_order_relaxed);
    }

    // gate/up row r (0 .. gu_rows - 1: gate rows first, or the up rows alone without a gate) of expert e
    const uint8_t * gu_row(const eh_layer & J, int32_t e, int64_t r) const {
        if (J.l.gate_up) {
            const ggml_tensor * t = J.l.gate_up;
            return expert_base(t, J.l.gate_up_src, e, 2) + r*t->nb[1];
        }
        if (!J.has_gate) {
            const ggml_tensor * t = J.l.up;
            return expert_base(t, J.l.up_src, e, 1) + r*t->nb[1];
        }
        const bool g = r < J.n_ff;
        const ggml_tensor * t = g ? J.l.gate : J.l.up;
        return expert_base(t, g ? J.l.gate_src : J.l.up_src, e, g ? 0 : 1) + (r % J.n_ff)*t->nb[1];
    }

    float gu_scale(const eh_layer & J, int32_t e, int64_t r) const {
        if (J.l.gate_up || !J.has_gate) {
            return J.l.up_scale ? J.l.up_scale[e] : 1.0f;
        }
        const float * s = r < J.n_ff ? J.l.gate_scale : J.l.up_scale;
        return s ? s[e] : 1.0f;
    }

    float gu_bias(const eh_layer & J, int32_t e, int64_t r) const {
        if (J.l.gate_up || !J.has_gate) {
            return J.l.up_bias ? J.l.up_bias[e*J.gu_rows + r] : 0.0f;
        }
        const float * b = r < J.n_ff ? J.l.gate_bias : J.l.up_bias;
        return b ? b[e*J.n_ff + r % J.n_ff] : 0.0f;
    }

    // the activation of one element: g (gate) and u (up), or u alone without a gate
    static float act1(const eh_layer & J, float g, float u) {
        const ggml_expert_handoff_layer & l = J.l;
        auto silu    = [](float v) { return v/(1.0f + expf(-v)); };
        auto gelu    = [](float v) { return 0.5f*v*(1.0f + tanhf(0.7978845608028654f*v*(1.0f + 0.044715f*v*v))); };
        auto sigmoid = [](float v) { return 1.0f/(1.0f + expf(-v)); };
        if (!J.has_gate) {
            switch (l.act) {
                case GGML_EXPERT_HANDOFF_ACT_UP_SILU:     return silu(u);
                case GGML_EXPERT_HANDOFF_ACT_UP_GELU:     return gelu(u);
                case GGML_EXPERT_HANDOFF_ACT_UP_RELU:     return std::max(u, 0.0f);
                case GGML_EXPERT_HANDOFF_ACT_UP_RELU_SQR: { const float r = std::max(u, 0.0f); return r*r; }
                default: return 0.0f;
            }
        }
        switch (l.act) {
            case GGML_EXPERT_HANDOFF_ACT_SILU: return silu(g)*u;
            case GGML_EXPERT_HANDOFF_ACT_GELU: return gelu(g)*u;
            case GGML_EXPERT_HANDOFF_ACT_RELU: return std::max(g, 0.0f)*u;
            case GGML_EXPERT_HANDOFF_ACT_SWIGLU_OAI: {
                const float x = std::min(g, l.limit);
                const float y = std::clamp(u, -l.limit, l.limit);
                return x/(1.0f + expf(-l.alpha*x))*(y + 1.0f);
            }
            case GGML_EXPERT_HANDOFF_ACT_SILU_CLAMP:
                return silu(std::min(g, l.limit))*std::clamp(u, -l.limit, l.limit);
            case GGML_EXPERT_HANDOFF_ACT_SILU_LIMIT:
                return std::min(silu(g), l.limit)*std::clamp(u, -l.limit, l.limit);
            case GGML_EXPERT_HANDOFF_ACT_SITU: {
                const float a = l.beta*tanhf(g/l.beta)*sigmoid(g);
                return a*(l.lb > 0.0f ? l.lb*tanhf(u/l.lb) : u);
            }
            default: return 0.0f;
        }
    }

    // inputs [i0, i1) of miss_tokens into the gate/up dot type
    void task_xq(int64_t i0, int64_t i1) {
        const eh_layer & J = *job_layer;
        const float * x = (const float *) (mb + ggml_expert_handoff_off_x(dims));
        for (int64_t i = i0; i < i1; ++i) {
            const int32_t t = miss_tokens[i];
            const float * xt = x + (int64_t) t*dims.max_embd;
#if defined(EH_INT8_FAST)
            if (fast_gu) {
                eh_act_quant(xt, (int) J.n_embd, xa_q.p + (size_t) t*J.n_embd, xa_d.data() + (size_t) t*J.n_embd/4,
                    xa_h.data() + (size_t) t*J.n_embd/32, eh_fast_biased(J.type_gu));
                continue;
            }
#endif
            uint8_t * dst = xq.data() + t*xq_row;
            if (J.vt_gu == GGML_TYPE_F32) {
                memcpy(dst, xt, J.n_embd*sizeof(float));
            } else {
                ggml_get_type_traits_cpu(J.vt_gu)->from_float(xt, dst, J.n_embd);
            }
        }
    }

    // gate/up units [u0, u1): unit = missed expert * n_ff + r; row r of the gate and of up, for every token that
    // misses the expert, then the activation (with weight_before, the routing weight scales the input: it is linear
    // up to here, so it scales the dot products)
    void task_gu(int64_t q0, int64_t q1) {
        const eh_layer & J = *job_layer;
        const int64_t n_ff = J.n_ff;
        const auto * tr_gu = ggml_get_type_traits_cpu(J.type_gu);
        for (int64_t q = q0; q < q1; ++q) {
            const int64_t u = q / n_ff;
            const int64_t r = q % n_ff;
            const int32_t e = uniq[u];
            const int64_t ru_i = J.has_gate ? n_ff + r : r;
            const uint8_t * rg = J.has_gate ? gu_row(J, e, r) : nullptr;
            const uint8_t * ru = gu_row(J, e, ru_i);
            const float sg = J.has_gate ? gu_scale(J, e, r) : 0.0f;
            const float bg = J.has_gate ? gu_bias(J, e, r)  : 0.0f;
            const float su = gu_scale(J, e, ru_i);
            const float bu = gu_bias(J, e, ru_i);
            for (int32_t k0 = uniq_begin[u]; k0 < uniq_begin[u + 1]; k0 += eh_multi_max) {
                const int nk = std::min<int>(eh_multi_max, uniq_begin[u + 1] - k0);
                float g[eh_multi_max] = {}, up[eh_multi_max];
#if defined(EH_INT8_FAST)
                if (fast_gu) {
                    eh_act a[eh_multi_max];
                    for (int j = 0; j < nk; ++j) {
                        const int32_t t = misses[by_expert[k0 + j]].t;
                        a[j] = { xa_q.p + (size_t) t*J.n_embd, xa_d.data() + (size_t) t*J.n_embd/4,
                                 xa_h.data() + (size_t) t*J.n_embd/32 };
                    }
                    if (rg) {
                        eh_fast_dot(J.type_gu, rg, (int) J.n_embd, a, nk, g);
                    }
                    eh_fast_dot(J.type_gu, ru, (int) J.n_embd, a, nk, up);
                } else
#endif
                {
                    const uint8_t * ys[eh_multi_max];
                    for (int j = 0; j < nk; ++j) {
                        ys[j] = xq.data() + misses[by_expert[k0 + j]].t*xq_row;
                    }
                    if (rg) {
                        eh_dot_n(tr_gu, (int) J.n_embd, rg, ys, nk, g);
                    }
                    eh_dot_n(tr_gu, (int) J.n_embd, ru, ys, nk, up);
                }
                for (int j = 0; j < nk; ++j) {
                    const int32_t m = by_expert[k0 + j];
                    const float wb = J.l.weight_before ? misses[m].w : 1.0f;
                    hf[(size_t) m*n_ff + r] = act1(J, g[j]*sg*wb + bg, up[j]*su*wb + bu);
                }
            }
        }
    }

    // activations [m0, m1) into the down dot type
    void task_hq(int64_t m0, int64_t m1) {
        const eh_layer & J = *job_layer;
        const int64_t n_ff = J.n_ff;
        for (int64_t m = m0; m < m1; ++m) {
            const float * h = hf.data() + (size_t) m*n_ff;
#if defined(EH_INT8_FAST)
            if (fast_dn) {
                eh_act_quant(h, (int) n_ff, ha_q.p + (size_t) m*n_ff, ha_d.data() + (size_t) m*n_ff/4,
                    ha_h.data() + (size_t) m*n_ff/32, eh_fast_biased(J.type_down));
                continue;
            }
#endif
            uint8_t * dst = hq.data() + (size_t) m*hq_row;
            if (J.vt_down == GGML_TYPE_F32) {
                memcpy(dst, h, n_ff*sizeof(float));
            } else {
                ggml_get_type_traits_cpu(J.vt_down)->from_float(h, dst, n_ff);
            }
        }
    }

    // down columns [c0, c1) for every token: each missed expert's rows in the range, read once for all the tokens
    // that miss it, summed per token here, then the columns of y (zeros for tokens without misses)
    void task_dn(int iw, int64_t c0, int64_t c1) {
        const eh_layer & J = *job_layer;
        const ggml_tensor * dn = J.l.down;
        const auto * tr_dn = ggml_get_type_traits_cpu(J.type_down);
        const int64_t nc = c1 - c0;
        const size_t  nt = miss_tokens.size();
        std::vector<float> & s_ = acc[iw];
        if (s_.size() < nt*nc) {
            s_.resize(nt*nc);
        }
        std::fill(s_.begin(), s_.begin() + nt*nc, 0.0f);
        for (size_t u = 0; u < uniq.size(); ++u) {
            const int32_t e = uniq[u];
            const uint8_t * base = expert_base(dn, J.l.down_src, e, 3);
            const float sc = J.l.down_scale ? J.l.down_scale[e] : 1.0f;
            const float * bias = J.l.down_bias ? J.l.down_bias + (int64_t) e*J.n_embd : nullptr;
            for (int32_t k0 = uniq_begin[u]; k0 < uniq_begin[u + 1]; k0 += eh_multi_max) {
                const int nk = std::min<int>(eh_multi_max, uniq_begin[u + 1] - k0);
#if defined(EH_INT8_FAST)
                eh_act a[eh_multi_max];
#endif
                const uint8_t * ys[eh_multi_max];
                float wm[eh_multi_max];
                int32_t ts[eh_multi_max];
                for (int j = 0; j < nk; ++j) {
                    const int32_t m = by_expert[k0 + j];
                    wm[j] = J.l.weight_before ? 1.0f : misses[m].w;
                    ts[j] = tok_slot[misses[m].t];
#if defined(EH_INT8_FAST)
                    if (fast_dn) {
                        a[j] = { ha_q.p + (size_t) m*J.n_ff, ha_d.data() + (size_t) m*J.n_ff/4,
                                 ha_h.data() + (size_t) m*J.n_ff/32 };
                        continue;
                    }
#endif
                    ys[j] = hq.data() + (size_t) m*hq_row;
                }
                for (int64_t c = c0; c < c1; ++c) {
                    const uint8_t * row = base + c*dn->nb[1];
                    float s[eh_multi_max];
#if defined(EH_INT8_FAST)
                    if (fast_dn) {
                        eh_fast_dot(J.type_down, row, (int) J.n_ff, a, nk, s);
                    } else
#endif
                    {
                        eh_dot_n(tr_dn, (int) J.n_ff, row, ys, nk, s);
                    }
                    const float b = bias ? bias[c] : 0.0f;
                    for (int j = 0; j < nk; ++j) {
                        s_[(size_t) ts[j]*nc + (c - c0)] += wm[j]*(s[j]*sc + b);
                    }
                }
            }
        }
        float * y = (float *) (mb + ggml_expert_handoff_off_y(dims));
        for (int32_t t = 0; t < n_tokens; ++t) {
            float * dst = y + (int64_t) t*dims.max_embd + c0;
            const int32_t ts = tok_slot[t];
            if (ts >= 0) {
                memcpy(dst, s_.data() + (size_t) ts*nc, nc*sizeof(float));
            } else {
                memset(dst, 0, nc*sizeof(float));
            }
        }
    }

    void worker(int iw) {
        uint64_t my_wake = 0;
        if (iw > 0) {
            uint32_t seen = gen.load(std::memory_order_acquire);
            for (;;) {
                uint32_t g = 0;
                if (!wait_gen(seen, g, my_wake)) {
                    return;
                }
                seen = g;
                run_tasks(iw, g);
            }
        }

        // worker 0: the sequence the pool was made at, not what this thread reads when it starts (a send may wait)
        uint32_t seen = seq_base;
        for (;;) {
            uint32_t s = 0;
            if (!wait_send(seen, s, my_wake)) {
                return;
            }
            prepare(s);
            seen = seq;
            const int64_t t_prep = eh_now_ns();
            int64_t t_step[3] = { t_prep, t_prep, t_prep };
            const bool ran = job_layer != nullptr && !misses.empty() && !sim_nohost;
            if (ran) {
                const eh_layer & J = *job_layer;
                if (!fetch.empty()) { // [TAG_EH_FILE_READ]
                    const int64_t tf = eh_now_ns();
                    run_phase(PH_FETCH, (int64_t) fetch.size(), 0);
                    st_file_ns.fetch_add((uint64_t) (eh_now_ns() - tf), std::memory_order_relaxed);
                }
                run_phase(PH_XQ, (int64_t) miss_tokens.size(), 4);
                t_step[0] = eh_now_ns();
                run_phase(PH_GU, (int64_t) uniq.size()*J.n_ff, 0);
                t_step[1] = eh_now_ns();
                run_phase(PH_HQ, (int64_t) misses.size(), 16);
                t_step[2] = eh_now_ns();
                run_phase(PH_DN, J.n_embd, 0);
            } else {
                float * y = (float *) (mb + ggml_expert_handoff_off_y(dims));
                const int64_t n_embd = job_layer ? job_layer->n_embd : hdr->n_embd;
                for (int32_t t = 0; t < n_tokens; ++t) {
                    memset(y + (int64_t) t*dims.max_embd, 0, n_embd*sizeof(float));
                }
            }
            // the answer: every task's writes of y are in (n_done, acquire)
            const int64_t gap = t_seen - t_prev_end;
            if (t_prev_end > 0 && gap >= 0 && gap < 5*1000*1000) {
                st_gap.fetch_add((uint64_t) gap, std::memory_order_relaxed);
                st_n_gap.fetch_add(1, std::memory_order_relaxed);
            }
            eh_store(&hdr->seq_done, seq);
            const int64_t t_end = eh_now_ns();
            t_prev_end = t_end;
            t_active.store(t_end, std::memory_order_relaxed);
            st_busy_ns.fetch_add((uint64_t) (t_end - t_seen), std::memory_order_relaxed);
            if (ran) {
                const int64_t ts[6] = { t_seen, t_prep, t_step[0], t_step[1], t_step[2], t_end };
                for (int k = 0; k < 5; ++k) {
                    st_step[k].fetch_add((uint64_t) std::max<int64_t>(0, ts[k + 1] - ts[k]), std::memory_order_relaxed);
                }
            }
            if (n_tokens > 1) {
                st_multi.fetch_add(1, std::memory_order_relaxed);
                st_busy_multi.fetch_add((uint64_t) (t_end - t_seen), std::memory_order_relaxed);
            }
            if ((uint64_t) (t_end - t_seen) > st_max_ns.load(std::memory_order_relaxed)) {
                st_max_ns.store((uint64_t) (t_end - t_seen), std::memory_order_relaxed);
                st_max_layer.store(job_il, std::memory_order_relaxed);
                st_max_tokens.store(n_tokens, std::memory_order_relaxed);
                st_max_miss.store((int32_t) misses.size(), std::memory_order_relaxed);
                st_max_uniq.store((int32_t) uniq.size(), std::memory_order_relaxed);
                const int64_t tk[4] = { t_step[0], t_step[1], t_step[2], t_end };
                int64_t prev = t_prep;
                for (int k = 0; k < 4; ++k) {
                    st_max_step[k].store((uint64_t) std::max<int64_t>(0, tk[k] - prev), std::memory_order_relaxed);
                    prev = std::max(prev, tk[k]);
                }
            }
            spin_until.store(std::max(spin_until.load(std::memory_order_relaxed), eh_now_ns() + linger_ns), std::memory_order_release);
        }
    }
};



ggml_expert_handoff_host_t ggml_expert_handoff_host_new(
        void * mailbox, int32_t max_tokens, int32_t max_embd, int32_t max_used, int32_t n_layer, int32_t n_workers) {
    GGML_ASSERT(mailbox != nullptr && n_layer > 0 && max_tokens > 0 && max_embd > 0 && max_used > 0);
    ggml_cpu_init();

    auto * h = new ggml_expert_handoff_host;
    h->mb   = (uint8_t *) mailbox;
    h->hdr  = (ggml_expert_handoff_header *) mailbox;
    h->dims = { max_tokens, max_embd, max_used };
    h->layers.resize(n_layer);
    // [TAG_EH_POOL2] n_workers <= 0: one per physical core but one (or per two logical processors less one)
    if (n_workers <= 0) {
        const int n_cores = eh_n_cores();
        n_workers = n_cores > 1 ? n_cores - 1 : std::max(1, (int32_t) std::thread::hardware_concurrency()/2 - 1);
    }
    h->n_workers = std::max(1, n_workers);
    h->acc.resize(h->n_workers);

    // a fresh mailbox: nothing sent, nothing answered
    h->seq_base = eh_load(&h->hdr->seq_send);
    h->seq      = h->seq_base; // [TAG_EH_TIMEOUT_DIAG] prepare() counts the sends it never took up from here
    if (const char * v = getenv("ROCMFPX_EH_SIM_RESIDENT")) {
        h->sim_resident = std::max<int64_t>(0, atoll(v));
    }
    if (const char * v = getenv("ROCMFPX_EH_SIM_NOHOST")) {
        h->sim_nohost = atoi(v) != 0;
    }
    if (h->sim_resident > 0 || h->sim_nohost) {
        GGML_LOG_WARN("%s: measuring: %s%s (the answers are wrong)\n", __func__,
            h->sim_nohost ? "the host answers zeros" : "", h->sim_resident > 0 ? "the host reads the first experts of each layer only" : "");
    }
    eh_store(&h->hdr->seq_done, h->seq_base);

    h->misses.reserve((size_t) max_tokens*max_used);
    h->threads.reserve(h->n_workers);
    for (int i = 0; i < h->n_workers; ++i) {
        h->threads.emplace_back([h, i] { h->worker(i); });
    }
    return h;
}

void ggml_expert_handoff_host_free(ggml_expert_handoff_host_t h) {
    if (h == nullptr) {
        return;
    }
    h->stop.store(true, std::memory_order_release);
    h->wake();
    for (auto & t : h->threads) {
        t.join();
    }
    delete h;
}

bool ggml_expert_handoff_host_set_layer(ggml_expert_handoff_host_t h, int32_t il, const ggml_expert_handoff_layer * l) {
    if (il < 0 || il >= (int32_t) h->layers.size() || l->down == nullptr || (l->gate_up == nullptr && l->up == nullptr)) {
        return false;
    }
    // a gate (separate or merged) for the gated activations, none for the activations of up alone
    const bool has_gate = l->gate_up != nullptr || l->gate != nullptr;
    const bool up_only_act = l->act >= GGML_EXPERT_HANDOFF_ACT_UP_SILU;
    if (has_gate == up_only_act) {
        return false;
    }
    const ggml_tensor * gu = l->gate_up ? l->gate_up : (l->gate ? l->gate : l->up);
    const ggml_tensor * dn = l->down;
    if (l->gate && !l->gate_up && (l->up == nullptr || l->up->type != l->gate->type || !ggml_are_same_shape(l->up, l->gate))) {
        return false;
    }
    if ((l->act == GGML_EXPERT_HANDOFF_ACT_SITU && l->beta == 0.0f) ||
        ((l->act == GGML_EXPERT_HANDOFF_ACT_SWIGLU_OAI || l->act == GGML_EXPERT_HANDOFF_ACT_SILU_CLAMP ||
          l->act == GGML_EXPERT_HANDOFF_ACT_SILU_LIMIT) && !(l->limit > 0.0f))) {
        return false;
    }

    eh_layer L;
    L.l        = *l;
    L.has_gate = has_gate;
    L.n_embd   = gu->ne[0];
    L.n_ff     = l->gate_up ? gu->ne[1]/2 : gu->ne[1];
    L.n_expert = gu->ne[2];
    L.gu_rows  = has_gate ? 2*L.n_ff : L.n_ff;
    L.type_gu   = gu->type;
    L.type_down = dn->type;
    if (dn->ne[0] != L.n_ff || dn->ne[1] != L.n_embd || dn->ne[2] != L.n_expert || L.n_embd > h->dims.max_embd) {
        return false;
    }
    const auto * tg = ggml_get_type_traits_cpu(L.type_gu);
    const auto * td = ggml_get_type_traits_cpu(L.type_down);
    if (tg->vec_dot == nullptr || td->vec_dot == nullptr) {
        return false;
    }
    L.vt_gu   = tg->vec_dot_type;
    L.vt_down = td->vec_dot_type;
    if ((L.vt_gu != GGML_TYPE_F32 && ggml_get_type_traits_cpu(L.vt_gu)->from_float == nullptr) ||
        (L.vt_down != GGML_TYPE_F32 && ggml_get_type_traits_cpu(L.vt_down)->from_float == nullptr)) {
        return false;
    }
    if (L.n_embd % ggml_blck_size(L.vt_gu) != 0 || L.n_ff % ggml_blck_size(L.vt_down) != 0) {
        return false;
    }
    L.counts.assign(L.n_expert, 0);
    L.set = true;
    h->layers[il] = std::move(L);
    return true;
}

void ggml_expert_handoff_host_wake(ggml_expert_handoff_host_t h) {
    h->wake();
}

void ggml_expert_handoff_host_idle(ggml_expert_handoff_host_t h) {
    h->spin_until.store(0, std::memory_order_release);
}

void ggml_expert_handoff_host_get_stats(ggml_expert_handoff_host_t h, ggml_expert_handoff_stats * st) {
    st->n_layers = h->st_layers.load();
    st->n_routed = h->st_routed.load();
    st->n_host   = h->st_host.load();
    st->busy_ns  = h->st_busy_ns.load();
    // [TAG_EH_TIMEOUT_DIAG]
    st->n_multi       = h->st_multi.load();
    st->busy_ns_multi = h->st_busy_multi.load();
    st->n_skipped     = h->st_skipped.load();
    st->max_ns        = h->st_max_ns.load();
    st->max_layer     = h->st_max_layer.load();
    st->max_tokens    = h->st_max_tokens.load();
    st->max_miss      = h->st_max_miss.load();
    st->max_uniq      = h->st_max_uniq.load();
    for (int k = 0; k < 4; ++k) {
        st->max_step_ns[k] = h->st_max_step[k].load();
    }
    for (int k = 0; k < 5; ++k) {
        st->step_ns[k] = h->st_step[k].load();
    }
    st->gather_ns     = h->st_gather.load();
    st->gap_ns        = h->st_gap.load();
    st->n_gap         = h->st_n_gap.load();
    st->n_read_copy   = h->st_read_copy.load();
    st->n_read_tensor = h->st_read_tensor.load();
    st->n_workers     = h->n_workers;
    st->file_bytes    = h->st_file_bytes.load(); // [TAG_EH_FILE_READ]
    st->file_ns       = h->st_file_ns.load();
    st->n_file_fail   = h->st_file_fail.load();
}

const uint64_t * ggml_expert_handoff_host_counts(ggml_expert_handoff_host_t h, int32_t il) {
    if (il < 0 || il >= (int32_t) h->layers.size() || !h->layers[il].set) {
        return nullptr;
    }
    return h->layers[il].counts.data();
}

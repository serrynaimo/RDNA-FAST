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
#define EH_Q2_FAST 1
// [TAG_EH_Q2_ACT] Q2_0 rows against int8 activations (AVX-512 VBMI/VNNI). An activation is int8 in chunks of 32 with
// an f32 scale d per chunk, 64-byte aligned, plus hx = d * (sum of the chunk's int8): a Q2_0 weight is (code - 1) * dw,
// so row . x = sum over blocks of dw * (d * sum(code * q) - hx), and the hx part is one FMA per eight blocks. Per
// 64-weight block that is one unpack (shared by all tokens), then per token one VNNI dot, one convert, one permute of
// the block's two scale products and one FMA into a float accumulator reduced once per row. ggml's Q2_0 x Q8_0 dot
// does two dots, scalar fp16 conversions and a blend per block, and reduces per call.
struct eh_act {
    const int8_t * q;     // n values, 64-byte aligned
    const float  * d16;   // per 64 values: the two chunk scales, each in 8 lanes
    const float  * hx16;  // per 64 values: the two chunks' hx/8, each in 8 lanes (the lanes of a chunk add up to hx)
};

// x (n values, n % 64 == 0) into q, d16, hx16 (n/64*16 floats each)
void eh_act_quant(const float * x, int n, int8_t * q, float * d16, float * hx16) {
    for (int k = 0; k < n/32; ++k) {
        const __m512 x0 = _mm512_loadu_ps(x + 32*k);
        const __m512 x1 = _mm512_loadu_ps(x + 32*k + 16);
        const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(x0), _mm512_abs_ps(x1)));
        const __m512 inv = _mm512_set1_ps(amax > 0.0f ? 127.0f/amax : 0.0f);
        const __m512i v0 = _mm512_cvtps_epi32(_mm512_mul_ps(x0, inv));
        const __m512i v1 = _mm512_cvtps_epi32(_mm512_mul_ps(x1, inv));
        _mm_storeu_si128((__m128i *) (q + 32*k),      _mm512_cvtsepi32_epi8(v0));
        _mm_storeu_si128((__m128i *) (q + 32*k + 16), _mm512_cvtsepi32_epi8(v1));
        const float d  = amax/127.0f;
        const float hx = d*(float) _mm512_reduce_add_epi32(_mm512_add_epi32(v0, v1));
        // chunk k is half k % 2 of block k / 2: lanes 8*(k % 2) .. + 7
        for (int l = 0; l < 8; ++l) {
            d16 [16*(k/2) + 8*(k % 2) + l] = d;
            hx16[16*(k/2) + 8*(k % 2) + l] = hx*0.125f;
        }
    }
}

// out[t] = row . a[t] for NT tokens; row: nb blocks of { f16 d; 16 bytes of 2-bit codes } (block_q2_0). Per block the
// weight scale is read once and the codes unpacked once; per token, lane by lane, (code . q) * d - hx/8, times the weight
// scale, into the token's accumulator. One or two tokens alone would chain every block's FMA on the last (4 cycles
// each): U blocks go to U accumulators. (A vector of eight blocks' scales built through small stack stores stalled: 79
// ns per row of 2560 against ggml's 63; per-token broadcasts of the two chunk scales: 176 ns for four tokens.)
template <int NT>
void eh_q2_rows(const uint8_t * row, int nb, const eh_act * a, float * out) {
    static_assert(sizeof(block_q2_0) == 18, "block_q2_0 layout");
    constexpr int U = NT == 1 ? 4 : NT == 2 ? 2 : 1;
    __m512 acc[U][NT];
    for (int u = 0; u < U; ++u) {
        for (int t = 0; t < NT; ++t) {
            acc[u][t] = _mm512_setzero_ps();
        }
    }
    const __m512i ctl   = _mm512_set1_epi64(0x0E0C0A0806040200LL);
    const __m512i three = _mm512_set1_epi8(3);
    const __m512i zero  = _mm512_setzero_si512();
    auto block = [&](int b, int u) {
        const uint8_t * blk = row + (size_t) b*18;
        uint16_t h;
        memcpy(&h, blk, 2);
        const __m512 dw = _mm512_set1_ps(_cvtsh_ss(h));
        // 16 code bytes as 8 u16 of 8 codes each -> 64 bytes, one code each
        const __m512i w = _mm512_and_si512(_mm512_multishift_epi64_epi8(ctl,
            _mm512_cvtepu16_epi64(_mm_loadu_si128((const __m128i *) (blk + 2)))), three);
        for (int t = 0; t < NT; ++t) {
            const __m512i dot = _mm512_dpbusd_epi32(zero, w, _mm512_load_si512((const void *) (a[t].q + (size_t) b*64)));
            const __m512 v = _mm512_fmsub_ps(_mm512_cvtepi32_ps(dot), _mm512_loadu_ps(a[t].d16 + 16*b), _mm512_loadu_ps(a[t].hx16 + 16*b));
            acc[u][t] = _mm512_fmadd_ps(dw, v, acc[u][t]);
        }
    };
    int b = 0;
    for (; b + U <= nb; b += U) {
        for (int u = 0; u < U; ++u) {
            block(b + u, u);
        }
    }
    for (; b < nb; ++b) {
        block(b, 0);
    }
    for (int t = 0; t < NT; ++t) {
        __m512 s = acc[0][t];
        for (int u = 1; u < U; ++u) {
            s = _mm512_add_ps(s, acc[u][t]);
        }
        out[t] = _mm512_reduce_add_ps(s);
    }
}

void eh_q2_dot(const uint8_t * row, int nb, const eh_act * a, int nt, float * out) {
    switch (nt) {
        case 1: eh_q2_rows<1>(row, nb, a, out); break;
        case 2: eh_q2_rows<2>(row, nb, a, out); break;
        case 3: eh_q2_rows<3>(row, nb, a, out); break;
        case 4: eh_q2_rows<4>(row, nb, a, out); break;
        case 5: eh_q2_rows<5>(row, nb, a, out); break;
        case 6: eh_q2_rows<6>(row, nb, a, out); break;
        case 7: eh_q2_rows<7>(row, nb, a, out); break;
        default: eh_q2_rows<8>(row, nb, a, out); break;
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
    enum { PH_XQ = 0, PH_GU = 1, PH_HQ = 2, PH_DN = 3 };
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
    bool fast_gu = false, fast_dn = false;       // [TAG_EH_Q2_ACT] the Q2_0 path for gate/up, for down
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
    eh_abuf xa_q;                          // Q2_0 path: [n_tokens][n_embd] int8, and per 64 values d16 and hx16
    std::vector<float> xa_d, xa_hx;        //   ([n_tokens][n_embd/4] each)
    std::vector<float> hf;                 // [n_miss][n_ff] the activations, f32
    std::vector<uint8_t> hq;               // generic: [n_miss] rows of the down vec_dot type
    size_t hq_row = 0;
    eh_abuf ha_q;                          // Q2_0 path: [n_miss][n_ff] int8, d16, hx16 ([n_miss][n_ff/4] each)
    std::vector<float> ha_d, ha_hx;
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

        st_layers.fetch_add(1, std::memory_order_relaxed);
        st_routed.fetch_add((uint64_t) n_tokens*n_used, std::memory_order_relaxed);
        st_host.fetch_add(misses.size(), std::memory_order_relaxed);

        // scratch
        const eh_layer & J = *job_layer;
#if defined(EH_Q2_FAST)
        fast_gu = J.type_gu   == GGML_TYPE_Q2_0 && J.n_embd % 64 == 0;
        fast_dn = J.type_down == GGML_TYPE_Q2_0 && J.n_ff   % 64 == 0;
#endif
        const size_t n_miss = misses.size();
        if (fast_gu) {
            xa_q.get((size_t) n_tokens*J.n_embd);
            if (xa_d.size() < (size_t) n_tokens*J.n_embd/4) {
                xa_d.resize((size_t) n_tokens*J.n_embd/4);
                xa_hx.resize((size_t) n_tokens*J.n_embd/4);
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
                ha_hx.resize(n_miss*J.n_ff/4);
            }
        } else {
            hq_row = ggml_row_size(J.vt_down, J.n_ff);
            if (hq.size() < n_miss*hq_row) {
                hq.resize(n_miss*hq_row);
            }
        }
    }

    // [TAG_EH_RESIDENT] where expert e of t starts: its copy if src has one, else in the tensor
    const uint8_t * expert_base(const ggml_tensor * t, const void * const * src, int32_t e) const {
        const int64_t ed = de(e);
        if (src != nullptr && src[ed] != nullptr) {
            return (const uint8_t *) src[ed];
        }
        return (const uint8_t *) t->data + ed*t->nb[2];
    }

    // gate/up row r (0 .. gu_rows - 1: gate rows first, or the up rows alone without a gate) of expert e
    const uint8_t * gu_row(const eh_layer & J, int32_t e, int64_t r) const {
        if (J.l.gate_up) {
            const ggml_tensor * t = J.l.gate_up;
            return expert_base(t, J.l.gate_up_src, e) + r*t->nb[1];
        }
        if (!J.has_gate) {
            const ggml_tensor * t = J.l.up;
            return expert_base(t, J.l.up_src, e) + r*t->nb[1];
        }
        const bool g = r < J.n_ff;
        const ggml_tensor * t = g ? J.l.gate : J.l.up;
        return expert_base(t, g ? J.l.gate_src : J.l.up_src, e) + (r % J.n_ff)*t->nb[1];
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
#if defined(EH_Q2_FAST)
            if (fast_gu) {
                eh_act_quant(xt, (int) J.n_embd, (int8_t *) xa_q.p + (size_t) t*J.n_embd,
                    xa_d.data() + (size_t) t*J.n_embd/4, xa_hx.data() + (size_t) t*J.n_embd/4);
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
#if defined(EH_Q2_FAST)
                if (fast_gu) {
                    eh_act a[eh_multi_max];
                    for (int j = 0; j < nk; ++j) {
                        const int32_t t = misses[by_expert[k0 + j]].t;
                        a[j] = { (const int8_t *) xa_q.p + (size_t) t*J.n_embd, xa_d.data() + (size_t) t*J.n_embd/4,
                                 xa_hx.data() + (size_t) t*J.n_embd/4 };
                    }
                    const int nb = (int) (J.n_embd/QK2_0);
                    if (rg) {
                        eh_q2_dot(rg, nb, a, nk, g);
                    }
                    eh_q2_dot(ru, nb, a, nk, up);
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
#if defined(EH_Q2_FAST)
            if (fast_dn) {
                eh_act_quant(h, (int) n_ff, (int8_t *) ha_q.p + (size_t) m*n_ff, ha_d.data() + (size_t) m*n_ff/4,
                    ha_hx.data() + (size_t) m*n_ff/4);
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
            const uint8_t * base = expert_base(dn, J.l.down_src, e);
            const float sc = J.l.down_scale ? J.l.down_scale[e] : 1.0f;
            const float * bias = J.l.down_bias ? J.l.down_bias + (int64_t) e*J.n_embd : nullptr;
            for (int32_t k0 = uniq_begin[u]; k0 < uniq_begin[u + 1]; k0 += eh_multi_max) {
                const int nk = std::min<int>(eh_multi_max, uniq_begin[u + 1] - k0);
#if defined(EH_Q2_FAST)
                eh_act a[eh_multi_max];
#endif
                const uint8_t * ys[eh_multi_max];
                float wm[eh_multi_max];
                int32_t ts[eh_multi_max];
                for (int j = 0; j < nk; ++j) {
                    const int32_t m = by_expert[k0 + j];
                    wm[j] = J.l.weight_before ? 1.0f : misses[m].w;
                    ts[j] = tok_slot[misses[m].t];
#if defined(EH_Q2_FAST)
                    if (fast_dn) {
                        a[j] = { (const int8_t *) ha_q.p + (size_t) m*J.n_ff, ha_d.data() + (size_t) m*J.n_ff/4,
                                 ha_hx.data() + (size_t) m*J.n_ff/4 };
                        continue;
                    }
#endif
                    ys[j] = hq.data() + (size_t) m*hq_row;
                }
                for (int64_t c = c0; c < c1; ++c) {
                    const uint8_t * row = base + c*dn->nb[1];
                    float s[eh_multi_max];
#if defined(EH_Q2_FAST)
                    if (fast_dn) {
                        eh_q2_dot(row, (int) (J.n_ff/QK2_0), a, nk, s);
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
}

const uint64_t * ggml_expert_handoff_host_counts(ggml_expert_handoff_host_t h, int32_t il) {
    if (il < 0 || il >= (int32_t) h->layers.size() || !h->layers[il].set) {
        return nullptr;
    }
    return h->layers[il].counts.data();
}

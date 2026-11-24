#include "conv-chain.cuh"
#include "gdn-conv.cuh"
#include "ggml-impl.h"

#include <cstring>

// [RDNA-FAST] [TAG_INKLING_CONV_CHAIN] see conv-chain.cuh. The kernel follows unslothai/llama.cpp #411 (MIT): one
// thread per channel keeps the conv window in registers and walks the tokens of its sequence; the arithmetic is the
// separate kernels' (SSM_CONV contracts the taps in tap order and adds its zero bias, every dynamic MUL and ADD rounds
// on its own, SCALE is scale_f32's FMA), so the results are bit-identical.

#define CONV_CHAIN_BLOCK_SIZE 128
#define CONV_CHAIN_MAX_TOKENS 32
#define CONV_CHAIN_MAX_WIN    8

struct conv_chain_params {
    const float * cur;      // [C, T*ns] contiguous
    const float * past;     // [n_conv - 1, C, ns], strides in floats
    int64_t       past_s1;
    int64_t       past_s2;
    const float * w;        // [n_conv, C], row stride in floats
    int64_t       w_s1;
    const float * dyn;      // [n_conv, n_head, T*ns] contiguous, or null
    int           ch_per_head;
    int           n_head;
    float       * dst;      // [C, T*ns] contiguous, may be cur
    float         scale;    // residual form: dst = scale*(cur + y) + bias
    float         bias;
    int           n_win;
    int           win_start[CONV_CHAIN_MAX_WIN]; // the first column of x each stored window starts at
    float       * win_dst[CONV_CHAIN_MAX_WIN];   // (column j, channel c) of sequence s at j + c*(n_conv - 1) + s*win_s2
    int64_t       win_s2[CONV_CHAIN_MAX_WIN];
    int           C;
    int           T;
};

template <int n_conv, bool has_dyn, bool has_res>
static __global__ void conv_chain_f32(const conv_chain_params p) {
    ggml_cuda_pdl_lc();
    const int c = blockIdx.y*CONV_CHAIN_BLOCK_SIZE + threadIdx.x;
    const int s = blockIdx.x;

    ggml_cuda_pdl_sync();
    if (c >= p.C) {
        return;
    }

    float w[n_conv];
#pragma unroll
    for (int j = 0; j < n_conv; ++j) {
        w[j] = p.w[c*p.w_s1 + j];
    }

    // xr[j] = column t + j of x; this thread reads its channel's inputs before it writes any output of the channel, so
    // the output may be cur and a stored window may be the history it read
    float xr[n_conv];
#pragma unroll
    for (int j = 0; j < n_conv - 1; ++j) {
        xr[j] = p.past[j + c*p.past_s1 + s*p.past_s2];
    }

    const int64_t tok0 = int64_t(s)*p.T;
    const int     h    = has_dyn ? c / p.ch_per_head : 0;

    auto store_windows = [&](const int col) {
        for (int k = 0; k < p.n_win; ++k) {
            if (p.win_start[k] == col) {
                float * d = p.win_dst[k] + s*p.win_s2[k] + c*(n_conv - 1);
#pragma unroll
                for (int j = 0; j < n_conv - 1; ++j) {
                    d[j] = xr[j];
                }
            }
        }
    };

    for (int t = 0; t < p.T; ++t) {
        xr[n_conv - 1] = p.cur[(tok0 + t)*p.C + c];

        store_windows(t);

        // SSM_CONV: products contracted in tap order, then the zero bias
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < n_conv; ++j) {
            sumf = __fmaf_rn(xr[j], w[j], sumf);
        }
        sumf = __fadd_rn(sumf, 0.0f);

        if (has_dyn) {
            // MUL, ADD per lag: the window of lag starts n_conv - 1 - lag columns into x
            const float * tap = p.dyn + ((tok0 + t)*p.n_head + h)*n_conv;
#pragma unroll
            for (int lag = 0; lag < n_conv; ++lag) {
                sumf = __fadd_rn(sumf, __fmul_rn(xr[n_conv - 1 - lag], tap[lag]));
            }
        }

        p.dst[(tok0 + t)*p.C + c] = has_res ? __fmaf_rn(p.scale, __fadd_rn(xr[n_conv - 1], sumf), p.bias) : sumf;

#pragma unroll
        for (int j = 0; j < n_conv - 1; ++j) {
            xr[j] = xr[j + 1];
        }
    }
    store_windows(p.T);
}

template <int n_conv>
static void conv_chain_launch(const conv_chain_params & p, const int ns, const bool has_dyn, const bool has_res, cudaStream_t stream) {
    const dim3 blocks(ns, (p.C + CONV_CHAIN_BLOCK_SIZE - 1)/CONV_CHAIN_BLOCK_SIZE, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(blocks, dim3(CONV_CHAIN_BLOCK_SIZE, 1, 1), 0, stream);
    if (has_dyn) {
        if (has_res) {
            ggml_cuda_kernel_launch(conv_chain_f32<n_conv, true,  true>,  lp, p);
        } else {
            ggml_cuda_kernel_launch(conv_chain_f32<n_conv, true,  false>, lp, p);
        }
    } else {
        if (has_res) {
            ggml_cuda_kernel_launch(conv_chain_f32<n_conv, false, true>,  lp, p);
        } else {
            ggml_cuda_kernel_launch(conv_chain_f32<n_conv, false, false>, lp, p);
        }
    }
}

static bool conv_chain_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_PERMUTE ||
        t->op == GGML_OP_NONE;
}

// the tensor a chain of RESHAPEs is made from
static const ggml_tensor * conv_chain_unreshape(const ggml_tensor * t) {
    while (t && t->op == GGML_OP_RESHAPE) {
        t = t->src[0];
    }
    return t;
}

static bool conv_chain_debug() {
    static const bool debug = getenv("GGML_CUDA_CONV_CHAIN_DEBUG") != nullptr;
    return debug;
}

#define CONV_CHAIN_DECLINE(reason) do { if (conv_chain_debug()) { GGML_LOG_WARN("%s: node %d (%s): %s\n", __func__, i, cgraph->nodes[i]->name, reason); } return 0; } while (0)

struct conv_chain_range {
    uintptr_t a = 0;
    uintptr_t b = 0;
};

static bool conv_chain_overlap(const conv_chain_range & x, const conv_chain_range & y) {
    return x.a < y.b && y.a < x.b;
}

int ggml_cuda_try_fuse_conv_chain(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    const int n_nodes = cgraph->n_nodes;
    ggml_tensor * x = cgraph->nodes[i];
    if (x->op != GGML_OP_CONCAT || ggml_get_op_params_i32(x, 0) != 0 || x->type != GGML_TYPE_F32) {
        return 0;
    }
    const ggml_tensor * past = x->src[0];
    const ggml_tensor * xT   = x->src[1];
    if (!past || !xT || xT->op != GGML_OP_TRANSPOSE || past->type != GGML_TYPE_F32 || xT->type != GGML_TYPE_F32) {
        return 0;
    }
    // the gated delta net convolutions (Qwen3.x, Swift) have fusions of their own (TAG_GDN_CONV_SNAP): leave them
    {
        ggml_cuda_gdn_conv_match gm;
        if (ggml_cuda_gdn_conv_match_at_concat(cgraph, i, gm)) {
            return 0;
        }
    }
    const int64_t nc1 = past->ne[0];
    const int64_t C   = past->ne[1];
    const int64_t ns  = past->ne[2];
    const int64_t T   = xT->ne[0];
    const int     nc  = (int) nc1 + 1;
    if (x->ne[0] != nc1 + T || x->ne[1] != C || x->ne[2] != ns || x->ne[3] != 1 || xT->ne[1] != C || xT->ne[2] != ns ||
        past->ne[3] != 1 || !ggml_is_contiguous(x)) {
        return 0;
    }
    // cur: the transposed [T, C, ns] view of a contiguous [C, T*ns]
    if (xT->nb[1] != sizeof(float) || xT->nb[0] != (size_t) C*sizeof(float) || xT->nb[2] != (size_t) (T*C)*sizeof(float) ||
        past->nb[0] != sizeof(float)) {
        CONV_CHAIN_DECLINE("layout of cur or the history");
    }
    if (T < 1 || T > CONV_CHAIN_MAX_TOKENS || nc < 2 || nc > 5 || C > INT_MAX || ns > 65535) {
        CONV_CHAIN_DECLINE("sizes");
    }

    conv_chain_params p = {};
    p.cur     = (const float *) xT->data;
    p.past    = (const float *) past->data;
    p.past_s1 = past->nb[1] / sizeof(float);
    p.past_s2 = past->nb[2] / sizeof(float);
    p.C       = (int) C;
    p.T       = (int) T;

    int j = i + 1;
    auto next_op = [&]() -> ggml_tensor * {
        while (j < n_nodes && conv_chain_is_view(cgraph->nodes[j])) {
            ++j;
        }
        return j < n_nodes ? cgraph->nodes[j] : nullptr;
    };

    // the state store: CPYs of windows of x into the state cells
    ggml_tensor * t = nullptr;
    while ((t = next_op()) != nullptr && t->op == GGML_OP_CPY) {
        const ggml_tensor * src  = t->src[0];
        const ggml_tensor * cell = t->src[1];
        if (p.n_win >= CONV_CHAIN_MAX_WIN || src->view_src != x || src->type != GGML_TYPE_F32 || cell->type != GGML_TYPE_F32 ||
            src->ne[0] != nc1 || src->ne[1] != C || src->ne[2] != ns || src->ne[3] != 1 ||
            src->nb[0] != sizeof(float) || src->nb[1] != x->nb[1] || src->nb[2] != x->nb[2] ||
            cell->ne[0] != nc1*C || cell->ne[1] != ns || cell->ne[2] != 1 || cell->ne[3] != 1 || cell->nb[0] != sizeof(float) ||
            cell->nb[1] % sizeof(float) != 0) {
            CONV_CHAIN_DECLINE("state store");
        }
        const int64_t off = (const char *) src->data - (const char *) x->data;
        if (off < 0 || off % sizeof(float) != 0 || off/(int64_t) sizeof(float) > T) {
            CONV_CHAIN_DECLINE("window start");
        }
        p.win_start[p.n_win] = (int) (off/sizeof(float));
        p.win_dst[p.n_win]   = (float *) cell->data;
        p.win_s2[p.n_win]    = cell->nb[1] / sizeof(float);
        p.n_win++;
        ++j;
    }
    if (p.n_win == 0) {
        return 0;
    }

    // the convolution
    if (!t || t->op != GGML_OP_SSM_CONV || t->src[0] != x) {
        CONV_CHAIN_DECLINE("no SSM_CONV after the state store");
    }
    const ggml_tensor * conv = t;
    const ggml_tensor * w    = conv->src[1];
    if (w->type != GGML_TYPE_F32 || w->ne[0] != nc || w->ne[1] != C || w->nb[0] != sizeof(float) || conv->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(conv)) {
        CONV_CHAIN_DECLINE("SSM_CONV weight or output");
    }
    p.w    = (const float *) w->data;
    p.w_s1 = w->nb[1] / sizeof(float);
    const ggml_tensor * y = conv;
    int end = j;
    ++j;

    // the dynamic taps: CONT(TRANSPOSE(window)), MUL by the tap, ADD per lag
    const ggml_tensor * dyn_base = nullptr;
    int n_lag = 0;
    while (true) {
        const int j0 = j;
        ggml_tensor * cont = next_op();
        if (!cont || cont->op != GGML_OP_CONT) {
            j = j0;
            break;
        }
        const ggml_tensor * tr  = cont->src[0];
        const ggml_tensor * win = tr ? tr->src[0] : nullptr;
        if (!tr || tr->op != GGML_OP_TRANSPOSE || !win || win->view_src != x || win->ne[0] != T || win->ne[1] != C ||
            win->ne[2] != ns || win->nb[1] != x->nb[1] || win->nb[2] != x->nb[2] ||
            (const char *) win->data - (const char *) x->data != (int64_t) (nc - 1 - n_lag)*(int64_t) sizeof(float) ||
            !ggml_is_contiguous(cont)) {
            CONV_CHAIN_DECLINE("dynamic window");
        }
        ++j;
        ggml_tensor * mul = next_op();
        if (!mul || mul->op != GGML_OP_MUL || conv_chain_unreshape(mul->src[0]) != cont) {
            CONV_CHAIN_DECLINE("dynamic MUL");
        }
        const ggml_tensor * tap = mul->src[1];
        const int64_t n_head = tap->ne[1];
        if (tap->type != GGML_TYPE_F32 || tap->ne[0] != 1 || n_head < 1 || C % n_head != 0 || tap->ne[2] != T*ns || tap->ne[3] != 1 ||
            tap->nb[1] != (size_t) nc*sizeof(float) || tap->nb[2] != (size_t) (nc*n_head)*sizeof(float) || !tap->view_src ||
            mul->src[0]->ne[0] != C/n_head || mul->src[0]->ne[1] != n_head || mul->src[0]->ne[2] != T*ns) {
            CONV_CHAIN_DECLINE("dynamic tap");
        }
        const float * base = (const float *) tap->data - n_lag;
        if (n_lag == 0) {
            dyn_base      = tap->view_src;
            p.dyn         = base;
            p.n_head      = (int) n_head;
            p.ch_per_head = (int) (C/n_head);
        } else if (tap->view_src != dyn_base || base != p.dyn || n_head != p.n_head) {
            CONV_CHAIN_DECLINE("dynamic taps of other tensors");
        }
        ++j;
        ggml_tensor * add = next_op();
        if (!add || add->op != GGML_OP_ADD || conv_chain_unreshape(add->src[0]) != y || conv_chain_unreshape(add->src[1]) != mul ||
            add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) || ggml_nelements(add) != C*T*ns) {
            CONV_CHAIN_DECLINE("dynamic ADD");
        }
        y   = add;
        end = j;
        ++j;
        n_lag++;
        if (n_lag == nc) {
            break;
        }
    }
    if (n_lag != 0 && n_lag != nc) {
        CONV_CHAIN_DECLINE("not every lag");
    }

    // the residual and the scale (the unfolded form)
    bool has_res = false;
    {
        const int j0 = j;
        ggml_tensor * add = next_op();
        if (add && add->op == GGML_OP_ADD && add->src[0]->data == xT->data && conv_chain_unreshape(add->src[1]) == y &&
            ggml_is_contiguous(add->src[0]) && ggml_nelements(add->src[0]) == C*T*ns) {
            ++j;
            ggml_tensor * scl = next_op();
            if (scl && scl->op == GGML_OP_SCALE && scl->src[0] == add && ggml_is_contiguous(scl)) {
                memcpy(&p.scale, (const float *) scl->op_params + 0, sizeof(float));
                memcpy(&p.bias,  (const float *) scl->op_params + 1, sizeof(float));
                has_res = true;
                y   = scl;
                end = j;
            }
        }
        if (!has_res) {
            j = j0;
        }
    }

    ggml_tensor * out = cgraph->nodes[end];
    if (out != y || out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || ggml_nelements(out) != C*T*ns) {
        CONV_CHAIN_DECLINE("output");
    }
    // a convolution followed by SiLU is a Mamba / gated delta net one, not Inkling's: leave it to its own kernels
    for (int k = end + 1; k < n_nodes && k <= end + 8; ++k) {
        const ggml_tensor * node = cgraph->nodes[k];
        if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SILU && conv_chain_unreshape(node->src[0]) == out) {
            CONV_CHAIN_DECLINE("SiLU after the convolution");
        }
    }

    // every node of the chain but the output is used inside it only
    for (int k = i; k < end; ++k) {
        const ggml_tensor * node = cgraph->nodes[k];
        if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
            CONV_CHAIN_DECLINE("an intermediate is a graph output");
        }
        int uses = 0;
        for (int l = k + 1; l <= end; ++l) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                uses += cgraph->nodes[l]->src[s] == node;
            }
        }
        if (uses != ggml_node_get_use_count(cgraph, k)) {
            CONV_CHAIN_DECLINE("an intermediate is used outside the chain");
        }
    }

    // memory: the output may be cur itself, a window the history it was read from; nothing else may overlap
    p.dst = (float *) out->data;
    auto span = [](const void * ptr, int64_t n_floats) {
        conv_chain_range r;
        r.a = (uintptr_t) ptr;
        r.b = r.a + n_floats*sizeof(float);
        return r;
    };
    const conv_chain_range r_cur  = span(p.cur, C*T*ns);
    const conv_chain_range r_past = span(p.past, (nc1 - 1) + (C - 1)*p.past_s1 + (ns - 1)*p.past_s2 + 1);
    const conv_chain_range r_w    = span(p.w, (C - 1)*p.w_s1 + nc);
    const conv_chain_range r_dyn  = p.dyn ? span(p.dyn, nc*p.n_head*T*ns) : conv_chain_range();
    const conv_chain_range r_dst  = span(p.dst, C*T*ns);
    if ((conv_chain_overlap(r_dst, r_cur) && p.dst != p.cur) || conv_chain_overlap(r_dst, r_past) || conv_chain_overlap(r_dst, r_w) ||
        (p.dyn && conv_chain_overlap(r_dst, r_dyn))) {
        CONV_CHAIN_DECLINE("output overlaps an input");
    }
    for (int k = 0; k < p.n_win; ++k) {
        const conv_chain_range r = span(p.win_dst[k], (ns - 1)*p.win_s2[k] + C*nc1);
        const bool same_as_past = p.win_dst[k] == p.past && p.past_s1 == nc1 && p.past_s2 == p.win_s2[k];
        if (conv_chain_overlap(r, r_cur) || conv_chain_overlap(r, r_w) || (p.dyn && conv_chain_overlap(r, r_dyn)) ||
            conv_chain_overlap(r, r_dst) || (conv_chain_overlap(r, r_past) && !same_as_past)) {
            CONV_CHAIN_DECLINE("a stored window overlaps an input or the output");
        }
        for (int l = 0; l < k; ++l) {
            if (conv_chain_overlap(r, span(p.win_dst[l], (ns - 1)*p.win_s2[l] + C*nc1))) {
                CONV_CHAIN_DECLINE("stored windows overlap");
            }
        }
    }

    const bool has_dyn = p.dyn != nullptr;
    switch (nc) {
        case 2: conv_chain_launch<2>(p, (int) ns, has_dyn, has_res, ctx.stream()); break;
        case 3: conv_chain_launch<3>(p, (int) ns, has_dyn, has_res, ctx.stream()); break;
        case 4: conv_chain_launch<4>(p, (int) ns, has_dyn, has_res, ctx.stream()); break;
        case 5: conv_chain_launch<5>(p, (int) ns, has_dyn, has_res, ctx.stream()); break;
        default: GGML_ABORT("fatal error");
    }
    if (conv_chain_debug()) {
        GGML_LOG_WARN("%s: node %d (%s): fused %d nodes, C %d, T %d, ns %d, n_conv %d, windows %d, dynamic %d, residual %d\n",
            __func__, i, x->name, end - i + 1, (int) C, (int) T, (int) ns, nc, p.n_win, (int) has_dyn, (int) has_res);
    }
    return end - i;
}

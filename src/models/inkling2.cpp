#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

#include <cstdlib>

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_exp    = hparams.n_ff_exp(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(i)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
    }
}

// [RDNA-FAST] [TAG_INKLING_FOLDS] the per-token constants folded into weights once the data is loaded (F32 copies,
// the loaded weights stay as they are). A short convolution adds its input: out = c*(x_t + sum_k w_k x_(t-n_conv+1+k))
// = sum_k c*(w_k + [k = n_conv - 1]) x_(...), the tap n_conv - 1 being the current input. The K and V convolutions
// keep c = 1 here: the graph folds their scale into the norm that follows (see build_layer). The relative bias gets
// its padding row of zeros and the relative clip threshold, which multiplies the clipped R it is multiplied with.
void llama_model_inkling2::load_arch_post() {
    const int64_t n_conv = hparams.n_shortconv_l_cache;
    const float   c_post = 1.0f/sqrtf((float) n_conv + 1.0f);

    std::vector<derived_f32> items;
    derived.assign(n_layer_phys(), {});

    auto read = [](const ggml_tensor * t, std::vector<float> & v) {
        if (!t || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || !t->buffer) {
            return false;
        }
        v.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        return true;
    };

    for (int il = 0; il < n_layer_phys(); ++il) {
        const auto & layer = layers[il];
        auto & d = derived[il];

        const std::pair<const ggml_tensor *, std::pair<float, ggml_tensor **>> convs[] = {
            { layer.attn_k_conv,    { 1.0f,   &d.k_conv    } },
            { layer.attn_v_conv,    { 1.0f,   &d.v_conv    } },
            { layer.attn_post_conv, { c_post, &d.attn_post } },
            { layer.ffn_post_conv,  { c_post, &d.ffn_post  } },
        };
        for (const auto & [w, how] : convs) {
            std::vector<float> v;
            if (!read(w, v) || w->ne[0] != n_conv) {
                continue;
            }
            for (int64_t ch = 0; ch < w->ne[1]; ++ch) {
                float * taps = v.data() + ch*n_conv;
                taps[n_conv - 1] += 1.0f;
                for (int64_t k = 0; k < n_conv; ++k) {
                    taps[k] *= how.first;
                }
            }
            items.push_back({ w, std::string(w->name) + ".folded", w->ne[0], w->ne[1], std::move(v), how.second });
        }

        std::vector<float> b;
        if (read(layer.attn_rel_bias, b)) {
            const ggml_tensor * w = layer.attn_rel_bias;
            std::vector<float> p((size_t) w->ne[0]*(w->ne[1] + 1), 0.0f);
            for (size_t i = 0; i < b.size(); ++i) {
                p[i] = b[i]*hparams.f_relative_clip;
            }
            items.push_back({ w, std::string(w->name) + ".padded", w->ne[0], w->ne[1] + 1, std::move(p), &d.rel_bias });
        }
    }

    make_derived_f32(items);
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// [RDNA-FAST] the clip "scale each row down to an L2 norm of at most t" was 8-9 ops (sqr, sum_rows, sqrt, +1e-6, fill,
// div, clamp, mul): it equals t * x/max(|x|, t), one L2_NORM with floor t (times t, which the caller folds)
ggml_tensor * llama_model_inkling2::graph::norm_floor(ggml_tensor * cur, ggml_tensor * scale, float floor) {
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }
    return ggml_l2_norm(ctx0, cur, floor);
}

// [RDNA-FAST] [TAG_INKLING_FUSIONS] the normalized stack of a block list, newest block first (the order of vanilla's
// concatenations): the stack of the list without its newest block, with that block's normalized row on top
const llama_model_inkling2::graph::block_stack * llama_model_inkling2::graph::stack(const std::vector<ggml_tensor *> & blocks) {
    if (blocks.empty()) {
        return nullptr;
    }
    auto it = block_stacks.find(blocks);
    if (it != block_stacks.end()) {
        return &it->second;
    }
    const block_stack * prev = stack(std::vector<ggml_tensor *>(blocks.begin(), blocks.end() - 1));

    ggml_tensor * normed   = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, blocks.back(), n_embd, 1, n_tokens), 1e-6f);
    ggml_tensor * normed_t = ggml_reshape_3d(ctx0, normed, 1, n_embd, n_tokens);

    block_stack st;
    st.states   = prev ? ggml_concat(ctx0, normed,   prev->states,   1) : normed;
    st.states_t = prev ? ggml_concat(ctx0, normed_t, prev->states_t, 0) : normed_t;
    return &block_stacks.emplace(blocks, st).first->second;
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
// [RDNA-FAST] [TAG_INKLING_FUSIONS] rows [blocks.back(), ..., blocks.front(), cur] as vanilla's, each RMS-normalized (the
// norm is per row, so normalizing the rows one by one gives the same values), from the shared stacks; on the GPU the
// concatenations, the scores, the softmax and the mix run as one kernel (depth mix fusion)
ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    std::vector<ggml_tensor *> key = blocks;
    key.push_back(cur);
    block_stack & all = mix_states[key];
    if (!all.states) {
        ggml_tensor * normed   = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), 1e-6f);
        ggml_tensor * normed_t = ggml_reshape_3d(ctx0, normed, 1, n_embd, n_tokens);

        const block_stack * st = stack(blocks);
        all.states   = st ? ggml_concat(ctx0, st->states,   normed,   1) : normed;
        all.states_t = st ? ggml_concat(ctx0, st->states_t, normed_t, 0) : normed_t;
    }

    // one token: the queries broadcast as they are
    ggml_tensor * queries = n_tokens == 1 ? query :
        ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, all.states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    // [A/B 11-25, to be removed] vanilla 396ed2cce's order (the scores as src0, n_embd the long dimension) instead of the
    // depth-mix fusion for prompt batches (more than 32 tokens; decode and drafts stay fused):
    //   LLAMA_INKLING_MIX_PROMPT=1 the loop-message mixes (4 queries), =2 every mix; unset/0: fused everywhere
    //   LLAMA_INKLING_MIX_SCORES_SRC0=1 (window w1 arm B): the 4-query mixes at every batch size
    static const int  mix_prompt  = getenv("LLAMA_INKLING_MIX_PROMPT") ? atoi(getenv("LLAMA_INKLING_MIX_PROMPT")) : 0;
    static const bool scores_src0 = getenv("LLAMA_INKLING_MIX_SCORES_SRC0") && atoi(getenv("LLAMA_INKLING_MIX_SCORES_SRC0")) != 0;
    const int64_t n_queries = query->ne[1];
    const bool use_src0 = (scores_src0 && n_queries > 3) ||
        (n_tokens > 32 && (mix_prompt >= 2 || (mix_prompt == 1 && n_queries > 3)));
    if (use_src0) {
        ggml_tensor * out = ggml_mul_mat(ctx0, scores, all.states_t); // [n_queries, n_embd, n_tokens]
        if (n_queries == 1) {
            return ggml_reshape_3d(ctx0, out, n_embd, 1, n_tokens);   // the same layout as [n_embd, 1, n_tokens]
        }
        return ggml_cont(ctx0, ggml_permute(ctx0, out, 1, 0, 2, 3));  // [n_embd, n_queries, n_tokens]
    }

    return ggml_mul_mat(ctx0, all.states_t, scores);
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * folded,
        float scale,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    // [RDNA-FAST] [TAG_INKLING_CONV_CHAIN] the inputs first and the output at the end, so that the chain from the
    // concatenation to the output is consecutive in the graph (one kernel on the GPU)
    ggml_build_forward_expand(gf, cur);
    if (dynamic) {
        ggml_build_forward_expand(gf, dynamic);
    }

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    // [RDNA-FAST] [TAG_INKLING_FOLDS] the folded weight carries the residual and the scale
    ggml_tensor * out = ggml_ssm_conv(ctx0, x, folded ? folded : weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);
    if (!folded) {
        out = ggml_add(ctx0, cur, out);
    }

    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_3d(ctx0, x, n_seq_tokens, n_channels, n_seqs, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);

            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, n_channels/n_heads, n_heads, n_tokens);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    // vanilla scales by 1/sqrt((dynamic ? 2 : 1)*n_conv + 1) here: the post convolutions pass it (folded: inside the
    // weight), the K and V ones pass 1 and fold it into the norm that follows
    if (!folded && scale != 1.0f) {
        out = ggml_scale(ctx0, out, scale);
    }
    ggml_build_forward_expand(gf, out);
    return out;
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * cur_n, ggml_tensor * message, ggml_tensor * message_n,
        const loop_layer & loop, int ctrl) {
    ggml_tensor * gate = ggml_concat(ctx0, cur_n, message_n, 0);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    ggml_tensor * gain   = nullptr;
    ggml_tensor * forget = nullptr;
    if (loop.ctrl_gain_forget[ctrl]) {
        // [RDNA-FAST] [TAG_INKLING_FOLDS] both projections in one product, [2*n_embd, n_tokens]; a tanh per half, so that
        // each tanh-mul-sub/add below is one kernel on the GPU ([TAG_INKLING_FUSIONS], the same values)
        ggml_tensor * both = build_lora_mm(loop.ctrl_gain_forget[ctrl], gate);
        gain   = ggml_tanh(ctx0, ggml_view_2d(ctx0, both, n_embd, n_tokens, both->nb[1], 0));
        forget = ggml_tanh(ctx0, ggml_view_2d(ctx0, both, n_embd, n_tokens, both->nb[1], n_embd*ggml_element_size(both)));
    } else {
        gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
        forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));
    }

    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    attn = inp->get_attn();
    recr = inp->get_recr();

    ggml_tensor * inp_pos = build_inp_pos();
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, const loop_layer * loop) {
    GGML_ASSERT(!messages == !loop);

    const auto & layer = model.layers[il];

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // [RDNA-FAST] [TAG_INKLING_FOLDS] the weights derived after the load (none in a dry run)
    const auto & m = static_cast<const llama_model_inkling2 &>(model);
    const derived_layer * fold = ip < (int) m.derived.size() ? &m.derived[ip] : nullptr;

    // the Q, K and V messages of [n_embd, LOOP_CTRL_COUNT, n_tokens] RMS-normalized at once (the norm is per row; the
    // FFN one is normalized where it is used, so this buffer dies with the attention inputs), and the views of
    // controller ctrl, [n_embd, n_tokens] (vanilla copies each out and normalizes it inside control)
    ggml_tensor * messages_n = messages ? ggml_rms_norm(ctx0,
            ggml_view_3d(ctx0, messages, n_embd, LOOP_CTRL_FFN, n_tokens, messages->nb[1], messages->nb[2], 0), 1e-6f) : nullptr;
    auto message = [&](ggml_tensor * msgs, int ctrl) {
        return ggml_view_2d(ctx0, msgs, n_embd, n_tokens, msgs->nb[2], ctrl*msgs->nb[1]);
    };

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

    ggml_tensor * inpSA = cur;

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * inputs[3] = {cur, cur, cur};
    if (layer.attn_res_query) {
        ggml_tensor * mixed = mix(blocks, cur, layer.attn_res_query);
        if (is_swa) {
            // [RDNA-FAST] one mixed input, contiguous [n_embd, 1, n_tokens]: a reshape, not three copies
            inputs[0] = inputs[1] = inputs[2] = ggml_reshape_2d(ctx0, mixed, n_embd, n_tokens);
        } else {
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], j*mixed->nb[1]));
            }
        }
    }

    if (messages) {
        // the messages make the Q, K and V inputs of a SWA layer differ (its three inputs are one: normalized once)
        ggml_tensor * shared   = is_swa ? inputs[0] : nullptr;
        ggml_tensor * shared_n = is_swa ? ggml_rms_norm(ctx0, shared, 1e-6f) : nullptr;
        for (int j = 0; j < 3; ++j) {
            ggml_tensor * in   = is_swa ? shared   : inputs[j];
            ggml_tensor * in_n = is_swa ? shared_n : ggml_rms_norm(ctx0, in, 1e-6f);
            inputs[j] = control(in, in_n, message(messages, LOOP_CTRL_Q + j), message(messages_n, LOOP_CTRL_Q + j),
                    *loop, LOOP_CTRL_Q + j);
        }
    }

    if (is_swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (is_swa) {
        inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }

    // self-attention
    {
        // [RDNA-FAST] [TAG_INKLING_FOLDS] exact rewrites of vanilla's per-token constants, the loop passes unchanged:
        // - the K and V convolutions end in a scale c = 1/sqrt(n_taps + 1) (n_taps: n_conv, twice that with the
        //   dynamic taps), vanilla's clip(c*y, t) = t*norm_floor(y, t/c) and rms_norm(c*y, eps) = rms_norm(y, eps/c^2)
        // - Q and K come out of their clips divided by t (norm_floor), t^2 joins the attention scale
        // - R's threshold multiplies the derived bias table instead (the table is linear in R)
        const float n_taps = (float) ((is_swa ? 1 : 2)*hparams.n_shortconv_l_cache + 1);
        const float t      = hparams.f_qk_clip;

        ggml_tensor * Qcur = build_lora_mm(layer.wq, inputs[0]);
        ggml_tensor * Kcur = convolve(build_lora_mm(layer.wk, inputs[1]), layer.attn_k_conv, fold ? fold->k_conv : nullptr, 1.0f,
                is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
        ggml_tensor * Vcur = convolve(build_lora_mm(layer.wv, inputs[2]), layer.attn_v_conv, fold ? fold->v_conv : nullptr, 1.0f,
                is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa);
        ggml_tensor * Rcur = build_lora_mm(layer.attn_rel, inputs[0]);

        Qcur = norm_floor(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), layer.attn_q_clip, t);
        Kcur = norm_floor(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), nullptr,           t*sqrtf(n_taps));
        Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps*n_taps);
        Rcur = norm_floor(ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens), layer.attn_rel_clip, hparams.f_relative_clip);
        if (!(fold && fold->rel_bias)) {
            Rcur = ggml_scale(ctx0, Rcur, hparams.f_relative_clip);
        }

        if (!is_swa) {
            Qcur = ggml_mul(ctx0, Qcur, tau);
            Rcur = ggml_mul(ctx0, Rcur, tau);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        ggml_tensor * sinks = build_lora_mm(layer.attn_sink_proj, inputs[0]);

        // project r on the bias table first, the distances pick from it: [n_extent + 1, n_head, n_tokens]
        // ([RDNA-FAST] the padded table is derived once after the load; vanilla pads the weight every pass)
        ggml_tensor * rel_bias = fold && fold->rel_bias ? fold->rel_bias : ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0);
        ggml_tensor * table = ggml_mul_mat(ctx0, rel_bias, Rcur);

        // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
        table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));

        // [RDNA-FAST] the zero sinks are made once per graph
        if (!sinks0 || sinks0->ne[0] != n_head) {
            sinks0 = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
        }

        // [RDNA-FAST] the gather is build_attn_rel_table's: dense like vanilla when small, per head when the dense
        // [n_kv, n_tokens, n_head] bias would be big (vanilla builds it here, 51 GB at 128k cells and 2048 tokens)
        cur = build_attn_rel_table(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, table, sinks0, t*t/sqrtf(n_embd_head), il);

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, inputs[0])));
        cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, fold ? fold->attn_post : nullptr,
                1.0f/sqrtf((float) hparams.n_shortconv_l_cache + 1.0f), nullptr, state, history, recr, 2*n_embd_k_gqa);
        cb(cur, "attn_out", il);
    }

    // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
    if (ip % hparams.attn_res_block_size == 0) {
        blocks.push_back(inpSA);
    } else {
        cur = ggml_add(ctx0, inpSA, cur);
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query), n_embd, n_tokens);
    if (messages) {
        ggml_tensor * msg_ffn = message(messages, LOOP_CTRL_FFN);
        cur = control(cur, ggml_rms_norm(ctx0, cur, 1e-6f), msg_ffn, ggml_rms_norm(ctx0, msg_ffn, 1e-6f),
                *loop, LOOP_CTRL_FFN);
    }
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    cur = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            nullptr,
            nullptr,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used(il),
            LLM_FFN_SITU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
            il,
            nullptr,
            layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);

    cur = convolve(cur, layer.ffn_post_conv, fold ? fold->ffn_post : nullptr, 1.0f/sqrtf((float) hparams.n_shortconv_l_cache + 1.0f),
            nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd);
    cb(cur, "ffn_out", il);

    cur = ggml_add(ctx0, ffn_inp, cur);
    cb(cur, "l_out", il);

    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_inkling2::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params), model(model) {
    ggml_tensor * cur = build_inputs();

    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < n_layer; ++il) {
        cur = build_layer(il, il, cur, blocks);
    }

    build_output(cur, blocks);
}

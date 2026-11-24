// [TAG_KV_LANES] [TAG_SHARED_PASS] A sequence must compute the same thing whichever slot of a shared KV pool it lives
// in, and whether its tokens are decoded alone or in a pass shared with other sequences.
//
// The test decodes fixed texts on the CPU in differently set up contexts and compares the logits bit for bit:
//   1. the same tokens as sequence 0 and as sequence 2 of a shared pool with slot ranges (both store their cells
//      bottom-up, the second range starts in the middle of the pool)
//   2. two different texts (different lengths), prompts first, then steps of k = 1..4 tokens per sequence:
//      a. one llama_decode per sequence  ==  one llama_decode for both with the shared pass off
//      b. the shared pass (the default, ids 0 and 2)  ==  the stock multi-stream cache (ids 0 and 1), which
//         evaluates the same number of columns per pass
//      c. the shared pass with ids 0 and 1: the bottom-up sequence  ==  the stock multi-stream cache
//      d. k = 1: the shared pass  ==  one llama_decode per sequence
//   3. as 2a and 2b with rejected draft tokens: after every step some of the last tokens of each sequence are removed
//      again (recurrent-state snapshots) and replaced by other tokens
//   4. three texts in three slots, k = 1 and 2: the shared pass  ==  the stock multi-stream cache
//
// Why not simply "everything equals everything": the CPU kernels are deterministic, but not invariant. A pass of 4 or
// more columns takes other matmul kernels than a narrower one, and a sequence with an odd id stores its cells top-down,
// so its attention sums in another order. Either moves last bits, and the 8-bit activation rounding of the quantized
// matmuls then drives two such runs as far apart as two independent roundings (several tenths of a logit). The checks
// above compare only runs with the same pass width and the same cell order, where every bit must match.
//
// Needs a model whose memory has slot ranges (plain, SWA, hybrid and hybrid SWA caches since 2026-11-24); others are
// skipped.
// usage: test-rocmfpx-seq-parity -m <model.gguf> [-t <threads>]

#include "llama.h"
#include "ggml.h"

#include "../src/llama-ext.h" // llama_kv_lanes_merge_max / llama_kv_lanes_set_merge

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void log_cb(enum ggml_log_level level, const char * text, void * user) {
    (void) user;
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

struct run_cfg {
    bool             unified;   // shared pool (slot ranges) or the stock multi-stream cache
    std::vector<int> ids;       // the sequence ids, one text each
    bool             shared;    // the shared pass stays on (the default) or is switched off
    int              k;         // tokens per sequence and step
    bool             joint;     // one llama_decode per step for all sequences
    bool             same_text; // all sequences decode the same tokens
    ggml_type        type_kv;
    std::vector<int> rb;        // per sequence: tokens removed again after every step (rejected draft tokens)
};

static const int n_prompt = 40;
static const int n_steps  = 3;

// the logits of every step (prompt, then n_steps) per sequence; n_merge = the token limit of a shared pass (0 = off)
static bool run(llama_model * model, const run_cfg & c, const std::vector<llama_token> & toks, int n_threads,
                std::vector<std::vector<float>> & out, uint32_t * n_merge = nullptr) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_seq   = (int) c.ids.size();

    bool has_rb  = false;
    int  id_max  = 0;
    for (int s = 0; s < n_seq; ++s) {
        has_rb = has_rb || (s < (int) c.rb.size() && c.rb[s] > 0);
        id_max = c.ids[s] > id_max ? c.ids[s] : id_max;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 8192;
    cp.n_batch         = 512;
    cp.n_ubatch        = 256;
    cp.n_seq_max       = (uint32_t) id_max + 1;
    cp.n_rs_seq        = has_rb ? 4 : 0;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.kv_unified      = c.unified;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k          = c.type_kv;
    cp.type_v          = c.type_kv;
    cp.offload_kqv     = false;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create a context\n");
        return false;
    }

    if (!c.shared) {
        llama_kv_lanes_set_merge(ctx, false);
    }
    if (n_merge) {
        *n_merge = llama_kv_lanes_merge_max(ctx);
    }

    // text s starts 35 tokens after text s - 1 and its prompt is 7 tokens longer
    std::vector<int> off(n_seq), np(n_seq), pos(n_seq, 0);
    for (int s = 0; s < n_seq; ++s) {
        off[s] = c.same_text ? 0 : 35*s;
        np[s]  = c.same_text ? n_prompt : n_prompt + 7*s;
    }

    llama_batch batch = llama_batch_init(512, 0, 1);
    bool ok = true;

    // moves the source of the tokens after a rollback, so that the replaced tokens differ from the removed ones
    int tok_shift = 0;

    const auto add = [&](int s, int n) {
        for (int i = 0; i < n; ++i) {
            const int b = batch.n_tokens++;
            batch.token[b]     = toks[off[s] + pos[s] + i + tok_shift];
            batch.pos[b]       = pos[s] + i;
            batch.n_seq_id[b]  = 1;
            batch.seq_id[b][0] = c.ids[s];
            batch.logits[b]    = i == n - 1;
        }
        return batch.n_tokens - 1;
    };
    const auto decode = [&]() {
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "llama_decode failed\n");
            ok = false;
        }
    };
    const auto keep = [&](int s, int i_batch) {
        if (ok) {
            const float * l = llama_get_logits_ith(ctx, i_batch);
            out[s].insert(out[s].end(), l, l + n_vocab);
        }
    };

    out.assign(n_seq, {});

    for (int s = 0; s < n_seq && ok; ++s) {
        batch.n_tokens = 0;
        const int i_last = add(s, np[s]);
        decode();
        keep(s, i_last);
        pos[s] = np[s];
    }
    for (int step = 0; step < n_steps && ok; ++step) {
        if (c.joint) {
            std::vector<int> i_last(n_seq);
            batch.n_tokens = 0;
            for (int s = 0; s < n_seq; ++s) {
                i_last[s] = add(s, c.k);
            }
            decode();
            for (int s = 0; s < n_seq; ++s) {
                keep(s, i_last[s]);
            }
        } else {
            for (int s = 0; s < n_seq && ok; ++s) {
                batch.n_tokens = 0;
                const int i_last = add(s, c.k);
                decode();
                keep(s, i_last);
            }
        }
        for (int s = 0; s < n_seq; ++s) {
            pos[s] += c.k;
        }

        // rejected draft tokens: the recurrent state goes back to a snapshot, the attention cells are freed
        for (int s = 0; s < n_seq && ok && has_rb; ++s) {
            const int n_rb = s < (int) c.rb.size() ? c.rb[s] : 0;
            if (n_rb > 0) {
                if (!llama_memory_seq_rm(llama_get_memory(ctx), c.ids[s], pos[s] - n_rb, -1)) {
                    fprintf(stderr, "rollback of %d tokens failed\n", n_rb);
                    ok = false;
                }
                pos[s] -= n_rb;
            }
        }
        if (has_rb) {
            tok_shift += 5;
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);

    return ok;
}

static bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

int main(int argc, char ** argv) {
    const char * path = nullptr;
    int n_threads = 4;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        }
    }
    if (!path) {
        fprintf(stderr, "usage: %s -m <model.gguf> [-t <threads>]\n", argv[0]);
        return 1;
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }
    // [TAG_KV_LANES] any model whose memory has slot ranges: the plain cache, the SWA caches, the hybrid memories
    {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = 1024;
        cp.n_seq_max       = 2;
        cp.kv_unified      = true;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.offload_kqv     = false;
        llama_context * ctx = llama_init_from_model(model, cp);
        const bool lanes = ctx != nullptr && llama_kv_lanes_enabled(ctx);
        if (ctx) {
            llama_free(ctx);
        }
        if (!lanes) {
            printf("the memory of this model has no slot ranges, nothing to test\n");
            llama_model_free(model);
            return 0;
        }
    }

    const std::string text =
        "The ship left the harbour two hours before dawn, and by the time the fog had lifted the coast was only a thin "
        "grey line behind it. The captain had sailed this route for thirty years and trusted neither the charts nor the "
        "weather reports; she trusted the colour of the water and the way the gulls turned. On the third day the wind "
        "dropped, the sea went flat as slate, and the crew began to repair the nets that the last storm had torn. Nobody "
        "spoke about the cargo in the forward hold, although everyone had seen the seals on the crates and the two men "
        "who never left the deck above them. In the evening the cook brought up bread and salted fish, and the youngest "
        "sailor asked where they were really going. The captain looked at him for a long time and then said that the "
        "harbour master of the northern port had paid for the voyage in advance, that the crates held instruments for "
        "the observatory on the island, and that the two men were astronomers who were afraid of the sea. The sailor "
        "did not believe a word of it, but he nodded, finished his bread and went back to the nets.";
    std::vector<llama_token> toks(1024);
    const int n_tok = llama_tokenize(llama_model_get_vocab(model), text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, false);
    if (n_tok < 35*2 + n_prompt + 7*2 + n_steps*4 + n_steps*5) {
        fprintf(stderr, "the text has only %d tokens\n", n_tok);
        return 1;
    }

    int n_fail = 0;
    const auto check = [&](bool ok, const char * type, int k, const char * what) {
        printf("%s  cache %-4s  k %d  %s\n", ok ? "ok  " : "FAIL", type, k, what);
        n_fail += ok ? 0 : 1;
    };
    // a comparison across pass widths: bit-exact only where every kernel is width-invariant (Qwen3.x with quantized
    // weights); the f16/f32 matmuls of the tiny test models take other kernels at 2 columns than at 1. Logged only.
    const auto info = [&](bool ok, const char * type, int k, const char * what) {
        printf("%s  cache %-4s  k %d  %s\n", ok ? "ok  " : "info", type, k, what);
    };

    typedef std::vector<std::vector<float>> rows;

    for (const ggml_type type : { GGML_TYPE_Q4_0, GGML_TYPE_F16 }) {
        const char * tn = ggml_type_name(type);

        // a K/V cache of this type for the head size of the model (q4_0 needs heads of whole blocks)
        {
            rows r;
            if (!run(model, { true, { 0 }, true, 1, false, false, type, {} }, toks, n_threads, r)) {
                printf("skip  cache %-4s  no context with this cache type for the model\n", tn);
                continue;
            }
        }

        // the shared pass is on by default in a pool with slot ranges where the model's attention builds it per
        // range, and can be switched off; models without it get the slot-range checks only
        bool shared = false;
        {
            rows r;
            uint32_t n_on = 0, n_off = 0, n_stream = 0;
            bool ok = run(model, { true,  { 0, 1 }, true,  1, true, false, type, {} }, toks, n_threads, r, &n_on);
            ok = ok && run(model, { true,  { 0, 1 }, false, 1, true, false, type, {} }, toks, n_threads, r, &n_off);
            ok = ok && run(model, { false, { 0, 1 }, true,  1, true, false, type, {} }, toks, n_threads, r, &n_stream);
            shared = n_on > 0;
            if (shared) {
                check(ok && n_off == 0 && n_stream == 0, tn, 1, "the shared pass is on by default with slot ranges, off on request, absent without them");
            } else {
                check(ok && n_stream == 0, tn, 1, "no shared pass for this model (slot ranges only): one pass per sequence");
            }
        }

        // 1. the same tokens in two bottom-up ranges of the shared pool
        {
            rows r;
            const bool ok = run(model, { true, { 0, 2 }, true, 1, false, true, type, {} }, toks, n_threads, r);
            check(ok && same_bits(r[0], r[1]), tn, 1, "the same tokens as sequence 0 and as sequence 2 of the shared pool");
        }

        for (int k = 1; k <= 4; ++k) {
            // 2. two texts
            rows apart, joint, merged02, merged01, stream;

            bool ok = run(model, { true,  { 0, 1 }, true,  k, false, false, type, {} }, toks, n_threads, apart);
            ok = ok && run(model, { true,  { 0, 1 }, false, k, true,  false, type, {} }, toks, n_threads, joint);

            check(ok && same_bits(apart[0], joint[0]) && same_bits(apart[1], joint[1]), tn, k,
                    "one decode per sequence == one decode for both with the shared pass off");

            if (!shared) {
                // with rollbacks, one pass per sequence
                if (k >= 2) {
                    rows rb_apart, rb_joint;
                    const std::vector<int> rb = { 1, k - 1 };
                    bool rok = run(model, { true,  { 0, 1 }, true,  k, false, false, type, rb }, toks, n_threads, rb_apart);
                    rok = rok && run(model, { true,  { 0, 1 }, false, k, true,  false, type, rb }, toks, n_threads, rb_joint);
                    check(rok && same_bits(rb_apart[0], rb_joint[0]) && same_bits(rb_apart[1], rb_joint[1]), tn, k,
                            "with rollbacks: one decode per sequence == one decode for both with the shared pass off");
                }
                continue;
            }

            ok = ok && run(model, { true,  { 0, 2 }, true,  k, true,  false, type, {} }, toks, n_threads, merged02);
            ok = ok && run(model, { true,  { 0, 1 }, true,  k, true,  false, type, {} }, toks, n_threads, merged01);
            ok = ok && run(model, { false, { 0, 1 }, true,  k, true,  false, type, {} }, toks, n_threads, stream);

            check(ok && same_bits(merged02[0], stream[0]) && same_bits(merged02[1], stream[1]), tn, k,
                    "shared pass on two bottom-up ranges == stock multi-stream cache, both texts");
            check(ok && same_bits(merged01[0], stream[0]), tn, k,
                    "shared pass on slots 0 and 1 == stock multi-stream cache, the bottom-up text");
            if (k == 1) {
                info(ok && same_bits(merged01[0], apart[0]) && same_bits(merged01[1], apart[1]), tn, k,
                        "shared pass on slots 0 and 1 == one decode per sequence, both texts (across pass widths)");
            }

            // 3. rejected draft tokens: 1 token of the first text and k - 1 of the second after every step
            if (k >= 2) {
                rows rb_apart, rb_joint, rb_merged, rb_stream;
                const std::vector<int> rb = { 1, k - 1 };

                bool rok = run(model, { true,  { 0, 1 }, true,  k, false, false, type, rb }, toks, n_threads, rb_apart);
                rok = rok && run(model, { true,  { 0, 1 }, false, k, true,  false, type, rb }, toks, n_threads, rb_joint);
                rok = rok && run(model, { true,  { 0, 2 }, true,  k, true,  false, type, rb }, toks, n_threads, rb_merged);
                rok = rok && run(model, { false, { 0, 1 }, true,  k, true,  false, type, rb }, toks, n_threads, rb_stream);

                check(rok && same_bits(rb_apart[0], rb_joint[0]) && same_bits(rb_apart[1], rb_joint[1]), tn, k,
                        "with rollbacks: one decode per sequence == one decode for both with the shared pass off");
                check(rok && same_bits(rb_merged[0], rb_stream[0]) && same_bits(rb_merged[1], rb_stream[1]), tn, k,
                        "with rollbacks: shared pass on two bottom-up ranges == stock multi-stream cache, both texts");
            }

            // 4. three texts: a pass of 3 and of 6 tokens
            if (k <= 2) {
                rows m024, m012, s012;

                bool tok = run(model, { true,  { 0, 2, 4 }, true, k, true, false, type, {} }, toks, n_threads, m024);
                tok = tok && run(model, { true,  { 0, 1, 2 }, true, k, true, false, type, {} }, toks, n_threads, m012);
                tok = tok && run(model, { false, { 0, 1, 2 }, true, k, true, false, type, {} }, toks, n_threads, s012);

                check(tok && same_bits(m024[0], s012[0]) && same_bits(m024[1], s012[1]) && same_bits(m024[2], s012[2]), tn, k,
                        "three texts: shared pass on three bottom-up ranges == stock multi-stream cache");
                check(tok && same_bits(m012[0], s012[0]) && same_bits(m012[2], s012[2]), tn, k,
                        "three texts: shared pass on slots 0, 1, 2 == stock multi-stream cache, the bottom-up texts");
            }
        }
    }

    llama_model_free(model);

    if (n_fail > 0) {
        printf("FAILED: %d check(s)\n", n_fail);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}

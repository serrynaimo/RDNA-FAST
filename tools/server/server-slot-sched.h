#pragma once

#include "server-common.h"
#include "server-task.h"

#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <vector>

// which prompt goes to which slot: slot prompt state, prompt cache and slot selection

// what a cached prompt can give to a new prompt
struct server_prompt_reuse {
    int32_t   n_past   = 0; // tokens to keep
    llama_pos pos_next = 0;

    // restore this checkpoint first, nullptr when the live state is used as it is
    // [RDNA-FAST] our checkpoints are held by value in server_prompt::checkpoints (the disk tier writes and reads
    // them), so this points into that list
    const common_prompt_checkpoint * ckpt = nullptr;
};

struct server_prompt {
    server_tokens tokens;

    std::list<common_prompt_checkpoint> checkpoints;

    // the lora config that the tokens were processed with, it follows the state to the prompt cache and back
    // TODO: slot files do not store it
    std::vector<common_adapter_lora_info> lora;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
            lora,
        };
    }

    // n_prefix: tokens shared with the new prompt, must be > 0
    // has_new_tokens: the new prompt is longer than n_prefix
    // n_swa: 0 if a state must end right before the next token (no SWA, or recurrent)
    // pos_min: min position of the live state
    // n_past == 0 without a checkpoint means that nothing can be reused
    server_prompt_reuse get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const;
};

struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;

    // speculative-impl state captured at the exact token boundary (e.g. the MTP
    // drafter's deferred boundary row); optional, restored on an exact-boundary hit
    std::vector<uint8_t> spec;

    size_t size() const {
        return main.size() + drft.size() + spec.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt.size();
        }

        return res;
    }
};

// The context checkpoint payloads may contain large host vectors. Disk-cache entries
// keep only the small scheduling metadata in RAM; the checkpoints themselves are persisted
// in full next to the entry (see save_disk) and restored as live checkpoints.
struct server_prompt_checkpoint_meta {
    int64_t   n_tokens = 0;
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;
};

// one context checkpoint persisted next to a disk entry (<stem>-ckpt<k>.bin)
struct server_prompt_disk_ckpt {
    std::string path;
    size_t      size = 0;
    server_prompt_checkpoint_meta meta;
};

struct server_prompt_disk_state {
    server_tokens tokens;
    std::vector<uint8_t> spec;

    std::string path_main;
    std::string path_drft;

    // The context checkpoints persisted alongside the entry, oldest first. A recurrent/hybrid
    // state cannot be truncated, so without them the entry is usable only when the request
    // extends it exactly. With them, a request that diverges anywhere past a checkpoint
    // (a client that re-sends the last assistant turn without reasoning_content, an agent
    // that rewrote or compacted older history, ...) still restores the entry: the slot rolls
    // back to the newest checkpoint at or before the divergence, exactly as it does in-slot,
    // and re-prefills from there.
    std::vector<server_prompt_disk_ckpt> ckpts;

    // speculative-impl state blob persisted next to the entry so it survives a restart
    std::string path_spec;

    size_t size_main = 0;
    size_t size_drft = 0;

    bool has_ckpt() const { return !ckpts.empty(); }

    size_t size_ckpt() const {
        size_t res = 0;
        for (const auto & ck : ckpts) {
            res += ck.size;
        }
        return res;
    }

    // newest persisted checkpoint boundary at or before `lcp` (0 = none)
    int64_t ckpt_at_or_below(size_t lcp) const {
        int64_t res = 0;
        for (const auto & ck : ckpts) {
            if (ck.meta.n_tokens > res && (size_t) ck.meta.n_tokens <= lcp) {
                res = ck.meta.n_tokens;
            }
        }
        return res;
    }

    uint64_t id = 0;
    bool usable = true;

    size_t size() const {
        return size_main + size_drft + size_ckpt();
    }

    int n_tokens() const {
        return tokens.size();
    }
};

// [TAG_REQUEST_HISTORY] the disk tier's counters (the ones log_disk_state prints) for GET /props. The cache lives
// on the main loop; it hands a copy of these to on_disk_stats after every call that may change them
struct server_prompt_disk_stats {
    bool     enabled       = false;
    size_t   entries       = 0;
    size_t   unusable      = 0;
    size_t   bytes         = 0;
    size_t   limit_bytes   = 0;
    size_t   tokens        = 0;
    uint64_t saves         = 0;
    uint64_t loads         = 0;
    uint64_t evictions     = 0;
    uint64_t save_failures = 0;
    bool     save_disabled = false;
    uint64_t bytes_written = 0;
    uint64_t bytes_read    = 0;
    uint64_t bytes_evicted = 0;
    double   restore_ms_total = 0.0; // disk restores: reading the entry's files into the slot
    uint64_t restores_timed   = 0;

    json to_json() const;
};

struct server_prompt_cache {
    server_prompt_cache(
            int32_t limit_size_mib,
             size_t limit_tokens,
        const std::string & disk_base_path = {},
            int32_t disk_limit_size_mib = 0,
        const std::string & disk_identity = {});

    ~server_prompt_cache();

    std::list<server_prompt_cache_state> states;

    // Cold automatic cache. Entries own only token/checkpoint metadata in RAM;
    // target and draft context payloads live in owner-only files.
    std::list<server_prompt_disk_state> disk_states;

    // Identity of the model/context this cache was written for (model path + size,
    // n_ctx, KV types). Written to <run>/identity.txt; a later run adopts a stale run
    // directory's entries only when its identity matches exactly.
    std::string disk_identity;

    // [TAG_CACHE_ALL_MODELS] the entries other identities (other models, or this model with other settings) left in
    // their abandoned run directories, oldest first: kept for their model, and counted against the limit, which covers
    // the whole cache directory. When room is needed they go first, one entry at a time, the oldest first, before any
    // entry of this run's model; a directory goes with its last entry.
    struct foreign_entry {
        std::string              dir;   // its run directory
        std::vector<std::string> files; // state-<id>-*: target, draft, spec, checkpoints (and a crash's temporaries)
        size_t                   bytes = 0;
    };
    std::vector<foreign_entry> disk_foreign;
    size_t disk_foreign_bytes = 0;

#if defined(_WIN32)
    // Held open with deny-all sharing for the life of the run: the Windows
    // counterpart of the POSIX flock() advisory lock, so a later startup can tell a
    // live run directory from an abandoned one.
    FILE * disk_lock_file = nullptr;
#endif

    bool ram_enabled = false;

    // true when both contexts support unbounded partial sequence removal (dense KV):
    // then a cached prompt that is longer than the request can be truncated in place.
    // Recurrent/hybrid targets cannot (their saved state carries no rollback
    // snapshots), so such entries are only usable at their exact boundary or through
    // a checkpoint at or before the divergence point.
    bool partial_seq_rm = false;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    // Disk fields are disabled when disk_owned_path is empty.
    std::string disk_base_path;
    std::string disk_owned_path;
    size_t disk_limit_size = 0;
    size_t disk_size_total = 0;
    int disk_lock_fd = -1;

    uint64_t disk_next_id       = 1;
    uint64_t disk_saves         = 0;
    uint64_t disk_loads         = 0;
    uint64_t disk_evictions     = 0;
    uint64_t disk_bytes_written = 0;
    uint64_t disk_bytes_read    = 0;
    uint64_t disk_bytes_evicted = 0;
    uint64_t disk_save_failures = 0;

    // [TAG_REQUEST_HISTORY] time of the successful disk restores (load_ms in the log)
    double   disk_restore_ms_total = 0.0;
    uint64_t disk_restores_timed   = 0;

    // [TAG_REQUEST_HISTORY] see server_prompt_disk_stats; called on the cache's thread at the end of save(),
    // save_prefix(), load() and update() while the disk tier is enabled
    std::function<void(const server_prompt_disk_stats &)> on_disk_stats;

    server_prompt_disk_stats disk_stats() const;

    // A durable write/removal failure opens this run-level circuit breaker.
    // Existing valid entries remain readable, but no further state files are
    // created for this server process.
    bool disk_save_disabled = false;

    // Prompts shorter than this are never written to disk: keep-alive pings, title
    // generation and similar one-shot requests would otherwise each leave an entry whose
    // size is dominated by the fixed recurrent state (~160 MiB for a few hundred tokens).
    size_t disk_min_tokens = 0;

    // max context checkpoints persisted with each entry, newest first (-1 = all, 0 = none)
    int32_t disk_max_ckpts = -1;

    // a disk entry replaces what the slot already holds only when it keeps at least this
    // many tokens more - below that the read costs more than the prefill it saves
    size_t disk_min_gain = 1024;

    bool disk_enabled() const { return !disk_owned_path.empty(); }

    size_t size() const;

    size_t n_tokens() const;

    size_t disk_size() const;

    size_t disk_n_tokens() const;

    // save the slot's state to the disk tier (if enabled) and the RAM tier (if enabled);
    // returns true when the state is now held by at least one tier
    bool save(
        const server_prompt & prompt,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec);

    server_prompt_cache_state * alloc(
        const server_prompt & prompt,
                    size_t state_size_main,
                    size_t state_size_drft,
        const std::vector<uint8_t> & state_spec);

    // Persist a shared exact-boundary entry for `tokens_prefix`. The contexts must hold
    // exactly these tokens for id_slot, i.e. this is called mid-prefill, between two batches.
    // Used for the system prompt + tools block, which many conversations have in common:
    // a new conversation then starts from the entry instead of prefilling the block again.
    // An existing entry with the same tokens is only touched (kept most-recently-used).
    bool save_prefix(
        const server_tokens & tokens_prefix,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec);

    // true when a usable disk entry holds exactly the first n_tokens of `tokens`
    bool has_disk_prefix(const server_tokens & tokens, size_t n_tokens) const;

    // [RDNA-FAST] vanilla 82dd97047 (#28992): true when a RAM or disk entry is a better start for tokens_new than
    // `prompt` by the rules of load() (no side effects: the disk files are checked only by the load)
    bool has_better(const server_prompt & prompt, const server_tokens & tokens_new,
                    const std::vector<common_adapter_lora_info> & lora_new) const;

    // restore the most useful entry (RAM or disk) for tokens_new into the slot, if any is
    // better than what the slot already holds. lora_new: the lora config of the new prompt, the states made with
    // another config are skipped; the disk tier holds only states made without an active adapter (its files do not
    // record the config), so it is used only when lora_new has none. On a hit, `prompt` is replaced and
    // `state_spec` (optional) receives the entry's exact-boundary speculative state.
    // `probe` marks a lookup made while the slot keeps serving the same conversation
    // (near-miss logging is then demoted to debug).
    // Returns false only when a restore was attempted and failed (the slot must be cleared).
    bool load(
              server_prompt & prompt,
        const server_tokens & tokens_new,
        const std::vector<common_adapter_lora_info> & lora_new,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
                    int32_t   id_slot,
       std::vector<uint8_t> * state_spec,
                       bool   probe = false);

    void update();

private:
    bool save_disk(
        const server_prompt & prompt,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec);

    bool load_disk(
        std::list<server_prompt_disk_state>::iterator it,
        server_prompt & prompt,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
         llama_seq_id   id_slot,
              size_t   lcp,
       std::vector<uint8_t> * state_spec);

    bool erase_disk_state(std::list<server_prompt_disk_state>::iterator it, bool eviction, const char * reason);

    // removes the oldest entry of another identity (its directory with its last one); false when none is left
    bool evict_foreign_entry(const char * reason);

    void disable_disk_saves(const char * reason, const std::string & path);

    void update_disk();

    void log_disk_state() const;
};


// what the scheduler needs to know about a slot
struct server_slot_info {
    int id = -1;
    const server_task * task = nullptr; // only for logging

    bool is_processing = false;
    int64_t t_last_used = -1;

    const server_prompt * prompt = nullptr;
};

struct server_slot_plan {
    // -1 if no slot is available; a busy slot if the task asks for it by id, the caller defers the task
    int id_slot = -1;

    // save the slot state to the prompt cache, then load the best match for the task
    bool update_cache = false;

    // [RDNA-FAST] look for a better cache entry without saving the slot first: the slot keeps most of its prompt
    // but the request diverges well past its newest usable checkpoint (see select)
    bool probe_cache = false;
};

// decides only, the caller applies the plan
struct server_slot_sched {
    float slot_prompt_similarity = 0.0f;

    // [RDNA-FAST] the server's prompt cache, nullptr without one: select() reads its disk-tier settings
    const server_prompt_cache * prompt_cache = nullptr;

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<int> get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const;

    // lora: the lora config of the task
    server_slot_plan select(const std::vector<server_slot_info> & slots, const server_task & task, const std::vector<common_adapter_lora_info> & lora) const;
};

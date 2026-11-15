#include "server-slot-sched.h"

#include "common.h"
#include "llama.h"
#include "log.h"
#include "speculative.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <system_error>

#if defined(_WIN32)
#include <share.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

//
// server_prompt
//

server_prompt_reuse server_prompt::get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const {
    server_prompt_reuse res;
    res.n_past   = n_prefix;
    res.pos_next = tokens.pos_next(n_prefix);

    // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
    // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
    const llama_pos pos_eval = res.pos_next - (has_new_tokens ? 0 : 1);

    // a state or a checkpoint is useful when its pos_min is below this threshold
    const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

    if (pos_min < pos_min_thold) {
        return res;
    }

    // search for a context checkpoint
    const auto it = std::find_if(
        checkpoints.rbegin(),
        checkpoints.rend(),
        [&](const auto & cur) {
            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
            SRV_TRC("checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
            if (cur.pos_max > res.pos_next) {
                return false;
            }
            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
        }
    );

    if (it == checkpoints.rend()) {
        res.n_past   = 0;
        res.pos_next = 0;
        return res;
    }

    const auto & cur = *it;

    res.pos_next = std::min(res.pos_next, std::max(cur.pos_min + 1, cur.pos_max));
    res.n_past   = std::min(tokens.size_up_to_pos(res.pos_next), (size_t) cur.n_tokens);
    res.ckpt     = &cur;

    return res;
}

//
// server_prompt_cache
//

namespace {

namespace fs = std::filesystem;

constexpr const char * SERVER_PROMPT_CACHE_DISK_NAMESPACE = ".llama-prompt-cache-v1";
constexpr const char * SERVER_PROMPT_CACHE_OWNER_MAGIC    = "llama.cpp automatic prompt cache v1";

// checkpoint sidecar: magic, version, n_tokens, pos_min, pos_max, size_tgt, size_dft, size_spec, payloads
constexpr uint32_t SERVER_PROMPT_CACHE_CKPT_MAGIC   = 0x50434b43u; // 'CKCP'
constexpr uint32_t SERVER_PROMPT_CACHE_CKPT_VERSION = 1;

static std::string server_prompt_cache_disk_path_utf8(const fs::path & path) {
#if defined(__cpp_lib_char8_t)
    const std::u8string value = path.u8string();
    return std::string(reinterpret_cast<const char *>(value.data()), value.size());
#else
    return path.u8string();
#endif
}

static bool server_prompt_cache_disk_owned(const fs::path & path) {
    std::ifstream owner(path / ".owner");
    std::string magic;
    return owner.good() && std::getline(owner, magic) && magic == SERVER_PROMPT_CACHE_OWNER_MAGIC;
}

static bool server_prompt_cache_disk_remove_file(const std::string & path) {
    if (path.empty()) {
        return true;
    }

    std::error_code ec;
    fs::remove(fs::u8path(path), ec);
    if (ec) {
        SRV_WRN("prompt cache disk cleanup failed: path=%s error=%s\n", path.c_str(), ec.message().c_str());
        return false;
    }

    // A missing file already satisfies the desired postcondition. This also
    // lets a later retry finish a pair after an earlier partial removal.
    return true;
}

static bool server_prompt_cache_disk_size_exact(
        const std::string & path,
                    size_t expected,
                    size_t * actual_out = nullptr) {
    if (path.empty()) {
        if (actual_out != nullptr) {
            *actual_out = 0;
        }
        return expected == 0;
    }

    std::error_code ec;
    const uintmax_t actual = fs::file_size(fs::u8path(path), ec);
    if (ec || actual > std::numeric_limits<size_t>::max()) {
        if (actual_out != nullptr) {
            *actual_out = 0;
        }
        return false;
    }

    if (actual_out != nullptr) {
        *actual_out = (size_t) actual;
    }
    return (size_t) actual == expected;
}

// llama_state_seq_save_file() closes the file before returning. Reopen it to
// force dirty pages to stable storage and immediately mark the cold state as
// reclaimable. This avoids replacing anonymous cache pressure with several GiB
// of sticky buffered page cache on UMA systems.
static bool server_prompt_cache_disk_flush_and_drop(const std::string & path, bool durable) {
#if !defined(_WIN32)
    const int fd = open(path.c_str(), (durable ? O_RDWR : O_RDONLY) | O_CLOEXEC);
    if (fd < 0) {
        SRV_ERR("prompt cache disk open failed: path=%s error=%s\n", path.c_str(), std::strerror(errno));
        return false;
    }

    bool ok = true;
    if (durable) {
#if defined(__APPLE__)
        const int sync_result = fsync(fd);
#else
        const int sync_result = fdatasync(fd);
#endif
        if (sync_result != 0) {
            SRV_ERR("prompt cache disk file sync failed: path=%s error=%s\n", path.c_str(), std::strerror(errno));
            ok = false;
        }
    }

#if defined(POSIX_FADV_DONTNEED)
    const int err = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (err != 0) {
        SRV_WRN("prompt cache disk fadvise failed: path=%s error=%s\n", path.c_str(), std::strerror(err));
    }
#endif

    close(fd);
    return ok;
#else
    GGML_UNUSED(path);
    GGML_UNUSED(durable);
    return true;
#endif
}

static bool server_prompt_cache_disk_sync_dir(const std::string & path) {
#if !defined(_WIN32)
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        SRV_ERR("prompt cache disk directory open failed: path=%s error=%s\n", path.c_str(), std::strerror(errno));
        return false;
    }

    const bool ok = fsync(fd) == 0;
    if (!ok) {
        SRV_ERR("prompt cache disk directory fsync failed: path=%s error=%s\n", path.c_str(), std::strerror(errno));
    }
    close(fd);
    return ok;
#else
    GGML_UNUSED(path);
    return true;
#endif
}

static bool server_prompt_cache_tokens_equal(const server_tokens & expected, const llama_tokens & actual) {
    return expected.get_tokens() == actual;
}

// Number of tokens of `tokens_new` that a cached state with `cached` tokens and common
// prefix `lcp` actually saves once restored into a slot:
//  - the whole prefix when the entry ends exactly there, or the target can drop the tail
//    in place (dense KV);
//  - otherwise the newest checkpoint at or before the divergence (`ckpt_n`, 0 = none):
//    recurrent/hybrid state cannot be truncated, so the slot rolls back to that
//    checkpoint and re-prefills from it;
//  - 0 when neither applies (the slot would have to start cold).
static size_t server_prompt_cache_eff_tokens(size_t cached, size_t lcp, bool partial_rm, int64_t ckpt_n) {
    if (lcp == cached || partial_rm) {
        return lcp;
    }
    if (ckpt_n > 0 && (size_t) ckpt_n <= lcp) {
        return (size_t) ckpt_n;
    }
    return 0;
}

// <stem>-ckpt<k>.bin, k = 0 is the oldest persisted checkpoint of the entry
constexpr int SERVER_PROMPT_CACHE_CKPT_MAX_FILES = 64;

static std::string server_prompt_cache_ckpt_suffix(int k) {
    return "-ckpt" + std::to_string(k) + ".bin";
}

// header: u32 magic, u32 version, u64 n_tokens, i32 pos_min, i32 pos_max, u64 size_tgt, u64 size_dft, u64 size_spec
static bool server_prompt_cache_ckpt_write(
        const fs::path & path_tmp, const fs::path & path, const common_prompt_checkpoint & ck, size_t & n_bytes) {
    const std::string path_tmp_utf8 = server_prompt_cache_disk_path_utf8(path_tmp);

    bool ok = false;
    n_bytes = 0;
    {
        const uint32_t magic   = SERVER_PROMPT_CACHE_CKPT_MAGIC;
        const uint32_t version = SERVER_PROMPT_CACHE_CKPT_VERSION;
        const uint64_t h_n_tokens = (uint64_t) ck.n_tokens;
        const int32_t  h_pos_min  = ck.pos_min;
        const int32_t  h_pos_max  = ck.pos_max;
        const uint64_t s_tgt = ck.data_tgt.size();
        const uint64_t s_dft = ck.data_dft.size();
        const uint64_t s_spc = ck.data_spec.size();
        FILE * f = fopen(path_tmp_utf8.c_str(), "wb");
        if (f) {
            ok = fwrite(&magic, sizeof(magic), 1, f) == 1 &&
                 fwrite(&version, sizeof(version), 1, f) == 1 &&
                 fwrite(&h_n_tokens, sizeof(h_n_tokens), 1, f) == 1 &&
                 fwrite(&h_pos_min, sizeof(h_pos_min), 1, f) == 1 &&
                 fwrite(&h_pos_max, sizeof(h_pos_max), 1, f) == 1 &&
                 fwrite(&s_tgt, sizeof(s_tgt), 1, f) == 1 &&
                 fwrite(&s_dft, sizeof(s_dft), 1, f) == 1 &&
                 fwrite(&s_spc, sizeof(s_spc), 1, f) == 1 &&
                 (s_tgt == 0 || fwrite(ck.data_tgt.data(), 1, s_tgt, f) == s_tgt) &&
                 (s_dft == 0 || fwrite(ck.data_dft.data(), 1, s_dft, f) == s_dft) &&
                 (s_spc == 0 || fwrite(ck.data_spec.data(), 1, s_spc, f) == s_spc);
            n_bytes = sizeof(magic) + sizeof(version) + sizeof(h_n_tokens) + sizeof(h_pos_min) + sizeof(h_pos_max) +
                      sizeof(s_tgt) + sizeof(s_dft) + sizeof(s_spc) + (size_t) (s_tgt + s_dft + s_spc);
            ok = (fclose(f) == 0) && ok;
        }
    }
    if (ok) {
        std::error_code ec;
        fs::rename(path_tmp, path, ec);
        ok = !ec;
    }
    if (!ok) {
        server_prompt_cache_disk_remove_file(path_tmp_utf8);
        n_bytes = 0;
    }
    return ok;
}

// reads only the header; false when the file is not a checkpoint of this format
static bool server_prompt_cache_ckpt_read_meta(const std::string & path, server_prompt_checkpoint_meta & meta) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    uint32_t magic = 0, version = 0; uint64_t n_tokens = 0; int32_t pos_min = 0, pos_max = 0;
    const bool ok = fread(&magic, sizeof(magic), 1, f) == 1 && fread(&version, sizeof(version), 1, f) == 1 &&
                    fread(&n_tokens, sizeof(n_tokens), 1, f) == 1 &&
                    fread(&pos_min, sizeof(pos_min), 1, f) == 1 && fread(&pos_max, sizeof(pos_max), 1, f) == 1 &&
                    magic == SERVER_PROMPT_CACHE_CKPT_MAGIC && version == SERVER_PROMPT_CACHE_CKPT_VERSION && n_tokens > 0;
    fclose(f);
    if (ok) {
        meta = {(int64_t) n_tokens, pos_min, pos_max};
    }
    return ok;
}

static bool server_prompt_cache_ckpt_read(
        const std::string & path, int64_t n_tokens_expected, common_prompt_checkpoint & ck, size_t & n_bytes) {
    n_bytes = 0;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    uint32_t magic = 0, version = 0; uint64_t h_n_tokens = 0, s_tgt = 0, s_dft = 0, s_spc = 0; int32_t h_pos_min = 0, h_pos_max = 0;
    bool ok = fread(&magic, sizeof(magic), 1, f) == 1 && magic == SERVER_PROMPT_CACHE_CKPT_MAGIC &&
              fread(&version, sizeof(version), 1, f) == 1 && version == SERVER_PROMPT_CACHE_CKPT_VERSION &&
              fread(&h_n_tokens, sizeof(h_n_tokens), 1, f) == 1 &&
              fread(&h_pos_min, sizeof(h_pos_min), 1, f) == 1 &&
              fread(&h_pos_max, sizeof(h_pos_max), 1, f) == 1 &&
              fread(&s_tgt, sizeof(s_tgt), 1, f) == 1 &&
              fread(&s_dft, sizeof(s_dft), 1, f) == 1 &&
              fread(&s_spc, sizeof(s_spc), 1, f) == 1 &&
              h_n_tokens == (uint64_t) n_tokens_expected;
    if (ok) {
        ck.n_tokens = (int64_t) h_n_tokens; ck.pos_min = h_pos_min; ck.pos_max = h_pos_max;
        ck.id_task = -1;
        ck.data_tgt.resize((size_t) s_tgt); ck.data_dft.resize((size_t) s_dft); ck.data_spec.resize((size_t) s_spc);
        ok = (s_tgt == 0 || fread(ck.data_tgt.data(), 1, (size_t) s_tgt, f) == s_tgt) &&
             (s_dft == 0 || fread(ck.data_dft.data(), 1, (size_t) s_dft, f) == s_dft) &&
             (s_spc == 0 || fread(ck.data_spec.data(), 1, (size_t) s_spc, f) == s_spc);
        n_bytes = (size_t) ftell(f);
    }
    fclose(f);
    return ok;
}

// newest checkpoint boundary at or before `lcp` (0 = none)
static int64_t server_prompt_cache_ckpt_at_or_below(const std::list<common_prompt_checkpoint> & checkpoints, size_t lcp) {
    int64_t res = 0;
    for (const auto & ck : checkpoints) {
        if (ck.n_tokens > res && (size_t) ck.n_tokens <= lcp) {
            res = ck.n_tokens;
        }
    }
    return res;
}

// Mark an entry's files as just used. External retention policies prune the cache directory
// by file age; without this a shared system-prompt entry that is restored every hour would
// still be deleted N hours after it was WRITTEN. With it, age means "time since last use".
static void server_prompt_cache_disk_touch_files(const server_prompt_disk_state & st) {
    const auto now = fs::file_time_type::clock::now();
    const auto touch = [&](const std::string & path) {
        if (!path.empty()) {
            std::error_code ec;
            fs::last_write_time(fs::u8path(path), now, ec);
        }
    };
    touch(st.path_main);
    touch(st.path_drft);
    touch(st.path_spec);
    for (const auto & ck : st.ckpts) {
        touch(ck.path);
    }
}

} // namespace

server_prompt_cache::server_prompt_cache(
        int32_t limit_size_mib,
         size_t limit_tokens,
    const std::string & disk_base_path,
        int32_t disk_limit_size_mib,
    const std::string & disk_identity_) {
    disk_identity      = disk_identity_;
    ram_enabled        = limit_size_mib != 0;
    limit_size         = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
    this->limit_tokens = limit_tokens;

    if (disk_base_path.empty() || disk_limit_size_mib <= 0) {
        return;
    }

    disk_limit_size = 1024ull*1024ull*disk_limit_size_mib;

    std::error_code ec;
    fs::path base = fs::absolute(fs::u8path(disk_base_path), ec);
    if (ec) {
        throw std::runtime_error("unable to resolve prompt cache disk path '" + disk_base_path + "': " + ec.message());
    }

    fs::create_directories(base, ec);
    if (ec || !fs::is_directory(base)) {
        throw std::runtime_error("unable to create prompt cache disk path '" + server_prompt_cache_disk_path_utf8(base) + "': " + ec.message());
    }

    const fs::path cache_root = base / SERVER_PROMPT_CACHE_DISK_NAMESPACE;
    fs::create_directories(cache_root, ec);
    if (ec || !fs::is_directory(cache_root)) {
        throw std::runtime_error("unable to create prompt cache namespace '" + server_prompt_cache_disk_path_utf8(cache_root) + "': " + ec.message());
    }
    fs::permissions(cache_root, fs::perms::owner_all, fs::perm_options::replace, ec);
    if (ec) {
        throw std::runtime_error("unable to secure prompt cache namespace '" + server_prompt_cache_disk_path_utf8(cache_root) + "': " + ec.message());
    }

    // Stale run directories from earlier runs are handled AFTER this run's own
    // directory exists (see below): their entries are adopted into it when the
    // identity matches, otherwise they are kept for their model ([TAG_CACHE_ALL_MODELS]).

    const auto stamp = (uint64_t) std::chrono::high_resolution_clock::now().time_since_epoch().count();
#if !defined(_WIN32)
    const auto pid = (uint64_t) getpid();
#else
    const uint64_t pid = 0;
#endif

    fs::path owned;
    for (uint32_t suffix = 0; suffix < 1000; ++suffix) {
        owned = cache_root / ("run-" + std::to_string(pid) + "-" + std::to_string(stamp) + "-" + std::to_string(suffix));
        if (fs::create_directory(owned, ec)) {
            break;
        }
        if (ec && ec != std::errc::file_exists) {
            throw std::runtime_error("unable to create owned prompt cache directory '" + server_prompt_cache_disk_path_utf8(owned) + "': " + ec.message());
        }
        ec.clear();
        owned.clear();
    }
    if (owned.empty() || !fs::is_directory(owned)) {
        throw std::runtime_error("unable to allocate a unique prompt cache run directory below '" + server_prompt_cache_disk_path_utf8(cache_root) + "'");
    }

    fs::permissions(owned, fs::perms::owner_all, fs::perm_options::replace, ec);
    if (ec) {
        fs::remove_all(owned);
        throw std::runtime_error("unable to secure owned prompt cache directory '" + server_prompt_cache_disk_path_utf8(owned) + "': " + ec.message());
    }

#if !defined(_WIN32)
    // Publish and hold the lock before publishing .owner. Stale cleanup only
    // considers magic-marked directories, so another startup can never see an
    // owned directory in the window before this process has acquired its lock.
    const fs::path lock_path = owned / ".lock";
    disk_lock_fd = open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (disk_lock_fd < 0 || flock(disk_lock_fd, LOCK_EX | LOCK_NB) != 0) {
        if (disk_lock_fd >= 0) {
            close(disk_lock_fd);
            disk_lock_fd = -1;
        }
        fs::remove_all(owned);
        throw std::runtime_error("unable to lock owned prompt cache directory '" + server_prompt_cache_disk_path_utf8(owned) + "'");
    }
#else
    {
        // Deny all sharing and keep the handle for the life of the run: any other
        // process trying to open this file fails while we are alive, which is how a
        // later startup distinguishes a live run directory from an abandoned one.
        const std::string lock_utf8 = server_prompt_cache_disk_path_utf8(owned / ".lock");
        disk_lock_file = _fsopen(lock_utf8.c_str(), "wb", _SH_DENYRW);
        if (disk_lock_file == nullptr) {
            fs::remove_all(owned);
            throw std::runtime_error("unable to create prompt cache lock file in '" + server_prompt_cache_disk_path_utf8(owned) + "'");
        }
    }
#endif

    {
        std::ofstream owner(owned / ".owner", std::ios::out | std::ios::trunc);
        owner << SERVER_PROMPT_CACHE_OWNER_MAGIC << '\n'
              << "pid=" << pid << '\n'
              << "created=" << stamp << '\n';
        owner.flush();
        if (!owner.good()) {
#if !defined(_WIN32)
            flock(disk_lock_fd, LOCK_UN);
            close(disk_lock_fd);
            disk_lock_fd = -1;
#endif
            fs::remove_all(owned);
            throw std::runtime_error("unable to write prompt cache ownership manifest in '" + server_prompt_cache_disk_path_utf8(owned) + "'");
        }
    }
    fs::permissions(owned / ".owner", fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
    if (ec) {
#if !defined(_WIN32)
        flock(disk_lock_fd, LOCK_UN);
        close(disk_lock_fd);
        disk_lock_fd = -1;
#endif
        fs::remove_all(owned);
        throw std::runtime_error("unable to secure prompt cache ownership manifest in '" + server_prompt_cache_disk_path_utf8(owned) + "': " + ec.message());
    }

    this->disk_base_path  = server_prompt_cache_disk_path_utf8(base);
    this->disk_owned_path = server_prompt_cache_disk_path_utf8(owned);

    {
        std::ofstream ident(owned / "identity.txt", std::ios::out | std::ios::trunc);
        ident << disk_identity << '\n';
    }

    SRV_INF("prompt cache disk enabled: path=%s owned_path=%s limit_mib=%d\n",
            this->disk_base_path.c_str(), this->disk_owned_path.c_str(), disk_limit_size_mib);

    // ---- adopt entries from abandoned run directories (survive server restarts) ----
    // A run directory is abandoned when its lock is no longer held (POSIX: flock
    // succeeds; Windows: the deny-all .lock can be opened). Entries whose identity
    // matches ours are MOVED into this run's directory and registered; the entries in
    // stale directories of other identities are kept for their model and counted against
    // the limit ([TAG_CACHE_ALL_MODELS]): a prod that switches between models keeps every
    // model's cache, and room is made from their oldest entries first. Every adopted entry is still fully validated by
    // load_disk() on first use (sizes, tokens, and llama's own state-file checks), so
    // a damaged file can at worst cost one rejected restore.
    struct adopted_t {
        server_prompt_disk_state st;
        fs::file_time_type       mtime;
    };
    std::vector<adopted_t> adopted;
    size_t n_stale_removed = 0;
    size_t n_adopt_bytes = 0;
    struct foreign_t {
        foreign_entry      e;
        fs::file_time_type mtime = fs::file_time_type::min(); // its newest file: when its model last used it
    };
    std::vector<foreign_t> foreign;
    size_t n_foreign_dirs = 0;

    std::error_code sc;
    for (const auto & entry : fs::directory_iterator(cache_root, sc)) {
        if (sc) {
            break;
        }
        if (entry.path() == owned) {
            continue;
        }
        const auto name = server_prompt_cache_disk_path_utf8(entry.path().filename());
        const bool is_run_dir      = name.rfind("run-", 0) == 0;
        const bool is_deleting_dir = name.rfind(".deleting-run-", 0) == 0;
        if (!entry.is_directory() || (!is_run_dir && !is_deleting_dir) || !server_prompt_cache_disk_owned(entry.path())) {
            continue;
        }

        bool stale = false;
#if !defined(_WIN32)
        {
            const fs::path lock_path = entry.path() / ".lock";
            const int fd = open(lock_path.c_str(), O_RDWR | O_CLOEXEC);
            if (fd < 0) {
                stale = true; // no lock file at all: nothing can be holding it
            } else {
                stale = flock(fd, LOCK_EX | LOCK_NB) == 0;
                if (stale) {
                    flock(fd, LOCK_UN);
                }
                close(fd);
            }
        }
#else
        {
            const std::string lock_utf8 = server_prompt_cache_disk_path_utf8(entry.path() / ".lock");
            FILE * probe = _fsopen(lock_utf8.c_str(), "rb", _SH_DENYNO);
            if (probe != nullptr) {
                fclose(probe);
                stale = true;             // opened: no live owner holds it deny-all
            } else {
                stale = (errno != EACCES); // EACCES = a live process holds it; anything else (e.g. missing) = stale
            }
        }
#endif
        if (!stale) {
            continue;
        }

        const auto stale_path = server_prompt_cache_disk_path_utf8(entry.path());

        std::string their_identity;
        {
            std::ifstream ident(entry.path() / "identity.txt");
            std::getline(ident, their_identity);
        }
        const bool same_identity = !disk_identity.empty() && their_identity == disk_identity;

        if (is_deleting_dir) {
            std::error_code rm_ec;
            const auto removed = fs::remove_all(entry.path(), rm_ec);
            if (!rm_ec) {
                n_stale_removed++;
                SRV_INF("prompt cache disk stale cleanup: path=%s files=%zu reason=half-deleted\n",
                        stale_path.c_str(), (size_t) removed);
            }
            continue;
        }
        if (!same_identity) {
            // another model's (or settings') entries: kept one by one, and counted against the limit. An entry is all
            // the files of one state-<id>
            std::vector<std::pair<std::string, foreign_t>> by_stem;
            std::error_code fec;
            for (const auto & f : fs::directory_iterator(entry.path(), fec)) {
                if (fec) {
                    break;
                }
                std::error_code zec;
                if (!f.is_regular_file(zec)) {
                    continue;
                }
                const std::string fname = server_prompt_cache_disk_path_utf8(f.path().filename());
                const size_t dash = fname.rfind("state-", 0) == 0 ? fname.find('-', 6) : std::string::npos;
                if (dash == std::string::npos) {
                    continue;
                }
                const std::string stem = fname.substr(0, dash);
                auto it = std::find_if(by_stem.begin(), by_stem.end(), [&](const auto & p) { return p.first == stem; });
                if (it == by_stem.end()) {
                    by_stem.emplace_back(stem, foreign_t{});
                    it = by_stem.end() - 1;
                    it->second.e.dir = stale_path;
                }
                foreign_t & fe = it->second;
                fe.e.files.push_back(server_prompt_cache_disk_path_utf8(f.path()));
                const auto sz = fs::file_size(f.path(), zec);
                fe.e.bytes += zec ? 0 : (size_t) sz;
                const auto mt = fs::last_write_time(f.path(), zec);
                if (!zec && mt > fe.mtime) {
                    fe.mtime = mt;
                }
            }
            if (by_stem.empty()) {
                std::error_code rm_ec;
                const auto removed = fs::remove_all(entry.path(), rm_ec);
                if (!rm_ec) {
                    n_stale_removed++;
                    SRV_INF("prompt cache disk stale cleanup: path=%s files=%zu reason=no-entries\n",
                            stale_path.c_str(), (size_t) removed);
                }
                continue;
            }
            n_foreign_dirs++;
            for (auto & p : by_stem) {
                foreign.push_back(std::move(p.second));
            }
            continue;
        }

        // adopt: every state-<id>-target.bin with its siblings
        size_t n_here = 0;
        std::error_code dc;
        for (const auto & f : fs::directory_iterator(entry.path(), dc)) {
            if (dc) {
                break;
            }
            const std::string fname = server_prompt_cache_disk_path_utf8(f.path().filename());
            if (fname.rfind("state-", 0) != 0 || fname.size() < 17 || fname.compare(fname.size() - 11, 11, "-target.bin") != 0) {
                continue;
            }
            const std::string old_stem = fname.substr(0, fname.size() - 11); // "state-<id>"

            // header: u32 magic, u32 version, u32 n_tokens, tokens[]
            llama_tokens toks;
            {
                FILE * fh = fopen(server_prompt_cache_disk_path_utf8(f.path()).c_str(), "rb");
                if (!fh) {
                    continue;
                }
                uint32_t magic = 0, version = 0, n_tok = 0;
                const bool hdr_ok = fread(&magic, 4, 1, fh) == 1 && fread(&version, 4, 1, fh) == 1 && fread(&n_tok, 4, 1, fh) == 1 &&
                                    magic == LLAMA_STATE_SEQ_MAGIC && version == LLAMA_STATE_SEQ_VERSION &&
                                    n_tok > 0 && n_tok <= (uint32_t) limit_tokens;
                if (hdr_ok) {
                    toks.resize(n_tok);
                    if (fread(toks.data(), sizeof(llama_token), n_tok, fh) != n_tok) {
                        toks.clear();
                    }
                }
                fclose(fh);
                if (toks.empty()) {
                    SRV_WRN("prompt cache disk adopt: skipping unreadable entry %s\n", server_prompt_cache_disk_path_utf8(f.path()).c_str());
                    continue;
                }
            }

            const fs::path p_main = f.path();
            const fs::path p_drft = entry.path() / (old_stem + "-draft.bin");
            const fs::path p_spec = entry.path() / (old_stem + "-spec.bin");
            const bool has_drft = fs::exists(p_drft);
            const bool has_spec = fs::exists(p_spec);

            server_prompt_disk_state st;
            st.tokens = server_tokens(toks, false);
            st.id     = disk_next_id++;
            st.usable = true;

            const std::string new_stem = "state-" + std::to_string(st.id);
            auto move_in = [&](const fs::path & from, const char * suffix, std::string & path_out, size_t & size_out) -> bool {
                std::error_code mec;
                const fs::path to = owned / (new_stem + suffix);
                fs::rename(from, to, mec);
                if (mec) {
                    return false;
                }
                path_out = server_prompt_cache_disk_path_utf8(to);
                std::error_code zec;
                const auto sz = fs::file_size(to, zec);
                size_out = zec ? 0 : (size_t) sz;
                return true;
            };

            size_t dummy = 0;
            if (!move_in(p_main, "-target.bin", st.path_main, st.size_main)) {
                continue;
            }
            if (has_drft && !move_in(p_drft, "-draft.bin", st.path_drft, st.size_drft)) {
                server_prompt_cache_disk_remove_file(st.path_main);
                continue;
            }
            // checkpoints: <stem>-ckpt<k>.bin, plus the single <stem>-ckpt.bin written by older builds
            {
                std::vector<fs::path> p_ckpts;
                p_ckpts.push_back(entry.path() / (old_stem + "-ckpt.bin"));
                for (int k = 0; k < SERVER_PROMPT_CACHE_CKPT_MAX_FILES; ++k) {
                    p_ckpts.push_back(entry.path() / (old_stem + server_prompt_cache_ckpt_suffix(k)));
                }
                for (const auto & p_ckpt : p_ckpts) {
                    std::error_code xec;
                    if (!fs::exists(p_ckpt, xec) || xec) {
                        continue;
                    }
                    server_prompt_disk_ckpt dck;
                    const std::string suffix = server_prompt_cache_ckpt_suffix((int) st.ckpts.size());
                    if (!move_in(p_ckpt, suffix.c_str(), dck.path, dck.size)) {
                        continue;
                    }
                    if (dck.size > 0 && server_prompt_cache_ckpt_read_meta(dck.path, dck.meta) &&
                        (size_t) dck.meta.n_tokens <= toks.size()) {
                        st.ckpts.push_back(std::move(dck));
                    } else {
                        server_prompt_cache_disk_remove_file(dck.path);
                    }
                }
                std::sort(st.ckpts.begin(), st.ckpts.end(), [](const server_prompt_disk_ckpt & a, const server_prompt_disk_ckpt & b) {
                    return a.meta.n_tokens < b.meta.n_tokens;
                });
            }
            if (has_spec && move_in(p_spec, "-spec.bin", st.path_spec, dummy)) {
                FILE * fsp = fopen(st.path_spec.c_str(), "rb");
                if (fsp) {
                    st.spec.resize(dummy);
                    if (dummy > 0 && fread(st.spec.data(), 1, dummy, fsp) != dummy) {
                        st.spec.clear();
                    }
                    fclose(fsp);
                }
            }

            std::error_code tec;
            const auto mt = fs::last_write_time(fs::u8path(st.path_main), tec);
            n_adopt_bytes += st.size();
            adopted.push_back({std::move(st), tec ? fs::file_time_type::min() : mt});
            n_here++;
        }

        // drop the emptied directory (ignore failures: leftovers are harmless)
        std::error_code rm_ec;
        fs::remove_all(entry.path(), rm_ec);
        SRV_INF("prompt cache disk adopt: path=%s entries=%zu\n", stale_path.c_str(), n_here);
    }

    std::sort(foreign.begin(), foreign.end(), [](const foreign_t & a, const foreign_t & b) { return a.mtime < b.mtime; });
    for (auto & f : foreign) {
        disk_foreign_bytes += f.e.bytes;
        disk_foreign.push_back(std::move(f.e));
    }
    if (!adopted.empty()) {
        std::sort(adopted.begin(), adopted.end(), [](const adopted_t & a, const adopted_t & b) { return a.mtime < b.mtime; });
        for (auto & a : adopted) {
            disk_size_total += a.st.size();
            disk_states.push_back(std::move(a.st));
        }
    }
    // respect the size limit, which covers the other models' entries too: those go first, then this model's oldest
    while (disk_limit_size > 0 && disk_size_total + disk_foreign_bytes > disk_limit_size) {
        if (evict_foreign_entry("adopt-limit")) {
            continue;
        }
        if (disk_states.empty() || !erase_disk_state(disk_states.begin(), true, "adopt-limit")) {
            break;
        }
    }
    SRV_INF("prompt cache disk adopted %zu entries (%zu bytes) from previous runs; kept %zu entries of other models in %zu dirs (%zu bytes); stale dirs removed: %zu\n",
            adopted.size(), n_adopt_bytes, disk_foreign.size(), n_foreign_dirs, disk_foreign_bytes, n_stale_removed);
}

bool server_prompt_cache::evict_foreign_entry(const char * reason) {
    if (disk_foreign.empty()) {
        return false;
    }
    const foreign_entry e = disk_foreign.front();
    disk_foreign.erase(disk_foreign.begin());
    disk_foreign_bytes -= std::min(disk_foreign_bytes, e.bytes);
    bool ok = true;
    for (const auto & f : e.files) {
        ok = server_prompt_cache_disk_remove_file(f) && ok;
    }
    // the directory goes with its last entry
    const bool last = std::none_of(disk_foreign.begin(), disk_foreign.end(), [&](const foreign_entry & o) { return o.dir == e.dir; });
    if (last) {
        std::error_code ec;
        fs::remove_all(fs::u8path(e.dir), ec);
    }
    SRV_INF("prompt cache disk evicted another model's entry: reason=%s dir=%s files=%zu bytes=%zu%s%s\n",
            reason, e.dir.c_str(), e.files.size(), e.bytes, last ? " (its last: directory removed)" : "",
            ok ? "" : " (not all files removed)");
    return true;
}

server_prompt_cache::~server_prompt_cache() {
    if (disk_owned_path.empty()) {
        return;
    }

    // The entries are deliberately RETAINED: the next run with the same identity
    // adopts them (see the constructor), so the cache survives restarts. Only the
    // lock is released. A run of another identity keeps them too, within the limit
    // ([TAG_CACHE_ALL_MODELS]).
    SRV_INF("prompt cache disk retained for the next run: path=%s entries=%zu bytes=%zu saves=%" PRIu64 " loads=%" PRIu64 " evictions=%" PRIu64 "\n",
            disk_owned_path.c_str(), disk_states.size(), disk_size_total, disk_saves, disk_loads, disk_evictions);

#if !defined(_WIN32)
    if (disk_lock_fd >= 0) {
        flock(disk_lock_fd, LOCK_UN);
        close(disk_lock_fd);
        disk_lock_fd = -1;
    }
#else
    if (disk_lock_file != nullptr) {
        fclose(disk_lock_file);
        disk_lock_file = nullptr;
    }
#endif
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

size_t server_prompt_cache::disk_size() const {
    return disk_size_total;
}

size_t server_prompt_cache::disk_n_tokens() const {
    size_t res = 0;
    for (const auto & state : disk_states) {
        res += state.n_tokens();
    }
    return res;
}

void server_prompt_cache::disable_disk_saves(const char * reason, const std::string & path) {
    disk_save_failures++;
    if (disk_save_disabled) {
        return;
    }

    disk_save_disabled = true;
    SRV_ERR("prompt cache disk writes disabled: reason=%s failures=%" PRIu64 " entries=%zu accounted_bytes=%zu path=%s cache_path=%s\n",
            reason, disk_save_failures, disk_states.size(), disk_size_total,
            path.empty() ? "-" : path.c_str(), disk_owned_path.c_str());
}

// [TAG_REQUEST_HISTORY] hands the disk counters to on_disk_stats when the public method that holds it returns
struct server_prompt_disk_publish {
    const server_prompt_cache & cache;
    ~server_prompt_disk_publish() {
        if (cache.on_disk_stats && cache.disk_enabled()) {
            cache.on_disk_stats(cache.disk_stats());
        }
    }
};

bool server_prompt_cache::save(
        const server_prompt & prompt,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec) {
    const server_prompt_disk_publish publish { *this }; // [TAG_REQUEST_HISTORY]

    if (prompt.tokens.size() == 0) {
        return false;
    }

    bool saved = false;

    if (disk_enabled()) {
        saved = save_disk(prompt, ctx_tgt, ctx_dft, id_slot, state_spec) || saved;
    }

    if (!ram_enabled) {
        return saved;
    }

    const size_t state_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
    const size_t state_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

    SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
            (int) prompt.tokens.size(), (state_size_tgt + state_size_dft) / (1024.0 * 1024.0), state_size_dft / (1024.0 * 1024.0));

    auto * cur = alloc(prompt, state_size_tgt, state_size_dft, state_spec);
    if (cur == nullptr) {
        return saved;
    }

    const size_t n_tgt = llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), state_size_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (n_tgt != state_size_tgt) {
        SRV_ERR("failed to save RAM prompt cache target state: expected=%zu saved=%zu\n", state_size_tgt, n_tgt);
        states.pop_back();
        return saved;
    }

    if (ctx_dft) {
        const size_t n_dft = llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), state_size_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (n_dft != state_size_dft) {
            SRV_ERR("failed to save RAM prompt cache draft state: expected=%zu saved=%zu\n", state_size_dft, n_dft);
            states.pop_back();
            return saved;
        }
    }

    return true;
}

bool server_prompt_cache::save_disk(
        const server_prompt & prompt,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec) {
    if (disk_owned_path.empty() || disk_limit_size == 0 || prompt.tokens.empty()) {
        return false;
    }

    if (prompt.tokens.has_media()) {
        SRV_WRN("prompt cache disk skip: reason=multimodal tokens=%zu path=%s\n",
                prompt.tokens.size(), disk_owned_path.c_str());
        return false;
    }

    if (prompt.tokens.size() < disk_min_tokens) {
        SRV_INF("prompt cache disk skip: reason=below-min-tokens tokens=%zu min_tokens=%zu\n",
                prompt.tokens.size(), disk_min_tokens);
        return false;
    }

    // If a usable entry already holds this prompt, retain it without rewriting the SSD.
    // An entry with the same tokens always qualifies; a longer entry only when the
    // target can truncate its tail in place (dense KV).
    for (auto it = disk_states.begin(); it != disk_states.end();) {
        if (!it->usable) {
            ++it;
            continue;
        }

        const int lcp = it->tokens.get_common_prefix(prompt.tokens);
        const bool exact_tokens = lcp == (int) prompt.tokens.size() && it->tokens.size() == prompt.tokens.size();
        const bool can_touch = exact_tokens || (partial_seq_rm && lcp == (int) prompt.tokens.size());
        if (!can_touch) {
            ++it;
            continue;
        }

        const bool pair_shape_ok = !it->path_main.empty() && it->size_main > 0 &&
            ((ctx_dft != nullptr) == (!it->path_drft.empty() && it->size_drft > 0));
        size_t actual_main = 0;
        size_t actual_drft = 0;
        const bool files_ok = pair_shape_ok &&
            server_prompt_cache_disk_size_exact(it->path_main, it->size_main, &actual_main) &&
            server_prompt_cache_disk_size_exact(it->path_drft, it->size_drft, &actual_drft);
        if (!files_ok) {
            SRV_WRN("prompt cache disk touch rejected: entry=%" PRIu64 " reason=unusable-pair target_bytes=%zu target_actual=%zu draft_bytes=%zu draft_actual=%zu path=%s\n",
                    it->id, it->size_main, actual_main, it->size_drft, actual_drft, disk_owned_path.c_str());
            auto bad = it++;
            bad->usable = false;
            if (!erase_disk_state(bad, false, "touch-unusable")) {
                disable_disk_saves("touch-unusable-removal", disk_owned_path);
            }
            continue;
        }

        {
            const auto id = it->id;
            server_prompt_cache_disk_touch_files(*it);
            disk_states.splice(disk_states.end(), disk_states, it);
            SRV_INF("prompt cache disk touch: entry=%" PRIu64 " lcp=%d tokens=%zu exact=%s path=%s\n",
                    id, lcp, prompt.tokens.size(), exact_tokens ? "true" : "false", disk_owned_path.c_str());
            return true;
        }
    }

    if (disk_save_disabled) {
        SRV_DBG("prompt cache disk save skip: reason=circuit-open tokens=%zu path=%s\n",
                prompt.tokens.size(), disk_owned_path.c_str());
        return false;
    }

    const auto & tokens = prompt.tokens.get_tokens();
    const size_t token_bytes = tokens.size()*sizeof(llama_token);
    const size_t file_overhead = 3*sizeof(uint32_t) + token_bytes;
    const size_t state_size_main = llama_state_seq_get_size_ext(ctx_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
    const size_t state_size_drft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
    const size_t predicted_main = state_size_main + file_overhead;
    const size_t predicted_drft = ctx_dft ? state_size_drft + file_overhead : 0;
    const size_t predicted_total = predicted_main + predicted_drft;

    if (predicted_total > disk_limit_size) {
        SRV_WRN("prompt cache disk skip: reason=oversize target_bytes=%zu draft_bytes=%zu total_bytes=%zu limit_bytes=%zu tokens=%zu path=%s\n",
                predicted_main, predicted_drft, predicted_total, disk_limit_size, tokens.size(), disk_owned_path.c_str());
        return false;
    }

    // [TAG_CACHE_ALL_MODELS] never fill the drive: with several models' entries kept, the limit can be more than the
    // drive has left (E: 66 GB free under a 128 GB limit, 2026-11-08). Room for the entry and 8 GiB more is made the
    // way the limit makes it (the other models' oldest entries first), else this entry is not written
    {
        const size_t keep_free = (size_t) 8 << 30;
        const auto available = [&]() -> size_t {
            std::error_code sec;
            const auto sp = fs::space(fs::u8path(disk_owned_path), sec);
            return sec ? SIZE_MAX : (size_t) sp.available;
        };
        while (available() < predicted_total + keep_free) {
            if (evict_foreign_entry("disk-space")) {
                continue;
            }
            if (disk_states.empty() || !erase_disk_state(disk_states.begin(), true, "disk-space")) {
                SRV_WRN("prompt cache disk skip: reason=disk-space total_bytes=%zu available_bytes=%zu tokens=%zu path=%s\n",
                        predicted_total, available(), tokens.size(), disk_owned_path.c_str());
                return false;
            }
        }
    }

    const uint64_t entry_id = disk_next_id++;
    const fs::path owned = fs::u8path(disk_owned_path);
    const std::string stem = "state-" + std::to_string(entry_id);
    const fs::path path_main_tmp = owned / (stem + "-target.bin.tmp");
    const fs::path path_main     = owned / (stem + "-target.bin");
    const fs::path path_drft_tmp = owned / (stem + "-draft.bin.tmp");
    const fs::path path_drft     = owned / (stem + "-draft.bin");
    const std::string path_main_tmp_utf8 = server_prompt_cache_disk_path_utf8(path_main_tmp);
    const std::string path_main_utf8     = server_prompt_cache_disk_path_utf8(path_main);
    const std::string path_drft_tmp_utf8 = server_prompt_cache_disk_path_utf8(path_drft_tmp);
    const std::string path_drft_utf8     = server_prompt_cache_disk_path_utf8(path_drft);

    const auto cleanup_temps = [&]() -> bool {
        const bool main_ok = server_prompt_cache_disk_remove_file(path_main_tmp_utf8);
        const bool drft_ok = server_prompt_cache_disk_remove_file(path_drft_tmp_utf8);
        return main_ok && drft_ok;
    };
    const auto fail_io = [&](const char * reason, const std::string & path) -> bool {
        const bool cleanup_ok = cleanup_temps();
        disable_disk_saves(reason, path);
        if (!cleanup_ok) {
            disable_disk_saves("temporary-cleanup", disk_owned_path);
        }
        return false;
    };

    const int64_t t_start = ggml_time_us();

    const size_t n_main = llama_state_seq_save_file(
        ctx_tgt, path_main_tmp_utf8.c_str(), id_slot, tokens.data(), tokens.size());
    size_t actual_main = 0;
    if (n_main == 0 ||
        !server_prompt_cache_disk_size_exact(path_main_tmp_utf8, n_main, &actual_main) ||
        !server_prompt_cache_disk_flush_and_drop(path_main_tmp_utf8, true)) {
        SRV_ERR("prompt cache disk save failed: entry=%" PRIu64 " component=target path=%s\n",
                entry_id, path_main_tmp_utf8.c_str());
        return fail_io("target-save", path_main_tmp_utf8);
    }

    size_t n_drft = 0;
    if (ctx_dft) {
        n_drft = llama_state_seq_save_file(
            ctx_dft, path_drft_tmp_utf8.c_str(), id_slot, tokens.data(), tokens.size());
        size_t actual_drft = 0;
        if (n_drft == 0 ||
            !server_prompt_cache_disk_size_exact(path_drft_tmp_utf8, n_drft, &actual_drft) ||
            !server_prompt_cache_disk_flush_and_drop(path_drft_tmp_utf8, true)) {
            SRV_ERR("prompt cache disk save failed: entry=%" PRIu64 " component=draft path=%s\n",
                    entry_id, path_drft_tmp_utf8.c_str());
            return fail_io("draft-save", path_drft_tmp_utf8);
        }
    }

    const size_t actual_total = n_main + n_drft;
    if (actual_total > disk_limit_size) {
        const bool cleanup_ok = cleanup_temps();
        SRV_WRN("prompt cache disk skip: reason=actual-oversize entry=%" PRIu64 " target_bytes=%zu draft_bytes=%zu total_bytes=%zu limit_bytes=%zu path=%s\n",
                entry_id, n_main, n_drft, actual_total, disk_limit_size, disk_owned_path.c_str());
        if (!cleanup_ok) {
            disable_disk_saves("actual-oversize-cleanup", disk_owned_path);
        }
        return false;
    }

    std::error_code ec;
    fs::permissions(path_main_tmp, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
    if (ec) {
        SRV_ERR("prompt cache disk permissions failed: entry=%" PRIu64 " component=target path=%s error=%s\n",
                entry_id, path_main_tmp_utf8.c_str(), ec.message().c_str());
        return fail_io("target-permissions", path_main_tmp_utf8);
    }
    if (ctx_dft) {
        fs::permissions(path_drft_tmp, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);
        if (ec) {
            SRV_ERR("prompt cache disk permissions failed: entry=%" PRIu64 " component=draft path=%s error=%s\n",
                    entry_id, path_drft_tmp_utf8.c_str(), ec.message().c_str());
            return fail_io("draft-permissions", path_drft_tmp_utf8);
        }
    }

    // The complete target/draft temporary pair is durable. Commit it before
    // touching older entries so a rename or directory-sync failure cannot
    // destroy a previously usable cache. This permits one incoming entry of
    // transient staging headroom above the configured payload limit.
    ec.clear();
    fs::rename(path_main_tmp, path_main, ec);
    if (ec) {
        SRV_ERR("prompt cache disk atomic rename failed: entry=%" PRIu64 " component=target path=%s error=%s\n",
                entry_id, path_main_utf8.c_str(), ec.message().c_str());
        return fail_io("target-rename", path_main_utf8);
    }

    if (ctx_dft) {
        ec.clear();
        fs::rename(path_drft_tmp, path_drft, ec);
        if (ec) {
            const bool main_cleanup_ok = server_prompt_cache_disk_remove_file(path_main_utf8);
            const bool temp_cleanup_ok = cleanup_temps();
            SRV_ERR("prompt cache disk atomic rename failed: entry=%" PRIu64 " component=draft path=%s error=%s\n",
                    entry_id, path_drft_utf8.c_str(), ec.message().c_str());
            disable_disk_saves("draft-rename", path_drft_utf8);
            if (!main_cleanup_ok || !temp_cleanup_ok) {
                disable_disk_saves("draft-rename-cleanup", disk_owned_path);
            }
            return false;
        }
    }

    if (!server_prompt_cache_disk_sync_dir(disk_owned_path) ||
        !server_prompt_cache_disk_flush_and_drop(path_main_utf8, false) ||
        (ctx_dft && !server_prompt_cache_disk_flush_and_drop(path_drft_utf8, false))) {
        const bool main_cleanup_ok = server_prompt_cache_disk_remove_file(path_main_utf8);
        const bool drft_cleanup_ok = server_prompt_cache_disk_remove_file(path_drft_utf8);
        disable_disk_saves("commit-sync", disk_owned_path);
        if (!main_cleanup_ok || !drft_cleanup_ok) {
            disable_disk_saves("commit-sync-cleanup", disk_owned_path);
        }
        return false;
    }

    server_prompt_disk_state state;
    state.tokens    = prompt.tokens.clone();
    state.path_main = path_main_utf8;
    state.path_drft = ctx_dft ? path_drft_utf8 : std::string();
    state.size_main = n_main;
    state.size_drft = n_drft;
    state.spec      = state_spec;
    state.id        = entry_id;
    state.usable    = true;

    disk_states.push_back(std::move(state));
    disk_size_total    += actual_total;
    disk_saves++;
    disk_bytes_written += actual_total;

    auto new_entry = std::prev(disk_states.end());

    // Persist the speculative-impl state blob too, so the entry is complete after a restart.
    if (!state_spec.empty()) {
        const fs::path p_spec_tmp = owned / (stem + "-spec.bin.tmp");
        const fs::path p_spec     = owned / (stem + "-spec.bin");
        const std::string p_spec_tmp_utf8 = server_prompt_cache_disk_path_utf8(p_spec_tmp);
        FILE * fsp = fopen(p_spec_tmp_utf8.c_str(), "wb");
        bool ok = false;
        if (fsp) {
            ok = fwrite(state_spec.data(), 1, state_spec.size(), fsp) == state_spec.size();
            ok = (fclose(fsp) == 0) && ok;
        }
        if (ok) {
            std::error_code sec;
            fs::rename(p_spec_tmp, p_spec, sec);
            ok = !sec;
        }
        if (ok) {
            new_entry->path_spec = server_prompt_cache_disk_path_utf8(p_spec);
        } else {
            server_prompt_cache_disk_remove_file(p_spec_tmp_utf8);
            SRV_WRN("prompt cache disk save: entry=%" PRIu64 " spec blob NOT persisted\n", entry_id);
        }
    }

    // Persist the context checkpoints with the entry, newest first up to disk_max_ckpts
    // (best effort: a failure here leaves a perfectly valid exact-boundary entry). The
    // newest one sits where a client that does not echo reasoning_content diverges; the
    // older ones are what a request that rewrote earlier history can still roll back to.
    // One checkpoint is ~150-200 MiB and is written without a durable flush.
    size_t n_ckpt_total = 0;
    if (disk_max_ckpts != 0 && !prompt.checkpoints.empty()) {
        std::vector<const common_prompt_checkpoint *> picked;
        for (auto ck = prompt.checkpoints.rbegin(); ck != prompt.checkpoints.rend(); ++ck) {
            if (disk_max_ckpts > 0 && (int32_t) picked.size() >= disk_max_ckpts) {
                break;
            }
            const bool ck_ok = ck->n_tokens > 0 && ck->n_tokens < (int64_t) tokens.size() &&
                               !ck->data_tgt.empty() && (ctx_dft == nullptr || !ck->data_dft.empty());
            if (!ck_ok) {
                continue;
            }
            // the pinned system-prompt checkpoint is redundant when a shared prefix entry
            // already ends exactly there: that entry is smaller to read and serves every conversation
            if (ck->pinned && has_disk_prefix(prompt.tokens, (size_t) ck->n_tokens)) {
                continue;
            }
            // Each request leaves several checkpoints within a few hundred tokens of its prompt end.
            // A rollback point that close to a newer one saves less prefill than it costs to write and
            // read (~200 MiB each), so keep only checkpoints at least disk_min_gain tokens apart.
            if (!ck->pinned && !picked.empty() &&
                picked.back()->n_tokens - ck->n_tokens < (int64_t) disk_min_gain &&
                picked.back()->n_tokens >= ck->n_tokens) {
                // Of the newest cluster keep the newest member: it sits where the next request of this
                // conversation diverges. Of an older cluster keep the EARLIEST: it still serves a prompt
                // that diverges anywhere after it, the later members miss whatever diverges just before them.
                if (picked.size() > 1 && !picked.back()->pinned) {
                    picked.back() = &*ck;
                }
                continue;
            }
            picked.push_back(&*ck);
        }
        if (picked.size() > (size_t) SERVER_PROMPT_CACHE_CKPT_MAX_FILES) {
            picked.resize(SERVER_PROMPT_CACHE_CKPT_MAX_FILES);
        }
        // oldest first on disk and in the entry
        std::sort(picked.begin(), picked.end(), [](const common_prompt_checkpoint * a, const common_prompt_checkpoint * b) {
            return a->n_tokens < b->n_tokens;
        });

        for (const auto * ck : picked) {
            const std::string suffix = server_prompt_cache_ckpt_suffix((int) new_entry->ckpts.size());
            const fs::path path_ckpt_tmp = owned / (stem + suffix + ".tmp");
            const fs::path path_ckpt     = owned / (stem + suffix);
            size_t n_ckpt = 0;
            if (server_prompt_cache_ckpt_write(path_ckpt_tmp, path_ckpt, *ck, n_ckpt)) {
                server_prompt_disk_ckpt dck;
                dck.path = server_prompt_cache_disk_path_utf8(path_ckpt);
                dck.size = n_ckpt;
                dck.meta = {ck->n_tokens, ck->pos_min, ck->pos_max};
                new_entry->ckpts.push_back(std::move(dck));
                disk_size_total    += n_ckpt;
                disk_bytes_written += n_ckpt;
                n_ckpt_total       += n_ckpt;
                SRV_INF("prompt cache disk save: entry=%" PRIu64 " checkpoint persisted n_tokens=%" PRId64 " pos=[%d,%d] bytes=%zu%s\n",
                        entry_id, ck->n_tokens, ck->pos_min, ck->pos_max, n_ckpt, ck->pinned ? " pinned" : "");
            } else {
                SRV_WRN("prompt cache disk save: entry=%" PRIu64 " checkpoint n_tokens=%" PRId64 " NOT persisted (write failed)\n",
                        entry_id, ck->n_tokens);
            }
        }
    }
    bool reclaim_ok = true;

    // With dense KV a longer entry supersedes shorter prefixes (their state is a
    // truncation of the new one). Recurrent/hybrid entries are exact-boundary states,
    // so shorter boundaries stay independently useful.
    if (partial_seq_rm) {
        for (auto it = disk_states.begin(); it != new_entry;) {
            const int lcp = it->tokens.get_common_prefix(prompt.tokens);
            if (lcp == (int) it->tokens.size()) {
                auto obsolete = it++;
                if (!erase_disk_state(obsolete, false, "obsolete-prefix")) {
                    disable_disk_saves("obsolete-reclaim", disk_owned_path);
                    reclaim_ok = false;
                    break;
                }
            } else {
                ++it;
            }
        }
    }

    while (reclaim_ok && disk_size_total + disk_foreign_bytes > disk_limit_size) {
        if (evict_foreign_entry("lru-limit")) {
            continue;
        }
        if (disk_states.begin() == new_entry) {
            SRV_ERR("prompt cache disk reclaim failed: entry=%" PRIu64 " reason=no-old-victim accounted_bytes=%zu limit_bytes=%zu path=%s\n",
                    entry_id, disk_size_total, disk_limit_size, disk_owned_path.c_str());
            disable_disk_saves("room-not-reclaimed", disk_owned_path);
            reclaim_ok = false;
            break;
        }
        if (!erase_disk_state(disk_states.begin(), true, "lru-limit")) {
            disable_disk_saves("lru-reclaim", disk_owned_path);
            reclaim_ok = false;
            break;
        }
    }

    if (!reclaim_ok) {
        SRV_WRN("prompt cache disk committed over limit: entry=%" PRIu64 " accounted_bytes=%zu limit_bytes=%zu save_disabled=true path=%s\n",
                entry_id, disk_size_total, disk_limit_size, disk_owned_path.c_str());
    }

    const double t_ms = (ggml_time_us() - t_start)/1000.0;
    SRV_INF("prompt cache disk save: entry=%" PRIu64 " tokens=%zu checkpoints=%zu/%zu ckpt_bytes=%zu target_bytes=%zu draft_bytes=%zu spec_bytes=%zu total_bytes=%zu save_ms=%.2f path=%s\n",
            entry_id, tokens.size(), new_entry->ckpts.size(), prompt.checkpoints.size(), n_ckpt_total, n_main, n_drft, state_spec.size(), actual_total + n_ckpt_total, t_ms, disk_owned_path.c_str());
    log_disk_state();

    return true;
}

server_prompt_cache_state * server_prompt_cache::alloc(
        const server_prompt & prompt,
                    size_t state_size_tgt,
                    size_t state_size_dft,
        const std::vector<uint8_t> & state_spec) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const size_t lcp = it->prompt.tokens.get_common_prefix(prompt.tokens);
        const bool covered = lcp == prompt.tokens.size() &&
            (partial_seq_rm || it->prompt.tokens.size() == prompt.tokens.size());

        if (covered) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + state_spec.size() + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt; with
    // recurrent/hybrid state a shorter boundary is not a truncation of the new one, so
    // it is kept
    if (partial_seq_rm) {
        for (auto it = states.begin(); it != states.end();) {
            const size_t len = it->prompt.tokens.get_common_prefix(prompt.tokens);

            if (len == it->prompt.tokens.size()) {
                SRV_TRC(" - removing obsolete cached prompt with length %zu\n", len);

                it = states.erase(it);
            } else {
                ++it;
            }
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens      =*/ prompt.tokens.clone(),
            /*.checkpoints =*/ prompt.checkpoints,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
            /*.spec =*/ state_spec,
        },
    });

    return &states.back();
}

bool server_prompt_cache::has_disk_prefix(const server_tokens & tokens, size_t n_tokens) const {
    if (n_tokens == 0 || n_tokens > tokens.size()) {
        return false;
    }
    for (const auto & st : disk_states) {
        if (st.usable && st.tokens.size() == n_tokens && st.tokens.get_common_prefix(tokens) == n_tokens) {
            return true;
        }
    }
    return false;
}

bool server_prompt_cache::save_prefix(
        const server_tokens & tokens_prefix,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
               llama_seq_id   id_slot,
        const std::vector<uint8_t> & state_spec) {
    const server_prompt_disk_publish publish { *this }; // [TAG_REQUEST_HISTORY]

    if (!disk_enabled() || tokens_prefix.empty()) {
        return false;
    }

    // an exact-boundary entry: no checkpoints, the contexts hold exactly these tokens
    server_prompt prefix;
    prefix.tokens = tokens_prefix.clone();

    return save_disk(prefix, ctx_tgt, ctx_dft, id_slot, state_spec);
}

bool server_prompt_cache::load_disk(
        std::list<server_prompt_disk_state>::iterator it,
        server_prompt & prompt,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
         llama_seq_id   id_slot,
              size_t   lcp,
       std::vector<uint8_t> * state_spec) {
    const uint64_t entry_id = it->id;
    const size_t target_bytes = it->size_main;
    const size_t draft_bytes  = it->size_drft;
    const size_t spec_bytes   = it->spec.size();
    const size_t total_bytes  = it->size();
    const size_t n_tokens_expected = it->tokens.size();
    const size_t n_checkpoints = it->ckpts.size();
    const std::string path_main = it->path_main;
    const std::string path_drft = it->path_drft;

    const auto reject_entry = [&](const char * reason) -> bool {
        it->usable = false;
        if (!erase_disk_state(it, false, reason)) {
            disable_disk_saves("invalid-entry-removal", disk_owned_path);
        }
        log_disk_state();
        return false;
    };

    // Validate the entire pair before mutating either context.
    size_t actual_main = 0;
    size_t actual_drft = 0;
    if (path_main.empty() || target_bytes == 0 ||
        !server_prompt_cache_disk_size_exact(path_main, target_bytes, &actual_main)) {
        SRV_ERR("prompt cache disk load failed: entry=%" PRIu64 " component=target reason=size-mismatch expected_bytes=%zu actual_bytes=%zu path=%s\n",
                entry_id, target_bytes, actual_main, path_main.c_str());
        return reject_entry("target-size-mismatch");
    }
    if (!path_drft.empty()) {
        if (ctx_dft == nullptr || draft_bytes == 0 ||
            !server_prompt_cache_disk_size_exact(path_drft, draft_bytes, &actual_drft)) {
            SRV_ERR("prompt cache disk load failed: entry=%" PRIu64 " component=draft reason=size-mismatch expected_bytes=%zu actual_bytes=%zu path=%s\n",
                    entry_id, draft_bytes, actual_drft, path_drft.c_str());
            return reject_entry("draft-size-mismatch");
        }
    } else if (ctx_dft != nullptr || draft_bytes != 0) {
        SRV_ERR("prompt cache disk load failed: entry=%" PRIu64 " component=draft reason=missing-draft-file expected_bytes=%zu path=%s\n",
                entry_id, draft_bytes, disk_owned_path.c_str());
        return reject_entry("missing-draft-file");
    }

    const int64_t t_start = ggml_time_us();

    llama_tokens tokens_main(n_tokens_expected);
    size_t n_tokens_main = 0;
    const size_t nread_main = llama_state_seq_load_file(
        ctx_tgt, path_main.c_str(), id_slot,
        tokens_main.data(), tokens_main.size(), &n_tokens_main);
    tokens_main.resize(n_tokens_main);
    server_prompt_cache_disk_flush_and_drop(path_main, false);

    if (nread_main != target_bytes || !server_prompt_cache_tokens_equal(it->tokens, tokens_main)) {
        SRV_ERR("prompt cache disk load failed: entry=%" PRIu64 " component=target expected_bytes=%zu read_bytes=%zu expected_tokens=%zu restored_tokens=%zu path=%s\n",
                entry_id, target_bytes, nread_main, n_tokens_expected, n_tokens_main, path_main.c_str());
        return reject_entry("corrupt-target");
    }

    size_t nread_drft = 0;
    if (!path_drft.empty()) {
        llama_tokens tokens_drft(n_tokens_expected);
        size_t n_tokens_drft = 0;
        nread_drft = llama_state_seq_load_file(
            ctx_dft, path_drft.c_str(), id_slot,
            tokens_drft.data(), tokens_drft.size(), &n_tokens_drft);
        tokens_drft.resize(n_tokens_drft);
        server_prompt_cache_disk_flush_and_drop(path_drft, false);

        if (nread_drft != draft_bytes ||
            !server_prompt_cache_tokens_equal(it->tokens, tokens_drft) ||
            tokens_drft != tokens_main) {
            SRV_ERR("prompt cache disk load failed: entry=%" PRIu64 " component=draft expected_bytes=%zu read_bytes=%zu expected_tokens=%zu restored_tokens=%zu path=%s\n",
                    entry_id, draft_bytes, nread_drft, n_tokens_expected, n_tokens_drft, path_drft.c_str());
            return reject_entry("corrupt-draft");
        }
    }

    server_prompt restored;
    restored.tokens = it->tokens.clone();

    // Restore the persisted checkpoints as real ones (see save_disk), oldest first. Those
    // past the divergence point would be invalidated by the slot right away, so they are
    // not even read. An unreadable file only costs that one rollback point.
    size_t nread_ckpt = 0;
    size_t n_ckpt_restored = 0;
    for (const auto & dck : it->ckpts) {
        if ((size_t) dck.meta.n_tokens > lcp) {
            continue;
        }
        common_prompt_checkpoint ck;
        size_t n_bytes = 0;
        if (server_prompt_cache_ckpt_read(dck.path, dck.meta.n_tokens, ck, n_bytes)) {
            nread_ckpt += n_bytes;
            n_ckpt_restored++;
            restored.checkpoints.push_back(std::move(ck));
            SRV_INF("prompt cache disk load: entry=%" PRIu64 " checkpoint restored n_tokens=%" PRId64 " bytes=%zu\n",
                    entry_id, dck.meta.n_tokens, n_bytes);
        } else {
            SRV_WRN("prompt cache disk load: entry=%" PRIu64 " checkpoint file unreadable (%s) - skipping this rollback point\n",
                    entry_id, dck.path.c_str());
        }
    }
    // a divergence inside the entry needs a rollback point; without one the slot would
    // hold a recurrent state it can neither truncate nor use
    if (!partial_seq_rm && lcp < n_tokens_expected && n_ckpt_restored == 0) {
        SRV_ERR("prompt cache disk load failed: entry=%" PRIu64 " reason=no-readable-checkpoint lcp=%zu tokens=%zu\n",
                entry_id, lcp, n_tokens_expected);
        return reject_entry("unreadable-checkpoints");
    }
    prompt = std::move(restored);

    if (state_spec != nullptr) {
        *state_spec = it->spec;
    }

    disk_bytes_read += nread_main + nread_drft + nread_ckpt;
    disk_loads++;

    // the entry stays on disk and becomes most-recently-used
    server_prompt_cache_disk_touch_files(*it);
    disk_states.splice(disk_states.end(), disk_states, it);

    const double t_ms = (ggml_time_us() - t_start)/1000.0;
    disk_restore_ms_total += t_ms; // [TAG_REQUEST_HISTORY]
    disk_restores_timed++;
    SRV_INF("prompt cache disk load: entry=%" PRIu64 " lcp=%zu tokens=%zu checkpoints=%zu/%zu target_bytes=%zu draft_bytes=%zu spec_bytes=%zu total_bytes=%zu read_bytes=%zu load_ms=%.2f path=%s\n",
            entry_id, lcp, n_tokens_expected, n_ckpt_restored, n_checkpoints, target_bytes, draft_bytes, spec_bytes, total_bytes,
            nread_main + nread_drft + nread_ckpt, t_ms, disk_owned_path.c_str());

    return true;
}

bool server_prompt_cache::erase_disk_state(
        std::list<server_prompt_disk_state>::iterator it,
        bool eviction,
        const char * reason) {
    const uint64_t entry_id    = it->id;
    const size_t target_bytes  = it->size_main;
    const size_t draft_bytes   = it->size_drft;
    const size_t total_bytes   = it->size();
    const size_t tokens        = it->tokens.size();
    const std::string path_main = it->path_main;
    const std::string path_drft = it->path_drft;

    // Quarantine before touching either component. If only one unlink works,
    // retain the full conservative accounting and metadata for a later retry.
    it->usable = false;
    const bool main_ok = server_prompt_cache_disk_remove_file(path_main);
    const bool drft_ok = server_prompt_cache_disk_remove_file(path_drft);
    for (const auto & ck : it->ckpts) {
        // best effort; the accounting below uses it->size() which includes the checkpoints
        server_prompt_cache_disk_remove_file(ck.path);
    }
    if (!it->path_spec.empty()) {
        server_prompt_cache_disk_remove_file(it->path_spec);
    }
    if (!main_ok || !drft_ok) {
        SRV_ERR("prompt cache disk removal failed: entry=%" PRIu64 " reason=%s target_removed=%s draft_removed=%s accounted_bytes=%zu path=%s\n",
                entry_id, reason, main_ok ? "true" : "false", drft_ok ? "true" : "false",
                disk_size_total, disk_owned_path.c_str());
        return false;
    }

    if (total_bytes > disk_size_total) {
        SRV_ERR("prompt cache disk accounting invariant failed: entry=%" PRIu64 " entry_bytes=%zu accounted_bytes=%zu path=%s\n",
                entry_id, total_bytes, disk_size_total, disk_owned_path.c_str());
        return false;
    }
    disk_size_total -= total_bytes;

    if (eviction) {
        disk_evictions++;
        disk_bytes_evicted += total_bytes;
        SRV_INF("prompt cache disk eviction: entry=%" PRIu64 " reason=%s tokens=%zu target_bytes=%zu draft_bytes=%zu total_bytes=%zu remaining_bytes=%zu path=%s\n",
                entry_id, reason, tokens, target_bytes, draft_bytes, total_bytes, disk_size_total, disk_owned_path.c_str());
    } else {
        SRV_INF("prompt cache disk remove: entry=%" PRIu64 " reason=%s tokens=%zu target_bytes=%zu draft_bytes=%zu total_bytes=%zu remaining_bytes=%zu path=%s\n",
                entry_id, reason, tokens, target_bytes, draft_bytes, total_bytes, disk_size_total, disk_owned_path.c_str());
    }

    disk_states.erase(it);
    return true;
}

void server_prompt_cache::update_disk() {
    bool changed = false;

    while (disk_size_total + disk_foreign_bytes > disk_limit_size) {
        if (evict_foreign_entry("lru-update-limit")) {
            changed = true;
            continue;
        }
        if (disk_states.empty()) {
            break;
        }
        if (!erase_disk_state(disk_states.begin(), true, "lru-update-limit")) {
            disable_disk_saves("update-limit-removal", disk_owned_path);
            break;
        }
        changed = true;
    }

    // called on every cache update: only report when something was evicted
    if (changed) {
        log_disk_state();
    }
}

// [TAG_REQUEST_HISTORY]
server_prompt_disk_stats server_prompt_cache::disk_stats() const {
    server_prompt_disk_stats s;
    if (disk_owned_path.empty()) {
        return s;
    }
    s.enabled          = true;
    s.entries          = disk_states.size();
    s.unusable         = std::count_if(disk_states.begin(), disk_states.end(),
                            [](const server_prompt_disk_state & state) { return !state.usable; });
    s.bytes            = disk_size_total;
    s.limit_bytes      = disk_limit_size;
    s.tokens           = disk_n_tokens();
    s.saves            = disk_saves;
    s.loads            = disk_loads;
    s.evictions        = disk_evictions;
    s.save_failures    = disk_save_failures;
    s.save_disabled    = disk_save_disabled;
    s.bytes_written    = disk_bytes_written;
    s.bytes_read       = disk_bytes_read;
    s.bytes_evicted    = disk_bytes_evicted;
    s.restore_ms_total = disk_restore_ms_total;
    s.restores_timed   = disk_restores_timed;
    return s;
}

json server_prompt_disk_stats::to_json() const {
    if (!enabled) {
        return { {"enabled", false} };
    }
    return {
        {"enabled",          true},
        {"entries",          entries},
        {"unusable",         unusable},
        {"bytes",            bytes},
        {"limit_bytes",      limit_bytes},
        {"over_limit",       bytes > limit_bytes},
        {"tokens",           tokens},
        {"saves",            saves},
        {"loads",            loads},
        {"evictions",        evictions},
        {"save_failures",    save_failures},
        {"save_disabled",    save_disabled},
        {"bytes_written",    bytes_written},
        {"bytes_read",       bytes_read},
        {"bytes_evicted",    bytes_evicted},
        {"restore_ms_total", restore_ms_total},
        {"restores_timed",   restores_timed},
    };
}

void server_prompt_cache::log_disk_state() const {
    if (disk_owned_path.empty()) {
        return;
    }

    const size_t unusable = std::count_if(disk_states.begin(), disk_states.end(),
        [](const server_prompt_disk_state & state) { return !state.usable; });
    SRV_INF("prompt cache disk state: entries=%zu unusable=%zu bytes=%zu limit_bytes=%zu over_limit=%s tokens=%zu saves=%" PRIu64 " loads=%" PRIu64 " evictions=%" PRIu64 " save_disabled=%s save_failures=%" PRIu64 " bytes_written=%" PRIu64 " bytes_read=%" PRIu64 " bytes_evicted=%" PRIu64 " path=%s\n",
            disk_states.size(), unusable, disk_size_total, disk_limit_size,
            disk_size_total > disk_limit_size ? "true" : "false", disk_n_tokens(),
            disk_saves, disk_loads, disk_evictions, disk_save_disabled ? "true" : "false", disk_save_failures,
            disk_bytes_written, disk_bytes_read, disk_bytes_evicted,
            disk_owned_path.c_str());
}

bool server_prompt_cache::load(
              server_prompt & prompt,
        const server_tokens & tokens_new,
              llama_context * ctx_tgt,
              llama_context * ctx_dft,
                    int32_t   id_slot,
       std::vector<uint8_t> * state_spec,
                       bool   probe) {
    const server_prompt_disk_publish publish { *this }; // [TAG_REQUEST_HISTORY]

    if (state_spec != nullptr) {
        state_spec->clear();
    }

    // Worth of what the slot already holds. Like every candidate below it is measured in
    // tokens the slot would actually keep: a diverging tail on a recurrent/hybrid target
    // means rolling back to the newest checkpoint at or before the divergence.
    const size_t lcp_base  = prompt.tokens.get_common_prefix(tokens_new);
    const size_t eff_base  = prompt.tokens.empty() ? 0 :
        server_prompt_cache_eff_tokens(prompt.tokens.size(), lcp_base, partial_seq_rm,
                                       server_prompt_cache_ckpt_at_or_below(prompt.checkpoints, lcp_base));

    // Candidates are ranked by eff tokens alone: a slot switch first saves the slot's own
    // state to the cache, so nothing is lost by taking the entry that keeps the most of
    // the request. (vanilla's additional f_keep criterion would prefer an older, shorter
    // version of the same conversation over a newer one with a longer discarded tail.)
    size_t eff_best = eff_base; // empty slot: any usable cache entry wins

    SRV_TRC(" - looking for better prompt, base eff = %zu (lcp = %zu, cached = %zu, request = %zu)\n", eff_base, lcp_base, prompt.tokens.size(), tokens_new.size());

    auto it_best_ram  = states.end();
    auto it_best_disk = disk_states.end();
    size_t lcp_selected = 0;

    // Find the most similar RAM prompt first. On an equal match, the hot RAM
    // copy wins and avoids SSD I/O.
    for (auto it = states.begin(); it != states.end(); ++it) {
        const size_t lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);
        const size_t eff_cur = server_prompt_cache_eff_tokens(it->prompt.tokens.size(), lcp_cur, partial_seq_rm,
                                                              server_prompt_cache_ckpt_at_or_below(it->prompt.checkpoints, lcp_cur));

        const float f_keep_cur = float(eff_cur) / std::max<size_t>(1, it->prompt.tokens.size());

        SRV_TRC("   - prompt with length %7zu, lcp = %7zu, eff = %7zu, f_keep = %.3f\n", it->prompt.tokens.size(), lcp_cur, eff_cur, f_keep_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (eff_cur > eff_best) {
            eff_best = eff_cur;

            it_best_ram  = it;
            lcp_selected = lcp_cur;
        }
    }

    // Disk candidates. Unlike a RAM entry, a disk entry is not consumed by a restore, so
    // there is no "don't trash large prompts" veto here: a 60k-token conversation whose
    // only usable checkpoint is the 14k system-prompt boundary is still worth 14k tokens.
    // What a disk restore must clear is the cost of the read: it has to keep at least
    // disk_min_gain tokens more than the slot already does. Among equals the smaller
    // entry wins (a shared prefix entry instead of a whole conversation).
    std::vector<std::list<server_prompt_disk_state>::iterator> dead;

    // entries that share a prefix with the request but cannot be resumed from it; logged
    // as one summary line per lookup (the per-entry detail is at debug level)
    size_t n_skip_boundary   = 0;
    size_t lcp_skip_boundary = 0;

    for (auto it = disk_states.begin(); it != disk_states.end(); ++it) {
        if (!it->usable) {
            continue;
        }

        const size_t  lcp_cur = it->tokens.get_common_prefix(tokens_new);
        const int64_t ckpt_n  = it->ckpt_at_or_below(lcp_cur);
        const size_t  eff_cur = server_prompt_cache_eff_tokens(it->tokens.size(), lcp_cur, partial_seq_rm, ckpt_n);

        if (eff_cur == 0) {
            if (lcp_cur >= disk_min_gain) {
                SRV_DBG("prompt cache skip: reason=boundary-mismatch source=disk entry=%" PRIu64 " lcp=%zu cached_tokens=%zu request_tokens=%zu checkpoints=%zu%s\n",
                        it->id, lcp_cur, it->tokens.size(), tokens_new.size(), it->ckpts.size(),
                        it->has_ckpt() ? " (request diverges before the oldest persisted checkpoint)" : "");
                n_skip_boundary++;
                lcp_skip_boundary = std::max(lcp_skip_boundary, lcp_cur);
            }
            continue;
        }

        if (eff_cur < eff_base + disk_min_gain) {
            continue;
        }

        const bool better = eff_cur > eff_best ||
            (eff_cur == eff_best && it_best_disk != disk_states.end() && it->size() < it_best_disk->size());
        if (!better) {
            continue;
        }

        // the files may have been removed under us (external retention policy, manual cleanup):
        // drop the entry and keep looking instead of failing the restore
        {
            size_t actual_main = 0;
            size_t actual_drft = 0;
            const bool files_ok =
                !it->path_main.empty() && it->size_main > 0 &&
                server_prompt_cache_disk_size_exact(it->path_main, it->size_main, &actual_main) &&
                (it->path_drft.empty() || server_prompt_cache_disk_size_exact(it->path_drft, it->size_drft, &actual_drft));
            if (!files_ok) {
                SRV_WRN("prompt cache skip: reason=files-missing source=disk entry=%" PRIu64 " target_bytes=%zu target_actual=%zu\n",
                        it->id, it->size_main, actual_main);
                dead.push_back(it);
                continue;
            }
        }

        if (ckpt_n > 0 && lcp_cur != it->tokens.size()) {
            SRV_DBG("prompt cache candidate: source=disk entry=%" PRIu64 " via checkpoint n_tokens=%" PRId64 " lcp=%zu cached_tokens=%zu request_tokens=%zu\n",
                    it->id, ckpt_n, lcp_cur, it->tokens.size(), tokens_new.size());
        }

        eff_best = eff_cur;

        it_best_ram  = states.end();
        it_best_disk = it;
        lcp_selected = lcp_cur;
    }

    for (auto & it : dead) {
        it->usable = false;
        if (!erase_disk_state(it, false, "files-missing")) {
            disable_disk_saves("missing-entry-removal", disk_owned_path);
        }
    }

    if (n_skip_boundary > 0 && !probe && it_best_disk == disk_states.end()) {
        SRV_INF("prompt cache: no disk entry usable, %zu share a prefix (longest lcp=%zu of %zu request tokens) but diverge before their oldest checkpoint\n",
                n_skip_boundary, lcp_skip_boundary, tokens_new.size());
    }

    if (it_best_disk != disk_states.end()) {
        SRV_INF(" - found better disk prompt: entry=%" PRIu64 " eff=%zu lcp=%zu cached_tokens=%zu request_tokens=%zu\n",
                it_best_disk->id, eff_best, lcp_selected, it_best_disk->tokens.size(), tokens_new.size());
        return load_disk(it_best_disk, prompt, ctx_tgt, ctx_dft, id_slot, lcp_selected, state_spec);
    }

    if (it_best_ram != states.end()) {
        SRV_TRC(" - found better RAM prompt with eff = %zu, lcp = %zu\n", eff_best, lcp_selected);

        {
            auto & data = it_best_ram->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best_ram->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        if (state_spec != nullptr) {
            *state_spec = std::move(it_best_ram->data.spec);
        }

        prompt = std::move(it_best_ram->prompt);

        states.erase(it_best_ram);
    }

    return true;
}

void server_prompt_cache::update() {
    const server_prompt_disk_publish publish { *this }; // [TAG_REQUEST_HISTORY]

    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }

    if (disk_enabled()) {
        update_disk();
    }
}

//
// server_slot_sched
//

static std::vector<const server_slot_info *> get_free_slot_infos(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) {
    std::vector<const server_slot_info *> free_slots;
    for (const auto & slot : slots) {
        if (!slot.is_processing && slot.id != exclude_id_slot) {
            free_slots.push_back(&slot);
        }
    }

    std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot_info * a, const server_slot_info * b) {
        if ((a->prompt->n_tokens() == 0) != (b->prompt->n_tokens() == 0)) {
            return a->prompt->n_tokens() == 0;
        }
        return a->t_last_used < b->t_last_used;
    });

    free_slots.resize(std::min(free_slots.size(), n_slots_needed));

    return free_slots;
}

std::vector<int> server_slot_sched::get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const {
    std::vector<int> res;
    for (const auto * slot : get_free_slot_infos(slots, n_slots_needed, exclude_id_slot)) {
        res.push_back(slot->id);
    }

    return res;
}

server_slot_plan server_slot_sched::select(const std::vector<server_slot_info> & slots, const server_task & task) const {
    const server_slot_info * ret = nullptr;

    bool update_cache = false;
    bool probe_cache  = false; // look for a better cache entry without saving the slot first

    // if a specific slot is requested, use it (still goes through cache update logic below)
    if (task.id_slot != -1 && !slots.empty()) {
        // note: allow id_slot to be out of bounds (wrap around)
        const int id_slot = task.id_slot % slots.size();

        for (const auto & slot : slots) {
            if (slot.id == id_slot) {
                ret = &slot;
                break;
            }
        }

        if (ret) {
            // a busy slot is returned untouched, the caller defers the task
            if (ret->is_processing) {
                return { ret->id, false, false };
            }

            SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
        }
    }

    // find the slot that has at least n% prompt similarity
    if (slot_prompt_similarity != 0.0f) {
        float f_sim_best = 0;

        for (const auto & slot : slots) {
            if (task.id_slot != -1 && slot.id != task.id_slot) {
                continue;
            }

            // skip the slot if it is not available
            if (slot.is_processing) {
                SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing);
                continue;
            }

            const auto & tokens = slot.prompt->tokens;

            // skip the slot if it does not contains cached tokens
            if (tokens.empty()) {
                SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                continue;
            }

            // fraction of the Longest Common Prefix length with respect to the input prompt length
            const size_t lcp_len = tokens.get_common_prefix(task.tokens);
            const float f_sim_cur = float(lcp_len) / task.tokens.size();

            SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

            // select the current slot if the criteria match
            if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                f_sim_best = f_sim_cur;

                ret = &slot;
            }
        }

        if (ret != nullptr) {
            const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt->tokens.size();

            if (task.id_slot == -1) {
                SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        f_sim_best, slot_prompt_similarity, f_keep);
            }

            // if we are about to lose a large portion of the existing context - save it in the prompt cache
            if (f_keep < 0.5f) {
                update_cache = true;
            } else if (prompt_cache && prompt_cache->disk_enabled() && !prompt_cache->partial_seq_rm) {
                // The slot keeps most of its tokens, but recurrent/hybrid memory can only keep
                // what lies before a checkpoint. When the request diverges well past the newest
                // usable one (an agent rewrote or compacted older history, a different
                // conversation shares the system prompt), the slot would re-prefill from there,
                // or from scratch - a cache entry may do much better.
                const auto & tokens  = ret->prompt->tokens;
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);

                if (lcp_len < tokens.size()) {
                    int64_t ckpt_n = 0;
                    for (const auto & ck : ret->prompt->checkpoints) {
                        if (ck.n_tokens > ckpt_n && (size_t) ck.n_tokens <= lcp_len) {
                            ckpt_n = ck.n_tokens;
                        }
                    }

                    if (lcp_len >= (size_t) ckpt_n + prompt_cache->disk_min_gain) {
                        // sharing nothing past the first user message means this is another
                        // conversation: its predecessor is worth keeping. Otherwise the slot
                        // holds a superseded version of this very conversation - do not
                        // spend seconds writing that to disk, only look for something better.
                        const int32_t first_user_pos = task.params.message_spans.first_user_message_pos();
                        if (first_user_pos < 0 || lcp_len <= (size_t) first_user_pos + 64) {
                            update_cache = true;
                        } else {
                            probe_cache = true;
                        }
                        SLT_INF(*ret, "prompt diverges %zu tokens past the newest usable checkpoint (lcp = %zu, checkpoint = %" PRId64 ") - %s\n",
                                lcp_len - (size_t) ckpt_n, lcp_len, ckpt_n,
                                update_cache ? "saving the slot and looking for a better cache entry" : "looking for a better cache entry");
                    }
                }
            }
        }
    }

    // find an empty slot, or the one that has been least recently used
    if (ret == nullptr) {
        const auto free_slots = get_free_slot_infos(slots, 1, -1);

        if (!free_slots.empty()) {
            ret = free_slots[0];

            SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt->n_tokens(), ret->t_last_used);

            update_cache = true;
        }
    }

    if (ret == nullptr) {
        return {};
    }

    // [RDNA-FAST] cache prompts only for completion tasks: decisions, realtime, transcription and TTS tasks neither
    // write nor restore entries (upstream: every task that is not stateless)
    const bool cacheable = prompt_cache != nullptr && task.type == SERVER_TASK_TYPE_COMPLETION;

    return { ret->id, update_cache && cacheable, !update_cache && probe_cache && cacheable };
}

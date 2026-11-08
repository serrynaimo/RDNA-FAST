#include "llama-expert-handoff.h"

#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-expert-handoff.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

// [TAG_EH_RESIDENT] host memory for the resident copy: committed, zeroed, page aligned
void * eh_ram_alloc(size_t size) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

void eh_ram_free(void * p, size_t size, size_t locked) {
    if (p == nullptr) {
        return;
    }
#if defined(_WIN32)
    if (locked > 0) {
        VirtualUnlock(p, locked);
    }
    GGML_UNUSED(size);
    VirtualFree(p, 0, MEM_RELEASE);
#else
    if (locked > 0) {
        munlock(p, locked);
    }
    munmap(p, size);
#endif
}

// the RAM the copy may take: physical memory available (on Windows: free and standby pages, bounded by the commit
// charge left)
bool eh_available_ram(uint64_t & avail) {
#if defined(_WIN32)
    MEMORYSTATUSEX ms = {};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) {
        return false;
    }
    avail = std::min<uint64_t>(ms.ullAvailPhys, ms.ullAvailPageFile);
    return true;
#else
    FILE * f = fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return false;
    }
    char line[256];
    bool ok = false;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long kb = 0;
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            avail = (uint64_t) kb*1024;
            ok = true;
            break;
        }
    }
    fclose(f);
    return ok;
#endif
}

// lock [p, p + size) in RAM so the memory manager never trims it (a trimmed page faults back in on the host's next
// read: what makes the mapped model file slow under memory pressure); returns the bytes locked from p on
size_t eh_ram_lock(void * p, size_t size, std::string & note) {
#if defined(_WIN32)
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        note = "GetProcessWorkingSetSizeEx failed (" + std::to_string(GetLastError()) + ")";
        return 0;
    }
    // locked pages count against the minimum working set: raise it by the copy and a margin (soft limits)
    const SIZE_T margin  = (SIZE_T) 512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T) size + margin;
    const SIZE_T new_max = std::max<SIZE_T>(max_ws, new_min + margin);
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        note = "SetProcessWorkingSetSizeEx failed (" + std::to_string(GetLastError()) + ")";
        return 0;
    }
    const size_t chunk = (size_t) 1 << 30;
    size_t done = 0;
    while (done < size) {
        const size_t n = std::min(chunk, size - done);
        if (!VirtualLock((uint8_t *) p + done, n)) {
            note = "VirtualLock stopped at " + std::to_string(done >> 20) + " MiB (" + std::to_string(GetLastError()) + ")";
            return done;
        }
        done += n;
    }
    return done;
#else
    if (mlock(p, size) != 0) {
        note = std::string("mlock failed: ") + strerror(errno);
        return 0;
    }
    return size;
#endif
}

// the host pool, through the CPU backend's proc addresses (the CPU backend may be a dynamically loaded library)
struct eh_host_api {
    decltype(ggml_expert_handoff_host_new)       * host_new   = nullptr;
    decltype(ggml_expert_handoff_host_free)      * host_free  = nullptr;
    decltype(ggml_expert_handoff_host_set_layer) * set_layer  = nullptr;
    decltype(ggml_expert_handoff_host_wake)      * wake       = nullptr;
    decltype(ggml_expert_handoff_host_idle)      * idle       = nullptr;
    decltype(ggml_expert_handoff_host_get_stats) * get_stats  = nullptr;
    decltype(ggml_expert_handoff_host_counts)    * counts     = nullptr;

    bool load() {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
        if (reg == nullptr) {
            return false;
        }
        host_new  = (decltype(host_new))  ggml_backend_reg_get_proc_address(reg, "ggml_expert_handoff_host_new");
        host_free = (decltype(host_free)) ggml_backend_reg_get_proc_address(reg, "ggml_expert_handoff_host_free");
        set_layer = (decltype(set_layer)) ggml_backend_reg_get_proc_address(reg, "ggml_expert_handoff_host_set_layer");
        wake      = (decltype(wake))      ggml_backend_reg_get_proc_address(reg, "ggml_expert_handoff_host_wake");
        idle      = (decltype(idle))      ggml_backend_reg_get_proc_address(reg, "ggml_expert_handoff_host_idle");
        get_stats = (decltype(get_stats)) ggml_backend_reg_get_proc_address(reg, "ggml_expert_handoff_host_get_stats");
        counts    = (decltype(counts))    ggml_backend_reg_get_proc_address(reg, "ggml_expert_handoff_host_counts");
        return host_new && host_free && set_layer && wake && idle && get_stats && counts;
    }
};

using eh_mapped_alloc_t = void * (*)(size_t);
using eh_mapped_free_t  = void   (*)(void *);

// [TAG_EH_COPY_ENGINE] the GPU backend's upload engine (ggml-expert-handoff.h)
struct eh_upload_api {
    ggml_upload_engine_new_t       make   = nullptr;
    ggml_upload_engine_free_t      free   = nullptr;
    ggml_upload_engine_push_t      push   = nullptr;
    ggml_upload_engine_gather_t    gather = nullptr;
    ggml_upload_engine_reset_t     reset  = nullptr;
    ggml_upload_engine_get_stats_t stats  = nullptr;

    bool load(ggml_backend_reg_t reg) {
        make   = (ggml_upload_engine_new_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_engine_new");
        free   = (ggml_upload_engine_free_t)      ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_engine_free");
        push   = (ggml_upload_engine_push_t)      ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_engine_push");
        gather = (ggml_upload_engine_gather_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_engine_gather");
        reset  = (ggml_upload_engine_reset_t)     ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_engine_reset");
        stats  = (ggml_upload_engine_get_stats_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_engine_get_stats");
        return make && free && push && gather && reset && stats;
    }
};

int64_t eh_env_int(const char * name, int64_t def) {
    const char * v = getenv(name);
    return v != nullptr && *v != '\0' ? std::strtoll(v, nullptr, 10) : def;
}

// the per-expert tensors of a layer, each cached in a bank: the weights (gate_up or gate and up, or up alone; down),
// then their biases ([n, n_expert], F32), which the graph adds by slot so that a miss adds nothing
std::vector<ggml_tensor *> eh_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps,
                             layer.ffn_gate_up_exps_b, layer.ffn_gate_exps_b, layer.ffn_up_exps_b, layer.ffn_down_exps_b }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

// bytes of one expert in t: weights [k, n, n_expert], biases [n, n_expert]
size_t eh_expert_stride(const ggml_tensor * t) {
    return ggml_n_dims(t) >= 3 ? t->nb[2] : t->nb[1];
}

int64_t eh_n_expert(const ggml_tensor * t) {
    return ggml_n_dims(t) >= 3 ? t->ne[2] : t->ne[1];
}

bool eh_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || eh_expert_stride(a[i]) != eh_expert_stride(b[i])) {
            return false;
        }
    }
    return true;
}

bool eh_valid_experts(const std::vector<ggml_tensor *> & experts) {
    if (experts.empty()) {
        return false;
    }
    const int64_t n_expert = eh_n_expert(experts[0]);
    for (const ggml_tensor * t : experts) {
        if (eh_n_expert(t) != n_expert || (ggml_n_dims(t) < 3 && (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)))) {
            return false;
        }
    }
    return true;
}

// plain host memory the host pool can read in place (not a repacked CPU buffer, which reports no host access)
bool eh_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

std::vector<float> eh_read_scale(const ggml_tensor * s, int64_t n_expert, bool & ok) {
    std::vector<float> res;
    if (s == nullptr) {
        return res;
    }
    if (s->type != GGML_TYPE_F32 || ggml_nelements(s) != n_expert) {
        ok = false;
        return res;
    }
    res.resize(n_expert);
    ggml_backend_tensor_get(s, res.data(), 0, n_expert*sizeof(float));
    return res;
}

} // namespace

struct llama_expert_handoff::impl {
    // a GPU holding layers of the model (layer split): its VRAM cache, the misses and routing counts of its layers
    struct device {
        ggml_backend_t             backend = nullptr;
        ggml_backend_dev_t         dev     = nullptr;
        ggml_backend_buffer_type_t buft    = nullptr;
        ggml_context_ptr           ctx;
        ggml_backend_buffer_ptr    buf;
        ggml_tensor * flags  = nullptr;     // I32 [n_layer]: the misses of each layer's last send
        ggml_tensor * counts = nullptr;     // I32 [counts_stride, n_layer]: routing counts
        size_t  host_bytes = 0;             // routed experts of its layers in host memory
        size_t  budget     = 0;             // its expert cache
        size_t  n_tensors  = 0;
    };

    struct group {
        int32_t id = -1;                    // the device
        std::vector<ggml_tensor *> ref;     // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks;   // device storage: n_slots real slots, the miss slot (zeros), one padding slot
        size_t  host_bytes = 0;
        size_t  slot_bytes = 0;             // one expert in all tensors
        int32_t n_slots    = 0;
        int32_t n_expert   = 0;
    };

    struct layer {
        int32_t ig = -1;
        ggml_tensor * slot_map = nullptr;           // device I32 [1, n_expert]
        mutable std::vector<int32_t> slots;         // host copy, miss slot for the experts not cached (adapt() changes it)
        std::vector<ggml_tensor *> experts;         // host experts in the order of the banks
        // registration with the host pool (first use)
        mutable int32_t state = 0;                  // 0 = not yet, 1 = ok, -1 = the host cannot compute it
        mutable llama_expert_handoff_form form;
        mutable std::vector<float> gate_s, up_s, down_s;
        // [TAG_EH_RESIDENT] [ip][e]: where the resident copy holds expert e of experts[ip], NULL if it does not (the
        // weights only; empty for the biases and without a copy)
        // adapt() moves slots between experts (the vectors keep their size: the host pool holds their data())
        mutable std::vector<std::vector<const void *>> res;
    };

    struct binding {
        int32_t il;
        int32_t ip;
        ggml_tensor * cached;
    };

    const llama_model & model;
    llama_expert_handoff_params params;

    std::vector<device> devices;

    std::vector<group> groups;
    std::vector<layer> layers;
    int32_t il_nextn = 0; // the first MTP draft layer (cached whole), the layer count if none
    std::unordered_map<const ggml_tensor *, binding> bindings;

    mutable ggml_context_ptr ctx_views; // copy_experts (the scheduler's thread)

    int32_t counts_stride = 0;

    const device & dev_of(int32_t il) const { return devices[groups[layers[il].ig].id]; }

    // the mailbox and the host pool
    eh_host_api api;
    eh_mapped_free_t mapped_free = nullptr;
    void * mailbox = nullptr;
    bool   mailbox_mapped = false;
    std::vector<uint8_t> mailbox_dummy; // no_alloc models
    ggml_expert_handoff_dims dims = {};
    ggml_expert_handoff_host_t host = nullptr;
    int32_t timeout_ms = 1000;
    mutable uint32_t timeouts_seen = 0;
    mutable int64_t t_report = 0;              // the periodic report in check()
    mutable ggml_expert_handoff_stats last = {};

    // routing counts loaded from the profile [il][e]
    std::vector<std::vector<uint64_t>> profile_counts;

    // [TAG_EH_RESIDENT] the resident RAM copy
    void * res_mem    = nullptr;
    size_t res_size   = 0;
    size_t res_locked = 0;

    // [TAG_EH_COPY_ENGINE] large batches: each GPU's upload engine, and the plan of the graph being computed: the
    // expert tensors its scheduler will upload, in the order it will ask for them (the order of their MUL_MAT_IDs).
    // A batch that routes to nearly every expert (all_experts) queues every expert outside VRAM of every planned
    // tensor before the graph starts, so the copies of later layers run while earlier ones compute (and no layer waits
    // to read its routing); a smaller one queues a layer's routed experts when the scheduler reaches the layer.
    eh_upload_api uapi;
    std::vector<ggml_upload_engine_t> engines;    // [device], NULL without
    struct plan_item {
        const ggml_tensor *  w;
        int32_t              il;
        int32_t              ip;
        ggml_backend_t       be;                    // the backend its split runs on
        ggml_upload_engine_t eng;
        bool                 queued = false;
        std::vector<int64_t> piece;                 // [e]: its piece in the engine, -1 if cached (or not queued)
    };
    // the host experts by name: an allocated graph's MUL_MAT_IDs read the scheduler's copies, named
    // "<backend>#<name>#<copy>" (ggml_backend_sched_split_graph)
    std::unordered_map<std::string, const ggml_tensor *> by_name;
    mutable std::mutex             plan_mtx;
    mutable std::vector<plan_item> plan;
    mutable size_t                 plan_next  = 0;
    mutable bool                   plan_all   = false;
    mutable const void *           plan_owner = nullptr; // the scheduler whose graph the plan is for
    mutable std::vector<ggml_upload_copy> copies; // scratch
    mutable ggml_upload_engine_stats up_last = {};
    mutable uint64_t n_up_graphs = 0, n_up_graphs_last = 0, n_up_fallback = 0;

    mutable std::mutex mtx;

    // batches of up to n_small tokens are always split; larger ones (prompts) where the host's estimated time beats
    // uploading the layer's uncached experts. The estimates learn from the run: the share of routed experts the
    // host computes (check() refreshes it) and the host's throughput (its stats)
    static constexpr int64_t n_small = 32;
    static constexpr double  upload_bps_default = 12e9;  // pageable host memory to the device over PCIe 4.0 x16
    mutable double est_miss = -1.0;                      // < 0: not yet measured

    // the adaptive cache (adapt()): decayed routing counts, the swaps in flight and the helper thread copying them
    struct swap {
        int32_t il;
        int32_t in;    // the expert copied in
        int32_t slot;  // the slot it lands in (that of the expert evicted)
    };
    mutable int64_t n_rounds = 0;                 // decode graphs seen
    mutable std::vector<float>   usage;           // [il*counts_stride + e]
    mutable std::vector<int32_t> counts_seen;     // the device counts at the last round
    mutable std::vector<swap>    in_flight;       // evicted, copying (the helper reads the list while busy)
    // [TAG_EH_RESIDENT] the adaptive resident copy: a swap gives the RAM slots of an expert (one per weight tensor)
    // to another of the same group, which the helper copies in from the tensor
    struct ram_swap {
        int32_t il;                     // the expert copied in
        int32_t in;
        std::vector<const void *> ptr;  // its slots, per weight tensor (NULL for the biases)
    };
    mutable std::vector<ram_swap> ram_in_flight;
    // the copy's own usage counts: the same routing counts with a slower decay (ram_decay per round). The cache's
    // decay (0.7 every 4 graphs) remembers a dozen tokens, about 10 routings per layer each, so nearly every expert of
    // the copy looked unused and the rounds traded it for whatever the last tokens had used twice
    mutable std::vector<float> ram_usage;
    static constexpr float ram_decay = 0.97f;
    mutable uint64_t n_ram_swaps = 0, n_ram_swaps_reported = 0;
    mutable std::thread          helper;
    mutable std::mutex           helper_mtx;
    mutable std::condition_variable helper_cv;
    mutable bool                 helper_job  = false;
    mutable bool                 helper_stop = false;
    mutable std::atomic<bool>    helper_busy{false};
    mutable uint64_t n_swaps = 0, n_swaps_reported = 0;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends,
            const std::vector<ggml_backend_buffer_type_t> & bufts, const llama_expert_handoff_params & params) :
            model(model), params(params), layers(model.layers.size()) {
        // the GPUs; each layer's experts are cached on the GPU that holds the rest of the layer (layer split). One
        // mailbox serves them all: a layer's join must finish before the next layer's send, on whichever GPU
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto t = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU) {
                device d;
                d.backend = backends[i];
                d.dev     = ggml_backend_get_device(backends[i]);
                d.buft    = bufts[i];
                devices.push_back(std::move(d));
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("expert handoff requires a GPU backend");
        }
        if (model.hparams.n_expert == 0) {
            throw std::runtime_error("expert handoff requires a MoE model");
        }
        if (!api.load()) {
            throw std::runtime_error("expert handoff: the CPU backend has no expert handoff pool");
        }
        // the mailbox comes from the first GPU's backend, mapped into all of its devices at the same address
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(devices[0].dev);
        auto mapped_alloc = (eh_mapped_alloc_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_mapped_alloc");
        mapped_free = (eh_mapped_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_host_mapped_free");
        if (mapped_alloc == nullptr || mapped_free == nullptr) {
            throw std::runtime_error(std::string("expert handoff: ") + ggml_backend_dev_name(devices[0].dev) + " cannot map host memory");
        }
        for (const device & d : devices) {
            if (ggml_backend_dev_backend_reg(d.dev) != reg) {
                throw std::runtime_error("expert handoff: the GPUs must use one backend");
            }
        }

        // the layers whose routed experts are all in plain host memory, and whose other weights are on a GPU
        size_t host_bytes_total = 0;
        size_t max_blob = 0;
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = eh_layer_experts(model.layers[il]);
            if (!eh_valid_experts(experts) || !std::all_of(experts.begin(), experts.end(), eh_is_host_weight)) {
                continue;
            }
            const ggml_backend_dev_t ldev = model.dev_layer((int) il);
            int32_t id = -1;
            for (size_t k = 0; k < devices.size(); ++k) {
                if (devices[k].dev == ldev) {
                    id = (int32_t) k;
                }
            }
            if (id < 0) {
                continue; // a layer on the CPU computes its experts there, as before
            }
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && eh_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
                it->n_expert = (int32_t) eh_n_expert(experts[0]);
                for (const ggml_tensor * t : experts) {
                    it->slot_bytes += eh_expert_stride(t);
                }
                max_blob = std::max(max_blob, it->slot_bytes);
            }
            it->layers.push_back((int32_t) il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes   += ggml_nbytes(t);
                host_bytes_total += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
            layers[il].experts = experts;
        }
        if (groups.empty()) {
            throw std::runtime_error("expert handoff: no GPU layer keeps its routed experts in host memory "
                    "(keep them there with --cpu-moe, and without weight repacking: --no-repack)");
        }

        load_profile();

        // [TAG_EH_COPY_ENGINE] before the cache takes the free VRAM (its ring) and before the RAM copy is locked (its
        // pinned buffers: on Windows the first pinned allocation after the lock stalls for a minute)
        if (!model.hparams.no_alloc && eh_env_int("ROCMFPX_EH_COPY_ENGINE", 1) != 0) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(devices[0].dev);
            engines.assign(devices.size(), nullptr);
            if (uapi.load(reg)) {
                ggml_upload_engine_params up = {};
                // test-expert-handoff --upload-bench, RX 7900 XT under Windows, 450 KiB pieces from pageable RAM:
                // 2/4/8/16/32 MiB per DMA moved 7.0/10.4/13.5/15.4/16.0 GB/s (each DMA costs ~0.2 ms here), plain
                // pageable uploads of the pieces 2.8 GB/s
                up.ring_bytes  = (size_t) eh_env_int("ROCMFPX_EH_RING_MIB", 128) << 20;
                up.stage_bytes = (size_t) eh_env_int("ROCMFPX_EH_STAGE_MIB", 16) << 20;
                up.n_stage     = (int) eh_env_int("ROCMFPX_EH_STAGES", 8);
                up.n_threads   = (int) eh_env_int("ROCMFPX_EH_COPY_THREADS", 4);
                for (const group & g : groups) { // a staging buffer holds at least one expert of any tensor
                    for (const ggml_tensor * t : g.ref) {
                        up.stage_bytes = std::max(up.stage_bytes, eh_expert_stride(t));
                    }
                }
                for (size_t id = 0; id < devices.size(); ++id) {
                    if (devices[id].host_bytes == 0) {
                        continue;
                    }
                    engines[id] = uapi.make(devices[id].backend, &up);
                    if (engines[id] == nullptr) {
                        LLAMA_LOG_WARN("%s: %s: no upload engine; large batches upload their experts directly\n", __func__,
                            ggml_backend_dev_name(devices[id].dev));
                    } else {
                        LLAMA_LOG_WARN("%s: %s: upload engine: %.0f MiB ring, %d x %.0f MiB pinned staging, %d copy threads\n",
                            __func__, ggml_backend_dev_name(devices[id].dev), up.ring_bytes/1048576.0, up.n_stage,
                            up.stage_bytes/1048576.0, up.n_threads);
                    }
                }
            } else {
                LLAMA_LOG_WARN("%s: the GPU backend has no upload engine\n", __func__);
            }
        }

        // each GPU's budget: its free VRAM less the reserve, or its share of an explicit size (n_slots of the
        // largest expert, or bytes) by the experts of its layers
        {
            // the context reserves its compute buffers before the cache, so this is only the slack for what comes
            // later (an MTP draft context's buffers, the backend's temporary pool)
            const size_t reserve = (size_t) (params.vram_reserve_mib >= 0 ? params.vram_reserve_mib : 1024)*1024*1024;
            const bool explicit_size = params.n_slots >= 0 || params.budget_bytes > 0;
            const size_t total_budget = params.n_slots >= 0 ? (size_t) params.n_slots*max_blob : params.budget_bytes;
            for (device & d : devices) {
                if (d.host_bytes == 0) {
                    continue;
                }
                size_t free = 0, total = 0;
                ggml_backend_dev_memory(d.dev, &free, &total);
                if (explicit_size) {
                    d.budget = (size_t) ((double) total_budget*d.host_bytes/host_bytes_total);
                    if (d.budget + reserve > free) {
                        LLAMA_LOG_WARN("%s: %s: an expert cache of %.0f MiB leaves %.0f MiB of the %.0f MiB free VRAM, less than the reserve of %.0f MiB\n",
                            __func__, ggml_backend_dev_name(d.dev), d.budget/1048576.0, free > d.budget ? (free - d.budget)/1048576.0 : 0.0,
                            free/1048576.0, reserve/1048576.0);
                    }
                } else {
                    d.budget = free > reserve ? free - reserve : 0;
                }
                LLAMA_LOG_INFO("%s: %s: %.0f MiB free, %.0f MiB for the expert cache (%.0f MiB of its layers' routed experts in host memory)\n",
                    __func__, ggml_backend_dev_name(d.dev), free/1048576.0, d.budget/1048576.0, d.host_bytes/1048576.0);
            }
        }

        // each group caches the same share of its experts; two extra slots per bank: the miss slot and padding
        // (MMQ reads past the last expert)
        // the MTP draft layers (the last n_layer_nextn) are always cached whole: their draft context, which shares
        // this cache, then never needs the host
        const int32_t n_nextn = (int32_t) model.hparams.n_layer_nextn;
        il_nextn = n_nextn > 0 ? (int32_t) model.layers.size() - n_nextn : (int32_t) model.layers.size();

        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t max_slots = (int32_t) g.layers.size()*g.n_expert;
            const size_t  g_budget  = d.host_bytes > 0 ? (size_t) ((double) d.budget*g.host_bytes/d.host_bytes) : 0;
            const size_t  fit       = g_budget / g.slot_bytes;
            g.n_slots = (int32_t) std::min<size_t>(max_slots, fit > 2 ? fit - 2 : 0);
            const int32_t forced = (int32_t) std::count_if(g.layers.begin(), g.layers.end(), [&](int32_t il) { return il >= il_nextn; })*g.n_expert;
            if (g.n_slots < forced) {
                LLAMA_LOG_WARN("%s: the expert cache is smaller than the MTP draft layer; it gets %d slots anyway\n", __func__, forced);
                g.n_slots = forced;
            }
            devices[g.id].n_tensors += g.ref.size()*(1 + g.layers.size()) + g.layers.size();
        }

        for (const group & g : groups) {
            counts_stride = std::max(counts_stride, g.n_expert);
        }
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_init_params ip = { (d.n_tensors + 2)*ggml_tensor_overhead(), nullptr, true };
            d.ctx.reset(ggml_init(ip));
            if (!d.ctx) {
                throw std::runtime_error("expert handoff: failed to create the context");
            }
            d.flags  = ggml_new_tensor_1d(d.ctx.get(), GGML_TYPE_I32, (int64_t) model.layers.size());
            d.counts = ggml_new_tensor_2d(d.ctx.get(), GGML_TYPE_I32, counts_stride, (int64_t) model.layers.size());
            ggml_format_name(d.flags,  "expert_cache.flags.%zu",  id);
            ggml_format_name(d.counts, "expert_cache.counts.%zu", id);
        }
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            ggml_context * gctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_n_dims(t) >= 3 ?
                    ggml_new_tensor_3d(gctx, t->type, t->ne[0], t->ne[1], g.n_slots + 2) :
                    ggml_new_tensor_2d(gctx, t->type, t->ne[0], g.n_slots + 2);
                GGML_ASSERT(eh_expert_stride(bank) == eh_expert_stride(t));
                ggml_format_name(bank, "expert_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig = (int32_t) ig;
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank = g.banks[ip];
                    // the miss slot is a valid expert index of the view
                    ggml_tensor * cached = ggml_n_dims(l.experts[ip]) >= 3 ?
                        ggml_view_3d(gctx, bank, bank->ne[0], bank->ne[1], g.n_slots + 1, bank->nb[1], bank->nb[2], 0) :
                        ggml_view_2d(gctx, bank, bank->ne[0], g.n_slots + 1, bank->nb[1], 0);
                    ggml_format_name(cached, "expert_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                    by_name[l.experts[ip]->name] = l.experts[ip]; // [TAG_EH_COPY_ENGINE]
                }
                l.slot_map = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, 1, g.n_expert);
                ggml_format_name(l.slot_map, "expert_cache.slot_map-%d", il);
                l.slots.assign(g.n_expert, g.n_slots);
            }
        }

        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            if (model.hparams.no_alloc) {
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            } else {
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error(std::string("expert handoff: failed to allocate the expert cache on ") + ggml_backend_dev_name(d.dev));
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                // read by the ops as weights: they run on the device
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        if (!model.hparams.no_alloc) {
            fill();
        }

        // the mailbox and the pool
        dims.max_tokens = (int32_t) params.max_tokens;
        dims.max_embd   = (int32_t) model.hparams.n_embd;
        dims.max_used   = (int32_t) model.hparams.n_expert_used_max();
        // a dual-channel DDR5 host is bandwidth-bound well before its cores run out: 6 workers beat 11 on a
        // 12-core Ryzen (Gemma 4 26B-A4B, half the experts cached: 67-74 vs 61-71 t/s), so one per 4 logical CPUs
        int32_t n_workers = params.n_workers;
        if (n_workers <= 0) {
            const int32_t hw = (int32_t) std::thread::hardware_concurrency();
            n_workers = std::max(2, hw/4);
        }
        if (model.hparams.no_alloc) {
            // a model loaded only to measure memory: the graphs are built, never run
            mailbox_dummy.assign(ggml_expert_handoff_size(dims) + GGML_EXPERT_HANDOFF_ALIGN, 0);
            mailbox = (void *) (((uintptr_t) mailbox_dummy.data() + GGML_EXPERT_HANDOFF_ALIGN - 1) & ~(uintptr_t) (GGML_EXPERT_HANDOFF_ALIGN - 1));
        } else {
            mailbox = mapped_alloc(ggml_expert_handoff_size(dims));
            if (mailbox == nullptr) {
                throw std::runtime_error("expert handoff: failed to allocate the mailbox");
            }
            mailbox_mapped = true;
            host = api.host_new(mailbox, dims.max_tokens, dims.max_embd, dims.max_used, (int32_t) model.layers.size(), n_workers);
            // [TAG_EH_RESIDENT] after the mailbox: on Windows the first pinned allocation after locking the copy took
            // as long as pinning all of it (Flash-Next, 15 GiB locked: 116 s for the mailbox's hipHostMalloc)
            if (params.resident_budget_gib != 0.0f) {
                make_resident();
            }
        }

        // WARN so the server log shows the split (its default verbosity drops libllama INFO) [TAG_EH_TIMEOUT_DIAG]
        size_t n_cached = 0, n_all = 0;
        for (const group & g : groups) {
            n_cached += g.n_slots;
            n_all    += (size_t) g.layers.size()*g.n_expert;
            LLAMA_LOG_WARN("%s: %s: %2zu layers, %s: %6d of %6d experts in VRAM (%.1f%%), %.2f MiB each\n", __func__,
                ggml_backend_dev_name(devices[g.id].dev), g.layers.size(), ggml_type_name(g.ref[0]->type), g.n_slots,
                (int32_t) (g.layers.size()*g.n_expert), 100.0*g.n_slots/std::max<size_t>(1, g.layers.size()*g.n_expert),
                g.slot_bytes/1048576.0);
        }
        LLAMA_LOG_WARN("%s: %zu of %zu experts cached%s, %d host workers, batches up to %d tokens split (above %d where "
            "the host is sooner than uploading)\n", __func__,
            n_cached, n_all, profile_counts.empty() ? " (no profile: the first experts of each layer)" : " by the profile",
            n_workers, dims.max_tokens, (int) n_small);
    }

    // the device's routing counts [il][e] since the start (decode batches; hits and misses)
    std::vector<int32_t> device_counts() const {
        std::vector<int32_t> res;
        std::vector<int32_t> tmp;
        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.counts == nullptr || d.counts->data == nullptr) {
                continue;
            }
            if (res.empty()) {
                res.assign(ggml_nelements(d.counts), 0);
            }
            tmp.resize(ggml_nelements(d.counts));
            ggml_backend_tensor_get(d.counts, tmp.data(), 0, ggml_nbytes(d.counts));
            for (size_t il = 0; il < layers.size(); ++il) {
                if (layers[il].ig >= 0 && groups[layers[il].ig].id == (int32_t) id) {
                    std::copy(tmp.begin() + il*counts_stride, tmp.begin() + (il + 1)*counts_stride, res.begin() + il*counts_stride);
                }
            }
        }
        return res;
    }

    ~impl() {
        for (ggml_upload_engine_t e : engines) { // [TAG_EH_COPY_ENGINE]
            if (e) {
                uapi.free(e);
            }
        }
        if (helper.joinable()) {
            {
                std::lock_guard<std::mutex> lk(helper_mtx);
                helper_stop = true;
            }
            helper_cv.notify_all();
            helper.join();
        }
        if (host) {
            ggml_expert_handoff_stats st = {};
            api.get_stats(host, &st);
            uint64_t n_routed = 0;
            for (int32_t c : device_counts()) {
                n_routed += (uint32_t) c;
            }
            if (n_routed > 0) {
                LLAMA_LOG_INFO("llama_expert_handoff: %" PRIu64 " routed experts, %" PRIu64 " computed on the host (VRAM hit rate %.2f%%); "
                    "%" PRIu64 " layers sent to the host, %.1f us each\n",
                    n_routed, st.n_host, 100.0*(1.0 - (double) st.n_host/n_routed), st.n_layers,
                    st.n_layers ? st.busy_ns/1000.0/st.n_layers : 0.0);
            }
            save_profile();
            api.host_free(host);
        }
        if (mailbox && mailbox_mapped) {
            mapped_free(mailbox);
        }
        // after the pool, which reads it
        eh_ram_free(res_mem, res_size, res_locked);
    }

    // the cache content: the most routed experts of each group by the profile, else the first experts of each layer
    void fill() {
        for (group & g : groups) {
            std::vector<std::pair<int32_t, int32_t>> chosen; // (il, e)
            // the MTP draft layers first, whole
            std::vector<int32_t> rest;
            for (int32_t il : g.layers) {
                if (il >= il_nextn) {
                    for (int32_t e = 0; e < g.n_expert; ++e) {
                        chosen.emplace_back(il, e);
                    }
                } else {
                    rest.push_back(il);
                }
            }
            const int32_t n_free = g.n_slots - (int32_t) chosen.size();
            if (!profile_counts.empty()) {
                // most routed first; equal counts (a short profile leaves most at zero) go round-robin over the
                // layers by their rank within the layer, so no layer is left out
                std::vector<std::tuple<uint64_t, int32_t, int32_t, int32_t>> rank; // count, rank in layer, il, e
                for (int32_t il : rest) {
                    const auto & c = (size_t) il < profile_counts.size() ? profile_counts[il] : std::vector<uint64_t>();
                    std::vector<int32_t> order(g.n_expert);
                    for (int32_t e = 0; e < g.n_expert; ++e) {
                        order[e] = e;
                    }
                    auto cnt = [&](int32_t e) { return e < (int32_t) c.size() ? c[e] : (uint64_t) 0; };
                    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return cnt(a) > cnt(b); });
                    for (int32_t r = 0; r < g.n_expert; ++r) {
                        rank.emplace_back(cnt(order[r]), r, il, order[r]);
                    }
                }
                std::stable_sort(rank.begin(), rank.end(), [](const auto & a, const auto & b) {
                    return std::get<0>(a) != std::get<0>(b) ? std::get<0>(a) > std::get<0>(b) : std::get<1>(a) < std::get<1>(b);
                });
                for (int32_t i = 0; i < n_free && i < (int32_t) rank.size(); ++i) {
                    chosen.emplace_back(std::get<2>(rank[i]), std::get<3>(rank[i]));
                }
            } else if (!rest.empty()) {
                const int32_t nl = (int32_t) rest.size();
                for (int32_t i = 0; i < nl; ++i) {
                    const int32_t n = n_free/nl + (i < n_free % nl ? 1 : 0);
                    for (int32_t e = 0; e < n && e < g.n_expert; ++e) {
                        chosen.emplace_back(rest[i], e);
                    }
                }
            }
            // slots in (layer, expert) order, so neighbours upload together
            std::sort(chosen.begin(), chosen.end());
            for (size_t s = 0; s < chosen.size(); ) {
                const int32_t il = chosen[s].first;
                const int32_t e0 = chosen[s].second;
                size_t n = 1;
                while (s + n < chosen.size() && chosen[s + n].first == il && chosen[s + n].second == e0 + (int32_t) n) {
                    n++;
                }
                layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t es = eh_expert_stride(w);
                    ggml_backend_tensor_set(g.banks[ip], (const uint8_t *) w->data + e0*es, s*es, n*es);
                }
                for (size_t k = 0; k < n; ++k) {
                    l.slots[e0 + k] = (int32_t) (s + k);
                }
                s += n;
            }
            for (int32_t il : g.layers) {
                ggml_backend_tensor_set(layers[il].slot_map, layers[il].slots.data(), 0, layers[il].slots.size()*sizeof(int32_t));
            }
        }
    }

    // [TAG_EH_RESIDENT] the resident RAM copy: the experts the VRAM cache misses, hottest first by the profile (the
    // fill's ranking: counts, then the rank within the layer), as many as the budget and the available RAM allow,
    // copied into one locked allocation. The host pool reads them there and the prompt uploads send them from there;
    // the others stay on the mapped file. On a box whose RAM cannot hold every expert the cache misses, the mapping
    // pages them in and out (the prompt uploads read all of them each ubatch), and even pages still cached cost a
    // soft fault per 4 KiB once the memory manager trims them from the working set: the host then takes ~5x its
    // in-RAM time per expert (Flash-Next Q2_0 on 32 GB, 2026-11-07: 175 vs 35 us).
    void make_resident() {
        const double gib = 1024.0*1024.0*1024.0;
#if defined(_WIN32)
        // the fill read the cached experts through the mapping, and Windows counts those file pages in this
        // process's working set, not as available: trimmed, they move to the standby list (still cached)
        SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T) -1, (SIZE_T) -1);
#endif
        uint64_t avail = 0;
        if (!eh_available_ram(avail)) {
            LLAMA_LOG_WARN("%s: cannot read the available RAM, no resident copy\n", __func__);
            return;
        }
        const uint64_t keep   = (uint64_t) (std::max(0.0f, params.ram_keep_gib)*gib) + ((uint64_t) 256 << 20);
        const uint64_t room   = avail > keep ? avail - keep : 0;
        const uint64_t budget = params.resident_budget_gib > 0.0f ? std::min<uint64_t>(room, (uint64_t) (params.resident_budget_gib*gib)) : room;

        // the weights of one expert of layer il (the biases are small and stay where they are)
        auto weights = [&](int32_t il) {
            size_t b = 0;
            for (const ggml_tensor * t : layers[il].experts) {
                b += ggml_n_dims(t) >= 3 ? eh_expert_stride(t) : 0;
            }
            return b;
        };

        // the candidates: what the cache misses, outside the MTP draft layers (cached whole)
        std::vector<std::tuple<uint64_t, int32_t, int32_t, int32_t>> rank; // count, rank in layer, il, e
        uint64_t missed_bytes = 0;
        int64_t  n_missed     = 0;
        for (int32_t il = 0; il < il_nextn && il < (int32_t) layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.ig < 0) {
                continue;
            }
            const group & g = groups[l.ig];
            static const std::vector<uint64_t> none;
            const auto & c = (size_t) il < profile_counts.size() ? profile_counts[il] : none;
            auto cnt = [&](int32_t e) { return e < (int32_t) c.size() ? c[e] : (uint64_t) 0; };
            std::vector<int32_t> order(g.n_expert);
            for (int32_t e = 0; e < g.n_expert; ++e) {
                order[e] = e;
            }
            std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return cnt(a) > cnt(b); });
            for (int32_t r = 0; r < g.n_expert; ++r) {
                const int32_t e = order[r];
                if (l.slots[e] >= g.n_slots) {
                    rank.emplace_back(cnt(e), r, il, e);
                    missed_bytes += weights(il);
                    n_missed++;
                }
            }
        }
        std::stable_sort(rank.begin(), rank.end(), [](const auto & a, const auto & b) {
            return std::get<0>(a) != std::get<0>(b) ? std::get<0>(a) > std::get<0>(b) : std::get<1>(a) < std::get<1>(b);
        });
        std::vector<std::pair<int32_t, int32_t>> chosen; // (il, e)
        uint64_t at = 0;
        for (const auto & r : rank) {
            const size_t b = weights(std::get<2>(r));
            if (at + b > budget) {
                continue;
            }
            at += b;
            chosen.emplace_back(std::get<2>(r), std::get<3>(r));
        }
        if (chosen.empty()) {
            LLAMA_LOG_WARN("%s: no room for a resident copy (%.2f GiB available, %.2f GiB kept)\n", __func__, avail/gib, keep/gib);
            return;
        }
        std::sort(chosen.begin(), chosen.end());

        // the layout: per layer and weight tensor, its chosen experts in order (consecutive experts upload as one
        // run), each block page aligned and followed by zeros the prompt uploads may read as an expert's padding
        struct piece { size_t off; const uint8_t * src; size_t bytes; };
        std::vector<piece> pieces;
        std::vector<std::tuple<int32_t, size_t, int32_t, size_t>> where; // il, ip, e, offset
        size_t off = 0;
        const size_t align = 4096, slack = 4096;
        const size_t piece_max = (size_t) 16 << 20;
        for (size_t i = 0; i < chosen.size(); ) {
            const int32_t il = chosen[i].first;
            size_t j = i;
            while (j < chosen.size() && chosen[j].first == il) {
                j++;
            }
            const layer & l = layers[il];
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * t = l.experts[ip];
                if (ggml_n_dims(t) < 3) {
                    continue;
                }
                const size_t es = eh_expert_stride(t);
                off = (off + align - 1) & ~(align - 1);
                for (size_t k = i; k < j; ) {
                    // a run of consecutive experts: one copy (cut in pieces for the threads)
                    size_t n = 1;
                    while (k + n < j && chosen[k + n].second == chosen[k].second + (int32_t) n) {
                        n++;
                    }
                    const uint8_t * src = (const uint8_t *) t->data + (size_t) chosen[k].second*es;
                    for (size_t done = 0; done < n*es; done += piece_max) {
                        pieces.push_back({ off + (k - i)*es + done, src + done, std::min(piece_max, n*es - done) });
                    }
                    for (size_t q = 0; q < n; ++q) {
                        where.emplace_back(il, ip, chosen[k + q].second, off + (k - i + q)*es);
                    }
                    k += n;
                }
                off += (j - i)*es + slack;
            }
            i = j;
        }
        const size_t total = (off + align - 1) & ~(align - 1);

        void * mem = eh_ram_alloc(total);
        if (mem == nullptr) {
            LLAMA_LOG_WARN("%s: cannot allocate %.2f GiB for the resident copy\n", __func__, total/gib);
            return;
        }
        // locked before the copy, so the pages are committed once and the memory manager never trims them (a pageable
        // copy was trimmed and faulted back in: Flash-Next decode 12-36 instead of 22-42 t/s). The price on Windows: the
        // GPU driver's next pinned allocation stalls for ~1 s per 150-400 MiB locked (15-17 GiB: 43-93 s, once, in the
        // warmup). ROCMFPX_EH_RES_LOCK=0 keeps the copy pageable.
        std::string note = "pageable";
        const char * lock_env = getenv("ROCMFPX_EH_RES_LOCK");
        const size_t locked = lock_env == nullptr || atoi(lock_env) != 0 ? eh_ram_lock(mem, total, note) : 0;

        const int64_t t0 = ggml_time_us();
        {
            // several readers, so the mapped file pages in from disk in parallel
            const int n_threads = std::max(1, std::min(8, (int) std::thread::hardware_concurrency()/2));
            std::atomic<size_t> next{0};
            std::vector<std::thread> th;
            for (int i = 0; i < n_threads; ++i) {
                th.emplace_back([&] {
                    for (size_t k; (k = next.fetch_add(1)) < pieces.size(); ) {
                        memcpy((uint8_t *) mem + pieces[k].off, pieces[k].src, pieces[k].bytes);
                    }
                });
            }
            for (auto & t : th) {
                t.join();
            }
        }
        const double secs = (ggml_time_us() - t0)*1e-6;

        // a slot array per weight tensor in every layer (adapt() may give any of them slots), then the copy's slots
        for (int32_t il = 0; il < il_nextn && il < (int32_t) layers.size(); ++il) {
            layer & l = layers[il];
            if (l.ig < 0) {
                continue;
            }
            l.res.assign(l.experts.size(), {});
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                if (ggml_n_dims(l.experts[ip]) >= 3) {
                    l.res[ip].assign(groups[l.ig].n_expert, nullptr);
                }
            }
        }
        for (const auto & [il, ip, e, o] : where) {
            layers[il].res[ip][e] = (const uint8_t *) mem + o;
        }
        res_mem    = mem;
        res_size   = total;
        res_locked = locked;

        const std::string how = locked == total ? std::string("locked") : locked == 0 && note == "pageable" ? note : "not locked: " + note;
        LLAMA_LOG_WARN("%s: resident RAM copy: %zu of the %" PRId64 " experts the VRAM cache misses (%.2f of %.2f GiB), %s; "
            "%.2f GiB of RAM were available, %.2f GiB kept; copied in %.1f s (%.2f GB/s)\n", __func__,
            chosen.size(), n_missed, at/gib, missed_bytes/gib, how.c_str(), avail/gib, keep/gib, secs,
            secs > 0.0 ? at/secs/1e9 : 0.0);
    }

    // the resident copy of expert e of experts[ip] of layer il, or NULL
    const void * resident(int32_t il, size_t ip, int64_t e) const {
        const layer & l = layers[il];
        return ip < l.res.size() && !l.res[ip].empty() ? l.res[ip][e] : nullptr;
    }

    const void * const * resident_of(int32_t il, const ggml_tensor * t) const {
        const layer & l = layers[il];
        for (size_t ip = 0; ip < l.experts.size() && t != nullptr; ++ip) {
            if (l.experts[ip] == t) {
                return ip < l.res.size() && !l.res[ip].empty() ? l.res[ip].data() : nullptr;
            }
        }
        return nullptr;
    }

    const void * host_source(const ggml_tensor * w, int64_t e, int64_t last, int64_t & n) const {
        n = last - e + 1;
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return nullptr;
        }
        const layer & l = layers[it->second.il];
        const size_t ip = (size_t) it->second.ip;
        if (ip >= l.res.size() || l.res[ip].empty()) {
            return nullptr;
        }
        const auto & r  = l.res[ip];
        const size_t es = eh_expert_stride(w);
        n = 1;
        if (r[e] == nullptr) {
            while (e + n <= last && r[e + n] == nullptr) {
                n++;
            }
            return nullptr;
        }
        while (e + n <= last && r[e + n] == (const uint8_t *) r[e] + n*es) {
            n++;
        }
        return r[e];
    }

    // profile: one line per layer, "layer <il> <n_expert> <count> ...", after a header line
    void load_profile() {
        if (params.profile.empty()) {
            return;
        }
        std::ifstream f(params.profile);
        if (!f) {
            LLAMA_LOG_WARN("%s: no expert profile at %s, the cache starts from the first experts of each layer\n", __func__, params.profile.c_str());
            return;
        }
        profile_counts.assign(model.layers.size(), {});
        std::string line;
        size_t n_read = 0;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string kw;
            ss >> kw;
            if (kw != "layer") {
                continue;
            }
            int64_t il = -1, n = 0;
            ss >> il >> n;
            if (il < 0 || il >= (int64_t) model.layers.size() || n <= 0) {
                continue;
            }
            auto & c = profile_counts[il];
            c.assign(n, 0);
            for (int64_t e = 0; e < n && ss; ++e) {
                ss >> c[e];
            }
            n_read++;
        }
        LLAMA_LOG_INFO("%s: expert profile %s: %zu layers\n", __func__, params.profile.c_str(), n_read);
    }

    void save_profile(bool verbose = true) const {
        const std::string & path = params.profile_save;
        if (path.empty() || host == nullptr) {
            return;
        }
        // written next to the target and renamed, so a reader never sees half a file
        const std::string tmp = path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "w");
        if (f == nullptr) {
            LLAMA_LOG_WARN("%s: cannot write the expert profile to %s\n", __func__, path.c_str());
            return;
        }
        fprintf(f, "# expert handoff routing counts, arch %s, %zu layers\n", llm_arch_name(model.arch), model.layers.size());
        const std::vector<int32_t> dc = device_counts();
        for (size_t il = 0; il < layers.size(); ++il) {
            if (layers[il].ig < 0) {
                continue;
            }
            const int32_t n = groups[layers[il].ig].n_expert;
            fprintf(f, "layer %zu %d", il, n);
            for (int32_t e = 0; e < n; ++e) {
                const size_t ic = il*counts_stride + e;
                uint64_t v = ic < dc.size() ? (uint32_t) dc[ic] : 0;
                if (il < profile_counts.size() && e < (int32_t) profile_counts[il].size()) {
                    v += profile_counts[il][e];
                }
                fprintf(f, " %" PRIu64, v);
            }
            fprintf(f, "\n");
        }
        fclose(f);
        std::remove(path.c_str());
        if (std::rename(tmp.c_str(), path.c_str()) != 0) {
            LLAMA_LOG_WARN("%s: cannot replace the expert profile %s\n", __func__, path.c_str());
            return;
        }
        if (verbose) {
            LLAMA_LOG_INFO("%s: expert profile written to %s\n", __func__, path.c_str());
        }
    }

    bool use(int32_t il, int64_t n_tokens, const llama_expert_handoff_form & form) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0 || n_tokens <= 0 || n_tokens > dims.max_tokens) {
            return false;
        }
        if (form.act < 0) {
            return false;
        }
        if (model.hparams.no_alloc) {
            // measuring only: the same graph, nothing read from the weights
            return true;
        }
        if (host == nullptr) {
            return false;
        }
        const layer & l = layers[il];
        if (n_tokens > n_small && !host_sooner(il, n_tokens)) {
            return false;
        }
        std::lock_guard<std::mutex> lk(mtx);
        if (l.state == 0) {
            const llama_layer & ml = model.layers[il];
            const int64_t n_expert = groups[l.ig].n_expert;
            bool ok = true;
            l.gate_s = eh_read_scale(ml.ffn_gate_exps_s, n_expert, ok);
            l.up_s   = eh_read_scale(ml.ffn_up_exps_s,   n_expert, ok);
            l.down_s = eh_read_scale(ml.ffn_down_exps_s, n_expert, ok);
            if (ok) {
                // the biases are host weights in the experts' list (eh_layer_experts), contiguous [n, n_expert] F32
                ggml_expert_handoff_layer hl = {};
                hl.gate          = ml.ffn_gate_exps;
                hl.up            = ml.ffn_up_exps;
                hl.gate_up       = ml.ffn_gate_up_exps;
                hl.down          = ml.ffn_down_exps;
                hl.gate_scale    = l.gate_s.empty() ? nullptr : l.gate_s.data();
                hl.up_scale      = l.up_s.empty()   ? nullptr : l.up_s.data();
                hl.down_scale    = l.down_s.empty() ? nullptr : l.down_s.data();
                hl.gate_bias     = ml.ffn_gate_exps_b ? (const float *) ml.ffn_gate_exps_b->data : nullptr;
                hl.up_bias       = ml.ffn_gate_up_exps_b ? (const float *) ml.ffn_gate_up_exps_b->data :
                                   ml.ffn_up_exps_b      ? (const float *) ml.ffn_up_exps_b->data : nullptr;
                hl.down_bias     = ml.ffn_down_exps_b ? (const float *) ml.ffn_down_exps_b->data : nullptr;
                hl.act           = (ggml_expert_handoff_act) form.act;
                hl.alpha         = form.alpha;
                hl.limit         = form.limit;
                hl.beta          = form.beta;
                hl.lb            = form.lb;
                hl.weight_before = form.weight_before;
                hl.n_used        = form.n_used;
                hl.miss_slot     = groups[l.ig].n_slots;
                // [TAG_EH_RESIDENT]
                hl.gate_src      = resident_of(il, ml.ffn_gate_exps);
                hl.up_src        = resident_of(il, ml.ffn_up_exps);
                hl.gate_up_src   = resident_of(il, ml.ffn_gate_up_exps);
                hl.down_src      = resident_of(il, ml.ffn_down_exps);
                ok = api.set_layer(host, il, &hl);
            }
            l.state = ok ? 1 : -1;
            l.form  = form;
            if (!ok) {
                LLAMA_LOG_WARN("%s: layer %d: the host cannot compute these experts, the layer runs without the handoff\n", __func__, il);
            }
        }
        return l.state == 1 && l.form == form;
    }

    // a batch of n_tokens in layer il: the host computing its misses vs the device uploading the experts it lacks
    // (those the batch routes to, uniform routing assumed), whichever is estimated sooner; the join's wait stays well
    // inside its timeout
    bool host_sooner(int32_t il, int64_t n_tokens) const {
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        int64_t n_cached = 0;
        for (int32_t s : l.slots) {
            n_cached += s < g.n_slots;
        }
        const double n_unc = (double) (g.n_expert - n_cached);
        if (n_unc <= 0.0) {
            return true; // nothing to upload, nothing for the host
        }
        // the host's throughput as measured so far (its experts may page in from disk); until it has computed some,
        // the batch takes the usual path
        ggml_expert_handoff_stats st = {};
        api.get_stats(host, &st);
        if (st.n_host < 1000 || st.busy_ns == 0) {
            return false;
        }
        const double host_bps = std::max(1e9, (double) st.n_host*g.slot_bytes/(st.busy_ns*1e-9));
        const double k    = (double) model.hparams.n_expert_used(il);
        const double miss = est_miss >= 0.0 ? est_miss : 0.5*n_unc/g.n_expert;
        const double used = 1.0 - std::pow(1.0 - k/g.n_expert, (double) n_tokens);
        const double t_host   = (double) n_tokens*k*miss*g.slot_bytes/host_bps;
        const double t_upload = n_unc*used*g.slot_bytes/upload_bps_default;
        return t_host < t_upload && t_host < 0.2*timeout_ms*1e-3;
    }

    int64_t copy_experts(ggml_backend_t be, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) const {
        // any backend of the cache's device (an MTP draft context has its own), copying on its stream
        const auto it = bindings.find(w);
        if (it == bindings.end() || ggml_backend_get_device(be) != dev_of(it->second.il).dev) {
            return 0;
        }
        const binding & b = it->second;
        const layer & l = layers[b.il];
        const group & g = groups[l.ig];
        if (l.slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && l.slots[e + n] == l.slots[e] + n) {
            n++;
        }
        ggml_tensor * bank = g.banks[b.ip];
        if (!ctx_views) {
            ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
            ctx_views.reset(ggml_init(ip));
        }
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], l.slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(be, be, src_view, dst_view);
        return n;
    }

    // [TAG_EH_COPY_ENGINE] before a graph: plan the uploads of its scheduler (see plan above)
    void uploads_begin(ggml_backend_sched_t sched, ggml_cgraph * gf) const {
        std::lock_guard<std::mutex> lk(plan_mtx);
        if (plan_owner != nullptr || engines.empty()) {
            return; // another context's graph holds the plan (an MTP draft's graphs never upload experts)
        }
        plan.clear();
        plan_next = 0;
        plan_all  = false;
        const int n = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n; ++i) {
            const ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            // the host experts this node reads: the tensor itself (not yet split), or the one the scheduler copies
            const ggml_tensor * w = node->src[0];
            if (!eh_is_host_weight(w)) {
                const char * a = strchr(w->name, '#');
                const char * b = strrchr(w->name, '#');
                const auto   f = a != nullptr && b > a ? by_name.find(std::string(a + 1, b)) : by_name.end();
                if (f == by_name.end()) {
                    continue; // the cache's banks, or a tensor of another kind
                }
                w = f->second;
            }
            const auto it = bindings.find(w);
            if (it == bindings.end()) {
                continue;
            }
            const int32_t id = groups[layers[it->second.il].ig].id;
            ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched, const_cast<ggml_tensor *>(node));
            if (engines[id] == nullptr || be == nullptr || ggml_backend_get_device(be) != devices[id].dev) {
                continue;
            }
            plan_item pi;
            pi.w   = w;
            pi.il  = it->second.il;
            pi.ip  = it->second.ip;
            pi.be  = be;
            pi.eng = engines[id];
            plan.push_back(std::move(pi));
            // ids [n_used, n_tokens]: at 4 routings per expert, ~98% of the experts are routed to
            const ggml_tensor * ids = node->src[2];
            plan_all = plan_all || ids->ne[0]*ids->ne[1] >= 4*w->ne[2];
        }
        if (plan.empty()) {
            return;
        }
        if (n_up_graphs == 0) {
            LLAMA_LOG_WARN("%s: the first graph through the upload engine: %zu expert tensors, %s\n", __func__, plan.size(),
                plan_all ? "every uncached expert queued at once" : "each layer's routed experts");
        }
        plan_owner = sched;
        n_up_graphs++;
        if (plan_all) {
            for (plan_item & pi : plan) {
                queue(pi, nullptr);
            }
        }
    }

    // under plan_mtx: the experts of pi outside VRAM (those used, with used) go to its engine
    void queue(plan_item & pi, const std::vector<bool> * used) const {
        const layer & l = layers[pi.il];
        const group & g = groups[l.ig];
        const size_t es = eh_expert_stride(pi.w);
        pi.piece.assign(g.n_expert, -1);
        for (int64_t e = 0; e < g.n_expert; ++e) {
            if (l.slots[e] < g.n_slots || (used != nullptr && !(*used)[e])) {
                continue;
            }
            const void * r = resident(pi.il, (size_t) pi.ip, e); // [TAG_EH_RESIDENT]
            pi.piece[e] = uapi.push(pi.eng, r ? r : (const uint8_t *) pi.w->data + (size_t) e*es, es);
        }
        pi.queued = true;
    }

    // the scheduler's upload of w into dst (its copy on be): true if the engine did it. Without used (the routing,
    // read from the device) only a batch that routes to every expert can be served
    bool upload(ggml_backend_t be, const ggml_tensor * w, ggml_tensor * dst, const std::vector<bool> * used) const {
        std::lock_guard<std::mutex> lk(plan_mtx);
        if (plan_owner == nullptr || (used == nullptr && !plan_all)) {
            return false;
        }
        size_t k = plan_next;
        while (k < plan.size() && plan[k].w != w) {
            k++;
        }
        if (k == plan.size() || plan[k].be != be) {
            n_up_fallback++;
            return false;
        }
        plan_next = k + 1;
        plan_item & pi = plan[k];
        if (!pi.queued) {
            // the layer's routing is known now: its tensors from here on (their gathers follow in this order)
            for (size_t j = k; j < plan.size() && plan[j].il == pi.il; ++j) {
                if (!plan[j].queued) {
                    queue(plan[j], plan_all ? nullptr : used);
                }
            }
        }
        const layer & l = layers[pi.il];
        const group & g = groups[l.ig];
        const ggml_tensor * bank = g.banks[pi.ip];
        const size_t es = eh_expert_stride(w);
        copies.clear();
        for (int64_t e = 0; e < g.n_expert; ++e) {
            const bool u = plan_all || (*used)[e];
            if (!u) {
                // MMQ reads a little past the last expert it uses: zeros there, not garbage that may be NaN
                if (e > 0 && (*used)[e - 1]) {
                    copies.push_back({ -1, nullptr, (size_t) e*es, std::min<size_t>(es, 512) });
                }
                continue;
            }
            if (l.slots[e] < g.n_slots) {
                copies.push_back({ -1, (const uint8_t *) bank->data + (size_t) l.slots[e]*es, (size_t) e*es, es });
            } else if (pi.piece[e] >= 0) {
                copies.push_back({ pi.piece[e], nullptr, (size_t) e*es, es });
            } else {
                n_up_fallback++;
                return false; // not queued (a different routing than its layer's first tensor)
            }
        }
        return uapi.gather(pi.eng, be, dst->data, copies.data(), (int) copies.size());
    }

    void uploads_end(ggml_backend_sched_t sched) const {
        std::lock_guard<std::mutex> lk(plan_mtx);
        if (plan_owner != sched) {
            return;
        }
        for (ggml_upload_engine_t e : engines) {
            if (e) {
                uapi.reset(e);
            }
        }
        plan.clear();
        plan_owner = nullptr;
    }

    // the helper thread: copies the experts of in_flight from host memory into their slots. The copies go through
    // the device's per-thread stream (ggml_backend_tensor_set), beside the graphs; nothing reads those slots
    // until adapt() admits them.
    void helper_main() const {
        std::unique_lock<std::mutex> lk(helper_mtx);
        while (true) {
            helper_cv.wait(lk, [&] { return helper_job || helper_stop; });
            if (helper_stop) {
                return;
            }
            helper_job = false;
            lk.unlock();
            for (const swap & s : in_flight) {
                const layer & l = layers[s.il];
                const group & g = groups[l.ig];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t es = eh_expert_stride(w);
                    const void * r = resident(s.il, ip, s.in); // [TAG_EH_RESIDENT]
                    ggml_backend_tensor_set(g.banks[ip], r ? r : (const uint8_t *) w->data + (size_t) s.in*es, (size_t) s.slot*es, es);
                }
            }
            // [TAG_EH_RESIDENT] into the resident copy: slots nothing reads until adapt() admits them
            for (const ram_swap & s : ram_in_flight) {
                const layer & l = layers[s.il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    if (s.ptr[ip] != nullptr) {
                        const ggml_tensor * w = l.experts[ip];
                        const size_t es = eh_expert_stride(w);
                        memcpy((uint8_t *) res_mem + ((const uint8_t *) s.ptr[ip] - (const uint8_t *) res_mem), (const uint8_t *) w->data + (size_t) s.in*es, es);
                    }
                }
            }
            helper_busy.store(false, std::memory_order_release);
            lk.lock();
        }
    }

    void upload_slot_maps(std::vector<int32_t> & touched) const {
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        for (int32_t il : touched) {
            const layer & l = layers[il];
            ggml_backend_tensor_set(l.slot_map, l.slots.data(), 0, l.slots.size()*sizeof(int32_t));
        }
    }

    // Every adapt_every decode graphs: the routing counts since the last round are added to the decayed usage; in
    // each layer, the most used experts the cache misses (used at least twice) replace the least used ones it holds
    // while the gain is at least 1.5 uses; the best adapt_swaps of all layers go. The evicted experts leave the slot
    // maps at once, so the graphs from here on send them to the host, and the device finishes the graphs that may
    // still read their slots; then the helper copies the new experts in. A later call admits them once copied.
    void adapt(int64_t n_tokens) const {
        if (params.adapt_every <= 0 || params.adapt_swaps <= 0 || host == nullptr ||
            n_tokens <= 0 || n_tokens > n_small) {
            return;
        }
        n_rounds++;

        // the last round's copies have landed: they serve from the next graph on
        if (!in_flight.empty() || !ram_in_flight.empty()) {
            if (helper_busy.load(std::memory_order_acquire)) {
                return;
            }
            // [TAG_EH_RESIDENT] the host pool reads the slot arrays only while it answers a graph's layers
            for (const ram_swap & s : ram_in_flight) {
                const layer & l = layers[s.il];
                for (size_t ip = 0; ip < l.res.size(); ++ip) {
                    if (s.ptr[ip] != nullptr) {
                        l.res[ip][s.in] = s.ptr[ip];
                    }
                }
            }
            n_ram_swaps += ram_in_flight.size();
            ram_in_flight.clear();
            std::lock_guard<std::mutex> lk(mtx);
            std::vector<int32_t> touched;
            for (const swap & s : in_flight) {
                layers[s.il].slots[s.in] = s.slot;
                touched.push_back(s.il);
            }
            upload_slot_maps(touched);
            n_swaps += in_flight.size();
            in_flight.clear();
        }
        if (n_rounds % params.adapt_every != 0) {
            return;
        }

        const std::vector<int32_t> dc = device_counts();
        if (dc.empty()) {
            return;
        }
        if (usage.empty()) {
            usage.assign(dc.size(), 0.0f);
            ram_usage.assign(dc.size(), 0.0f);
            counts_seen.assign(dc.size(), 0);
        }
        for (size_t i = 0; i < dc.size(); ++i) {
            const float d = (float) (uint32_t) (dc[i] - counts_seen[i]);
            usage[i]     += d;
            ram_usage[i] += d;
            counts_seen[i] = dc[i];
        }

        struct cand { float gain; int32_t il, in, out; };
        std::vector<cand> swaps;
        std::vector<std::pair<float, int32_t>> miss, held;
        const float min_gain = 1.5f;
        for (int32_t il = 0; il < il_nextn && il < (int32_t) layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.ig < 0 || l.state != 1) {
                continue;
            }
            const group & g = groups[l.ig];
            const float * u = usage.data() + (size_t) il*counts_stride;
            miss.clear();
            held.clear();
            for (int32_t e = 0; e < g.n_expert; ++e) {
                if (l.slots[e] >= g.n_slots) {
                    if (u[e] >= 2.0f) {
                        miss.emplace_back(u[e], e);
                    }
                } else {
                    held.emplace_back(u[e], e);
                }
            }
            if (miss.empty() || held.empty()) {
                continue;
            }
            const size_t n = std::min(miss.size(), held.size());
            std::partial_sort(miss.begin(), miss.begin() + n, miss.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            std::partial_sort(held.begin(), held.begin() + n, held.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
            for (size_t i = 0; i < n && miss[i].first >= held[i].first + min_gain; ++i) {
                swaps.push_back({ miss[i].first - held[i].first, il, miss[i].second, held[i].second });
            }
        }
        // [TAG_EH_RESIDENT] the resident copy, per group (its experts share a size): the most used experts the host
        // reads from the tensor take the slots of the least used ones in the copy, and first those of experts the
        // VRAM cache holds by now (the host never reads them)
        std::vector<std::tuple<float, int32_t, int32_t, int32_t, int32_t>> ram_swaps; // gain, il in, in, il out, out
        if (res_mem != nullptr) {
            auto in_ram = [&](const layer & l, int32_t e) {
                for (const auto & r : l.res) {
                    if (!r.empty()) {
                        return r[e] != nullptr;
                    }
                }
                return false;
            };
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                const group & g = groups[ig];
                std::vector<std::tuple<float, int32_t, int32_t>> rin, rout; // usage, il, e
                for (int32_t il : g.layers) {
                    const layer & l = layers[il];
                    if (il >= il_nextn || l.state != 1 || l.res.empty()) {
                        continue;
                    }
                    const float * u = ram_usage.data() + (size_t) il*counts_stride;
                    for (int32_t e = 0; e < g.n_expert; ++e) {
                        const bool vram = l.slots[e] < g.n_slots;
                        if (in_ram(l, e)) {
                            rout.emplace_back(vram ? -1.0f : u[e], il, e);
                        } else if (!vram && u[e] >= 2.0f) {
                            rin.emplace_back(u[e], il, e);
                        }
                    }
                }
                const size_t n = std::min({ rin.size(), rout.size(), (size_t) params.adapt_swaps });
                std::partial_sort(rin.begin(),  rin.begin()  + n, rin.end(),  [](const auto & a, const auto & b) { return std::get<0>(a) > std::get<0>(b); });
                std::partial_sort(rout.begin(), rout.begin() + n, rout.end(), [](const auto & a, const auto & b) { return std::get<0>(a) < std::get<0>(b); });
                for (size_t i = 0; i < n && std::get<0>(rin[i]) >= std::get<0>(rout[i]) + min_gain; ++i) {
                    ram_swaps.emplace_back(std::get<0>(rin[i]) - std::get<0>(rout[i]), std::get<1>(rin[i]), std::get<2>(rin[i]),
                                           std::get<1>(rout[i]), std::get<2>(rout[i]));
                }
            }
            std::sort(ram_swaps.begin(), ram_swaps.end(), [](const auto & a, const auto & b) { return std::get<0>(a) > std::get<0>(b); });
            if ((int32_t) ram_swaps.size() > params.adapt_swaps) {
                ram_swaps.resize(params.adapt_swaps);
            }
            // evict: the host pool reads the tensor for those experts from the next graph on, and the helper fills
            // their slots for the experts coming in
            for (const auto & [gain, il_in, e_in, il_out, e_out] : ram_swaps) {
                const layer & lo = layers[il_out];
                ram_swap s;
                s.il = il_in;
                s.in = e_in;
                s.ptr.assign(lo.res.size(), nullptr);
                for (size_t ip = 0; ip < lo.res.size(); ++ip) {
                    if (!lo.res[ip].empty()) {
                        s.ptr[ip] = lo.res[ip][e_out];
                        lo.res[ip][e_out] = nullptr;
                    }
                }
                ram_in_flight.push_back(std::move(s));
            }
        }

        for (float & v : usage) {
            v *= params.adapt_decay;
        }
        for (float & v : ram_usage) {
            v *= ram_decay;
        }
        if (swaps.empty() && ram_swaps.empty()) {
            return;
        }
        std::sort(swaps.begin(), swaps.end(), [](const cand & a, const cand & b) { return a.gain > b.gain; });
        if ((int32_t) swaps.size() > params.adapt_swaps) {
            swaps.resize(params.adapt_swaps);
        }

        // evict: the victims go to the miss slot in the maps; the graphs queued before may still read their slots
        if (!swaps.empty()) {
            std::lock_guard<std::mutex> lk(mtx);
            std::vector<int32_t> touched;
            for (const cand & c : swaps) {
                const layer & l = layers[c.il];
                in_flight.push_back({ c.il, c.in, l.slots[c.out] });
                l.slots[c.out] = groups[l.ig].n_slots;
                touched.push_back(c.il);
            }
            upload_slot_maps(touched);
        }
        for (const device & d : devices) {
            if (d.ctx) {
                ggml_backend_synchronize(d.backend);
            }
        }

        if (!helper.joinable()) {
            helper = std::thread([this] { helper_main(); });
        }
        helper_busy.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(helper_mtx);
            helper_job = true;
        }
        helper_cv.notify_one();
    }

    void check() const {
        if (mailbox == nullptr) {
            return;
        }
        const volatile ggml_expert_handoff_header * hdr = (const volatile ggml_expert_handoff_header *) mailbox;
        const uint32_t n = hdr->n_timeouts;
        if (n != timeouts_seen) {
            LLAMA_LOG_ERROR("%s: %u layer(s) gave up waiting for the host experts (%d ms each); their output lacks those experts\n",
                __func__, n - timeouts_seen, timeout_ms);
            timeouts_seen = n;
            // [TAG_EH_TIMEOUT_DIAG] whether the host was slow, never saw the send, or answered another one
            if (host != nullptr) {
                ggml_expert_handoff_stats st = {};
                api.get_stats(host, &st);
                LLAMA_LOG_ERROR("%s: the last one: layer %d waited for send %u, the host's last answer was %u (now: send %u, answer %u); "
                    "host: %" PRIu64 " sends never taken up, %" PRIu64 " multi-token answers at %.1f ms, slowest %.1f ms "
                    "(layer %d, %d tokens, %d misses of %d experts; inputs %.1f, gate/up %.1f, act %.1f, down %.1f ms)\n",
                    __func__, (int) hdr->to_layer, (unsigned) hdr->to_want, (unsigned) hdr->to_done,
                    (unsigned) hdr->seq_send, (unsigned) hdr->seq_done, st.n_skipped, st.n_multi,
                    st.n_multi ? st.busy_ns_multi/1e6/st.n_multi : 0.0, st.max_ns/1e6, st.max_layer, st.max_tokens,
                    st.max_miss, st.max_uniq, st.max_step_ns[0]/1e6, st.max_step_ns[1]/1e6, st.max_step_ns[2]/1e6,
                    st.max_step_ns[3]/1e6);
            }
        }

        // every 30 s of use: the hit rate and the host's time per layer since the last report
        const int64_t now = ggml_time_us();
        if (host != nullptr && now - t_report > 30*1000*1000) {
            ggml_expert_handoff_stats st = {};
            api.get_stats(host, &st);
            uint64_t n_routed = 0;
            for (int32_t c : device_counts()) {
                n_routed += (uint32_t) c;
            }
            const uint64_t d_routed = n_routed - last.n_routed;
            const uint64_t d_host   = st.n_host - last.n_host;
            const uint64_t d_layers = st.n_layers - last.n_layers;
            const uint64_t d_busy   = st.busy_ns - last.busy_ns;
            if (d_routed > 0) {
                // WARN, not INFO: the server's log drops libllama's INFO lines at its default verbosity, and this is the one
                // line that shows how the cache does (owner, 2026-11-07: tuning Flash-Next in prod)
                const uint64_t d_multi = st.n_multi - last.n_multi;
                LLAMA_LOG_WARN("%s: %" PRIu64 " routed experts, VRAM hit rate %.1f%%, %" PRIu64 " layers sent to the host, %.0f us each "
                    "(%" PRIu64 " of them multi-token, %.0f us each), %" PRIu64 " experts swapped in\n",
                    __func__, d_routed, 100.0*(1.0 - (double) d_host/d_routed), d_layers, d_layers ? d_busy/1000.0/d_layers : 0.0,
                    d_multi, d_multi ? (st.busy_ns_multi - last.busy_ns_multi)/1000.0/d_multi : 0.0, n_swaps - n_swaps_reported);
                if (res_mem != nullptr) {
                    LLAMA_LOG_WARN("%s: %" PRIu64 " experts swapped into the resident RAM copy\n", __func__, n_ram_swaps - n_ram_swaps_reported);
                }
                // [TAG_EH_LOCAL_SUM] where the host's time goes, per layer it computed
                if (d_layers > 0) {
                    double sp[5];
                    for (int k = 0; k < 5; ++k) {
                        sp[k] = (st.step_ns[k] - last.step_ns[k])/1000.0/d_layers;
                    }
                    const uint64_t d_copy = st.n_read_copy - last.n_read_copy;
                    const uint64_t d_tens = st.n_read_tensor - last.n_read_tensor;
                    LLAMA_LOG_WARN("%s: host steps per layer: send %.0f, inputs %.0f, gate/up %.0f, activation %.0f, down %.0f us; "
                        "%.2f missed experts per layer, %.1f%% of the distinct ones read from the resident copy\n", __func__,
                        sp[0], sp[1], sp[2], sp[3], sp[4], (double) d_host/d_layers,
                        d_copy + d_tens ? 100.0*d_copy/(d_copy + d_tens) : 0.0);
                    const uint64_t d_ngap = st.n_gap - last.n_gap;
                    LLAMA_LOG_WARN("%s: host gather %.0f us per layer; the device's time from an answer to its next send %.0f us\n",
                        __func__, (st.gather_ns - last.gather_ns)/1000.0/d_layers,
                        d_ngap ? (st.gap_ns - last.gap_ns)/1000.0/d_ngap : 0.0);
                }
            }
            // [TAG_EH_COPY_ENGINE] the uploads of large batches since the last report
            {
                ggml_upload_engine_stats us = {};
                for (ggml_upload_engine_t e : engines) {
                    if (e) {
                        ggml_upload_engine_stats s = {};
                        uapi.stats(e, &s);
                        us.n_pieces += s.n_pieces; us.n_dma += s.n_dma; us.bytes += s.bytes; us.n_gathers += s.n_gathers;
                        us.copy_ns += s.copy_ns; us.wait_ring_ns += s.wait_ring_ns; us.wait_copy_ns += s.wait_copy_ns;
                        us.wait_gather_ns += s.wait_gather_ns; us.dropped += s.dropped;
                    }
                }
                if (us.bytes > up_last.bytes) {
                    LLAMA_LOG_WARN("%s: uploads: %" PRIu64 " graphs, %.2f GB in %" PRIu64 " DMAs (%" PRIu64 " experts), %" PRIu64
                        " gathers; host copies %.1f s (all threads), the issuer waited %.1f s for copies and %.1f s for ring "
                        "space, the graphs %.1f s for DMAs; %" PRIu64 " experts dropped, %" PRIu64 " uploads not planned\n",
                        __func__, n_up_graphs - n_up_graphs_last, (us.bytes - up_last.bytes)/1e9, us.n_dma - up_last.n_dma,
                        us.n_pieces - up_last.n_pieces, us.n_gathers - up_last.n_gathers, (us.copy_ns - up_last.copy_ns)/1e9,
                        (us.wait_copy_ns - up_last.wait_copy_ns)/1e9, (us.wait_ring_ns - up_last.wait_ring_ns)/1e9,
                        (us.wait_gather_ns - up_last.wait_gather_ns)/1e9, us.dropped - up_last.dropped, n_up_fallback);
                }
                up_last = us;
                n_up_graphs_last = n_up_graphs;
            }
            n_swaps_reported = n_swaps;
            n_ram_swaps_reported = n_ram_swaps;
            // the estimates of host_sooner(): the share of routed experts the host computed, its bytes per second
            if (n_routed > 10000) {
                est_miss = (double) st.n_host/n_routed;
            }
            last = st;
            last.n_routed = n_routed;
            t_report = now;
            // keep what was learned if the process ends without a clean exit
            save_profile(false);
        }
    }
};

llama_expert_handoff::llama_expert_handoff(const llama_model & model, const std::vector<ggml_backend_t> & backends,
        const std::vector<ggml_backend_buffer_type_t> & bufts, const llama_expert_handoff_params & params) :
    pimpl(new impl(model, backends, bufts, params)) {
}

llama_expert_handoff::~llama_expert_handoff() = default;

bool llama_expert_handoff::use(int32_t il, int64_t n_tokens, const llama_expert_handoff_form & form) const {
    return pimpl->use(il, n_tokens, form);
}

ggml_backend_dev_t llama_expert_handoff::device(int32_t il) const {
    const auto & p = *pimpl;
    return il >= 0 && il < (int32_t) p.layers.size() && p.layers[il].ig >= 0 ? p.dev_of(il).dev : nullptr;
}

ggml_tensor * llama_expert_handoff::slot_map(int32_t il) const {
    return il >= 0 && il < (int32_t) pimpl->layers.size() ? pimpl->layers[il].slot_map : nullptr;
}

ggml_tensor * llama_expert_handoff::get_experts(const ggml_tensor * w) const {
    const auto it = pimpl->bindings.find(w);
    return it != pimpl->bindings.end() ? it->second.cached : nullptr;
}

ggml_expert_handoff_params llama_expert_handoff::params(int32_t il) const {
    const auto & p = *pimpl;
    ggml_expert_handoff_params res = {};
    res.mailbox       = p.mailbox;
    res.layer         = il;
    res.max_tokens    = p.dims.max_tokens;
    res.max_embd      = p.dims.max_embd;
    res.max_used      = p.dims.max_used;
    res.miss_slot     = p.layers[il].ig >= 0 ? p.groups[p.layers[il].ig].n_slots : 0;
    // a model loaded only to measure memory has no data: any address does, the graph never runs
    const auto & d = p.dev_of(il);
    res.flags         = d.flags->data  ? (int32_t *) d.flags->data  : (int32_t *) p.mailbox;
    res.counts        = d.counts->data ? (int32_t *) d.counts->data : nullptr;
    res.counts_stride = p.counts_stride;
    res.timeout_ms    = p.timeout_ms;
    return res;
}

int64_t llama_expert_handoff::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) const {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

const void * llama_expert_handoff::host_source(const ggml_tensor * w, int64_t e, int64_t last, int64_t & n) const {
    return pimpl->host_source(w, e, last, n);
}

void llama_expert_handoff::uploads_begin(ggml_backend_sched_t sched, ggml_cgraph * gf) const {
    pimpl->uploads_begin(sched, gf);
}

bool llama_expert_handoff::upload(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, const std::vector<bool> * used) const {
    return pimpl->upload(backend, w, dst, used);
}

void llama_expert_handoff::uploads_end(ggml_backend_sched_t sched) const {
    pimpl->uploads_end(sched);
}

void llama_expert_handoff::wake() const {
    if (pimpl->host) {
        pimpl->api.wake(pimpl->host);
    }
}

void llama_expert_handoff::idle() const {
    if (pimpl->host) {
        pimpl->api.idle(pimpl->host);
    }
}

void llama_expert_handoff::check() const {
    pimpl->check();
}

void llama_expert_handoff::adapt(int64_t n_tokens) const {
    pimpl->adapt(n_tokens);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_expert_handoff::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += ggml_backend_buffer_get_size(d.buf.get());
        }
    }
    return res;
}

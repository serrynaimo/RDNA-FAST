// ROCmFPX [TAG_EH_COPY_ENGINE] the upload engine: expert bytes from pageable host memory into a VRAM ring ahead of the
// graph, then gathered into the tensors the graph reads (protocol in ggml-expert-handoff.h).
//
// Why this shape (Flash-Next Q2_0 on an RX 7900 XT under Windows, 2026-11-08): a copy from pageable memory makes the
// HIP runtime stage it through its own small buffers while the calling thread waits, ~1 GB/s for the experts of a
// prompt, and pinning the 15 GiB the experts live in is not an option on Windows (minutes, then failed launches).
// So the bytes go through a few pinned buffers instead: host threads memcpy runs of pieces into them, a thread of its
// own issues one DMA per buffer on a stream of its own, and the graph waits on events only where it reads. The ring
// lets the copies of later layers run while earlier layers compute; the graph's thread never waits for a memcpy.
// Under WDDM an event record costs ~80 us and a wait between streams ~45 us, so the engine records one event per DMA
// (a run of up to stage_bytes) and one per gather, never per piece.

#include "upload-engine.cuh"
#include "expert-handoff.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

// the copy threads wait for DMAs on these events: sleeping, not spinning on the cores that copy
#if defined(GGML_USE_HIP)
#define UE_EVENT_SLEEP hipEventBlockingSync
#else
#define UE_EVENT_SLEEP cudaEventBlockingSync
#endif

namespace {

struct ue_entry {
    const uint8_t * src;   // NULL: zeros
    uint8_t       * dst;
    uint64_t        bytes;
};

constexpr int64_t ue_chunk = 64*1024; // bytes per block of the gather kernel

static __global__ void k_ue_gather(const ue_entry * __restrict__ tab) {
    __shared__ ue_entry e;
    if (threadIdx.x == 0) {
        e = tab[blockIdx.x]; // mapped host memory: read once per block
    }
    __syncthreads();
    const uint64_t b0 = (uint64_t) blockIdx.y*ue_chunk;
    if (b0 >= e.bytes) {
        return;
    }
    const uint64_t b1 = min(e.bytes, b0 + (uint64_t) ue_chunk);
    const bool aligned = ((uintptr_t) e.src % 16 == 0) && ((uintptr_t) e.dst % 16 == 0) && (b0 % 16 == 0);
    uint64_t b = b0;
    if (aligned) {
        const uint64_t n16 = (b1 - b0)/16;
        int4 * d = (int4 *) (e.dst + b0);
        if (e.src != nullptr) {
            const int4 * s = (const int4 *) (e.src + b0);
            for (uint64_t i = threadIdx.x; i < n16; i += blockDim.x) {
                d[i] = s[i];
            }
        } else {
            const int4 z = make_int4(0, 0, 0, 0);
            for (uint64_t i = threadIdx.x; i < n16; i += blockDim.x) {
                d[i] = z;
            }
        }
        b = b0 + n16*16;
    }
    for (uint64_t i = b + threadIdx.x; i < b1; i += blockDim.x) {
        e.dst[i] = e.src != nullptr ? e.src[i] : 0;
    }
}

int64_t ue_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

struct ggml_upload_engine {
    struct piece {
        const uint8_t * src;
        size_t  bytes;
        int64_t job;
        size_t  job_off;
    };
    struct job {
        int64_t  p0 = 0, p1 = 0;  // its pieces [p0, p1)
        size_t   bytes = 0;
        size_t   ring_off = 0;
        uint64_t vstart = 0;      // its ring positions (the ring's bytes counted from the start, wraps included)
        uint64_t vend = 0;
        bool     closed = false;  // no more pieces: the copy threads may take it
        bool     copied = false;  // in its staging buffer
    };
    // a gather's end: the ring before position v and the table before tpos are free once ev has completed
    struct done {
        uint64_t    v;
        uint64_t    tpos;
        cudaEvent_t ev;
    };

    int device = 0;
    ggml_upload_engine_params p = {};

    uint8_t *              ring = nullptr;
    std::vector<uint8_t *> stage;
    cudaStream_t           stream = nullptr;
    static constexpr int   n_job_ev = 1024;
    std::vector<cudaEvent_t> job_ev;     // [job % n_job_ev]: the job's DMA is done
    std::vector<cudaEvent_t> free_ev;    // for the gathers' done records

    // the gather tables, mapped host memory the kernel reads
    ue_entry * tab   = nullptr;
    static constexpr int64_t n_tab = 1 << 16;
    uint64_t   tpos  = 0;                // entries written so far

    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<piece>       pieces;      // pieces[i - p_base]
    int64_t                 p_base = 0;
    std::deque<job>         jobs;        // jobs[j - j_base]
    int64_t                 j_base = 0;
    int64_t                 n_jobs = 0;      // jobs made (the last may be open)
    int64_t                 n_claimed = 0;   // taken by a copy thread
    int64_t                 n_issued = 0;    // DMA queued
    int64_t                 claim_limit = INT64_MAX; // reset(): no claims at or past this
    uint64_t                vhead = 0;       // ring position of the next job
    int64_t                 p_consumed = 0;  // pieces before this are gathered or dropped
    int64_t                 j_consumed = 0;  // the first job with a piece not yet gathered
    std::deque<done>        dones;
    bool                    stop = false;

    std::vector<std::thread> threads;
    std::thread              issuer;

    std::atomic<uint64_t> st_pieces{0}, st_dma{0}, st_bytes{0}, st_gathers{0}, st_copy{0}, st_wait_ring{0},
                          st_wait_copy{0}, st_wait_gather{0}, st_dropped{0};

    piece & P(int64_t i) { return pieces[(size_t) (i - p_base)]; }
    job   & J(int64_t j) { return jobs[(size_t) (j - j_base)]; }

    bool init(int dev, const ggml_upload_engine_params & params) {
        device = dev;
        p = params;
        p.stage_bytes = (p.stage_bytes + 255) & ~(size_t) 255;
        // a gather part (half the ring) plus the jobs it straddles and a wrap must fit, or a gather would wait for
        // ring space only its own reads free
        p.ring_bytes  = std::max(p.ring_bytes, 8*p.stage_bytes);
        p.n_stage     = std::max(2, p.n_stage);
        p.n_threads   = std::max(1, p.n_threads);
        ggml_cuda_set_device(device);
        if (cudaMalloc((void **) &ring, p.ring_bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_ERROR("%s: failed to allocate a %.0f MiB ring in VRAM\n", __func__, p.ring_bytes/1048576.0);
            ring = nullptr;
            return false;
        }
        for (int i = 0; i < p.n_stage; ++i) {
            void * b = nullptr;
            if (cudaHostAlloc(&b, p.stage_bytes, cudaHostAllocPortable) != cudaSuccess) {
                (void) cudaGetLastError();
                GGML_LOG_ERROR("%s: failed to allocate %d pinned buffers of %.1f MiB\n", __func__, p.n_stage, p.stage_bytes/1048576.0);
                return false;
            }
            stage.push_back((uint8_t *) b);
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        job_ev.resize(n_job_ev);
        for (cudaEvent_t & e : job_ev) {
            CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming | UE_EVENT_SLEEP));
        }
        tab = (ue_entry *) ggml_cuda_host_mapped_alloc(n_tab*sizeof(ue_entry));
        if (tab == nullptr) {
            return false;
        }
        for (int i = 0; i < p.n_threads; ++i) {
            threads.emplace_back([this] { copy_main(); });
        }
        issuer = std::thread([this] { issue_main(); });
        return true;
    }

    ~ggml_upload_engine() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
        }
        cv.notify_all();
        for (std::thread & t : threads) {
            t.join();
        }
        if (issuer.joinable()) {
            issuer.join();
        }
        ggml_cuda_set_device(device);
        if (stream) {
            (void) cudaStreamSynchronize(stream);
        }
        (void) cudaDeviceSynchronize(); // the gathers' reads of the ring and the tables
        for (cudaEvent_t e : job_ev)  { (void) cudaEventDestroy(e); }
        for (cudaEvent_t e : free_ev) { (void) cudaEventDestroy(e); }
        for (const done & d : dones)  { if (d.ev) { (void) cudaEventDestroy(d.ev); } }
        for (uint8_t * b : stage) { (void) cudaFreeHost(b); }
        if (ring)   { (void) cudaFree(ring); }
        if (stream) { (void) cudaStreamDestroy(stream); }
        if (tab)    { ggml_cuda_host_mapped_free(tab); }
    }

    // under mtx: the open job (the last one, not closed) gets its place in the ring and goes to the copy threads
    void close_open() {
        if (n_jobs == 0) {
            return;
        }
        job & jb = J(n_jobs - 1);
        if (jb.closed) {
            return;
        }
        size_t off = (size_t) (vhead % p.ring_bytes);
        if (off + jb.bytes > p.ring_bytes) {
            vhead += p.ring_bytes - off; // the ring's tail stays unused this round
            off = 0;
        }
        jb.ring_off = off;
        jb.vstart   = vhead;
        vhead      += jb.bytes;
        jb.vend     = vhead;
        jb.closed   = true;
        cv.notify_all();
    }

    int64_t push(const void * src, size_t bytes) {
        if (bytes == 0 || bytes > p.stage_bytes) {
            return -1;
        }
        std::lock_guard<std::mutex> lk(mtx);
        const int64_t ip = p_base + (int64_t) pieces.size();
        if (n_jobs == 0 || J(n_jobs - 1).closed || J(n_jobs - 1).bytes + bytes > p.stage_bytes) {
            close_open();
            jobs.emplace_back();
            job & jb = jobs.back();
            jb.p0 = jb.p1 = ip;
            n_jobs++;
        }
        job & jb = J(n_jobs - 1);
        pieces.push_back({ (const uint8_t *) src, bytes, n_jobs - 1, jb.bytes });
        jb.p1     = ip + 1;
        jb.bytes += bytes;
        if (jb.bytes + 4096 > p.stage_bytes) {
            close_open();
        }
        return ip;
    }

    // the copy threads: a closed job into its staging buffer, once the DMA that last read the buffer is done
    void copy_main() {
        ggml_cuda_set_device(device);
        std::unique_lock<std::mutex> lk(mtx);
        for (;;) {
            cv.wait(lk, [&] {
                return stop || (n_claimed < n_jobs && n_claimed < claim_limit && J(n_claimed).closed &&
                                (n_claimed < p.n_stage || n_issued > n_claimed - p.n_stage));
            });
            if (stop) {
                return;
            }
            const int64_t j = n_claimed++;
            const int s = (int) (j % p.n_stage);
            // the buffer's last DMA (job j - n_stage) was queued; its event is not recorded again before job j is
            // issued (n_job_ev >= n_stage)
            cudaEvent_t prev = j >= p.n_stage ? job_ev[(size_t) ((j - p.n_stage) % n_job_ev)] : nullptr;
            struct cp { const uint8_t * src; size_t off; size_t bytes; };
            std::vector<cp> cps;
            for (int64_t i = J(j).p0; i < J(j).p1; ++i) {
                cps.push_back({ P(i).src, P(i).job_off, P(i).bytes });
            }
            lk.unlock();
            if (prev) {
                CUDA_CHECK(cudaEventSynchronize(prev));
            }
            const int64_t t0 = ue_now_ns();
            for (const cp & c : cps) {
                memcpy(stage[s] + c.off, c.src, c.bytes);
            }
            st_copy.fetch_add((uint64_t) (ue_now_ns() - t0), std::memory_order_relaxed);
            lk.lock();
            J(j).copied = true;
            cv.notify_all();
        }
    }

    // the issuer: each job's DMA in order, once it is copied and its ring space is free
    void issue_main() {
        ggml_cuda_set_device(device);
        std::unique_lock<std::mutex> lk(mtx);
        for (;;) {
            // counted as waiting for copies only while a job is queued (not between graphs)
            const bool queued = n_issued < n_jobs && J(n_issued).closed;
            int64_t t0 = ue_now_ns();
            cv.wait(lk, [&] { return stop || (n_issued < n_claimed && J(n_issued).copied); });
            if (stop) {
                return;
            }
            if (queued) {
                st_wait_copy.fetch_add((uint64_t) (ue_now_ns() - t0), std::memory_order_relaxed);
            }
            const int64_t j = n_issued;
            const job & jb = J(j);
            // ring space: the gathers of what was there before must have read it; and the job event's slot must be
            // past the gather that waited on its last use
            const uint64_t need = jb.vend > p.ring_bytes ? jb.vend - p.ring_bytes : 0;
            const int64_t  old  = j - n_job_ev;
            t0 = ue_now_ns();
            cv.wait(lk, [&] {
                return stop || ((need == 0 || (!dones.empty() && dones.back().v >= need)) &&
                                (old < 0 || p_consumed >= J(old).p1));
            });
            if (stop) {
                return;
            }
            st_wait_ring.fetch_add((uint64_t) (ue_now_ns() - t0), std::memory_order_relaxed);
            cudaEvent_t wait_ev = nullptr;
            if (need > 0) {
                // the oldest gather that freed enough; the older ones are implied by it
                while (dones.size() > 1 && dones[1].v >= need) {
                    if (dones.front().ev) {
                        free_ev.push_back(dones.front().ev);
                    }
                    dones.pop_front();
                }
                wait_ev = dones.front().ev;
            }
            uint8_t * dst = ring + jb.ring_off;
            const uint8_t * src = stage[(size_t) (j % p.n_stage)];
            const size_t bytes = jb.bytes;
            const int64_t np = jb.p1 - jb.p0;
            // outside the lock: under WDDM these calls take tens of microseconds each. The job stays (reset() waits
            // for it); if a gather recycles wait_ev meanwhile, the wait covers a later gather, which is only stricter
            lk.unlock();
            if (wait_ev) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, wait_ev, 0));
            }
            CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(job_ev[(size_t) (j % n_job_ev)], stream));
            lk.lock();
            n_issued = j + 1;
            st_pieces.fetch_add((uint64_t) np, std::memory_order_relaxed);
            st_dma.fetch_add(1, std::memory_order_relaxed);
            st_bytes.fetch_add(bytes, std::memory_order_relaxed);
            cv.notify_all();
            trim();
        }
    }

    // under mtx: forget the pieces and jobs nothing refers to any more (the last n_job_ev issued jobs stay: the
    // issuer looks at the one whose event slot it reuses)
    void trim() {
        while (!jobs.empty() && j_base < j_consumed && j_base < n_issued && j_base + n_job_ev < n_issued) {
            jobs.pop_front();
            j_base++;
        }
        const int64_t keep = jobs.empty() ? p_consumed : std::min(p_consumed, jobs.front().p0);
        while (!pieces.empty() && p_base < keep) {
            pieces.pop_front();
            p_base++;
        }
    }

    // under mtx: the ring position below which every job's pieces are all gathered
    uint64_t consumed_v() {
        while (j_consumed < n_jobs && J(j_consumed).closed && J(j_consumed).p1 <= p_consumed) {
            j_consumed++;
        }
        if (j_consumed >= n_jobs || !J(j_consumed).closed) {
            return vhead; // every closed job
        }
        return J(j_consumed).vstart;
    }

    bool gather(ggml_backend_cuda_context & ctx, void * dst, const ggml_upload_copy * copies, int n) {
        cudaStream_t cs = ctx.stream();
        std::unique_lock<std::mutex> lk(mtx);
        int i0 = 0;
        while (i0 < n) {
            // a part: up to half the ring of pieces and a table's worth of entries
            int64_t pmax = -1;
            size_t  pbytes = 0;
            int i1 = i0;
            while (i1 < n && (i1 - i0) < 4096) {
                const ggml_upload_copy & c = copies[i1];
                if (c.piece >= 0) {
                    if (c.piece < p_consumed || c.piece >= p_base + (int64_t) pieces.size()) {
                        GGML_LOG_ERROR("%s: piece %lld is not queued\n", __func__, (long long) c.piece);
                        return false;
                    }
                    if (pbytes > 0 && pbytes + P(c.piece).bytes > p.ring_bytes/2) {
                        break;
                    }
                    pbytes += P(c.piece).bytes;
                    pmax = std::max(pmax, c.piece);
                }
                i1++;
            }
            // its pieces' DMAs: the last one's event covers the others (one stream)
            if (pmax >= 0) {
                const int64_t jl = P(pmax).job;
                if (!J(jl).closed) {
                    close_open();
                }
                const int64_t t0 = ue_now_ns();
                cv.wait(lk, [&] { return stop || n_issued > jl; });
                if (stop) {
                    return false;
                }
                st_wait_gather.fetch_add((uint64_t) (ue_now_ns() - t0), std::memory_order_relaxed);
                CUDA_CHECK(cudaStreamWaitEvent(cs, job_ev[(size_t) (jl % n_job_ev)], 0));
            }
            // the table: its slots must not be read by a gather still in flight
            const int64_t nt = i1 - i0;
            uint64_t t = tpos;
            if (t % n_tab + nt > (uint64_t) n_tab) {
                t += n_tab - t % n_tab;
            }
            const uint64_t tneed = t + nt > (uint64_t) n_tab ? t + nt - n_tab : 0;
            for (const done & d : dones) {
                if (d.tpos >= tneed) {
                    if (tneed > 0 && d.ev) {
                        CUDA_CHECK(cudaEventSynchronize(d.ev));
                    }
                    break;
                }
            }
            ue_entry * te = tab + t % n_tab;
            uint64_t max_bytes = 0;
            for (int i = i0; i < i1; ++i) {
                const ggml_upload_copy & c = copies[i];
                ue_entry & e = te[i - i0];
                if (c.piece >= 0) {
                    const piece & pc = P(c.piece);
                    e.src   = ring + J(pc.job).ring_off + pc.job_off;
                    e.bytes = pc.bytes;
                } else {
                    e.src   = (const uint8_t *) c.src;
                    e.bytes = c.bytes;
                }
                e.dst = (uint8_t *) dst + c.dst_off;
                max_bytes = std::max<uint64_t>(max_bytes, e.bytes);
            }
            tpos = t + nt;
            if (nt > 0 && max_bytes > 0) {
                const dim3 grid((unsigned) nt, (unsigned) ((max_bytes + ue_chunk - 1)/ue_chunk), 1);
                k_ue_gather<<<grid, 256, 0, cs>>>(te);
                CUDA_CHECK(cudaGetLastError());
                st_gathers.fetch_add(1, std::memory_order_relaxed);
            }
            // what this part frees: its pieces and every piece pushed before them
            if (pmax >= 0) {
                p_consumed = std::max(p_consumed, pmax + 1);
            }
            cudaEvent_t ev = nullptr;
            if (!free_ev.empty()) {
                ev = free_ev.back();
                free_ev.pop_back();
            } else {
                CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventRecord(ev, cs));
            dones.push_back({ consumed_v(), tpos, ev });
            // keep the queue short: a record is implied by a later one once the issuer no longer needs to tell them apart
            while (dones.size() > 64) {
                if (dones.front().ev) {
                    free_ev.push_back(dones.front().ev);
                }
                dones.pop_front();
            }
            cv.notify_all();
            i0 = i1;
        }
        return true;
    }

    void reset() {
        std::unique_lock<std::mutex> lk(mtx);
        const int64_t n_pieces = p_base + (int64_t) pieces.size();
        if (n_pieces > p_consumed) {
            st_dropped.fetch_add((uint64_t) (n_pieces - p_consumed), std::memory_order_relaxed);
        }
        // no new claims. Everything pushed counts as read: the copies under way are issued into ring space nothing
        // will read, after the last gather's reads (its event) and after the DMAs before them (one stream). Their
        // ring space must be granted first, or the issuer would wait for gathers that never come.
        claim_limit = n_claimed;
        p_consumed  = n_pieces;
        if (!dones.empty()) {
            dones.back().v = std::max(dones.back().v, vhead);
        } else {
            dones.push_back({ vhead, tpos, nullptr });
        }
        cv.notify_all();
        cv.wait(lk, [&] { return stop || n_issued >= n_claimed; });
        // the jobs nobody claimed are dropped with their pieces (vhead stays: their ring space is simply skipped)
        while (n_jobs > n_issued) {
            jobs.pop_back();
            n_jobs--;
        }
        const int64_t keep_pieces = n_jobs > j_base ? J(n_jobs - 1).p1 : p_base;
        while (p_base + (int64_t) pieces.size() > keep_pieces && !pieces.empty()) {
            pieces.pop_back();
        }
        p_consumed  = p_base + (int64_t) pieces.size();
        j_consumed  = n_jobs;
        claim_limit = INT64_MAX;
        trim();
        cv.notify_all();
    }
};

ggml_upload_engine_t ggml_backend_cuda_upload_engine_new(void * backend, const ggml_upload_engine_params * params) {
    ggml_backend_t be = (ggml_backend_t) backend;
    if (be == nullptr || !ggml_backend_is_cuda(be) || params == nullptr) {
        return nullptr;
    }
    auto * ctx = (ggml_backend_cuda_context *) be->context;
    auto * eng = new ggml_upload_engine();
    if (!eng->init(ctx->device, *params)) {
        delete eng;
        return nullptr;
    }
    return eng;
}

void ggml_backend_cuda_upload_engine_free(ggml_upload_engine_t eng) {
    delete eng;
}

int64_t ggml_backend_cuda_upload_engine_push(ggml_upload_engine_t eng, const void * src, size_t bytes) {
    return eng->push(src, bytes);
}

bool ggml_backend_cuda_upload_engine_gather(ggml_upload_engine_t eng, void * backend, void * dst,
                                            const ggml_upload_copy * copies, int n_copies) {
    ggml_backend_t be = (ggml_backend_t) backend;
    auto * ctx = (ggml_backend_cuda_context *) be->context;
    if (ctx->device != eng->device) {
        return false;
    }
    ggml_cuda_set_device(ctx->device);
    return eng->gather(*ctx, dst, copies, n_copies);
}

void ggml_backend_cuda_upload_engine_reset(ggml_upload_engine_t eng) {
    eng->reset();
}

void ggml_backend_cuda_upload_engine_get_stats(ggml_upload_engine_t eng, ggml_upload_engine_stats * st) {
    st->n_pieces       = eng->st_pieces.load();
    st->n_dma          = eng->st_dma.load();
    st->bytes          = eng->st_bytes.load();
    st->n_gathers      = eng->st_gathers.load();
    st->copy_ns        = eng->st_copy.load();
    st->wait_ring_ns   = eng->st_wait_ring.load();
    st->wait_copy_ns   = eng->st_wait_copy.load();
    st->wait_gather_ns = eng->st_wait_gather.load();
    st->dropped        = eng->st_dropped.load();
}

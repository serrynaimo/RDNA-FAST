#pragma once

// ROCmFPX [TAG_EXPERT_HANDOFF] expert handoff: the layout of the mailbox the device and the host share.
//
// The mailbox is host memory both sides read and write in place (pinned, mapped into the device, coherent). For each
// MoE layer the device's send op writes the routed experts' input, their ids, cache slots and weights here and bumps
// seq_send; the host computes the experts the device does not hold (the slots equal to the layer's miss slot), writes
// their weighted sum to y and sets seq_done = seq_send; the device's join op waits for that and adds y to its own
// experts' sum. One layer is in flight at a time per mailbox, because the next layer's send needs this layer's join.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_EXPERT_HANDOFF_ALIGN 256

// the header, one 64-byte line per writer so neither side's writes evict the other's line
struct ggml_expert_handoff_header {
    uint32_t seq_send;      // device: bumped once the inputs of a layer are in place
    uint32_t pad_send[15];
    uint32_t seq_done;      // host: set to seq_send once y is in place
    uint32_t pad_done[15];
    uint32_t n_timeouts;    // device: joins that gave up waiting for the host (the host's experts are then missing)
    uint32_t to_want;       // device, the last join that gave up [TAG_EH_TIMEOUT_DIAG]: the send it waited for,
    uint32_t to_done;       //   the host's last answer it saw,
    int32_t  to_layer;      //   its layer
    uint32_t pad_err[12];
    int32_t  layer;         // device, written before seq_send
    int32_t  n_tokens;
    int32_t  n_embd;
    int32_t  n_used;
    uint32_t pad_info[12];
};

// capacities, fixed when the mailbox is allocated (passed to the ops in their op params)
struct ggml_expert_handoff_dims {
    int32_t max_tokens;
    int32_t max_embd;
    int32_t max_used;
};

static inline size_t ggml_expert_handoff_pad(size_t n) {
    return (n + GGML_EXPERT_HANDOFF_ALIGN - 1) / GGML_EXPERT_HANDOFF_ALIGN * GGML_EXPERT_HANDOFF_ALIGN;
}

// region offsets in bytes from the start of the mailbox:
// x [max_tokens][max_embd] f32, ids [max_tokens][max_used] i32, slots (same), weights [max_tokens][max_used] f32,
// y [max_tokens][max_embd] f32
static inline size_t ggml_expert_handoff_off_x(struct ggml_expert_handoff_dims d) {
    (void) d;
    return ggml_expert_handoff_pad(sizeof(struct ggml_expert_handoff_header));
}
static inline size_t ggml_expert_handoff_off_ids(struct ggml_expert_handoff_dims d) {
    return ggml_expert_handoff_off_x(d) + ggml_expert_handoff_pad((size_t) d.max_tokens*d.max_embd*sizeof(float));
}
static inline size_t ggml_expert_handoff_off_slots(struct ggml_expert_handoff_dims d) {
    return ggml_expert_handoff_off_ids(d) + ggml_expert_handoff_pad((size_t) d.max_tokens*d.max_used*sizeof(int32_t));
}
static inline size_t ggml_expert_handoff_off_weights(struct ggml_expert_handoff_dims d) {
    return ggml_expert_handoff_off_slots(d) + ggml_expert_handoff_pad((size_t) d.max_tokens*d.max_used*sizeof(int32_t));
}
static inline size_t ggml_expert_handoff_off_y(struct ggml_expert_handoff_dims d) {
    return ggml_expert_handoff_off_weights(d) + ggml_expert_handoff_pad((size_t) d.max_tokens*d.max_used*sizeof(float));
}
static inline size_t ggml_expert_handoff_size(struct ggml_expert_handoff_dims d) {
    return ggml_expert_handoff_off_y(d) + ggml_expert_handoff_pad((size_t) d.max_tokens*d.max_embd*sizeof(float));
}

// op params of GGML_OP_EXPERT_HANDOFF (int32 slots)
enum ggml_expert_handoff_param {
    GGML_EXPERT_HANDOFF_P_MODE       = 0, // GGML_EXPERT_HANDOFF_SEND or _JOIN
    GGML_EXPERT_HANDOFF_P_PTR_LO     = 1, // the mailbox (the same address on the host and the device)
    GGML_EXPERT_HANDOFF_P_PTR_HI     = 2,
    GGML_EXPERT_HANDOFF_P_LAYER      = 3,
    GGML_EXPERT_HANDOFF_P_MAX_TOKENS = 4,
    GGML_EXPERT_HANDOFF_P_MAX_EMBD   = 5,
    GGML_EXPERT_HANDOFF_P_MAX_USED   = 6,
    GGML_EXPERT_HANDOFF_P_TIMEOUT_MS = 7, // join: give up after this long (the GPU watchdog must never see a stuck kernel)
    GGML_EXPERT_HANDOFF_P_MISS_SLOT  = 8, // send: the slot of the experts the host computes
    GGML_EXPERT_HANDOFF_P_FLAGS_LO   = 9, // device memory [n_layer] int32: the misses of each layer's last send
    GGML_EXPERT_HANDOFF_P_FLAGS_HI   = 10,
    GGML_EXPERT_HANDOFF_P_COUNTS_LO  = 11, // device memory [n_layer][stride] int32 or NULL: routing counts
    GGML_EXPERT_HANDOFF_P_COUNTS_HI  = 12,
    GGML_EXPERT_HANDOFF_P_COUNTS_STRIDE = 13,
};

// A send whose selection has no miss writes nothing to the mailbox and does not ring it, and its join returns y
// without waiting: the host is only involved in layers it has experts to compute for. The device counts the
// misses into flags[layer] (the join reads it) and every routed expert into counts (for routing profiles).

enum ggml_expert_handoff_mode {
    GGML_EXPERT_HANDOFF_SEND = 0,
    GGML_EXPERT_HANDOFF_JOIN = 1,
};

// [TAG_EH_COPY_ENGINE] the upload engine of a GPU backend (get_proc_address "ggml_backend_upload_engine_*"): moves
// expert bytes from pageable host memory into a ring in VRAM ahead of the graph that reads them, and gathers them
// (with ranges already in VRAM) into a destination tensor on the backend's stream.
//
// Pieces are pushed in the order the gathers will read them. Host threads copy runs of consecutive pieces into pinned
// staging buffers, an issuer thread sends each run with one DMA on a stream of its own into the ring, and a gather
// waits on the DMAs of its pieces only, so the copies of later layers run while earlier layers compute. A gather
// frees the ring space of its pieces and of every piece pushed before them (pieces a gather skips are dropped).

typedef struct ggml_upload_engine * ggml_upload_engine_t;

struct ggml_upload_engine_params {
    size_t ring_bytes;   // the VRAM ring; a gather larger than about half of it runs in parts
    size_t stage_bytes;  // one pinned staging buffer: the largest piece and the most bytes per DMA
    int    n_stage;      // pinned staging buffers (how far the host copies run ahead of the DMAs)
    int    n_threads;    // host copy threads
};

// one range of a gather: from a pushed piece (piece >= 0, its whole size), from device memory (piece < 0, src), or
// zeros (piece < 0, src NULL), to dst + dst_off
struct ggml_upload_copy {
    int64_t      piece;
    const void * src;
    size_t       dst_off;
    size_t       bytes;
};

struct ggml_upload_engine_stats {
    uint64_t n_pieces;        // pieces sent
    uint64_t n_dma;           // DMAs (runs of pieces)
    uint64_t bytes;           // bytes sent
    uint64_t n_gathers;       // gather kernels
    uint64_t copy_ns;         // the host copy threads' time in memcpy and file reads (summed over the threads)
    uint64_t wait_ring_ns;    // the issuer waiting for gathers to free ring space (the device is the bottleneck)
    uint64_t wait_copy_ns;    // the issuer waiting for host copies (the host memory is the bottleneck)
    uint64_t wait_gather_ns;  // gathers waiting on the host for their pieces' DMAs to be sent
    uint64_t dropped;         // pieces pushed and never gathered
    uint64_t read_bytes;      // [TAG_EH_DIRECT_READ] bytes read from files (alignment included)
    uint64_t read_ns;         // the copy threads' time in those reads (summed over the threads)
    uint64_t n_reads;         // reads
    uint64_t n_read_failed;   // reads that failed or came back short (their pieces were copied from memory instead)
};

// [TAG_EH_DIRECT_READ] a file the engine reads pieces from, straight into its pinned staging buffers: the experts of a
// prompt that are neither in VRAM nor in the RAM copy (llama.cpp: from a copy of the model file, --expert-copy).
// Through the mapping they came in by page faults, a few dozen KiB per fault; as positioned reads the pieces of a layer
// that are adjacent in the file go as one read. The reads start, end and land on multiples of align (1 for buffered
// reads; unbuffered ones need the sector size, and the engine reads a little around each run of pieces).
struct ggml_upload_file {
    // reads bytes at offset off into dst: the bytes read (fewer at the end of the file), negative on an error
    int64_t (*read)(void * ctx, void * dst, uint64_t off, size_t bytes);
    void *  ctx;
    size_t  align;
};

typedef ggml_upload_engine_t (*ggml_upload_engine_new_t)      (void * backend, const struct ggml_upload_engine_params * params);
typedef void                 (*ggml_upload_engine_free_t)     (ggml_upload_engine_t eng);
// returns the piece's index, or -1 if it is larger than a staging buffer
typedef int64_t              (*ggml_upload_engine_push_t)     (ggml_upload_engine_t eng, const void * src, size_t bytes);
// [TAG_EH_DIRECT_READ] a piece read from file at off; src holds the same bytes in host memory (the mapping), copied
// instead if the read fails. The file must outlive the pieces pushed from it.
typedef int64_t              (*ggml_upload_engine_push_file_t)(ggml_upload_engine_t eng, const struct ggml_upload_file * file,
                                                               uint64_t off, const void * src, size_t bytes);
// on backend's stream (a backend of the engine's device): dst[copies[i].dst_off ...] = the copies, after their DMAs
typedef bool                 (*ggml_upload_engine_gather_t)   (ggml_upload_engine_t eng, void * backend, void * dst,
                                                               const struct ggml_upload_copy * copies, int n_copies);
// drops the pieces not yet gathered (the end of a graph); the engine is ready for the next one
typedef void                 (*ggml_upload_engine_reset_t)    (ggml_upload_engine_t eng);
typedef void                 (*ggml_upload_engine_get_stats_t)(ggml_upload_engine_t eng, struct ggml_upload_engine_stats * st);

#ifdef __cplusplus
}
#endif

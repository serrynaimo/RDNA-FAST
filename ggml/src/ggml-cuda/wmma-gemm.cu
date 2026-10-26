#include "wmma-gemm.cuh"
#include "convert.cuh"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// [TAG_WMMA_GEMM] The library is loaded at run time from its per-arch build (rocm_wmma_gemm_<arch>_accf32: f16 inputs,
// f32 accumulation and output), next to the executable or in ROCMFPX_WMMA_GEMM_DIR. C entry point, layouts 0 = row
// major, 1 = column major, stream passed by address.
typedef void (*wmma_gemm_f32_f16_t)(void * C, void * A, void * B, size_t M, size_t N, size_t K, size_t batch_count,
                                    int layout_c, int layout_a, int layout_b, void * stream);

static int wmma_gemm_env(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

// The archs with tuned configurations upstream; RDNA3 cards with other chips (gfx1101, gfx1102) are not tuned for.
static const char * wmma_gemm_arch(const int cc) {
    if (cc == GGML_CUDA_CC_OFFSET_AMD + 0x1100) {
        return "gfx1100";
    }
    if (cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151) {
        return "gfx1151";
    }
    return nullptr;
}

// The directory of the module this code is in (ggml-hip): the build puts the library next to it.
static std::string wmma_gemm_module_dir() {
#ifdef _WIN32
    HMODULE self = nullptr;
    char path[MAX_PATH] = {};
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR) &wmma_gemm_module_dir, &self) && GetModuleFileNameA(self, path, MAX_PATH)) {
        std::string p(path);
        const size_t slash = p.find_last_of("\\/");
        return slash == std::string::npos ? std::string() : p.substr(0, slash);
    }
#else
    Dl_info info;
    if (dladdr((void *) &wmma_gemm_module_dir, &info) && info.dli_fname) {
        std::string p(info.dli_fname);
        const size_t slash = p.find_last_of('/');
        return slash == std::string::npos ? std::string() : p.substr(0, slash);
    }
#endif
    return std::string();
}

static wmma_gemm_f32_f16_t wmma_gemm_load(const char * arch) {
    static std::mutex mutex;
    static std::map<std::string, wmma_gemm_f32_f16_t> loaded;
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = loaded.find(arch);
    if (it != loaded.end()) {
        return it->second;
    }
    const char * env_dir = getenv("ROCMFPX_WMMA_GEMM_DIR");
    const std::string dir = env_dir ? std::string(env_dir) : wmma_gemm_module_dir();
    const std::string name = std::string("rocm_wmma_gemm_") + arch + "_accf32";
    wmma_gemm_f32_f16_t fn = nullptr;
#ifdef _WIN32
    const std::string path = (dir.empty() ? std::string() : dir + "\\") + name + ".dll";
    HMODULE h = LoadLibraryA(path.c_str());
    if (h) {
        fn = (wmma_gemm_f32_f16_t) GetProcAddress(h, "rocm_wmma_gemm_f32_f16");
    }
#else
    const std::string path = (dir.empty() ? std::string() : dir + "/") + "lib" + name + ".so";
    void * h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h) {
        fn = (wmma_gemm_f32_f16_t) dlsym(h, "rocm_wmma_gemm_f32_f16");
    }
#endif
    GGML_LOG_INFO("%s: %s: %s\n", __func__, path.c_str(),
        fn ? "loaded, prompt-sized FP4/FP6 products use the WMMA GEMM" : "not available, MMQ as before");
    loaded[arch] = fn;
    return fn;
}

bool ggml_cuda_should_use_wmma_gemm(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    static const bool    enabled   = wmma_gemm_env("ROCMFPX_WMMA_GEMM", 1) != 0;
    // at 256 tokens the unpack costs about what the GEMM saves (0.68x-1.25x by shape on gfx1100); at 512 it is 1.1x-1.4x
    static const int64_t min_batch = wmma_gemm_env("ROCMFPX_WMMA_GEMM_MIN_BATCH", 512);
    if (!enabled) {
        return false;
    }
    const char * arch = wmma_gemm_arch(cc);
    if (!arch) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_Q4_0_ROCMFP4:
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
        case GGML_TYPE_Q6_0_ROCMFPX:
        // [TAG_WMMA_GEMM_STOCK] stock quants, against int8 MMQ (their path on gfx1100 since TAG_MMB_RDNA3_0_MMQ), RX
        // 7900 XT 2026-10-24, 4096 x 14336 at 512 rows: Q4_0 48.1 vs 47.5 TFLOPS, Q4_K 47.2 vs 42.3, Q5_K 45.2 vs 38.7,
        // Q6_K 46.7 vs 42.8, IQ4_NL 48.0 vs 46.4, IQ4_XS 48.0 vs 46.5; Qwen3.8-27B IQ4_XS cold 17k prompt 864 vs 781 t/s
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        // [TAG_WMMA_GEMM_Q3] the 3-bit stock types (IQ3_S / IQ3_XXS of the IQ3 quants, Q3_K of Q3_K_S/M/L), unpacked to f16
        // by their existing to_fp16 kernels like the 4-bit ones above. RX 7900 XT 2026-10-26, 22016 x 7168 at 512 rows
        // against int8 MMQ: IQ3_S 55.1 vs 51.9 TFLOPS, IQ3_XXS 55.2 vs 52.7, Q3_K 51.2 vs 45.4
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_Q3_K:
            break;
        default:
            return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->ne[1] < min_batch) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // the kernels read K and write N in whole tiles (at most 256 wide); token rows past M are clamped by the hardware
    if (src0->ne[0] % 256 != 0 || src0->ne[1] % 256 != 0) {
        return false;
    }
    return wmma_gemm_load(arch) != nullptr;
}

void ggml_cuda_mul_mat_wmma_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t K = src0->ne[0];   // input width
    const int64_t N = src0->ne[1];   // weight rows = output width
    const int64_t M = src1->ne[1];   // tokens

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const wmma_gemm_f32_f16_t gemm = wmma_gemm_load(wmma_gemm_arch(cc));
    GGML_ASSERT(gemm);

    cudaStream_t stream = ctx.stream();

    // activations: f32 -> f16, row major M x K
    ggml_cuda_pool_alloc<half> a16(ctx.pool(), M*K);
    ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1->data, a16.get(), M*K, stream);

    // weights: a chunk of whole rows at a time, unpacked to f16 into one scratch of at most ROCMFPX_WMMA_GEMM_SCRATCH_MB
    // (64): a full f16 copy of a 17408 x 5120 matrix is 178 MB, which a 20 GB card running a 15 GB model has not got.
    // Reading the FP4 codes inside the GEMM instead (2026-10-23) was bit-exact but slower, 14-49 vs 54-67 TFLOPS:
    // every block of activation rows decodes its weight tiles again (8-22x the unpacks), where this unpacks once.
    static const int64_t budget = (int64_t) std::max(1, wmma_gemm_env("ROCMFPX_WMMA_GEMM_SCRATCH_MB", 64)) << 20;
    const int64_t rows = std::min<int64_t>(N, std::max<int64_t>(256, budget / (K*(int64_t) sizeof(half)) / 256 * 256));
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(src0->type);
    GGML_ASSERT(to_fp16);
    ggml_cuda_pool_alloc<half> w16(ctx.pool(), rows*K);

    // a chunk's output columns are not contiguous in dst: compute them into a temporary and copy them across
    ggml_cuda_pool_alloc<float> c32(ctx.pool());
    if (rows < N) {
        c32.alloc(M*rows);
    }

    float * dst_d = (float *) dst->data;
    for (int64_t n0 = 0; n0 < N; n0 += rows) {
        const int64_t nc = std::min(rows, N - n0);
        to_fp16((const char *) src0->data + n0*src0->nb[1], w16.get(), nc*K, stream);
        float * c = rows < N ? c32.get() : dst_d;
        // C (M x nc, row major) = A (M x K, row major) * B (K x nc, column major: the weight rows as stored)
        hipStream_t s = stream;
        gemm(c, a16.get(), w16.get(), (size_t) M, (size_t) nc, (size_t) K, 1, 0, 0, 1, &s);
        CUDA_CHECK(cudaGetLastError());
        if (rows < N) {
            CUDA_CHECK(cudaMemcpy2DAsync(dst_d + n0, N*sizeof(float), c, nc*sizeof(float), nc*sizeof(float), M,
                                         cudaMemcpyDeviceToDevice, stream));
        }
    }
}

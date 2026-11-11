#pragma once
#include "common.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * x, T * y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
typedef to_t_cuda_t<nv_bfloat16> to_bf16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

// TODO more general support for non-contiguous inputs

template<typename T>
using to_t_nc_cuda_t = void (*)(const void * x, T * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

typedef to_t_nc_cuda_t<float> to_fp32_nc_cuda_t;
typedef to_t_nc_cuda_t<half> to_fp16_nc_cuda_t;
typedef to_t_nc_cuda_t<nv_bfloat16> to_bf16_nc_cuda_t;

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);
to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type);

// [TAG_FP8_KV] OCP FP8 E4M3: the scalar conversions are upstream's (common.cuh, through ggml_cuda_cast, the
// same bytes as the CPU for every finite value); the packed decoders below serve the attention kernels.

// Four E4M3 codes (the low byte first) to two half2 holding their values * 2^-8, for the attention kernels: a code's
// magnitude moved into an f16's exponent and mantissa fields reads as its value * 2^-8 (bias 15 vs 7), normal and
// subnormal alike, exactly (every E4M3 value is an f16 value). On AMD one byte permute puts two codes into the high
// bytes of the two f16 lanes; the sign stays at bit 15, the magnitude moves down one bit. The caller applies the 256
// (once per dot product for K, see vec_dot_fattn_vec_KQ_f8_e4m3).
static __device__ __forceinline__ void ggml_cuda_fp8x4_e4m3_to_half2x2_div256(const uint32_t v, half2 & lo, half2 & hi) {
#ifdef GGML_USE_HIP
    const uint32_t x0 = __builtin_amdgcn_perm(0u, v, 0x010C000Cu); // bytes: 0, code0, 0, code1
    const uint32_t x1 = __builtin_amdgcn_perm(0u, v, 0x030C020Cu); // bytes: 0, code2, 0, code3
#else
    const uint32_t x0 = ((v & 0x000000FFu) << 8) | ((v & 0x0000FF00u) << 16);
    const uint32_t x1 = ((v & 0x00FF0000u) >> 8) | (v & 0xFF000000u);
#endif // GGML_USE_HIP
    const uint32_t b0 = (x0 & 0x80008000u) | ((x0 >> 1) & 0x3F803F80u);
    const uint32_t b1 = (x1 & 0x80008000u) | ((x1 >> 1) & 0x3F803F80u);
    memcpy(&lo, &b0, sizeof(lo));
    memcpy(&hi, &b1, sizeof(hi));
}

// Two E4M3 codes (the low byte first) to a half2 of their values: the magnitude in an f16's fields times 256.
static __device__ __forceinline__ half2 ggml_cuda_fp8x2_e4m3_to_half2(const uint32_t v) {
    const uint32_t bits = ((v & 0x0080u) << 8) | ((v & 0x007Fu) << 7) | ((v & 0x8000u) << 16) | ((v & 0x7F00u) << 15);
    half2 h;
    memcpy(&h, &bits, sizeof(h));
    return __hmul2(h, make_half2(256.0f, 256.0f));
}

template<typename dst_t, typename src_t>
 __host__ __device__ inline dst_t ggml_cuda_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return x;
    } else if constexpr (std::is_same_v<dst_t, ggml_fp8_e4m3_t>) {
        return { ggml_cuda_fp32_to_f8_e4m3(ggml_cuda_cast<float>(x)) };
    } else if constexpr (std::is_same_v<src_t, ggml_fp8_e4m3_t>) {
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if CUDART_VERSION >= 13040
        if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
            return static_cast<nv_bfloat16>(__nv_cvt_fp8_to_bf16raw(x.bits, __NV_E4M3));
        }
#endif
        if constexpr (std::is_same_v<dst_t, half>) {
            return static_cast<half>(__nv_cvt_fp8_to_halfraw(x.bits, __NV_E4M3));
        }
#endif
        return ggml_cuda_cast<dst_t>(ggml_cuda_f8_e4m3_to_fp32(x.bits));
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    } else if constexpr (std::is_same_v<src_t, __nv_fp8x2_e4m3> && std::is_same_v<dst_t, nv_bfloat162>) {
#if CUDART_VERSION >= 13040
        return static_cast<nv_bfloat162>(__nv_cvt_fp8x2_to_bf162raw(x.__x, __NV_E4M3));
#else
        return make_bfloat162(
            ggml_cuda_cast<nv_bfloat16>(ggml_cuda_f8_e4m3_to_fp32(x.__x & 0xFF)),
            ggml_cuda_cast<nv_bfloat16>(ggml_cuda_f8_e4m3_to_fp32(x.__x >> 8)));
#endif
    } else if constexpr (std::is_same_v<src_t, __nv_fp8x2_e4m3> && std::is_same_v<dst_t, float2>) {
#if CUDART_VERSION >= 13040
        return ggml_cuda_cast<float2>(ggml_cuda_cast<nv_bfloat162>(x));
#else
        return __half22float2(static_cast<half2>(__nv_cvt_fp8x2_to_halfraw2(x.__x, __NV_E4M3)));
#endif
#endif
    } else if constexpr(std::is_same_v<dst_t, nv_bfloat16>) {
        return __float2bfloat16(float(x));
    } else if constexpr(std::is_same_v<src_t, nv_bfloat16>) {
        return __bfloat162float(x);
    } else if constexpr(std::is_same_v<src_t, float2> && std::is_same_v<dst_t, half2>) {
        return __float22half2_rn(x);
    } else if constexpr(std::is_same_v<src_t, nv_bfloat162> && std::is_same_v<dst_t, float2>) {
#ifdef GGML_USE_HIP
        return make_float2(__bfloat162float(__low2bfloat16(x)), __bfloat162float(__high2bfloat16(x)));
#else
#if !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
        return __bfloat1622float2(x);
#else
        return make_float2(__bfloat162float(x.x), __bfloat162float(x.y));
#endif // !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
#endif // GGML_USE_HIP
    } else if constexpr(std::is_same_v<src_t, float2> && std::is_same_v<dst_t, nv_bfloat162>) {
        // bypass compile error on cuda 12.0.1
#ifdef GGML_USE_HIP
        return __float22bfloat162_rn(x);
#else
        return {x.x, x.y};
#endif // GGML_USE_HIP
    } else if constexpr(std::is_same_v<dst_t, int32_t>) {
        return int32_t(x);
    } else {
        return float(x);
    }
}

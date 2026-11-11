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

// [TAG_FP8_KV] OCP FP8 E4M3 (#27512) on the GPU: 1 sign, 4 exponent (bias 7), 3 mantissa bits, max 448, no inf.
// The encoder is the CPU's (ggml_fp32_to_f8_e4m3: round to nearest even, saturating to +-448, NaN -> 0x7F), so a
// cache written here holds the same bytes as one written on the CPU. Decoding is exact for every finite code; the
// NaN codes (0x7F / 0xFF, never written from finite values) decode as +-480.
static __host__ __device__ __forceinline__ float ggml_cuda_fp8_e4m3_to_float(const uint8_t b) {
    const uint32_t mag = b & 0x7Fu;
    float v;
    if (mag < 8u) {
        // subnormal: M * 2^-9, kept out of f32 subnormals (a GPU may flush those to zero)
        v = (float) mag * 0.001953125f;
    } else {
        // normal: exponent E - 7 + 127 = E + 120, the 3 mantissa bits on top of the f32 mantissa
        const uint32_t bits = (mag + (120u << 3)) << 20;
        memcpy(&v, &bits, sizeof(v));
    }
    return b & 0x80u ? -v : v;
}

static __host__ __device__ __forceinline__ uint8_t ggml_cuda_fp32_to_fp8_e4m3(float x) {
    const uint8_t sign = signbit(x) ? 0x80 : 0;
    x = fabsf(x);
    if (isnan(x)) {
        return 0x7F;
    }
    if (x == 0.0f) {
        return sign;
    }
    if (isinf(x) || x >= 448.0f) {
        return sign | 0x7E;
    }
    const auto round_even = [](float y) -> int {
        const int value = (int) floorf(y);
        const float fraction = y - value;
        return fraction > 0.5f || (fraction == 0.5f && (value & 1)) ? value + 1 : value;
    };
    if (x < 0.015625f) {
        // subnormal: M * 2^-9; a carry to 8 is the smallest normal, 0x08
        return sign | (uint8_t) round_even(x * 512.0f);
    }
    int exp;
    const float mantissa = frexpf(x, &exp) * 2.0f;
    int encoded_exp = exp + 6;
    int encoded_man = round_even((mantissa - 1.0f) * 8.0f);
    if (encoded_man == 8) {
        encoded_man = 0;
        encoded_exp++;
    }
    if (encoded_exp > 15 || (encoded_exp == 15 && encoded_man > 6)) {
        return sign | 0x7E;
    }
    return sign | (uint8_t) (encoded_exp << 3) | (uint8_t) encoded_man;
}

// Two E4M3 codes (the low byte first) to a half2, for the attention kernels: a code's magnitude moved into an f16's
// exponent and mantissa fields reads as its value * 2^-8 (bias 15 vs 7), normal and subnormal alike, so one packed
// multiply by 256 restores both exactly (every E4M3 value is an f16 value).
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
    } else if constexpr(std::is_same_v<src_t, ggml_fp8_e4m3_t>) {
        const float f = ggml_cuda_fp8_e4m3_to_float(x.bits);
        if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
            return __float2bfloat16(f);
        } else if constexpr (std::is_same_v<dst_t, half>) {
            return __float2half(f);
        } else {
            return dst_t(f);
        }
    } else if constexpr(std::is_same_v<dst_t, ggml_fp8_e4m3_t>) {
        ggml_fp8_e4m3_t r;
        if constexpr (std::is_same_v<src_t, nv_bfloat16>) {
            r.bits = ggml_cuda_fp32_to_fp8_e4m3(__bfloat162float(x));
        } else {
            r.bits = ggml_cuda_fp32_to_fp8_e4m3(float(x));
        }
        return r;
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

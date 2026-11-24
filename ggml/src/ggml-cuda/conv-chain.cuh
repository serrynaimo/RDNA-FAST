#pragma once

#include "common.cuh"

// [RDNA-FAST] [TAG_INKLING_CONV_CHAIN] Inkling 2's short convolution (llama_model_inkling2::graph::convolve) as one
// kernel per sequence and channel, after unslothai/llama.cpp #411 (MIT), for our graph:
//   x = CONCAT(past, TRANSPOSE(cur), 0)                  the history before the new inputs
//   CPY(window of x, state cell) x K                     the state store, one window per rollback slot
//   y = SSM_CONV(x, w)                                   taps contracted in tap order, as ssm_conv_f32
//   [y = ADD(y, MUL(CONT(TRANSPOSE(window_lag)), tap_lag)) for lag = 0 .. n_conv - 1]    the dynamic taps
//   [y = SCALE(ADD(cur, y))]                             the residual (folded into w in production)
// Up to 32 tokens per sequence (decode, batched decode, verify); longer ubatches keep the separate kernels.
// Returns the number of nodes after node i (the CONCAT) taken by the fused kernel, 0 if none.
int ggml_cuda_try_fuse_conv_chain(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);

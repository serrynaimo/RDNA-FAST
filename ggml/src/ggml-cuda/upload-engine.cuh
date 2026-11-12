#pragma once

#include "common.cuh"
#include "ggml-expert-handoff.h"

// ROCmFPX [TAG_EH_COPY_ENGINE] the upload engine (API and protocol in ggml-expert-handoff.h)
ggml_upload_engine_t ggml_backend_cuda_upload_engine_new(void * backend, const ggml_upload_engine_params * params);
void    ggml_backend_cuda_upload_engine_free(ggml_upload_engine_t eng);
int64_t ggml_backend_cuda_upload_engine_push(ggml_upload_engine_t eng, const void * src, size_t bytes);
int64_t ggml_backend_cuda_upload_engine_push_file(ggml_upload_engine_t eng, const ggml_upload_file * file, uint64_t off,
                                               const void * src, size_t bytes); // [TAG_EH_DIRECT_READ]
bool    ggml_backend_cuda_upload_engine_gather(ggml_upload_engine_t eng, void * backend, void * dst,
                                               const ggml_upload_copy * copies, int n_copies);
void    ggml_backend_cuda_upload_engine_reset(ggml_upload_engine_t eng);
void    ggml_backend_cuda_upload_engine_get_stats(ggml_upload_engine_t eng, ggml_upload_engine_stats * st);

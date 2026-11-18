#include "llama-memory.h"

#include "llama-ext.h"
#include "llama-io.h"

#include <set>
#include <vector>

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}

// collects the tensors that a state write visits, without keeping the data
struct llama_memory_tensors_writer : public llama_io_write_i {
    std::set<ggml_tensor *> tensors;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        tensors.insert(tensor);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem) {
    std::set<ggml_backend_buffer_t> buffers;
    for (ggml_tensor * tensor : llama_memory_get_tensors(mem)) {
        buffers.insert(tensor->buffer);
    }

    return std::vector<ggml_backend_buffer_t>(buffers.begin(), buffers.end());
}

std::vector<ggml_tensor *> llama_memory_get_tensors(const llama_memory_t mem) {
    if (mem == nullptr) {
        return {};
    }

    llama_memory_tensors_writer writer;

    mem->state_write(writer);

    return std::vector<ggml_tensor *>(writer.tensors.begin(), writer.tensors.end());
}

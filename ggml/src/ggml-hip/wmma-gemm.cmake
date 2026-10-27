# ROCmFPX [TAG_WMMA_GEMM]: build the vendored rocm_wmma_gemm (vendor/rocm_wmma_gemm, MIT) for the gfx11 GPU targets:
# one shared library per architecture, rocm_wmma_gemm_<arch>_accf32 (f16 inputs, f32 accumulation and output), next to
# the other binaries. ggml-hip loads it at run time (ggml-cuda/wmma-gemm.cu) and uses it for prompt-sized products of
# the fork's FP4/FP6 weight types and most stock quants; without it those stay on MMQ. The configurations are tuned for
# gfx1100 and gfx1151 only: the desktop chips gfx1101 / gfx1102 build with the gfx1100 ones, the iGPUs gfx1103 /
# gfx1150 / gfx1152 with the gfx1151 ones (not tuned for, not measured). RDNA4 (gfx12) has other WMMA instructions.
# Included from ggml-hip/CMakeLists.txt when GGML_HIP_WMMA_GEMM is ON. Needs Python 3 at configure time: the kernel
# instances are generated from the tuned configurations by the library's own generate_config.py.

set(WMMA_GEMM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor/rocm_wmma_gemm")

set(WMMA_GEMM_ARCHS "")
foreach (arch gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152)
    if ("${arch}" IN_LIST GPU_TARGETS OR "${arch}" IN_LIST AMDGPU_TARGETS OR "${arch}" IN_LIST CMAKE_HIP_ARCHITECTURES)
        list(APPEND WMMA_GEMM_ARCHS ${arch})
    endif()
endforeach()

if (NOT WMMA_GEMM_ARCHS)
    message(STATUS "rocm_wmma_gemm: no gfx11 target among the GPU targets, not built")
    return()
endif()

find_package(Python3 COMPONENTS Interpreter)
if (NOT Python3_Interpreter_FOUND)
    message(WARNING "rocm_wmma_gemm: Python 3 not found (needed to generate its kernels), not built; "
                    "prompt-sized FP4/FP6 products stay on MMQ")
    return()
endif()

foreach (arch ${WMMA_GEMM_ARCHS})
    if (arch MATCHES "^gfx110[012]$")
        set(tuned gfx1100)
    else()
        set(tuned gfx1151)
    endif()
    set(gen "${CMAKE_CURRENT_BINARY_DIR}/rocm_wmma_gemm_${arch}")
    file(MAKE_DIRECTORY "${gen}/include/rocm_wmma_gemm/kernel" "${gen}/src")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "${WMMA_GEMM_DIR}/config/generate_config.py"
                --config "${WMMA_GEMM_DIR}/config/gemm_config_${tuned}_f32.json" --group accf32
                "${gen}/include/rocm_wmma_gemm/kernel/config_generated.hpp" --kernel-dir "${gen}/src"
        RESULT_VARIABLE wmma_gen_result
        OUTPUT_QUIET
        ERROR_VARIABLE  wmma_gen_error)
    if (NOT wmma_gen_result EQUAL 0)
        message(FATAL_ERROR "rocm_wmma_gemm: kernel generation for ${arch} failed: ${wmma_gen_error}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${WMMA_GEMM_DIR}/config/gemm_config_${tuned}_f32.json")

    set(KERNEL_INST_SOURCES "")
    include("${gen}/src/kernel_sources.cmake")   # sets KERNEL_INST_SOURCES
    set(wmma_sources "${WMMA_GEMM_DIR}/src/gemm.cpp" "${gen}/src/kernel_lookup.cpp" ${KERNEL_INST_SOURCES})
    list(LENGTH KERNEL_INST_SOURCES wmma_kernel_count)

    set(lib rocm_wmma_gemm_${arch}_accf32)
    add_library(${lib} SHARED ${wmma_sources})
    target_include_directories(${lib} PRIVATE "${WMMA_GEMM_DIR}/include" "${gen}/include")
    target_compile_definitions(${lib} PRIVATE WMMA_ACC_GROUP_ACCF32)
    # the library's own build flags; -mcumode: the kernels assume one compute unit per workgroup
    target_compile_options(${lib} PRIVATE -O3 -ffast-math -mcumode)
    set_target_properties(${lib} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if (CXX_IS_HIPCC)
        # as for ggml-hip on Windows: HIP sources compiled as C++, but for this library's single architecture only
        set_source_files_properties(${wmma_sources} PROPERTIES LANGUAGE CXX)
        target_compile_options(${lib} PRIVATE -x hip --offload-arch=${arch})
    else()
        set_source_files_properties(${wmma_sources} PROPERTIES LANGUAGE HIP)
        set_target_properties(${lib} PROPERTIES HIP_ARCHITECTURES ${arch})
    endif()
    target_link_libraries(${lib} PRIVATE hip::host)
    add_dependencies(ggml-hip ${lib})
    message(STATUS "rocm_wmma_gemm: ${lib} (${wmma_kernel_count} kernels, ${tuned} configurations)")
endforeach()

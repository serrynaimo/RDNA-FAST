#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

# Accumulator groups: map a group name to the (output C type, input A/B type) pairs it owns.
# Each group is generated and compiled independently into its own library, so a config
# tuned for one accumulator never spawns kernels for another. New groups (e.g. int8/int4
# inputs accumulating into float) slot in as additional entries with their own config JSON,
# leaving every existing group's generated files untouched.
ACC_GROUPS = {
    "acc16":  [("half", "half"), ("__hip_bfloat16", "__hip_bfloat16")],
    "accf32": [("float", "half"), ("float", "__hip_bfloat16")],
}

# Synthetic default config appended to every group's config set. Its index is used as the
# closest-match fallback (DEFAULT_CONFIG_IDX) when no tuned entry matches a query.
DEFAULT_CONFIGS = [(4, 4, 4, 4, 1, 0, 8, 256)]


def type_tag(cpp_type):
    """Short tag used in generated symbol names: half->f16, __hip_bfloat16->bf16, float->f32."""
    return cpp_type.replace('__hip_bfloat16', 'bf16').replace('half', 'f16').replace('float', 'f32')


def load_config(config_file):
    """Load a single accumulator group's config JSON.
    Returns (configurations, unique_configs_sorted)."""
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    confs = cfg['configurations']

    unique_configs = set()
    for conf in confs:
        c = conf['config']
        unique_configs.add((c['warps_m'], c['warps_n'],
                            c['warp_tile_m'], c['warp_tile_n'],
                            c['k_slices'], c['single_buffer'], c['swizzle'], c['bits']))

    for dc in DEFAULT_CONFIGS:
        unique_configs.add(dc)

    return confs, sorted(list(unique_configs))


def generate_config_header(config_file, output_file):
    confs, unique_configs = load_config(config_file)
    num_configs = len(unique_configs)

    # Build sorted_config_map entries: one per (M, N, K, layouts).
    size_layout_configs = {}
    for conf in confs:
        range_info = conf['range']
        M, N, K = range_info['M'], range_info['N'], range_info['K']
        layout_dict = conf['layout']
        layout_key = (
            layout_dict.get('A', 'any'),
            layout_dict.get('B', 'any'),
            layout_dict.get('C', 'any'),
        )
        c = conf['config']
        config_tuple = (c['warps_m'], c['warps_n'],
                        c['warp_tile_m'], c['warp_tile_n'],
                        c['k_slices'], c['single_buffer'], c['swizzle'], c['bits'])
        config_idx = unique_configs.index(config_tuple)
        size_layout_configs.setdefault((M, N, K), {})[layout_key] = config_idx

    sorted_configs = []
    for (M, N, K), tl_map in size_layout_configs.items():
        for (a_layout, b_layout, c_layout), config_idx in tl_map.items():
            sorted_configs.append((M, N, K, a_layout, b_layout, c_layout, config_idx))

    def sort_key(x):
        M, N, K, a, b, c, _ = x
        return (M, N, K,
                0 if a == 'row_major' else 1,
                0 if b == 'row_major' else 1,
                0 if c == 'row_major' else 1)

    sorted_configs.sort(key=sort_key)

    # Generate code
    code = f"""// Auto-generated file - DO NOT EDIT
#ifndef ROCM_WMMA_GEMM_CONFIG_GENERATED_HPP
#define ROCM_WMMA_GEMM_CONFIG_GENERATED_HPP

#include <rocm_wmma_gemm/kernel/common.hpp>
#include <array>
#include <tuple>
#include <cstddef>
#include <cmath>
#include <limits>
#include <algorithm>

namespace rocm_wmma_gemm
{{

// Number of unique kernel variants
static constexpr size_t KERNEL_VARIANTS = {num_configs};

// Configuration parameters for a specific problem size
struct gemm_params
{{
    int warps_m;
    int warps_n;
    int warp_tile_m;
    int warp_tile_n;
    int k_slices;
    int single_buffer;
    int swizzle;
    int bits;
}};

namespace detail
{{
    // Kernel configuration tuple
    using kernel_config = std::tuple<int, int, int, int, int, int, int, int>;

    // All unique kernel configurations
    static constexpr std::array<kernel_config, KERNEL_VARIANTS> kernel_configs = {{
"""

    # Generate config array
    for i, (wm, wn, wtm, wtn, k_slices, single_buffer, swizzle, bits) in enumerate(unique_configs):
        code += f"        std::tuple<int, int, int, int, int, int, int, int>{{{wm}, {wn}, {wtm}, {wtn}, {k_slices}, {single_buffer}, {swizzle}, {bits}}}"
        code += "," if i < len(unique_configs) - 1 else ""
        code += f" // Config {i}: warps({wm},{wn}) tiles({wtm},{wtn}) k_slices({k_slices}) single_buffer({single_buffer}) swizzle({swizzle}) bits({bits})\n"

    code += f"""    }};

    // Default config (last in the array)
    static constexpr size_t DEFAULT_CONFIG_IDX = KERNEL_VARIANTS - 1;

    // Configuration lookup key.
    struct config_key
    {{
        size_t m, n, k;
        m_layout layout_a, layout_b, layout_c;

        constexpr bool operator<(const config_key& other) const
        {{
            if(m != other.m) return m < other.m;
            if(n != other.n) return n < other.n;
            if(k != other.k) return k < other.k;
            if(layout_a != other.layout_a) return layout_a < other.layout_a;
            if(layout_b != other.layout_b) return layout_b < other.layout_b;
            return layout_c < other.layout_c;
        }}

        constexpr bool operator==(const config_key& other) const
        {{
            return m == other.m && n == other.n && k == other.k &&
                   layout_a == other.layout_a && layout_b == other.layout_b &&
                   layout_c == other.layout_c;
        }}
    }};

    // Sorted configuration map for binary search
    static constexpr std::array<std::pair<config_key, size_t>, {len(sorted_configs)}> sorted_config_map = {{{{
"""

    # Generate sorted config array
    for i, (M, N, K, a_layout, b_layout, c_layout, config_idx) in enumerate(sorted_configs):
        a_enum = f"m_layout::{a_layout}" if a_layout != "any" else "m_layout::row_major"
        b_enum = f"m_layout::{b_layout}" if b_layout != "any" else "m_layout::row_major"
        c_enum = f"m_layout::{c_layout}" if c_layout != "any" else "m_layout::row_major"

        code += f"        {{{{{M}, {N}, {K}, {a_enum}, {b_enum}, {c_enum}}}, {config_idx}}}"
        code += "," if i < len(sorted_configs) - 1 else ""
        code += f" // {M}x{N}x{K}, A={a_layout}, B={b_layout}, C={c_layout}\n"

    code += """    }};

    // Find closest configuration for a given layout.
    // Strategy: closest K first, then closest M,N. Falls back to the default config only
    // when no entry shares the requested layout.
    constexpr size_t find_closest_config(size_t m, size_t n, size_t k,
                                         m_layout layout_a,
                                         m_layout layout_b,
                                         m_layout layout_c)
    {
        if(sorted_config_map.empty())
        {
            return DEFAULT_CONFIG_IDX;
        }

        auto mn_distance = [](size_t m1, size_t n1, size_t m2, size_t n2) -> double
        {
            double dm = static_cast<double>(m1) - static_cast<double>(m2);
            double dn = static_cast<double>(n1) - static_cast<double>(n2);
            return dm * dm + dn * dn;
        };

        size_t closest_k = 0;
        size_t min_k_diff = std::numeric_limits<size_t>::max();
        bool found = false;

        for(size_t i = 0; i < sorted_config_map.size(); ++i)
        {
            const auto& key = sorted_config_map[i].first;
            if(key.layout_a == layout_a && key.layout_b == layout_b && key.layout_c == layout_c)
            {
                found = true;
                size_t kd = (key.k > k) ? (key.k - k) : (k - key.k);
                if(kd < min_k_diff) { min_k_diff = kd; closest_k = key.k; }
            }
        }

        if(!found) return DEFAULT_CONFIG_IDX;

        double min_dist = std::numeric_limits<double>::max();
        size_t best = DEFAULT_CONFIG_IDX;
        for(size_t i = 0; i < sorted_config_map.size(); ++i)
        {
            const auto& key = sorted_config_map[i].first;
            if(key.k == closest_k &&
               key.layout_a == layout_a && key.layout_b == layout_b && key.layout_c == layout_c)
            {
                double d = mn_distance(m, n, key.m, key.n);
                if(d < min_dist) { min_dist = d; best = i; }
            }
        }
        return sorted_config_map[best].second;
    }

    // Find the best configuration for a problem size and layout using binary search,
    // falling back to closest-match when no exact entry exists.
    constexpr size_t find_best_config(size_t m, size_t n, size_t k,
                                      m_layout layout_a,
                                      m_layout layout_b,
                                      m_layout layout_c)
    {
        config_key target{m, n, k, layout_a, layout_b, layout_c};
        auto it = std::lower_bound(
            sorted_config_map.begin(), sorted_config_map.end(),
            std::make_pair(target, size_t(0)),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        if(it != sorted_config_map.end() && it->first == target)
        {
            return it->second;
        }

        return find_closest_config(m, n, k, layout_a, layout_b, layout_c);
    }

} // namespace detail

/**
 * @brief Get the optimal configuration parameters for a specific problem size and layout
 *
 * @param m Number of rows in matrices C and A
 * @param n Number of columns in matrices C and B
 * @param k Number of columns in matrix A / rows in matrix B
 * @param layout_c Layout of matrix C
 * @param layout_a Layout of matrix A
 * @param layout_b Layout of matrix B
 * @return Tuned parameters for the given problem
 */
constexpr gemm_params get_gemm_params(size_t m, size_t n, size_t k,
                                       m_layout layout_c,
                                       m_layout layout_a,
                                       m_layout layout_b)
{
    const size_t config_idx = detail::find_best_config(m, n, k,
                                                        layout_a, layout_b, layout_c);
    const auto& config = detail::kernel_configs[config_idx];
    return gemm_params{
        std::get<0>(config),
        std::get<1>(config),
        std::get<2>(config),
        std::get<3>(config),
        std::get<4>(config),
        std::get<5>(config),
        std::get<6>(config),
        std::get<7>(config)
    };
}

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_CONFIG_GENERATED_HPP"""

    write_if_changed(Path(output_file), code)


def config_signature(config_tuple, layout_str):
    """Build a stable identity string for a (config, layout) from its parameter values.
    Used for both filenames and getter symbols so a kernel keeps the same identity
    regardless of its position in the config list. Only genuinely new or changed
    configs churn; inserting or reordering others leaves their files untouched."""
    wm, wn, wtm, wtn, k_slices, single_buffer, swizzle, bits = config_tuple
    return f"w{wm}x{wn}_t{wtm}x{wtn}_k{k_slices}_sb{single_buffer}_s{swizzle}_b{bits}_{layout_str}"


def write_if_changed(filepath, content):
    """Write content only when it differs from what's on disk, so unchanged files keep
    their mtime and the build system does not recompile them. Returns True if written."""
    if filepath.exists() and filepath.read_text() == content:
        return False
    filepath.write_text(content)
    return True


def layout_str_of(layout):
    """Three-letter layout tag, e.g. row/col/col -> 'rcc'."""
    return f"{layout['A'][0]}{layout['B'][0]}{layout['C'][0]}"


def generate_kernel_sources(config_file, output_dir, type_pairs):
    """Generate one source file per (config, layout) holding this group's type pairs
    (aligned and unaligned variants)."""
    confs, _ = load_config(config_file)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seen_combinations = set()
    file_list = []
    written = 0

    for conf in confs:
        cfg = conf['config']
        layout = conf['layout']

        config_tuple = (
            cfg['warps_m'], cfg['warps_n'],
            cfg['warp_tile_m'], cfg['warp_tile_n'],
            cfg['k_slices'], cfg['single_buffer'], cfg['swizzle'], cfg['bits']
        )
        layout_key = (layout['A'], layout['B'], layout['C'])

        if (config_tuple, layout_key) in seen_combinations:
            continue
        seen_combinations.add((config_tuple, layout_key))

        wm, wn, wtm, wtn, k_slices, single_buffer, swizzle_val, bits = config_tuple
        layout_a, layout_b, layout_c = layout_key
        layout_str = layout_str_of(layout)
        sig        = config_signature(config_tuple, layout_str)

        filename = f"kernel_inst_{sig}.cpp"
        filepath = output_path / filename
        file_list.append(filename)

        code = f"""// Auto-generated kernel instantiation file - DO NOT EDIT
// Config: warps({wm},{wn}) tiles({wtm},{wtn}) k_slices({k_slices}) single_buffer({single_buffer}) swizzle({swizzle_val}) bits({bits})
// Layout: A={layout_a}, B={layout_b}, C={layout_c}

#include <rocm_wmma_gemm/kernel/kernel.hpp>

namespace rocm_wmma_gemm
{{

"""
        for t_type, u_type in type_pairs:
            t_name = type_tag(t_type)
            u_name = type_tag(u_type)

            code += f"""extern "C" void* get_kernel_inst_{sig}_{t_name}_{u_name}_aligned() {{
    return (void*)&kernel_gemm_impl<{t_type}, {u_type},
        m_layout::{layout_c}, m_layout::{layout_a}, m_layout::{layout_b},
        {wm}, {wn}, {wtm}, {wtn}, {k_slices}, {single_buffer}, {swizzle_val}, {bits}, 1>::run;
}}

extern "C" void* get_kernel_inst_{sig}_{t_name}_{u_name}_unaligned() {{
    return (void*)&kernel_gemm_impl<{t_type}, {u_type},
        m_layout::{layout_c}, m_layout::{layout_a}, m_layout::{layout_b},
        {wm}, {wn}, {wtm}, {wtn}, {k_slices}, {single_buffer}, {swizzle_val}, {bits}, 0>::run;
}}

"""

        code += """} // namespace rocm_wmma_gemm
"""
        if write_if_changed(filepath, code):
            written += 1

    # Prune stale kernel files from previous generations that no longer match any config.
    keep = set(file_list)
    removed = 0
    for stale in output_path.glob("kernel_inst_*.cpp"):
        if stale.name not in keep:
            stale.unlink()
            removed += 1

    cmake_lines = ["# Auto-generated list of kernel source files\n", "set(KERNEL_INST_SOURCES\n"]
    cmake_lines += [f"    {output_path.as_posix()}/{name}\n" for name in file_list]  # local: forward slashes for CMake on Windows
    cmake_lines.append(")\n")
    cmake_file = output_path / "kernel_sources.cmake"
    write_if_changed(cmake_file, "".join(cmake_lines))

    print(f"Kernel sources: {len(file_list)} total, {written} written/updated, {removed} pruned")

    generate_kernel_lookup(config_file, output_path, type_pairs)

    return file_list


def generate_kernel_lookup(config_file, output_dir, type_pairs):
    """Generate the kernel lookup implementation with a static
    [config][type][layout][alignment] table scoped to this accumulator group."""
    _, unique_configs = load_config(config_file)
    num_configs = len(unique_configs)

    type_tags = [(type_tag(t), type_tag(u)) for t, u in type_pairs]
    num_types = len(type_tags)

    layouts = [
        ('row_major', 'row_major', 'row_major', 'rrr'),
        ('row_major', 'row_major', 'col_major', 'rrc'),
        ('row_major', 'col_major', 'row_major', 'rcr'),
        ('row_major', 'col_major', 'col_major', 'rcc'),
        ('col_major', 'row_major', 'row_major', 'crr'),
        ('col_major', 'row_major', 'col_major', 'crc'),
        ('col_major', 'col_major', 'row_major', 'ccr'),
        ('col_major', 'col_major', 'col_major', 'ccc'),
    ]

    # Build mapping from (config_tuple, layout_tuple) -> signature string.
    # Signature-based identity keeps getter symbols stable across config insert/reorder.
    confs, _ = load_config(config_file)
    config_layout_to_sig = {}
    seen_combinations = set()

    for conf in confs:
        cfg = conf['config']
        layout = conf['layout']

        config_tuple = (
            cfg['warps_m'], cfg['warps_n'],
            cfg['warp_tile_m'], cfg['warp_tile_n'],
            cfg['k_slices'], cfg['single_buffer'], cfg['swizzle'], cfg['bits']
        )
        layout_tuple = (layout['A'], layout['B'], layout['C'])

        if (config_tuple, layout_tuple) in seen_combinations:
            continue
        seen_combinations.add((config_tuple, layout_tuple))

        config_layout_to_sig[(config_tuple, layout_tuple)] = config_signature(
            config_tuple, layout_str_of(layout))

    filepath = output_dir / "kernel_lookup.cpp"

    code = """// Auto-generated kernel lookup - DO NOT EDIT
#include <rocm_wmma_gemm/kernel/common.hpp>
#include <cstddef>

namespace rocm_wmma_gemm
{

"""

    # Forward declare all getters
    for (config_tuple, layout_tuple), sig in config_layout_to_sig.items():
        for t_name, u_name in type_tags:
            code += f'extern "C" void* get_kernel_inst_{sig}_{t_name}_{u_name}_aligned();\n'
            code += f'extern "C" void* get_kernel_inst_{sig}_{t_name}_{u_name}_unaligned();\n'

    type_idx_comment = ", ".join(f"{i}={t}-{u}" for i, (t, u) in enumerate(type_tags))

    code += f"""
// Static kernel lookup table: [config_idx][type_idx][layout_idx][alignment_idx]
// config_idx: 0 to {num_configs-1}
// type_idx: {type_idx_comment}
// layout_idx: 0=rrr, 1=rrc, 2=rcr, 3=rcc, 4=crr, 5=crc, 6=ccr, 7=ccc
// alignment_idx: 0=unaligned, 1=aligned
static void* kernel_table[{num_configs}][{num_types}][8][2] = {{
"""

    for config_idx, config_tuple in enumerate(unique_configs):
        code += f"    // Config {config_idx}: warps({config_tuple[0]},{config_tuple[1]}) tiles({config_tuple[2]},{config_tuple[3]}) k_slices({config_tuple[4]}) single_buffer({config_tuple[5]}) swizzle({config_tuple[6]}) bits({config_tuple[7]})\n"
        code += "    {\n"

        for type_idx, (t_name, u_name) in enumerate(type_tags):
            code += "        {"

            for layout_idx, (layout_a, layout_b, layout_c, layout_str) in enumerate(layouts):
                layout_tuple = (layout_a, layout_b, layout_c)
                key = (config_tuple, layout_tuple)

                code += "{"
                if key in config_layout_to_sig:
                    sig = config_layout_to_sig[key]
                    code += f"get_kernel_inst_{sig}_{t_name}_{u_name}_unaligned(), "
                    code += f"get_kernel_inst_{sig}_{t_name}_{u_name}_aligned()"
                else:
                    code += "nullptr, nullptr"
                code += "}"

                if layout_idx < 7:
                    code += ", "

            code += "}"
            if type_idx < num_types - 1:
                code += ","
            code += "\n"

        code += "    }"
        if config_idx < num_configs - 1:
            code += ","
        code += "\n"

    code += """};

void* lookup_kernel(size_t config_idx, size_t type_idx, size_t layout_idx, size_t alignment_idx)
{
    return kernel_table[config_idx][type_idx][layout_idx][alignment_idx];
}

} // namespace rocm_wmma_gemm
"""

    write_if_changed(filepath, code)

    print(f"Generated kernel lookup table in {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate WMMA GEMM configuration header and kernel sources for one '
                    'accumulator group')
    parser.add_argument('--config', type=str, required=True,
                        help='JSON config file for this accumulator group')
    parser.add_argument('--group', type=str, required=True, choices=list(ACC_GROUPS.keys()),
                        help='Accumulator group name (selects the type pairs to generate)')
    parser.add_argument('output_file', type=str, help='Output header file')
    parser.add_argument('--kernel-dir', type=str, help='Output directory for kernel source files')

    args = parser.parse_args()

    type_pairs = ACC_GROUPS[args.group]

    generate_config_header(args.config, args.output_file)

    if args.kernel_dir:
        generate_kernel_sources(args.config, args.kernel_dir, type_pairs)


if __name__ == '__main__':
    main()

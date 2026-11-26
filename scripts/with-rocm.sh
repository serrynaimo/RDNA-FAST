#!/usr/bin/env bash
# Run anything RDNA-FAST builds or runs against AMD's ROCm, fetching the
# toolchain first if it is not on the machine yet.
#
#   scripts/with-rocm.sh scripts/build-rdna3.sh
#   scripts/with-rocm.sh --arch gfx1200 scripts/build-rdna4.sh
#   scripts/with-rocm.sh ./build-rdna3/bin/llama-server -m model.gguf -ngl 999
#
# AMD ships ROCm 10 for Linux as a TheRock tarball, one per GPU family. This
# unpacks the one matching the GPU into $ROCM_ROOT/<version>-<family>, exports
# the environment the HIP build and the binaries need, and runs the command.
# Nothing is installed system-wide and no apt source is touched. An existing
# ROCm (ROCM_PATH, or hipconfig on PATH) is used as is and nothing downloads.
#
# Environment:
#   ROCM_ARCH     gfx target to fetch for (same as --arch)
#   ROCM_VERSION  ROCm release (default 10.2.0)
#   ROCM_ROOT     prefix root (default $XDG_DATA_HOME/rocm)

set -euo pipefail

VERSION="${ROCM_VERSION:-10.2.0}"
ROOT="${ROCM_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/rocm}"

usage() {
    echo "usage: $(basename "$0") [--arch <gfx>] <command> [args...]" >&2
    exit 2
}

[[ $# -gt 0 ]] || usage

ARCH="${ROCM_ARCH:-}"
if [[ "${1:-}" == "--arch" ]]; then
    [[ -n "${2:-}" ]] || usage
    ARCH="$2"
    shift 2
fi
[[ $# -gt 0 ]] || usage

# One tarball per GPU family. RDNA3.5 has one per chip (gfx1150 Strix Point,
# gfx1151 Strix Halo, gfx1152 Krackan): the libraries in each are built for that
# chip only.
family() {
    case "$1" in
        gfx115[0-2])     printf '%s\n' "$1" ;;
        gfx110[0-3])     printf 'gfx110X-all\n' ;;
        gfx120[0-1])     printf 'gfx120X-all\n' ;;
        *) return 1 ;;
    esac
}

prefix_ready() {
    [[ -x "$1/bin/hipconfig" && -x "$1/lib/llvm/bin/clang++" ]]
}

# A ROCm already on the machine: ROCM_PATH, then hipconfig on PATH.
find_rocm() {
    local path

    if [[ -n "${ROCM_PATH:-}" ]] && prefix_ready "$ROCM_PATH"; then
        printf '%s\n' "$ROCM_PATH"
        return 0
    fi

    if command -v hipconfig >/dev/null 2>&1; then
        path="$(cd "$(dirname "$(command -v hipconfig)")/.." && pwd)"
        if prefix_ready "$path"; then
            printf '%s\n' "$path"
            return 0
        fi
    fi

    return 1
}

# The installed card, asked through any prefix this script unpacked before.
arch_from_prefix() {
    local p
    for p in "$ROOT/$VERSION"-*; do
        [[ -x "$p/bin/rocm_agent_enumerator" ]] || continue
        LD_LIBRARY_PATH="$p/lib:$p/lib/llvm/lib" PATH="$p/bin:$PATH" \
            "$p/bin/rocm_agent_enumerator" 2>/dev/null \
            | grep -oE '^gfx[0-9a-f]{3,4}$' | grep -vx gfx000 | head -1 && return 0
    done
    return 1
}

arch_from_command() {
    case "$(basename "$1")" in
        build-rdna2*) printf 'gfx1030\n' ;;
        build-rdna3*) printf 'gfx1100\n' ;;
        build-rdna4*) printf 'gfx1200\n' ;;
        build-strix*) printf 'gfx1151\n' ;;
        *) return 1 ;;
    esac
}

ROC="$(find_rocm || true)"
if [[ -z "$ROC" ]]; then
    # The GPU picks the tarball: --arch or ROCM_ARCH, then the installed card,
    # then the generation the command builds for.
    if [[ -z "$ARCH" ]]; then
        ARCH="$(arch_from_prefix || arch_from_command "$1" || true)"
    fi
    if [[ -z "$ARCH" ]]; then
        echo "Cannot tell which GPU to fetch ROCm for; pass --arch <gfx>." >&2
        exit 2
    fi
    if ! FAMILY="$(family "$ARCH")"; then
        echo "$ARCH has no ROCm tarball. Supported: RDNA3 gfx1100-gfx1103, RDNA3.5 gfx1150-gfx1152, RDNA4 gfx1200-gfx1201." >&2
        exit 2
    fi

    PREFIX="$ROOT/$VERSION-$FAMILY"
    if ! prefix_ready "$PREFIX"; then
        URL="${ROCM_TARBALL_URL:-https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-${FAMILY}-${VERSION}.tar.gz}"
        TARBALL="$ROOT/$(basename "$URL")"

        if [[ ! -s "$TARBALL" ]]; then
            mkdir -p "$ROOT"
            echo "Downloading $URL"
            curl -fL -C - "$URL" -o "$TARBALL.part"
            mv "$TARBALL.part" "$TARBALL"
        fi

        echo "Unpacking into $PREFIX"
        mkdir -p "$PREFIX"
        tar -xzf "$TARBALL" -C "$PREFIX" --strip-components=1
        if ! prefix_ready "$PREFIX"; then
            echo "$PREFIX carries no bin/hipconfig and lib/llvm/bin/clang++, so the unpack failed." >&2
            exit 1
        fi
    fi

    ROC="$PREFIX"
    echo "ROCm $VERSION ($FAMILY) at $ROC"
else
    echo "ROCm at $ROC"
fi

export ROCM_PATH="$ROC"
export HIP_PATH="$ROC"
export HIP_PLATFORM="${HIP_PLATFORM:-amd}"
export PATH="$ROC/bin:$PATH"
export CMAKE_PREFIX_PATH="$ROC${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$ROC/lib:$ROC/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$@"

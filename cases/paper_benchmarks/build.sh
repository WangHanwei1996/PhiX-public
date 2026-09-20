#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Standalone build for the paper benchmarks.
#
# These programs are deliberately NOT registered in the root CMakeLists.txt and
# NOT placed under test/ — they link against an already-built libphix.a, so the
# core library and its test suite stay untouched.
#
#   ./build.sh [BUILD_DIR] [ARCH]
#
#   BUILD_DIR   directory holding libphix.a   (default: ../../build)
#   ARCH        CUDA compute capability       (default: probed via nvidia-smi)
#
# Example:
#   ./build.sh ../../build 120
#   ./bench_fusion | tee results/bench_fusion.txt
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHIX_ROOT="$(cd "$HERE/../.." && pwd)"

BUILD_DIR="${1:-$PHIX_ROOT/build}"
if [[ -n "${2:-}" ]]; then
    ARCH="$2"
else
    ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | head -1 | tr -d '.' || true)"
    ARCH="${ARCH:-75}"
fi

if [[ ! -f "$BUILD_DIR/libphix.a" ]]; then
    echo "error: libphix.a not found in $BUILD_DIR" >&2
    echo "       build the library first, e.g." >&2
    echo "         mkdir -p $PHIX_ROOT/build && cd \$_ && cmake .. -DPHIX_CUDA_ARCH=$ARCH && make -j" >&2
    exit 1
fi

# nlohmann/json is header-only and pulled in by IO/ConfigFile.h; this repo
# supplies it through a conda env.  Override with JSON_INC=... if it lives
# somewhere else.
JSON_INC="${JSON_INC:-}"
if [[ -z "$JSON_INC" ]]; then
    CAND="$(find "$HOME" -maxdepth 6 -name json.hpp -path '*nlohmann*' 2>/dev/null | head -1 || true)"
    [[ -n "$CAND" ]] && JSON_INC="$(dirname "$(dirname "$CAND")")"
fi
[[ -n "$JSON_INC" ]] || { echo "error: nlohmann/json.hpp not found; set JSON_INC" >&2; exit 1; }

NVCC="${NVCC:-}"
if [[ -z "$NVCC" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        NVCC=nvcc
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        NVCC=/usr/local/cuda/bin/nvcc
    else
        echo "error: nvcc not found; put it on PATH or set NVCC=..." >&2
        exit 1
    fi
fi
echo "PhiX root : $PHIX_ROOT"
echo "library   : $BUILD_DIR/libphix.a"
echo "arch      : sm_$ARCH"
echo "json      : $JSON_INC"
echo

for src in bench_fusion bench_mg_sweep bench_scaling bench_hostsync bench_timestepping; do
    echo "building $src ..."
    "$NVCC" -O3 -std=c++17 -arch="sm_$ARCH" \
        --expt-extended-lambda --expt-relaxed-constexpr -rdc=true \
        -I"$PHIX_ROOT/include" -I"$JSON_INC" \
        "$HERE/$src.cu" -L"$BUILD_DIR" -lphix -lcufft \
        -o "$HERE/$src"
done

echo
echo "done. binaries in $HERE"

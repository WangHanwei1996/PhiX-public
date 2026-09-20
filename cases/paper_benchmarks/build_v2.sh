#!/usr/bin/env bash
# Build bench_fusion_v2 against an already-built libphix.a (v3.10+: needs nvrtc, cuda driver, cupti).
#   ./build_v2.sh [BUILD_DIR] [ARCH]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHIX_ROOT="$(cd "$HERE/../.." && pwd)"
BUILD_DIR="${1:-$PHIX_ROOT/build}"
ARCH="${2:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUPTI=/usr/local/cuda/extras/CUPTI
[[ -f "$BUILD_DIR/libphix.a" ]] || { echo "libphix.a not in $BUILD_DIR" >&2; exit 1; }
echo "lib: $BUILD_DIR/libphix.a  arch: sm_$ARCH"
"$NVCC" -O3 -std=c++17 -arch="sm_$ARCH" --expt-extended-lambda --expt-relaxed-constexpr -rdc=true \
    -I"$PHIX_ROOT/include" -I"$BUILD_DIR/generated" -I"$CUPTI/include" \
    "$HERE/bench_fusion_v2.cu" -L"$BUILD_DIR" -L"$CUPTI/lib64" \
    -lphix -lcufft -lnvrtc -lcuda -lcupti -Xlinker -rpath="$CUPTI/lib64" \
    -o "$HERE/bench_fusion_v2"
echo "built $HERE/bench_fusion_v2"

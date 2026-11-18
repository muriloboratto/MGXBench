#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# MGXBench - Multi-GPU Execution Benchmark
# Installer: CUDA / MPI / NCCL / NVSHMEM
#
# Supported GPUs:
#   NVIDIA V100       -> sm_70
#   NVIDIA H100/GH200 -> sm_90
#
# Usage:
#   bash install.sh
#   CUDA_ARCH=90 bash install.sh
#   CUDA_ARCH=70 bash install.sh
#
# Optional:
#   JOBS=16 CUDA_ARCH=90 bash install.sh
# ============================================================

echo ""
echo "============================================================"
echo "  MGXBench | Multi-GPU Execution Benchmark"
echo "  Installer | CUDA / MPI / NCCL / NVSHMEM"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# 1. Check required tools
# ------------------------------------------------------------

for tool in cmake nvcc mpicxx; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "[ERROR] Required tool '$tool' not found."
        echo "Please load the appropriate environment modules."
        exit 1
    fi
done

echo "[INFO] CMake : $(cmake --version | head -n 1)"
echo "[INFO] CUDA  : $(nvcc --version | tail -n 1)"
echo "[INFO] MPI   : $(command -v mpicxx)"

# ------------------------------------------------------------
# 2. Detect CUDA architecture
# ------------------------------------------------------------

detect_cuda_arch() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local capability

        capability=$(nvidia-smi \
            --query-gpu=compute_cap \
            --format=csv,noheader,nounits \
            2>/dev/null | head -n 1 || true)

        capability=$(echo "$capability" | tr -d '[:space:]')

        if [[ "$capability" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
            echo "${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
            return 0
        fi
    fi

    return 1
}

if [[ -n "${CUDA_ARCH:-}" ]]; then
    echo "[INFO] CUDA architecture manually selected."
else
    CUDA_ARCH=$(detect_cuda_arch) || {
        echo "[ERROR] Unable to detect GPU architecture."
        echo "Set CUDA_ARCH manually, for example:"
        echo "  CUDA_ARCH=90 bash install.sh"
        exit 1
    }
fi

echo "[INFO] CUDA Architecture : sm_${CUDA_ARCH}"

# ------------------------------------------------------------
# 3. Validate CUDA architecture
# ------------------------------------------------------------

if ! [[ "$CUDA_ARCH" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid CUDA_ARCH: $CUDA_ARCH"
    exit 1
fi

CUDA_MAJOR=$(nvcc --version |
    sed -n 's/.*release \([0-9]*\)\..*/\1/p')

if [[ "$CUDA_MAJOR" =~ ^[0-9]+$ ]] &&
   (( CUDA_MAJOR >= 13 && CUDA_ARCH < 75 )); then
    echo "[ERROR] CUDA ${CUDA_MAJOR} does not support sm_${CUDA_ARCH}."
    echo "Use a supported GPU architecture or older CUDA toolkit."
    exit 1
fi

# ------------------------------------------------------------
# 4. Build configuration
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
JOBS="${JOBS:-8}"

echo "[INFO] Source directory : ${SCRIPT_DIR}"
echo "[INFO] Build directory  : ${BUILD_DIR}"
echo "[INFO] Parallel jobs    : ${JOBS}"

# ------------------------------------------------------------
# 5. Clean previous CMake configuration
# ------------------------------------------------------------

echo "[INFO] Cleaning previous build..."

if [[ "$BUILD_DIR" == "$SCRIPT_DIR" || "$BUILD_DIR" == "/" ]]; then
    echo "[ERROR] Unsafe build directory."
    exit 1
fi

rm -rf -- "$BUILD_DIR"

# ------------------------------------------------------------
# 6. Configure MGXBench
# ------------------------------------------------------------

echo "[INFO] Configuring MGXBench..."
echo ""
echo "cmake -S . -B build \\"
echo "    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \\"
echo "    -DCMAKE_CXX_STANDARD=17 \\"
echo "    -DCMAKE_CXX_STANDARD_REQUIRED=ON \\"
echo "    -DCMAKE_CUDA_STANDARD=17 \\"
echo "    -DCMAKE_CUDA_STANDARD_REQUIRED=ON \\"
echo "    -DCMAKE_BUILD_TYPE=Release"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DCMAKE_CUDA_STANDARD=17 \
    -DCMAKE_CUDA_STANDARD_REQUIRED=ON \
    -DCMAKE_BUILD_TYPE=Release

# ------------------------------------------------------------
# 7. Compile
# ------------------------------------------------------------

echo "[INFO] Building MGXBench..."

cmake --build "$BUILD_DIR" \
    --parallel "$JOBS"

# ------------------------------------------------------------
# 8. Verify executables
# ------------------------------------------------------------

echo "[INFO] Checking executables..."

for executable in mm_1d mm_2d; do
    if [[ -x "${BUILD_DIR}/${executable}" ]]; then
        echo "[OK] ${executable}"
    else
        echo "[ERROR] ${executable} not found."
        exit 1
    fi
done

# ------------------------------------------------------------
# 9. Build summary
# ------------------------------------------------------------

echo "============================================================"
echo " MGXBench | Build completed successfully"
echo "============================================================"
echo " CUDA Architecture : sm_${CUDA_ARCH}"
echo " C++ Standard      : C++17"
echo " CUDA C++ Standard : C++17"
echo " Build Directory   : ${BUILD_DIR}"
echo " Executables:"
echo "   ${BUILD_DIR}/mm_1d"
echo "   ${BUILD_DIR}/mm_2d"
echo "============================================================"

# ------------------------------------------------------------
# 10. Installation success message
# ------------------------------------------------------------

printf 'MGXBench installation successful.\nExecutables: build/mm_1d and build/mm_2d\nRun: python3 benchmark.py --decomposition 1d --size 4096 --ranks 4 --libraries MMM YYY CCC XXX NNN WWW SSS --warmup 2 --repeats 3 --output results/1d\n'



#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
printf "\n%s\n" "============================================================" "  MGXBench | Multi-GPU Execution Benchmark" "  Installer | CUDA / MPI / NCCL / NVSHMEM" "============================================================"

# Usage: bash install.sh [--modules] [--clean]
# --modules: load tested SDumont modules in this shell (if available).
# --clean: remove CMake build directory before configuring.
LOAD_MODULES=0
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --modules) LOAD_MODULES=1 ;;
    --clean) CLEAN=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if (( LOAD_MODULES )); then
  if ! type module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
    fi
  fi
  type module >/dev/null 2>&1 || { echo "Environment Modules is unavailable. Load modules in your login shell first." >&2; exit 1; }
  module load gcc/9.3_sequana
  module load cuda/11.2_sequana
  module load openmpi/gnu/4.1.4+cuda-11.2_sequana
  module load nccl/2.13_cuda-11.2_sequana
  module load nvshmem/3.1.7_cuda-11.2_sequana
  module load cmake/3.30.3_sequana
fi

for tool in cmake nvcc mpicxx g++ python3; do
  command -v "$tool" >/dev/null || { echo "Missing required command: $tool" >&2; exit 1; }
done

if [[ -z "${NVSHMEM_HOME:-}" || ! -f "$NVSHMEM_HOME/lib/libnvshmem.a" ]]; then
  echo "Expected SDumont NVSHMEM monolithic archive at NVSHMEM_HOME/lib/libnvshmem.a" >&2
  echo "Load the NVSHMEM module and check: find "\$NVSHMEM_HOME" -name 'libnvshmem*'" >&2
  exit 1
fi
if (( CLEAN )); then rm -rf build; fi

cmake -S . -B build \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-70}" \
  -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)" \
  -DNVSHMEM_LIBRARY="$NVSHMEM_HOME/lib/libnvshmem.a"
cmake --build build -j "${BUILD_JOBS:-4}"
for exe in build/mm_1d build/mm_2d; do
  [[ -x "$exe" ]] || { echo "Executable not generated: $exe" >&2; exit 1; }
done
printf '\nMGXBench installation successful.\nExecutables: build/mm_1d and build/mm_2d\nRun: python3 benchmark.py --decomposition 1d --size 4096 --ranks 4 --libraries MMM YYY CCC XXX NNN WWW SSS --warmup 2 --repeats 3 --output results/1d\n'

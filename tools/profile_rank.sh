#!/usr/bin/env bash
# Profile a SEPARATE diagnostic run, never include profiler timings in benchmark statistics.
set -euo pipefail
: "${MGXBENCH_PROFILE_DIR:?Set an absolute shared output directory}"
command -v nsys >/dev/null || { echo 'NVIDIA Nsight Systems is required' >&2; exit 1; }
mkdir -p "$MGXBENCH_PROFILE_DIR"
rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-}}}"
[[ -n "$rank" ]] || { echo 'Cannot determine rank for unique profile filename' >&2; exit 1; }
exec nsys profile --trace=cuda,mpi,nvtx --force-overwrite=false -o "$MGXBENCH_PROFILE_DIR/rank-$rank" "$@"

#!/usr/bin/env bash
# Analyze an existing Nsight report; no GPU, MPI launch or graphical UI required.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$project_dir/configuration/plot_nsys_stats.py" "$@"

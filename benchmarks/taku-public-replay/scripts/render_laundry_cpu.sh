#!/usr/bin/env bash
set -euo pipefail

candidate_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 3 ]]; then
  echo 'Usage: render_laundry_cpu.sh PREPARED_DIRECTORY OUTPUT_DIRECTORY OSMESA_LIBRARY_DIRECTORY' >&2
  exit 2
fi
ulimit -v 6291456
export CUDA_VISIBLE_DEVICES=''
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LP_NUM_THREADS=2
export PYTHONNOUSERSITE=1
export DISPLAY=''
export LD_LIBRARY_PATH="$3"

exec timeout --signal=TERM --kill-after=5s 600s taskset -c 84,85 \
  "$candidate_dir/runtime/.venv/bin/taku-public-replay" render \
  --prepared "$1" \
  --output "$2" \
  --wall-seconds 540

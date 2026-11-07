#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${WOOF_STEP_RUNTIME:?set the local runtime root}"
ulimit -n 65536
export CUPY_GPU_MEMORY_LIMIT=21474836480
export WOOF_NATIVE_STEP_LIBRARY=$PWD/receipts/libawstep-host-argument.so
export LD_PRELOAD=$WOOF_NATIVE_STEP_LIBRARY
export CUDA_PATH=$WOOF_STEP_RUNTIME/venv/lib/python3.11/site-packages/nvidia/cu13
export LD_LIBRARY_PATH=$CUDA_PATH/lib
export PYTHONPATH=$PWD:$WOOF_STEP_RUNTIME/data
export WOOF_STEP_BINDING=$PWD/receipts/canary-local-binding.json
export CUPY_CACHE_DIR=$PWD/cache/cupy
# The engine plan cache does not follow XDG_CACHE_HOME. Keep replayed
# compiler include paths on this runtime, away from shared-home plans.
export WOOF_KERNEL_PLAN_DIR=$PWD/cache/kernel-plans
exec "$WOOF_STEP_RUNTIME/venv/bin/python" -B tools/native_step/run_swiss_step.py "$@"

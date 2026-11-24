#!/usr/bin/env bash
# Four roots locate everything outside the repository.  Set them, or accept the
# defaults below, which assume sibling directories next to the checkout.
#
#   INTEGRAL_REPO   this checkout
#   INTEGRAL_VENV   the Python environment (torch, triton, fla, transformers)
#   INTEGRAL_RUNS   where training writes runs/<name>/checkpoint-<step>.pt
#   INTEGRAL_DATA   tokenized FineWeb-Edu and the HF cache
#
# The Triton kernels are tuned for Blackwell (sm 10.3), Hopper (sm 9.0) and
# A6000 (sm 8.6); integral_deltanet/ops/tuning holds the launch tables per GPU
# family.  On other architectures the fused paths fall back to the unfused ones,
# which are numerically equivalent and slower; nothing needs to change for that
# to happen.
INTEGRAL_REPO="${INTEGRAL_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export INTEGRAL_REPO
export INTEGRAL_VENV="${INTEGRAL_VENV:-$INTEGRAL_REPO/../venv}"
export INTEGRAL_RUNS="${INTEGRAL_RUNS:-$INTEGRAL_REPO/../runs}"
export INTEGRAL_DATA="${INTEGRAL_DATA:-$INTEGRAL_REPO/../data}"
export INTEGRAL_ROOT="${INTEGRAL_ROOT:-$INTEGRAL_REPO/..}"

export PY="${PY:-$INTEGRAL_VENV/bin/python}"
export PYTHONPATH="$INTEGRAL_REPO:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="$INTEGRAL_DATA/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export FINEWEB_EDU_LOCAL_ROOT="$INTEGRAL_DATA/fineweb-edu"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Slurm reads this, so the batch scripts carry no site-specific partition name.
# Set it to a partition that has the GPUs; leave it unset off Slurm.
export SBATCH_PARTITION="${SBATCH_PARTITION:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$INTEGRAL_DATA/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$INTEGRAL_DATA/cache/inductor}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Which kernel mode the readout path runs.  Both compute the same
# function; they differ in what the forward pass keeps for the backward pass.
#
#   throughput-first  store the mixer's intermediates (the default, and the
#                     setting the paper's throughput numbers were measured with)
#   memory-first      recompute the mixer body during backpropagation, which
#                     drops activation memory well below throughput-first at a
#                     moderate slowdown (on A6000, only the readout operator)
#
# This is independent of `gradient_checkpointing`, which recomputes whole
# blocks including the FFN and is available to every model.
export INTEGRAL_KERNEL_MODE="${INTEGRAL_KERNEL_MODE:-throughput-first}"

# Readout-path kernel selection.  These are the values every number in the
# paper was produced with; they are defaults, not tuning knobs.
export INTEGRAL_READOUT_KERNEL=rows
export INTEGRAL_READOUT_DOT=tf32
export INTEGRAL_READOUT_BUILDER=triton
export INTEGRAL_READOUT_SCAN=triangular
export INTEGRAL_READOUT_LOCAL=pairwise

# Fused paths for Blackwell and Hopper, the gate fusions also for A6000.  All
# are opt-in, and each falls back to a numerically equivalent
# unfused path when the device, dtype or gate configuration does not qualify.
export INTEGRAL_FUSE_GATE_PAIR="${INTEGRAL_FUSE_GATE_PAIR:-0}"
export INTEGRAL_FUSE_DECAY_GATE="${INTEGRAL_FUSE_DECAY_GATE:-0}"
export INTEGRAL_FUSE_INPUT_PROJ="${INTEGRAL_FUSE_INPUT_PROJ:-0}"
export INTEGRAL_MF_NOGRAD_FWD="${INTEGRAL_MF_NOGRAD_FWD:-0}"
export INTEGRAL_AUTO_OP_CHUNK="${INTEGRAL_AUTO_OP_CHUNK:-0}"

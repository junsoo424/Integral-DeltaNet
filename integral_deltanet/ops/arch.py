"""Which architectures run the tuned fast paths.

B300 (SM103) and GH200 (SM90) run every fast path: the readout ones (the
reload backward, the head-major scan output and the pinned readout scan
tiles), the fused gate kernels and the C16 register policy.  A6000 (SM86)
runs the readout fast paths and the fused gates, plus four paths of its own;
it keeps the split C16 kernels and the separate input projections.  Every
other architecture keeps the generic paths, which compute the same function.
The launch tables (tiles, warps, stages) live per GPU family in
``ops/tuning``.

The generated row kernels hold each of their implementations behind constexpr
flags (ops/gen_merge.py); the selectors at the end of this module return the
flags a call site passes, as FLA's ``triton.heuristics`` pick its kernels'
branches.
"""
from __future__ import annotations

import os

import torch

from .tuning import by_arch

B300 = (10, 3)
GH200 = (9, 0)
A6000 = (8, 6)

# Every fast path: the readout ones, the fused gates, the C16 register policy
# and the one-GEMM input projection.
TUNED_ARCHS = (B300, GH200)
# The readout fast paths and the fused gate-pair / decay-gate kernels.
READOUT_ARCHS = {*TUNED_ARCHS, A6000}
FUSED_GATE_ARCHS = {*TUNED_ARCHS, A6000}
# A6000 only:
# - the block RMSNorm as one Triton kernel per direction (ops/rmsnorm_triton.py)
#   instead of the PyTorch norm's dozen fp32 passes per row;
TRITON_NORM_ARCHS = {A6000}
# - the three short convolutions of a ``fuse_qkv_conv`` recipe run separately:
#   the dense convolution over the concatenated projections costs as much as
#   the three, so the concatenations around it are pure overhead;
SEPARATE_QKV_CONV_ARCHS = {A6000}
# - the readout path's transport skips its constant-zero second erase operand
#   (the READOUT_GATE kernel paths; exact, every dropped term is a zero).
TRANSPORT_READOUT_GATE_ARCHS = {A6000}
# - memory-first recomputes only the readout operator (operand builders, block
#   composition, solve and two-state scan) and stores the rest of the mixer,
#   instead of recomputing the whole mixer body.
READOUT_RECOMPUTE_ARCHS = {A6000}

# Shortest sequence that takes the readout fast paths: T=4096 on B300 and
# GH200, every length on A6000.
READOUT_MIN_LENGTH = {A6000: 0}
_DEFAULT_READOUT_MIN_LENGTH = 4096


def on_arch(tensor: torch.Tensor, archs) -> bool:
    """Whether ``tensor`` is on a CUDA device whose capability is in ``archs``."""
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device) in archs


def tuned_arch(tensor: torch.Tensor) -> bool:
    return on_arch(tensor, TUNED_ARCHS)


def optimized_readout(tensor: torch.Tensor, length: int) -> bool:
    """Whether the readout fast paths apply to a ``length``-token sequence here.

    ``INTEGRAL_READOUT_OPTIMIZED=0`` turns them off and ``force`` takes them at
    every length; either way only on ``READOUT_ARCHS``, and each path also
    needs its own launch table where it has one.
    """
    flag = os.environ.get("INTEGRAL_READOUT_OPTIMIZED", "1")
    if flag == "0" or not on_arch(tensor, READOUT_ARCHS):
        return False
    if flag == "force":
        return True
    capability = torch.cuda.get_device_capability(tensor.device)
    return length >= READOUT_MIN_LENGTH.get(capability, _DEFAULT_READOUT_MIN_LENGTH)


# The CP8 register-reloading readout backward runs where the GPU has a
# RELOAD_BWD launch table; RELOAD_BWD_PRODUCT adds its running-product decays.
RELOAD_BWD_CONFIGS = by_arch("RELOAD_BWD")
RELOAD_BWD_PRODUCT_ARCHS = {cap for cap, on in by_arch("RELOAD_BWD_PRODUCT").items() if on}


def readout_rows_forward_flags() -> dict[str, bool]:
    """Flags of ``_readout_rows_fwd_kernel``: PRODUCT (running-product decays)
    unless ``INTEGRAL_READOUT_FORWARD=exp`` asks for relative exponentials."""
    return {"PRODUCT": os.environ.get("INTEGRAL_READOUT_FORWARD") != "exp"}


def readout_rows_backward_flags(tensor: torch.Tensor, length: int, key_dim: int) -> dict[str, bool]:
    """Flags of ``_readout_rows_bwd_kernel``.

    RELOAD (the CP8 reload backward) at K=128 on a GPU with a RELOAD_BWD table,
    where the readout fast paths apply at ``length``; PRODUCT with it on
    RELOAD_BWD_PRODUCT GPUs, or as ``INTEGRAL_READOUT_BWD_PRODUCT=0/1`` forces.
    """
    capability = torch.cuda.get_device_capability(tensor.device) if tensor.is_cuda else None
    reload = (key_dim == 128 and capability in RELOAD_BWD_CONFIGS
              and optimized_readout(tensor, length))
    forced = os.environ.get("INTEGRAL_READOUT_BWD_PRODUCT")
    product = reload and (forced == "1" if forced is not None
                          else capability in RELOAD_BWD_PRODUCT_ARCHS)
    return {"RELOAD": reload, "PRODUCT": product}


def c16_fused(tensor: torch.Tensor | None = None) -> bool:
    """FUSED_PREFIX of the direct C16 row kernels: one kernel per direction
    with the prefix fused in.

    Default on TUNED_ARCHS; other architectures keep the split kernels.
    ``INTEGRAL_C16=fused|split`` overrides either way.
    """
    override = os.environ.get("INTEGRAL_C16")
    if override is not None:
        if override not in ("fused", "split"):
            raise ValueError("INTEGRAL_C16 must be 'fused' or 'split'")
        return override == "fused"
    return tensor is not None and tuned_arch(tensor)


__all__ = [
    "FUSED_GATE_ARCHS",
    "READOUT_ARCHS",
    "READOUT_MIN_LENGTH",
    "READOUT_RECOMPUTE_ARCHS",
    "RELOAD_BWD_CONFIGS",
    "RELOAD_BWD_PRODUCT_ARCHS",
    "SEPARATE_QKV_CONV_ARCHS",
    "TRANSPORT_READOUT_GATE_ARCHS",
    "TRITON_NORM_ARCHS",
    "TUNED_ARCHS",
    "c16_fused",
    "on_arch",
    "optimized_readout",
    "readout_rows_backward_flags",
    "readout_rows_forward_flags",
    "tuned_arch",
]

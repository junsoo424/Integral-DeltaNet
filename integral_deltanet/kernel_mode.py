"""Which of the two kernels the integral readout path runs.

Both kernels compute the same function; they differ in what the forward pass
keeps for the backward pass.

``throughput-first``
    Store the mixer's intermediates.  This is the default and the setting the
    throughput numbers in the paper were measured with.

``memory-first``
    Recompute the mixer body during backpropagation instead of storing it.
    Activation memory drops well below the throughput-first kernel at the cost
    of a moderate slowdown.  On A6000 (``READOUT_RECOMPUTE_ARCHS`` in
    ``ops/arch.py``) it recomputes only the readout operator (operand
    builders, block composition, solve and two-state scan) and stores the
    projections, convolution, normalisations and gates.

The backbone kernels this work builds on already recompute their own
kernel-internal intermediates.  ``memory-first`` is a wider scope than that: it
recomputes the whole mixer body, projections, convolution, normalisations and
gates included, so almost nothing the mixer produces is held until backward.

Select a mode with ``INTEGRAL_KERNEL_MODE``, or per model with ``kernel_mode``
in the model config.  This is orthogonal to ``gradient_checkpointing``, which
recomputes whole blocks including the FFN and is available to every model.
"""
from __future__ import annotations

import os

THROUGHPUT_FIRST = "throughput-first"
MEMORY_FIRST = "memory-first"
MODES = (THROUGHPUT_FIRST, MEMORY_FIRST)

ENV_VAR = "INTEGRAL_KERNEL_MODE"
DEFAULT = THROUGHPUT_FIRST


def normalize(mode: str | None) -> str:
    """Accept a mode name, also with '_' for '-', and reject anything else."""
    if mode is None:
        return DEFAULT
    candidate = str(mode).strip().lower().replace("_", "-")
    if candidate not in MODES:
        raise ValueError(
            f"unknown kernel mode {mode!r}; expected one of {', '.join(MODES)}"
        )
    return candidate


def kernel_mode(config: dict | None = None) -> str:
    """The active mode: the environment wins, then the config, then the default.

    The environment comes first so the mode can be switched without editing
    configs.
    """
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return normalize(from_env)
    if config is not None and config.get("kernel_mode") is not None:
        return normalize(config.get("kernel_mode"))
    return DEFAULT


def recomputes_mixer_body(config: dict | None = None) -> bool:
    return kernel_mode(config) == MEMORY_FIRST


__all__ = [
    "THROUGHPUT_FIRST",
    "MEMORY_FIRST",
    "MODES",
    "ENV_VAR",
    "DEFAULT",
    "normalize",
    "kernel_mode",
    "recomputes_mixer_body",
]

"""Where the channel-wise direct C16 kernels recompute their segment starts."""
from __future__ import annotations

import os

import torch

from .arch import tuned_arch


_FORCE_RECOMPUTE = "INTEGRAL_DELTANET_FORCE_RECOMPUTE_SEGMENTS"


def needs_recomputed_segment_starts(tensor: torch.Tensor) -> bool:
    """Avoid long-lived Triton checkpoint registers on the tuned architectures."""
    if os.environ.get(_FORCE_RECOMPUTE) == "1":
        return True
    return tuned_arch(tensor)


__all__ = ["needs_recomputed_segment_starts"]

"""Low-level kernels for the Integral DeltaNet paper recurrence."""

from .chunk import (
    chunk_integral_delta,
    chunk_integral_gdn2,
    chunk_integral_kda,
)
from .reference import integral_delta_reference
from .readout import (
    chunk_readout_delta,
    chunk_readout_gdn2,
    chunk_readout_kda,
    readout_delta_reference,
)
from .preprocess_triton import (
    fused_channel_preprocess,
    fused_scalar_preprocess,
)

__all__ = [
    "chunk_integral_delta",
    "chunk_integral_gdn2",
    "chunk_integral_kda",
    "integral_delta_reference",
    "chunk_readout_delta",
    "chunk_readout_gdn2",
    "chunk_readout_kda",
    "readout_delta_reference",
    "fused_channel_preprocess",
    "fused_scalar_preprocess",
]

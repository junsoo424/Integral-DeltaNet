"""A portable RMSNorm for the block skeleton.

``fla.modules.RMSNorm`` dispatches to ``layer_norm_fwd_kernel1``, which fails to
launch under Triton 3.7.1 (``TypeError: failed to specialize argument of type:
str``), the version the Blackwell kernels use.  The block norms are a
rounding error next to the mixers, so rather than take on
another kernel dependency this computes the norm in PyTorch, in float32, which
runs identically on every Triton version.

The call signature matches the subset of FLA's RMSNorm the block uses,
including the fused ``(x, residual, prenorm)`` form that returns the updated
residual alongside the normalised output.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from .ops.arch import TRITON_NORM_ARCHS, on_arch

# On TRITON_NORM_ARCHS the norm runs as one fused Triton kernel per direction
# (ops/rmsnorm_triton.py) instead of the PyTorch ops below.
# INTEGRAL_TRITON_NORM=0 turns it off.
_TRITON_NORM_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _use_triton_norm(x, residual, prenorm):
    return (
        os.environ.get("INTEGRAL_TRITON_NORM", "1") != "0"
        and not (prenorm and residual is None)
        and x.dtype in _TRITON_NORM_DTYPES
        and (residual is None or residual.dtype in _TRITON_NORM_DTYPES)
        and x.shape[-1] <= 16384
        and on_arch(x, TRITON_NORM_ARCHS)
    )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if _use_triton_norm(x, residual, prenorm):
            from .ops.rmsnorm_triton import fused_rmsnorm

            return fused_rmsnorm(x, residual, self.weight, self.eps, prenorm)
        if residual is not None:
            x = x + residual
        residual_out = x
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        y = (x * self.weight.float()).to(dtype)
        return (y, residual_out) if prenorm else y

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"

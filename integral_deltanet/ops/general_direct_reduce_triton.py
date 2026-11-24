"""One-pass FP32 reduction for joint prefix and relative gate gradients."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_joint_scalar_grads_kernel(
    REL_DGAMMA,
    REL_DLAM,
    PREFIX_DGAMMA,
    PREFIX_DLAM,
    DGAMMA,
    DLAM,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    REL_NK: tl.constexpr,
    PREFIX_NK: tl.constexpr,
    BLOCK_NK: tl.constexpr,
):
    token_system = tl.program_id(0)
    row = token_system % C
    system = token_system // C
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    key_blocks = tl.arange(0, BLOCK_NK)
    relative_offsets = (system * REL_NK + key_blocks) * C + row
    prefix_offsets = (system * PREFIX_NK + key_blocks) * C + row
    relative_mask = key_blocks < REL_NK
    prefix_mask = key_blocks < PREFIX_NK
    dgamma = tl.sum(
        tl.load(
            REL_DGAMMA + relative_offsets,
            mask=relative_mask,
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    dgamma += tl.sum(
        tl.load(
            PREFIX_DGAMMA + prefix_offsets,
            mask=prefix_mask,
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    dlam = tl.sum(
        tl.load(
            REL_DLAM + relative_offsets,
            mask=relative_mask,
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    dlam += tl.sum(
        tl.load(
            PREFIX_DLAM + prefix_offsets,
            mask=prefix_mask,
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    output = (batch * T + chunk * C + row) * H + head
    tl.store(DGAMMA + output, dgamma)
    tl.store(DLAM + output, dlam)


def reduce_joint_scalar_grads(
    relative_dgamma,
    relative_dlam,
    prefix_dgamma,
    prefix_dlam,
    gamma,
    lam,
    *,
    batch,
    length,
    heads,
    chunks,
    chunk_size,
):
    """Reduce both recurrences in FP32 and cast each input gradient once."""
    relative_nk = relative_dgamma.shape[1]
    prefix_nk = prefix_dgamma.shape[1]
    dgamma = torch.empty_like(gamma)
    dlam = torch.empty_like(lam)
    block_nk = triton.next_power_of_2(max(relative_nk, prefix_nk))
    _reduce_joint_scalar_grads_kernel[(batch * heads * length,)](
        relative_dgamma,
        relative_dlam,
        prefix_dgamma,
        prefix_dlam,
        dgamma,
        dlam,
        T=length,
        H=heads,
        N=chunks,
        C=chunk_size,
        REL_NK=relative_nk,
        PREFIX_NK=prefix_nk,
        BLOCK_NK=block_nk,
        num_warps=1,
        num_stages=1,
    )
    return dgamma, dlam


__all__ = ["reduce_joint_scalar_grads"]

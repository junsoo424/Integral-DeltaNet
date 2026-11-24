"""Inverse-free builders for integral KDA and GDN-2.

Every diagonal block is the exact direct-relative C16 recurrence.  C32/C64
reuse the optimized block transport, TensorCore off-diagonal products, fused
write solve, and scan; only forward 2x2 products cross a C16 boundary.
"""
from __future__ import annotations

import torch

from .general_direct_block_triton import _GeneralDirectBlockedIntra
from .general_direct_c16_triton import stable_direct_c16
from .general_fused_write_triton import solve_general_operands_fused_write


_LOCAL_BLOCK = 16


@torch.compiler.disable
def build_general_direct_operands_triton(
    q, k, v, g, b, w, gamma, lam, scale, chunk_size
):
    """Build stable BF16 operands without inverse or centered coordinates."""
    if chunk_size not in (16, 32, 64) or q.shape[1] % chunk_size:
        raise ValueError("direct general builder requires C16/C32/C64")
    (
        local_a1,
        local_a2,
        local_q1,
        local_q2,
        local_transport,
        local_tail1,
        local_tail2,
        local_matrix,
        local_causal,
    ) = stable_direct_c16(q, k, g, b, gamma, lam, scale)

    if chunk_size == _LOCAL_BLOCK:
        u0, state_weight1, state_weight2 = solve_general_operands_fused_write(
            local_matrix, v, w, local_a1, local_a2
        )
        return (
            u0,
            state_weight1,
            state_weight2,
            local_q1,
            local_q2,
            local_tail1,
            local_tail2,
            local_causal,
            local_transport,
        )

    # Direct diagonals are already complete.  The transport stores only strict
    # lower pairs; transported tails form every off-diagonal block.
    from .general_block_triton import _compose_general_blocks_triton

    empty_coordinates = local_tail1.new_empty(0)
    transported = _compose_general_blocks_triton(
        local_a1,
        local_a2,
        empty_coordinates,
        empty_coordinates,
        local_q1,
        local_q2,
        local_tail1,
        local_tail2,
        local_transport,
        chunk_size=chunk_size,
        direct=True,
    )
    (
        global_a1,
        global_a2,
        global_q1,
        global_q2,
        pair_c1,
        pair_c2,
        tail1,
        tail2,
        transport,
    ) = transported
    matrix, causal = _GeneralDirectBlockedIntra.apply(
        local_a1,
        local_a2,
        local_q1,
        local_q2,
        pair_c1,
        pair_c2,
        local_matrix,
        local_causal,
        chunk_size,
    )
    u0, state_weight1, state_weight2 = solve_general_operands_fused_write(
        matrix, v, w, global_a1, global_a2
    )
    return (
        u0,
        state_weight1,
        state_weight2,
        global_q1,
        global_q2,
        tail1,
        tail2,
        causal,
        transport,
    )


__all__ = ["build_general_direct_operands_triton"]

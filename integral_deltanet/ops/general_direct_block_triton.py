"""Direct C32/C64 intra-chunk assembly without diagonal recomputation.

The inverse-free C16 recurrence already produces complete diagonal matrix and
causal blocks.  This module copies those blocks exactly and launches TensorCore
products only for strict lower off-diagonal block pairs.  Backward mirrors the
same split: diagonal gradients are copied to the C16 recurrence while GEMMs
differentiate only real off-diagonal work.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import safe_dot

from .general_block_triton import (
    _INTRA_BACKWARD_AUTOTUNE_CONFIGS,
    _INTRA_CONFIGS,
    _pinned_intra_configs,
)


_LOCAL_BLOCK = 16


@triton.jit
def _direct_block_diagonal_fwd_kernel(
    LOCAL_MATRIX,
    LOCAL_CAUSAL,
    MATRIX,
    CAUSAL,
    BLOCKS: tl.constexpr,
    C: tl.constexpr,
):
    pid = tl.program_id(0)
    block = pid % BLOCKS
    outer = pid // BLOCKS
    rows = tl.arange(0, 16)[:, None]
    cols = tl.arange(0, 16)[None, :]
    local_offsets = ((outer * BLOCKS + block) * 16 + rows) * 16 + cols
    global_offsets = (
        outer * C * C
        + (block * 16 + rows) * C
        + block * 16
        + cols
    )
    tl.store(MATRIX + global_offsets, tl.load(LOCAL_MATRIX + local_offsets))
    tl.store(CAUSAL + global_offsets, tl.load(LOCAL_CAUSAL + local_offsets))


@triton.autotune(configs=_INTRA_CONFIGS, key=["K", "BLOCKS", "READOUT_GATE"])
@triton.jit
def _direct_block_offdiag_fwd_kernel(
    A1,
    A2,
    Q1,
    Q2,
    PAIR_C1,
    PAIR_C2,
    MATRIX,
    CAUSAL,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAIRS: tl.constexpr,
    OFF_PAIRS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    READOUT_GATE: tl.constexpr = False,
):
    pid = tl.program_id(0)
    pair = pid % OFF_PAIRS
    outer = pid // OFF_PAIRS
    if BLOCKS == 2:
        target = 1
        source = 0
    else:
        target = tl.where(pair < 1, 1, tl.where(pair < 3, 2, 3))
        source = pair - target * (target - 1) // 2
    rows = tl.arange(0, 16)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    target_token = outer * C + target * 16 + rows
    pair_token = outer * PAIRS * 16 + pair * 16 + rows
    matrix = tl.zeros((16, 16), dtype=tl.float32)
    causal = tl.zeros((16, 16), dtype=tl.float32)
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        target_offsets = target_token[:, None] * K + keys[None, :]
        source_offsets = pair_token[:, None] * K + keys[None, :]
        a1 = tl.load(A1 + target_offsets, mask=key_mask[None, :], other=0.0)
        if not READOUT_GATE:
            a2 = tl.load(A2 + target_offsets, mask=key_mask[None, :], other=0.0)
        q1 = tl.load(Q1 + target_offsets, mask=key_mask[None, :], other=0.0)
        q2 = tl.load(Q2 + target_offsets, mask=key_mask[None, :], other=0.0)
        c1 = tl.load(PAIR_C1 + source_offsets, mask=key_mask[None, :], other=0.0)
        c2 = tl.load(PAIR_C2 + source_offsets, mask=key_mask[None, :], other=0.0)
        matrix += safe_dot(a1, tl.trans(c1), allow_tf32=False)
        if not READOUT_GATE:
            matrix += safe_dot(a2, tl.trans(c2), allow_tf32=False)
        causal += safe_dot(q1, tl.trans(c1), allow_tf32=False)
        causal += safe_dot(q2, tl.trans(c2), allow_tf32=False)
    offsets = (
        outer * C * C
        + (target * 16 + row_ids) * C
        + source * 16
        + col_ids
    )
    tl.store(MATRIX + offsets, matrix)
    tl.store(CAUSAL + offsets, causal)
    if pair == 0:
        for upper_target in range(BLOCKS):
            for upper_source in range(upper_target + 1, BLOCKS):
                upper_offsets = (
                    outer * C * C
                    + (upper_target * 16 + row_ids) * C
                    + upper_source * 16
                    + col_ids
                )
                tl.store(MATRIX + upper_offsets, 0.0)
                tl.store(CAUSAL + upper_offsets, 0.0)


@triton.autotune(
    configs=_INTRA_BACKWARD_AUTOTUNE_CONFIGS,
    key=["K", "BLOCKS", "TARGET", "READOUT_GATE"],
)
@triton.jit
def _direct_block_intra_bwd_target_kernel(
    A1,
    A2,
    Q1,
    Q2,
    PAIR_C1,
    PAIR_C2,
    DMATRIX,
    DCAUSAL,
    DA1,
    DA2,
    DQ1,
    DQ2,
    DPAIR_C1,
    DPAIR_C2,
    DLOCAL_MATRIX,
    DLOCAL_CAUSAL,
    K: tl.constexpr,
    TARGET: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAIRS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    READOUT_GATE: tl.constexpr = False,
):
    outer = tl.program_id(0)
    rows = tl.arange(0, 16)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    target_token = outer * C + TARGET * 16 + rows

    diagonal_offsets = (
        outer * C * C
        + (TARGET * 16 + row_ids) * C
        + TARGET * 16
        + col_ids
    )
    local_offsets = ((outer * BLOCKS + TARGET) * 16 + row_ids) * 16 + col_ids
    tl.store(DLOCAL_MATRIX + local_offsets, tl.load(DMATRIX + diagonal_offsets))
    tl.store(DLOCAL_CAUSAL + local_offsets, tl.load(DCAUSAL + diagonal_offsets))

    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        target_offsets = target_token[:, None] * K + keys[None, :]
        a1 = tl.load(A1 + target_offsets, mask=key_mask[None, :], other=0.0)
        if not READOUT_GATE:
            a2 = tl.load(A2 + target_offsets, mask=key_mask[None, :], other=0.0)
        q1 = tl.load(Q1 + target_offsets, mask=key_mask[None, :], other=0.0)
        q2 = tl.load(Q2 + target_offsets, mask=key_mask[None, :], other=0.0)
        da1 = tl.zeros((16, BK), dtype=tl.float32)
        da2 = tl.zeros((16, BK), dtype=tl.float32)
        dq1 = tl.zeros((16, BK), dtype=tl.float32)
        dq2 = tl.zeros((16, BK), dtype=tl.float32)

        for source in tl.static_range(0, TARGET):
            pair = TARGET * (TARGET - 1) // 2 + source
            pair_token = outer * PAIRS * 16 + pair * 16 + rows
            source_offsets = pair_token[:, None] * K + keys[None, :]
            c1 = tl.load(PAIR_C1 + source_offsets, mask=key_mask[None, :], other=0.0)
            c2 = tl.load(PAIR_C2 + source_offsets, mask=key_mask[None, :], other=0.0)
            matrix_offsets = (
                outer * C * C
                + (TARGET * 16 + row_ids) * C
                + source * 16
                + col_ids
            )
            grad_matrix = tl.load(DMATRIX + matrix_offsets).to(c1.dtype)
            grad_causal = tl.load(DCAUSAL + matrix_offsets).to(q1.dtype)
            da1 += safe_dot(grad_matrix, c1, allow_tf32=False)
            if not READOUT_GATE:
                da2 += safe_dot(grad_matrix, c2, allow_tf32=False)
            dq1 += safe_dot(grad_causal, c1, allow_tf32=False)
            dq2 += safe_dot(grad_causal, c2, allow_tf32=False)
            dc1 = safe_dot(tl.trans(grad_matrix), a1, allow_tf32=False)
            dc1 += safe_dot(tl.trans(grad_causal), q1, allow_tf32=False)
            if READOUT_GATE:
                dc2 = safe_dot(tl.trans(grad_causal), q2, allow_tf32=False)
            else:
                dc2 = safe_dot(tl.trans(grad_matrix), a2, allow_tf32=False)
                dc2 += safe_dot(tl.trans(grad_causal), q2, allow_tf32=False)
            tl.store(DPAIR_C1 + source_offsets, dc1, mask=key_mask[None, :])
            tl.store(DPAIR_C2 + source_offsets, dc2, mask=key_mask[None, :])

        tl.store(DA1 + target_offsets, da1, mask=key_mask[None, :])
        if not READOUT_GATE:
            tl.store(DA2 + target_offsets, da2, mask=key_mask[None, :])
        tl.store(DQ1 + target_offsets, dq1, mask=key_mask[None, :])
        tl.store(DQ2 + target_offsets, dq2, mask=key_mask[None, :])


class _GeneralDirectBlockedIntra(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a1,
        a2,
        q1,
        q2,
        pair_c1,
        pair_c2,
        local_matrix,
        local_causal,
        chunk_size,
        readout_gate=False,
    ):
        ctx.readout_gate = readout_gate
        ctx.argument_count = len(ctx.needs_input_grad)
        if readout_gate:
            a2 = a1  # Dummy pointer; the constant zero second erase path is eliminated.
        tensors = tuple(
            tensor.contiguous()
            for tensor in (
                a1,
                a2,
                q1,
                q2,
                pair_c1,
                pair_c2,
                local_matrix,
                local_causal,
            )
        )
        (
            a1,
            a2,
            q1,
            q2,
            pair_c1,
            pair_c2,
            local_matrix,
            local_causal,
        ) = tensors
        batch, heads, local_chunks, _, key_dim = a1.shape
        blocks = chunk_size // _LOCAL_BLOCK
        chunks = local_chunks // blocks
        pairs = blocks * (blocks - 1) // 2
        off_pairs = pairs
        shape = (batch, heads, chunks, chunk_size, chunk_size)
        matrix = torch.empty(*shape, device=a1.device, dtype=torch.float32)
        causal = torch.empty(*shape, device=a1.device, dtype=local_causal.dtype)
        systems = batch * heads * chunks
        _direct_block_diagonal_fwd_kernel[(systems * blocks,)](
            local_matrix,
            local_causal,
            matrix,
            causal,
            BLOCKS=blocks,
            C=chunk_size,
            num_warps=1,
            num_stages=1,
        )
        pinned = _pinned_intra_configs(a1, key_dim, chunk_size)
        launch = dict(
            K=key_dim,
            BLOCKS=blocks,
            PAIRS=pairs,
            OFF_PAIRS=off_pairs,
            C=chunk_size,
            READOUT_GATE=readout_gate,
        )
        kernel = _direct_block_offdiag_fwd_kernel
        if pinned is not None:
            block_key, warps, stages = pinned[0][chunk_size]
            kernel = getattr(kernel, "fn", kernel)
            launch.update(BK=block_key, num_warps=warps, num_stages=stages)
        kernel[(systems * off_pairs,)](
            a1,
            a2,
            q1,
            q2,
            pair_c1,
            pair_c2,
            matrix,
            causal,
            **launch,
        )
        ctx.save_for_backward(a1, a2, q1, q2, pair_c1, pair_c2)
        ctx.dimensions = (systems, key_dim, blocks, pairs, chunk_size)
        ctx.input_dtypes = tuple(tensor.dtype for tensor in tensors)
        ctx.local_metadata = (
            local_matrix.shape,
            local_matrix.dtype,
            local_causal.shape,
            local_causal.dtype,
        )
        ctx.pinned = pinned
        return matrix, causal

    @staticmethod
    def backward(ctx, dmatrix, dcausal):
        a1, a2, q1, q2, pair_c1, pair_c2 = ctx.saved_tensors
        systems, key_dim, blocks, pairs, chunk_size = ctx.dimensions
        matrix_shape, matrix_dtype, causal_shape, causal_dtype = ctx.local_metadata
        local_grads = tuple(torch.empty_like(tensor) for tensor in (a1, a2, q1, q2))
        pair_grads = tuple(torch.empty_like(tensor) for tensor in (pair_c1, pair_c2))
        dlocal_matrix = torch.empty(
            matrix_shape, device=a1.device, dtype=matrix_dtype
        )
        dlocal_causal = torch.empty(
            causal_shape, device=a1.device, dtype=causal_dtype
        )
        dmatrix = dmatrix.contiguous()
        dcausal = dcausal.contiguous()
        for target in range(blocks):
            launch = dict(
                K=key_dim,
                TARGET=target,
                BLOCKS=blocks,
                PAIRS=pairs,
                C=chunk_size,
                READOUT_GATE=ctx.readout_gate,
            )
            kernel = _direct_block_intra_bwd_target_kernel
            if ctx.pinned is not None:
                block_key, warps, stages = ctx.pinned[1][chunk_size][target]
                kernel = getattr(kernel, "fn", kernel)
                launch.update(BK=block_key, num_warps=warps, num_stages=stages)
            kernel[(systems,)](
                a1,
                a2,
                q1,
                q2,
                pair_c1,
                pair_c2,
                dmatrix,
                dcausal,
                *local_grads,
                *pair_grads,
                dlocal_matrix,
                dlocal_causal,
                **launch,
            )
        grads = (*local_grads, *pair_grads, dlocal_matrix, dlocal_causal)
        converted = tuple(
            grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)
        )
        converted = list(converted)
        if ctx.readout_gate:
            converted[1] = None
        return (*converted, None, None)[:ctx.argument_count]


__all__ = ["_GeneralDirectBlockedIntra"]

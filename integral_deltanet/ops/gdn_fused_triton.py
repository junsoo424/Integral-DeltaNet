"""Fused C16/C32 operand construction for inverse-free scalar GDN."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import safe_dot

from .tuning import by_arch


@triton.jit
def _gdn_direct_builder_fwd_kernel(
    Q,
    K_IN,
    V_IN,
    BETA,
    PI1,
    PI2,
    INTERACTION_COEFF,
    OUTPUT_COEFF,
    TAIL1_COEFF,
    TAIL2_COEFF,
    INVERSE,
    U0,
    W1,
    W2,
    Q_OUT,
    K1,
    K2,
    CAUSAL,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Keep the direct Gram matrix in registers through the triangular solve."""
    system = tl.program_id(0)
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    rows = tl.arange(0, C)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    matrix_offsets = system * C * C + row_ids * C + col_ids
    interaction = tl.load(INTERACTION_COEFF + matrix_offsets).to(tl.float32)
    output = tl.load(OUTPUT_COEFF + matrix_offsets).to(tl.float32)
    tail1 = tl.load(TAIL1_COEFF + system * C + rows).to(tl.float32)
    tail2 = tl.load(TAIL2_COEFF + system * C + rows).to(tl.float32)
    gram_kk = tl.zeros((C, C), dtype=tl.float32)
    gram_qk = tl.zeros((C, C), dtype=tl.float32)
    sequence_key_base = ((batch * T + chunk * C) * H + head) * K
    chunk_key_base = system * C * K
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        sequence_offsets = (
            sequence_key_base + rows[:, None] * H * K + keys[None, :]
        )
        key = tl.load(
            K_IN + sequence_offsets, mask=key_mask[None, :], other=0.0
        )
        query = tl.load(
            Q + sequence_offsets, mask=key_mask[None, :], other=0.0
        )
        query_scaled = (query * SCALE).to(tl.bfloat16)
        gram_kk += safe_dot(key, tl.trans(key), allow_tf32=False)
        gram_qk += safe_dot(query_scaled, tl.trans(key), allow_tf32=False)
        chunk_offsets = chunk_key_base + rows[:, None] * K + keys[None, :]
        tl.store(Q_OUT + chunk_offsets, query_scaled, mask=key_mask[None, :])
        tl.store(
            K1 + chunk_offsets,
            tail1[:, None] * key,
            mask=key_mask[None, :],
        )
        tl.store(
            K2 + chunk_offsets,
            tail2[:, None] * key,
            mask=key_mask[None, :],
        )

    matrix = tl.where(row_ids > col_ids, interaction * gram_kk, 0.0)
    causal = tl.where(row_ids >= col_ids, output * gram_qk, 0.0)
    tl.store(CAUSAL + matrix_offsets, causal)
    inverse = -matrix
    for row in range(2, C):
        current = tl.sum(
            tl.where(row_ids == row, -matrix, 0.0), axis=0
        )
        current += tl.sum(current[:, None] * inverse, axis=0)
        inverse = tl.where(row_ids == row, current[None, :], inverse)
    inverse += row_ids == col_ids
    inverse_bf16 = inverse.to(tl.bfloat16)
    tl.store(INVERSE + matrix_offsets, inverse_bf16)

    beta_base = (batch * T + chunk * C) * H + head
    beta = tl.load(BETA + beta_base + rows * H).to(tl.float32)
    pi1 = tl.load(PI1 + system * C + rows).to(tl.float32)
    pi2 = tl.load(PI2 + system * C + rows).to(tl.float32)
    sequence_value_base = ((batch * T + chunk * C) * H + head) * V
    chunk_value_base = system * C * V
    for value_block in range(tl.cdiv(V, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < V
        input_offsets = (
            sequence_value_base + rows[:, None] * H * V + values[None, :]
        )
        value = tl.load(
            V_IN + input_offsets, mask=value_mask[None, :], other=0.0
        )
        solution = safe_dot(
            inverse_bf16,
            (beta[:, None] * value).to(tl.bfloat16),
            allow_tf32=False,
        )
        output_offsets = (
            chunk_value_base + rows[:, None] * V + values[None, :]
        )
        tl.store(U0 + output_offsets, solution, mask=value_mask[None, :])

    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        input_offsets = (
            sequence_key_base + rows[:, None] * H * K + keys[None, :]
        )
        key = tl.load(
            K_IN + input_offsets, mask=key_mask[None, :], other=0.0
        )
        solution1 = safe_dot(
            inverse_bf16,
            (pi1[:, None] * key).to(tl.bfloat16),
            allow_tf32=False,
        )
        solution2 = safe_dot(
            inverse_bf16,
            (pi2[:, None] * key).to(tl.bfloat16),
            allow_tf32=False,
        )
        output_offsets = chunk_key_base + rows[:, None] * K + keys[None, :]
        tl.store(W1 + output_offsets, solution1, mask=key_mask[None, :])
        tl.store(W2 + output_offsets, solution2, mask=key_mask[None, :])


# Pinned fused-builder launches per architecture: ops/tuning.
_PINNED_BUILDER_FORWARD_CONFIGS = by_arch("BUILDER_FORWARD")


def _builder_forward_launch(tensor, chunk_size, key_dim, value_dim):
    """(BK, BV, warps, stages) for the fused builder; defaults unless pinned."""
    default = (128, 128, 2 if chunk_size == 16 else 4, 2 if chunk_size == 16 else 1)
    if not tensor.is_cuda or key_dim != 128 or value_dim != 128:
        return default
    table = _PINNED_BUILDER_FORWARD_CONFIGS.get(
        torch.cuda.get_device_capability(tensor.device)
    )
    if table is None:
        return default
    return table.get(chunk_size, default)


class GDNDirectBuild(torch.autograd.Function):
    """Fused forward with FLA-style recompute kernels in backward."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        beta,
        pi1,
        pi2,
        interaction_coeff,
        output_coeff,
        tail1,
        tail2,
        scale,
    ):
        tensors = tuple(
            tensor.contiguous()
            for tensor in (
                q, k, v, beta, pi1, pi2,
                interaction_coeff, output_coeff, tail1, tail2,
            )
        )
        (
            q, k, v, beta, pi1, pi2,
            interaction_coeff, output_coeff, tail1, tail2,
        ) = tensors
        batch, length, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        chunks, chunk_size = interaction_coeff.shape[2:4]
        if chunk_size not in (16, 32):
            raise ValueError("the fused direct builder supports C16/C32")
        shape = (batch, heads, chunks, chunk_size)
        inverse = torch.empty(
            *shape, chunk_size, dtype=torch.bfloat16, device=q.device
        )
        u0 = torch.empty(*shape, value_dim, dtype=v.dtype, device=v.device)
        w1 = torch.empty(*shape, key_dim, dtype=k.dtype, device=k.device)
        w2 = torch.empty_like(w1)
        q_out = torch.empty_like(w1)
        k1 = torch.empty_like(w1)
        k2 = torch.empty_like(w1)
        causal = torch.empty(
            *shape, chunk_size, dtype=q.dtype, device=q.device
        )
        systems = batch * heads * chunks
        block_key, block_value, warps, stages = _builder_forward_launch(
            q, chunk_size, key_dim, value_dim
        )
        _gdn_direct_builder_fwd_kernel[(systems,)](
            q, k, v, beta, pi1, pi2,
            interaction_coeff, output_coeff, tail1, tail2,
            inverse, u0, w1, w2, q_out, k1, k2, causal,
            T=length, H=heads, N=chunks, K=key_dim, V=value_dim,
            C=chunk_size, SCALE=float(scale), BK=block_key, BV=block_value,
            num_warps=warps,
            num_stages=stages,
        )
        ctx.save_for_backward(
            inverse, k, v, beta, pi1, pi2,
            interaction_coeff, output_coeff, tail1, tail2,
            u0, w1, w2, q_out,
        )
        ctx.dimensions = (
            batch, length, heads, key_dim, value_dim, chunks, chunk_size
        )
        ctx.scale = float(scale)
        ctx.input_dtypes = tuple(tensor.dtype for tensor in tensors)
        return u0, w1, w2, q_out, k1, k2, causal

    @staticmethod
    def backward(ctx, du0, dw1, dw2, dq_direct, dk1, dk2, dcausal):
        (
            inverse, k, v, beta, pi1, pi2,
            interaction_coeff, output_coeff, tail1, tail2,
            u0, w1, w2, q_out,
        ) = ctx.saved_tensors
        (
            batch, length, heads, key_dim, value_dim, chunks, chunk_size
        ) = ctx.dimensions
        from .delta_triton import _delta_solve_backward_launch
        from .gdn_direct_triton import _gdn_direct_intra_bwd_kernel

        dmatrix = torch.empty(
            batch, heads, chunks, chunk_size, chunk_size,
            dtype=torch.float32, device=k.device,
        )
        dk_solve = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbeta = torch.empty_like(beta)
        dpi1 = torch.empty_like(pi1)
        dpi2 = torch.empty_like(pi2)
        solve_kernel, solve_launch = _delta_solve_backward_launch(
            k, chunk_size, key_dim, value_dim
        )
        solve_kernel[(batch * heads * chunks,)](
            inverse, k, v, beta, pi1, pi2, u0, w1, w2,
            du0.contiguous(), dw1.contiguous(), dw2.contiguous(),
            dmatrix, dk_solve, dv, dbeta, dpi1, dpi2,
            T=length, H=heads, N=chunks, K=key_dim, V=value_dim,
            C=chunk_size, **solve_launch,
        )
        # C16/C32 have one program per chunk/head. Write Q/K directly in the
        # consumer dtype and fold the solve contribution into the K store.
        dq = torch.empty(
            batch, length, heads, key_dim,
            dtype=q_out.dtype, device=k.device,
        )
        dk_intra = torch.empty_like(k)
        dinteraction = torch.empty_like(interaction_coeff)
        doutput = torch.empty_like(output_coeff)
        dtail1 = torch.empty_like(tail1)
        dtail2 = torch.empty_like(tail2)
        _gdn_direct_intra_bwd_kernel[(batch * heads * chunks,)](
            q_out, k, interaction_coeff, output_coeff, tail1, tail2,
            dmatrix, dcausal.contiguous(), dq_direct.contiguous(),
            dk1.contiguous(), dk2.contiguous(),
            dk_solve, dq, dk_intra, dk_intra,
            dinteraction, doutput, dtail1, dtail2,
            T=length, H=heads, N=chunks, K=key_dim,
            C=chunk_size, BC=chunk_size, PAIRS=1,
            SCALE=ctx.scale, ADD_DK_BASE=True, BK=128,
            num_warps=4 if chunk_size == 32 else 2,
            num_stages=2 if chunk_size == 32 else 1,
        )
        grads = (
            dq,
            dk_intra,
            dv,
            dbeta,
            dpi1,
            dpi2,
            dinteraction,
            doutput,
            dtail1,
            dtail2,
        )
        return tuple(
            grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)
        ) + (None,)


__all__ = ["GDNDirectBuild"]

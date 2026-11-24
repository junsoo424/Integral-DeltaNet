"""TensorCore operand builder for inverse-free Integral GDN.

The scalar relative coefficients are already materialized as bounded lower
triangular matrices.  This module multiplies them by raw K/K and Q/K Gram
tiles, then reuses the shared fused triangular solve and boundary scan.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import safe_dot

from .gdn_relative_triton import gdn_relative_coefficients_triton


@triton.jit
def _gdn_direct_intra_fwd_kernel(
    Q,
    K_IN,
    INTERACTION_COEFF,
    OUTPUT_COEFF,
    TAIL1_COEFF,
    TAIL2_COEFF,
    MATRIX,
    CAUSAL,
    Q_OUT,
    K1,
    K2,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    PAIRS: tl.constexpr,
    SCALE: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = pid % PAIRS
    system = pid // PAIRS
    if C == 64:
        target = tl.where(pair == 0, 0, 1)
        source = tl.where(pair == 2, 1, 0)
    else:
        target = 0
        source = 0
    diagonal = target == source
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    rows = tl.arange(0, BC)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    target_row = target * BC + rows
    source_row = source * BC + rows

    gram_kk = tl.zeros((BC, BC), dtype=tl.float32)
    gram_qk = tl.zeros((BC, BC), dtype=tl.float32)
    target_token = chunk * C + target * BC
    source_token = chunk * C + source * BC
    target_base = ((batch * T + target_token) * H + head) * K
    source_base = ((batch * T + source_token) * H + head) * K
    output_base = system * C * K
    tail1 = tl.load(TAIL1_COEFF + system * C + target_row).to(tl.float32)
    tail2 = tl.load(TAIL2_COEFF + system * C + target_row).to(tl.float32)
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        target_offsets = target_base + rows[:, None] * H * K + keys[None, :]
        source_offsets = source_base + rows[:, None] * H * K + keys[None, :]
        key_target = tl.load(
            K_IN + target_offsets, mask=key_mask[None, :], other=0.0
        )
        key_source = tl.load(
            K_IN + source_offsets, mask=key_mask[None, :], other=0.0
        )
        query = tl.load(Q + target_offsets, mask=key_mask[None, :], other=0.0)
        query_scaled = (query * SCALE).to(tl.bfloat16)
        gram_kk += safe_dot(
            key_target, tl.trans(key_source), allow_tf32=False
        )
        gram_qk += safe_dot(
            query_scaled, tl.trans(key_source), allow_tf32=False
        )
        output_offsets = (
            output_base + target_row[:, None] * K + keys[None, :]
        )
        tl.store(
            Q_OUT + output_offsets,
            query_scaled,
            mask=diagonal & key_mask[None, :],
        )
        tl.store(
            K1 + output_offsets,
            tail1[:, None] * key_target,
            mask=diagonal & key_mask[None, :],
        )
        tl.store(
            K2 + output_offsets,
            tail2[:, None] * key_target,
            mask=diagonal & key_mask[None, :],
        )

    matrix_offsets = (
        system * C * C
        + target_row[:, None] * C
        + source_row[None, :]
    )
    interaction = tl.load(INTERACTION_COEFF + matrix_offsets).to(tl.float32)
    output = tl.load(OUTPUT_COEFF + matrix_offsets).to(tl.float32)
    matrix = interaction * gram_kk
    causal = output * gram_qk
    if C == 64:
        matrix = tl.where((~diagonal) | (row_ids > col_ids), matrix, 0.0)
        causal = tl.where((~diagonal) | (row_ids >= col_ids), causal, 0.0)
    else:
        matrix = tl.where(row_ids > col_ids, matrix, 0.0)
        causal = tl.where(row_ids >= col_ids, causal, 0.0)
    tl.store(MATRIX + matrix_offsets, matrix)
    tl.store(CAUSAL + matrix_offsets, causal)


@triton.jit
def _gdn_direct_intra_bwd_kernel(
    Q_OUT,
    K_IN,
    INTERACTION_COEFF,
    OUTPUT_COEFF,
    TAIL1_COEFF,
    TAIL2_COEFF,
    DMATRIX,
    DCAUSAL,
    DQ_DIRECT,
    DK1,
    DK2,
    DK_BASE,
    DQ,
    DK,
    DK_SOURCE,
    DINTERACTION,
    DOUTPUT,
    DTAIL1,
    DTAIL2,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    PAIRS: tl.constexpr,
    SCALE: tl.constexpr,
    ADD_DK_BASE: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = pid % PAIRS
    system = pid // PAIRS
    if C == 64:
        target = tl.where(pair == 0, 0, 1)
        source = tl.where(pair == 2, 1, 0)
    else:
        target = 0
        source = 0
    diagonal = target == source
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    rows = tl.arange(0, BC)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    target_row = target * BC + rows
    source_row = source * BC + rows
    matrix_offsets = (
        system * C * C
        + target_row[:, None] * C
        + source_row[None, :]
    )
    grad_matrix = tl.load(DMATRIX + matrix_offsets).to(tl.float32)
    grad_causal = tl.load(DCAUSAL + matrix_offsets).to(tl.float32)
    if C == 64:
        grad_matrix = tl.where(
            (~diagonal) | (row_ids > col_ids), grad_matrix, 0.0
        )
        grad_causal = tl.where(
            (~diagonal) | (row_ids >= col_ids), grad_causal, 0.0
        )
    else:
        grad_matrix = tl.where(row_ids > col_ids, grad_matrix, 0.0)
        grad_causal = tl.where(row_ids >= col_ids, grad_causal, 0.0)
    interaction = tl.load(INTERACTION_COEFF + matrix_offsets).to(tl.float32)
    output = tl.load(OUTPUT_COEFF + matrix_offsets).to(tl.float32)
    grad_gram_kk = (grad_matrix * interaction).to(tl.bfloat16)
    grad_gram_qk = (grad_causal * output).to(tl.bfloat16)
    gram_kk = tl.zeros((BC, BC), dtype=tl.float32)
    gram_qk = tl.zeros((BC, BC), dtype=tl.float32)
    grad_tail1 = tl.zeros((BC,), dtype=tl.float32)
    grad_tail2 = tl.zeros((BC,), dtype=tl.float32)

    target_token = chunk * C + target * BC
    source_token = chunk * C + source * BC
    target_base = ((batch * T + target_token) * H + head) * K
    source_base = ((batch * T + source_token) * H + head) * K
    output_base = system * C * K
    tail1 = tl.load(TAIL1_COEFF + system * C + target_row).to(tl.float32)
    tail2 = tl.load(TAIL2_COEFF + system * C + target_row).to(tl.float32)
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        target_offsets = target_base + rows[:, None] * H * K + keys[None, :]
        source_offsets = source_base + rows[:, None] * H * K + keys[None, :]
        key_target = tl.load(
            K_IN + target_offsets, mask=key_mask[None, :], other=0.0
        )
        key_source = tl.load(
            K_IN + source_offsets, mask=key_mask[None, :], other=0.0
        )
        chunk_offsets = (
            output_base + target_row[:, None] * K + keys[None, :]
        )
        query = tl.load(
            Q_OUT + chunk_offsets, mask=key_mask[None, :], other=0.0
        )
        grad_query_direct = tl.load(
            DQ_DIRECT + chunk_offsets,
            mask=diagonal & key_mask[None, :],
            other=0.0,
        )
        grad_k1 = tl.load(
            DK1 + chunk_offsets,
            mask=diagonal & key_mask[None, :],
            other=0.0,
        )
        grad_k2 = tl.load(
            DK2 + chunk_offsets,
            mask=diagonal & key_mask[None, :],
            other=0.0,
        )
        gram_kk += safe_dot(
            key_target, tl.trans(key_source), allow_tf32=False
        )
        gram_qk += safe_dot(
            query, tl.trans(key_source), allow_tf32=False
        )
        grad_query = safe_dot(
            grad_gram_qk, key_source, allow_tf32=False
        )
        grad_query = (grad_query + grad_query_direct) * SCALE
        grad_key_target = safe_dot(
            grad_gram_kk, key_source, allow_tf32=False
        )
        grad_key_source = safe_dot(
            tl.trans(grad_gram_kk), key_target, allow_tf32=False
        )
        grad_key_source += safe_dot(
            tl.trans(grad_gram_qk), query, allow_tf32=False
        )
        grad_key_target += tl.where(
            diagonal,
            tail1[:, None] * grad_k1 + tail2[:, None] * grad_k2,
            0.0,
        )
        if C == 64:
            # Each C32 pair writes a private FP32 tile. A small deterministic
            # reduction below replaces contended global atomics, which are
            # disproportionately expensive at C64 and can stall on Blackwell.
            partial_offsets = (
                (pid * BC + rows[:, None]) * K + keys[None, :]
            )
            source_partial_offsets = (
                (system * BC + rows[:, None]) * K + keys[None, :]
            )
            tl.store(
                DQ + partial_offsets,
                grad_query,
                mask=key_mask[None, :],
            )
            tl.store(
                DK + partial_offsets,
                grad_key_target
                + tl.where(diagonal, grad_key_source, 0.0),
                mask=key_mask[None, :],
            )
            tl.store(
                DK_SOURCE + source_partial_offsets,
                grad_key_source,
                mask=(~diagonal) & key_mask[None, :],
            )
        else:
            tl.store(
                DQ + target_offsets,
                grad_query,
                mask=key_mask[None, :],
            )
            grad_key = grad_key_target + grad_key_source
            if ADD_DK_BASE:
                grad_key += tl.load(
                    DK_BASE + target_offsets,
                    mask=key_mask[None, :],
                    other=0.0,
                )
            tl.store(
                DK + target_offsets,
                grad_key,
                mask=key_mask[None, :],
            )
        grad_tail1 += tl.sum(
            grad_k1.to(tl.float32) * key_target.to(tl.float32), axis=1
        )
        grad_tail2 += tl.sum(
            grad_k2.to(tl.float32) * key_target.to(tl.float32), axis=1
        )

    tl.store(DINTERACTION + matrix_offsets, grad_matrix * gram_kk)
    tl.store(DOUTPUT + matrix_offsets, grad_causal * gram_qk)
    if C == 64:
        # The three lower block-pair programs write every coefficient tile
        # except the structurally zero upper-right C32 tile. Pair 0 owns that
        # tile, so writing it here removes two full-matrix zero-fill launches.
        upper_offsets = (
            system * C * C
            + rows[:, None] * C
            + (BC + rows)[None, :]
        )
        upper_mask = pair == 0
        tl.store(DINTERACTION + upper_offsets, 0.0, mask=upper_mask)
        tl.store(DOUTPUT + upper_offsets, 0.0, mask=upper_mask)
    tail_offsets = system * C + target_row
    tl.store(DTAIL1 + tail_offsets, grad_tail1, mask=diagonal)
    tl.store(DTAIL2 + tail_offsets, grad_tail2, mask=diagonal)


@triton.jit
def _gdn_direct_c64_reduce_bwd_kernel(
    DQ_PARTIAL,
    DK_TARGET_PARTIAL,
    DK_SOURCE_PARTIAL,
    DQ,
    DK,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    """Reduce three private C32 pair tiles into one C64 Q/K gradient."""
    system = tl.program_id(0)
    key_block = tl.program_id(1)
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    rows = tl.arange(0, BC)
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    partial_mask = key_mask[None, :]

    partial0 = (
        ((system * 3 + 0) * BC + rows[:, None]) * K + keys[None, :]
    )
    partial1 = (
        ((system * 3 + 1) * BC + rows[:, None]) * K + keys[None, :]
    )
    partial2 = (
        ((system * 3 + 2) * BC + rows[:, None]) * K + keys[None, :]
    )

    token0 = chunk * (2 * BC)
    output0 = (
        ((batch * T + token0) * H + head) * K
        + rows[:, None] * H * K
        + keys[None, :]
    )
    output1 = output0 + BC * H * K

    q0 = tl.load(DQ_PARTIAL + partial0, mask=partial_mask, other=0.0)
    tl.store(DQ + output0, q0, mask=partial_mask)
    q1 = tl.load(DQ_PARTIAL + partial1, mask=partial_mask, other=0.0)
    q1 += tl.load(DQ_PARTIAL + partial2, mask=partial_mask, other=0.0)
    tl.store(DQ + output1, q1, mask=partial_mask)

    source_offsets = (
        (system * BC + rows[:, None]) * K + keys[None, :]
    )
    k0 = tl.load(
        DK_TARGET_PARTIAL + partial0, mask=partial_mask, other=0.0
    )
    k0 += tl.load(
        DK_SOURCE_PARTIAL + source_offsets, mask=partial_mask, other=0.0
    )
    tl.store(DK + output0, k0, mask=partial_mask)
    k1 = tl.load(
        DK_TARGET_PARTIAL + partial1, mask=partial_mask, other=0.0
    )
    k1 += tl.load(
        DK_TARGET_PARTIAL + partial2, mask=partial_mask, other=0.0
    )
    tl.store(DK + output1, k1, mask=partial_mask)


class _GDNDirectIntra(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        interaction_coeff,
        output_coeff,
        tail1,
        tail2,
        scale,
    ):
        q, k, interaction_coeff, output_coeff, tail1, tail2 = (
            tensor.contiguous()
            for tensor in (q, k, interaction_coeff, output_coeff, tail1, tail2)
        )
        batch, length, heads, key_dim = q.shape
        chunks, chunk_size = interaction_coeff.shape[2:4]
        block_size = 16 if chunk_size == 16 else 32
        pairs = 3 if chunk_size == 64 else 1
        shape = (batch, heads, chunks, chunk_size)
        matrix = torch.zeros(
            *shape, chunk_size, dtype=torch.float32, device=q.device
        )
        causal = torch.zeros(
            *shape, chunk_size, dtype=q.dtype, device=q.device
        )
        q_out = torch.empty(*shape, key_dim, dtype=q.dtype, device=q.device)
        k1 = torch.empty_like(q_out)
        k2 = torch.empty_like(q_out)
        systems = batch * heads * chunks
        _gdn_direct_intra_fwd_kernel[(systems * pairs,)](
            q, k, interaction_coeff, output_coeff, tail1, tail2,
            matrix, causal, q_out, k1, k2,
            T=length, H=heads, N=chunks, K=key_dim,
            C=chunk_size, BC=block_size, PAIRS=pairs,
            SCALE=float(scale), BK=128 if chunk_size == 64 else 64,
            num_warps=2 if chunk_size == 64 else (
                4 if block_size == 32 else 2
            ),
            num_stages=2,
        )
        ctx.save_for_backward(
            q_out, k, interaction_coeff, output_coeff, tail1, tail2
        )
        ctx.dimensions = (batch, length, heads, key_dim, chunks, chunk_size)
        ctx.scale = float(scale)
        ctx.input_dtypes = (
            q.dtype, k.dtype, interaction_coeff.dtype, output_coeff.dtype,
            tail1.dtype, tail2.dtype,
        )
        return matrix, causal, q_out, k1, k2

    @staticmethod
    def backward(ctx, dmatrix, dcausal, dq_direct, dk1, dk2):
        q_out, k, interaction_coeff, output_coeff, tail1, tail2 = ctx.saved_tensors
        batch, length, heads, key_dim, chunks, chunk_size = ctx.dimensions
        block_size = 16 if chunk_size == 16 else 32
        pairs = 3 if chunk_size == 64 else 1
        dmatrix = dmatrix.contiguous()
        dcausal = dcausal.contiguous()
        dq_direct = dq_direct.contiguous()
        dk1 = dk1.contiguous()
        dk2 = dk2.contiguous()
        systems = batch * heads * chunks
        if chunk_size == 64:
            # Pair-private FP32 tiles avoid global atomics. Only the final
            # BF16 Q/K outputs and coefficient upper triangles need storage.
            dq = torch.empty(
                batch, length, heads, key_dim,
                dtype=q_out.dtype, device=k.device,
            )
            dk = torch.empty_like(k)
            dq_direct_out = torch.empty(
                systems, pairs, block_size, key_dim,
                dtype=torch.float32, device=k.device,
            )
            dk_direct_out = torch.empty_like(dq_direct_out)
            dk_source_out = torch.empty(
                systems, block_size, key_dim,
                dtype=torch.float32, device=k.device,
            )
            # Every element now has one deterministic writer: the pair
            # programs cover the lower block triangle and pair 0 clears the
            # structurally zero upper-right tile.
            dinteraction = torch.empty_like(interaction_coeff)
            doutput = torch.empty_like(output_coeff)
        else:
            # Unique writers cast once at the final store, as an FP32 buffer
            # cast afterwards would, without materializing FP32 Q/K.
            dq = torch.empty(
                batch, length, heads, key_dim,
                dtype=q_out.dtype, device=k.device,
            )
            dk = torch.empty_like(k)
            dinteraction = torch.empty_like(interaction_coeff)
            doutput = torch.empty_like(output_coeff)
            dq_direct_out = dq
            dk_direct_out = dk
            dk_source_out = dk
        dtail1 = torch.empty_like(tail1)
        dtail2 = torch.empty_like(tail2)
        _gdn_direct_intra_bwd_kernel[(systems * pairs,)](
            q_out, k, interaction_coeff, output_coeff, tail1, tail2,
            dmatrix, dcausal, dq_direct, dk1, dk2,
            k, dq_direct_out, dk_direct_out, dk_source_out,
            dinteraction, doutput, dtail1, dtail2,
            T=length, H=heads, N=chunks, K=key_dim,
            C=chunk_size, BC=block_size, PAIRS=pairs,
            SCALE=ctx.scale, ADD_DK_BASE=False,
            BK=32 if chunk_size == 64 else 128,
            num_warps=2 if chunk_size == 64 else (
                4 if block_size == 32 else 2
            ),
            num_stages=3 if chunk_size == 64 else (
                1 if chunk_size == 16 else 2
            ),
        )
        if chunk_size == 64:
            _gdn_direct_c64_reduce_bwd_kernel[
                (systems, triton.cdiv(key_dim, 128))
            ](
                dq_direct_out, dk_direct_out, dk_source_out, dq, dk,
                T=length, H=heads, N=chunks, K=key_dim, BC=block_size, BK=128,
                num_warps=4, num_stages=3,
            )
        grads = (dq, dk, dinteraction, doutput, dtail1, dtail2)
        return tuple(
            grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)
        ) + (None,)


def build_gdn_relative_operands_triton(
    q, k, v, g, beta, gamma, lam, scale, chunk_size=16
):
    """Return inverse-free operands consumed by the fused boundary scan."""
    if not (q.is_cuda and q.dtype == torch.bfloat16):
        raise ValueError("the inverse-free GDN builder requires CUDA BF16")
    if chunk_size not in (16, 32, 64) or q.shape[1] % chunk_size:
        raise ValueError("length must be divisible by a 16/32/64 chunk")
    (
        pi1,
        pi2,
        eta1,
        eta2,
        interaction_coeff,
        output_coeff,
        tail1,
        tail2,
        transport,
    ) = gdn_relative_coefficients_triton(
        g, beta, gamma, lam, chunk_size=chunk_size
    )
    if chunk_size in (16, 32):
        from .gdn_fused_triton import GDNDirectBuild

        (
            u0,
            state_weight1,
            state_weight2,
            query,
            k1,
            k2,
            causal,
        ) = GDNDirectBuild.apply(
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
        )
    else:
        matrix, causal, query, k1, k2 = _GDNDirectIntra.apply(
            q,
            k,
            interaction_coeff,
            output_coeff,
            tail1,
            tail2,
            scale,
        )
        from .delta_triton import _DeltaSolve

        u0, state_weight1, state_weight2 = _DeltaSolve.apply(
            matrix, k, v, beta, pi1, pi2
        )
    query_coefficients = torch.stack((eta1, eta2), dim=-1).unsqueeze(-2)
    return (
        u0,
        state_weight1,
        state_weight2,
        query,
        query_coefficients,
        k1,
        k2,
        causal,
        transport,
    )


__all__ = ["build_gdn_relative_operands_triton"]

"""Fused value-write product and multi-RHS solve for general variants."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import safe_dot

from .general_triton import (
    _GENERAL_SOLVE_BACKWARD_CONFIGS,
    _GENERAL_SOLVE_FORWARD_CONFIGS,
)
from .tuning import by_arch


# Pinned (forward, backward) solve launches per architecture: ops/tuning.
_PINNED_FUSED_WRITE_SOLVE_CONFIGS = by_arch("FUSED_WRITE_SOLVE")


def _pinned_fused_write_solve_configs(tensor, chunk_size, key_dim, value_dim):
    """(forward, backward) launch tables for this device, or None."""
    if not tensor.is_cuda or key_dim != 128 or value_dim != 128:
        return None
    tables = _PINNED_FUSED_WRITE_SOLVE_CONFIGS.get(
        torch.cuda.get_device_capability(tensor.device)
    )
    if tables is None or chunk_size not in tables[0]:
        return None
    return tables


@triton.autotune(
    configs=_GENERAL_SOLVE_FORWARD_CONFIGS,
    key=["C", "K", "V", "READOUT_GATE"],
)
@triton.jit
def _fused_write_solve_fwd_kernel(
    MATRIX,
    V_IN,
    WRITE_GATE,
    A1,
    A2,
    INVERSE,
    U0,
    W1,
    W2,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    SV0: tl.constexpr,
    SV1: tl.constexpr,
    SV2: tl.constexpr,
    SV3: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    SW2: tl.constexpr,
    SW3: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    READOUT_GATE: tl.constexpr = False,
):
    """Invert the local system and apply ``v * write_gate`` in one pass."""
    pid = tl.program_id(0)
    chunk = pid % N
    head = (pid // N) % H
    batch = pid // (N * H)
    rows = tl.arange(0, C)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    matrix_offsets = pid * C * C + row_ids * C + col_ids
    matrix = tl.load(MATRIX + matrix_offsets).to(tl.float32)
    matrix = tl.where(row_ids > col_ids, matrix, 0.0)
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

    value_base = pid * C * V
    tokens = chunk * C + rows
    for value_block in range(tl.cdiv(V, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < V
        output_offsets = value_base + rows[:, None] * V + values[None, :]
        v_offsets = (
            batch * SV0
            + tokens[:, None] * SV1
            + head * SV2
            + values[None, :] * SV3
        )
        gate_offsets = (
            batch * SW0
            + tokens[:, None] * SW1
            + head * SW2
            + values[None, :] * SW3
        )
        value = tl.load(
            V_IN + v_offsets, mask=value_mask[None, :], other=0.0
        )
        gate = tl.load(
            WRITE_GATE + gate_offsets,
            mask=value_mask[None, :],
            other=0.0,
        )
        write = (value.to(tl.float32) * gate.to(tl.float32)).to(tl.bfloat16)
        solution = safe_dot(inverse_bf16, write, allow_tf32=False)
        tl.store(U0 + output_offsets, solution, mask=value_mask[None, :])

    key_base = pid * C * K
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        offsets = key_base + rows[:, None] * K + keys[None, :]
        a1 = tl.load(A1 + offsets, mask=key_mask[None, :], other=0.0)
        solution1 = safe_dot(
            inverse_bf16, a1.to(tl.bfloat16), allow_tf32=False
        )
        if READOUT_GATE:
            solution2 = tl.zeros((C, BK), tl.float32)
        else:
            a2 = tl.load(A2 + offsets, mask=key_mask[None, :], other=0.0)
            solution2 = safe_dot(inverse_bf16, a2.to(tl.bfloat16), allow_tf32=False)
        tl.store(W1 + offsets, solution1, mask=key_mask[None, :])
        if not READOUT_GATE:
            tl.store(W2 + offsets, solution2, mask=key_mask[None, :])


@triton.autotune(
    configs=_GENERAL_SOLVE_BACKWARD_CONFIGS,
    key=["C", "K", "V", "READOUT_GATE"],
)
@triton.jit
def _fused_write_solve_bwd_kernel(
    INVERSE,
    U0,
    W1,
    W2,
    V_IN,
    WRITE_GATE,
    DU0,
    DW1,
    DW2,
    DMATRIX,
    DV_IN,
    DWRITE_GATE,
    DA1,
    DA2,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    SV0: tl.constexpr,
    SV1: tl.constexpr,
    SV2: tl.constexpr,
    SV3: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    SW2: tl.constexpr,
    SW3: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    READOUT_GATE: tl.constexpr = False,
):
    """Backpropagate solve and write product without a ``dwrite`` buffer."""
    pid = tl.program_id(0)
    chunk = pid % N
    head = (pid // N) % H
    batch = pid // (N * H)
    rows = tl.arange(0, C)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    matrix_offsets = pid * C * C + row_ids * C + col_ids
    inverse = tl.load(INVERSE + matrix_offsets)
    inverse_t = tl.trans(inverse)
    grad_matrix = tl.zeros((C, C), dtype=tl.float32)

    value_base = pid * C * V
    tokens = chunk * C + rows
    for value_block in range(tl.cdiv(V, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < V
        output_offsets = value_base + rows[:, None] * V + values[None, :]
        grad_solution = tl.load(
            DU0 + output_offsets, mask=value_mask[None, :], other=0.0
        )
        grad_rhs = safe_dot(inverse_t, grad_solution, allow_tf32=False)
        solution = tl.load(
            U0 + output_offsets, mask=value_mask[None, :], other=0.0
        )
        grad_rhs_bf16 = grad_rhs.to(tl.bfloat16)
        grad_matrix += safe_dot(
            grad_rhs_bf16, tl.trans(solution), allow_tf32=False
        )
        v_offsets = (
            batch * SV0
            + tokens[:, None] * SV1
            + head * SV2
            + values[None, :] * SV3
        )
        gate_offsets = (
            batch * SW0
            + tokens[:, None] * SW1
            + head * SW2
            + values[None, :] * SW3
        )
        value = tl.load(
            V_IN + v_offsets, mask=value_mask[None, :], other=0.0
        )
        gate = tl.load(
            WRITE_GATE + gate_offsets,
            mask=value_mask[None, :],
            other=0.0,
        )
        logical_offsets = (
            ((batch * T + tokens[:, None]) * H + head) * V
            + values[None, :]
        )
        tl.store(
            DV_IN + logical_offsets,
            (grad_rhs_bf16.to(tl.float32) * gate.to(tl.float32)).to(tl.bfloat16),
            mask=value_mask[None, :],
        )
        tl.store(
            DWRITE_GATE + logical_offsets,
            (grad_rhs_bf16.to(tl.float32) * value.to(tl.float32)).to(tl.bfloat16),
            mask=value_mask[None, :],
        )

    key_base = pid * C * K
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        offsets = key_base + rows[:, None] * K + keys[None, :]
        grad_solution1 = tl.load(
            DW1 + offsets, mask=key_mask[None, :], other=0.0
        )
        grad_rhs1 = safe_dot(inverse_t, grad_solution1, allow_tf32=False)
        solution1 = tl.load(W1 + offsets, mask=key_mask[None, :], other=0.0)
        grad_matrix += safe_dot(
            grad_rhs1.to(tl.bfloat16), tl.trans(solution1), allow_tf32=False
        )
        if not READOUT_GATE:
            grad_solution2 = tl.load(DW2 + offsets, mask=key_mask[None, :], other=0.0)
            grad_rhs2 = safe_dot(inverse_t, grad_solution2, allow_tf32=False)
            solution2 = tl.load(W2 + offsets, mask=key_mask[None, :], other=0.0)
            grad_matrix += safe_dot(
                grad_rhs2.to(tl.bfloat16), tl.trans(solution2), allow_tf32=False
            )
        tl.store(DA1 + offsets, grad_rhs1, mask=key_mask[None, :])
        if not READOUT_GATE:
            tl.store(DA2 + offsets, grad_rhs2, mask=key_mask[None, :])

    grad_matrix = tl.where(row_ids > col_ids, -grad_matrix, 0.0)
    tl.store(DMATRIX + matrix_offsets, grad_matrix)


class _GeneralFusedWriteSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, value, write_gate, a1, a2, readout_gate):
        ctx.readout_gate = readout_gate
        chunk_size = matrix.shape[-1]
        batch, length, heads, value_dim = value.shape
        key_dim = a1.shape[-1]
        chunks = length // chunk_size
        systems = batch * heads * chunks
        expected_matrix = (batch, heads, chunks, chunk_size, chunk_size)
        expected_rhs = (batch, heads, chunks, chunk_size, key_dim)
        if matrix.shape != expected_matrix or a1.shape != expected_rhs:
            raise ValueError("matrix/key RHS shapes do not match value sequence")
        if a2.shape != a1.shape or write_gate.shape != value.shape:
            raise ValueError("a1/a2 and value/write_gate shapes must match")
        matrix, a1, a2 = (
            tensor.contiguous() for tensor in (matrix, a1, a2)
        )
        inverse = torch.empty_like(matrix, dtype=torch.bfloat16)
        output_shape = matrix.shape[:-1]
        u0 = torch.empty(
            *output_shape,
            value_dim,
            device=value.device,
            dtype=value.dtype,
        )
        state_weight1 = torch.empty(
            *output_shape,
            key_dim,
            device=a1.device,
            dtype=torch.bfloat16,
        )
        if readout_gate:
            state_weight2 = torch.zeros((), device=a1.device, dtype=torch.bfloat16).expand_as(state_weight1)
            ctx.mark_non_differentiable(state_weight2)
            ctx.set_materialize_grads(False)
        else:
            state_weight2 = torch.empty_like(state_weight1)
        pinned = _pinned_fused_write_solve_configs(
            matrix, chunk_size, key_dim, value_dim
        )
        kernel = _fused_write_solve_fwd_kernel
        launch = {}
        if pinned is not None:
            block_key, block_value, warps, stages = pinned[0][chunk_size]
            kernel = getattr(kernel, "fn", kernel)
            launch.update(
                BK=block_key,
                BV=block_value,
                num_warps=warps,
                num_stages=stages,
            )
        kernel[(systems,)](
            matrix,
            value,
            write_gate,
            a1,
            a2,
            inverse,
            u0,
            state_weight1,
            state_weight2,
            T=length,
            H=heads,
            N=chunks,
            K=key_dim,
            V=value_dim,
            C=chunk_size,
            SV0=value.stride(0),
            SV1=value.stride(1),
            SV2=value.stride(2),
            SV3=value.stride(3),
            SW0=write_gate.stride(0),
            SW1=write_gate.stride(1),
            SW2=write_gate.stride(2),
            SW3=write_gate.stride(3),
            READOUT_GATE=ctx.readout_gate,
            **launch,
        )
        ctx.save_for_backward(
            inverse, u0, state_weight1, state_weight2, value, write_gate
        )
        ctx.input_dtypes = tuple(
            tensor.dtype for tensor in (matrix, value, write_gate, a1, a2)
        )
        ctx.dimensions = (
            systems,
            chunk_size,
            key_dim,
            value_dim,
            length,
            heads,
            chunks,
        )
        ctx.pinned = pinned
        return u0, state_weight1, state_weight2

    @staticmethod
    def backward(ctx, du0, dw1, dw2):
        inverse, u0, w1, w2, value, write_gate = ctx.saved_tensors
        (
            systems,
            chunk_size,
            key_dim,
            value_dim,
            length,
            heads,
            chunks,
        ) = ctx.dimensions
        if du0 is None:
            du0 = torch.zeros_like(u0)
        if dw1 is None:
            dw1 = torch.zeros_like(w1)
        dmatrix = torch.empty_like(inverse, dtype=torch.float32)
        dvalue = torch.empty(
            value.shape, device=value.device, dtype=value.dtype
        )
        dwrite_gate = torch.empty(
            write_gate.shape,
            device=write_gate.device,
            dtype=write_gate.dtype,
        )
        da1 = torch.empty_like(w1, dtype=ctx.input_dtypes[3])
        da2 = None if ctx.readout_gate else torch.empty_like(w2, dtype=ctx.input_dtypes[4])
        kernel = _fused_write_solve_bwd_kernel
        launch = {}
        if ctx.pinned is not None:
            block_key, block_value, warps, stages = ctx.pinned[1][chunk_size]
            kernel = getattr(kernel, "fn", kernel)
            launch.update(
                BK=block_key,
                BV=block_value,
                num_warps=warps,
                num_stages=stages,
            )
        kernel[(systems,)](
            inverse,
            u0,
            w1,
            w2,
            value,
            write_gate,
            du0.contiguous(),
            dw1.contiguous(),
            dw1 if ctx.readout_gate else dw2.contiguous(),
            dmatrix,
            dvalue,
            dwrite_gate,
            da1,
            da1 if ctx.readout_gate else da2,
            T=length,
            H=heads,
            N=chunks,
            K=key_dim,
            V=value_dim,
            C=chunk_size,
            SV0=value.stride(0),
            SV1=value.stride(1),
            SV2=value.stride(2),
            SV3=value.stride(3),
            SW0=write_gate.stride(0),
            SW1=write_gate.stride(1),
            SW2=write_gate.stride(2),
            SW3=write_gate.stride(3),
            READOUT_GATE=ctx.readout_gate,
            **launch,
        )
        grads = (dmatrix, dvalue, dwrite_gate, da1, da2)
        return tuple(
            None if grad is None else grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)
        ) + (None,)


@torch.compiler.disable
def solve_general_operands_fused_write(matrix, value, write_gate, a1, a2=None):
    """Omitting a2 selects the structurally-zero readout-side RHS."""
    if not matrix.is_cuda or matrix.shape[-1] not in (16, 32, 64):
        raise ValueError("the fused general solve requires CUDA C16/C32/C64")
    if value.dtype != torch.bfloat16 or write_gate.dtype != torch.bfloat16:
        raise ValueError("the fused general solve requires BF16 value/write gates")
    return _GeneralFusedWriteSolve.apply(
        matrix, value, write_gate, a1, a1 if a2 is None else a2, a2 is None
    )


__all__ = ["solve_general_operands_fused_write"]

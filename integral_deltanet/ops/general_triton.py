"""Fused multi-RHS solve for the diagonal KDA/GDN-2 operand builder."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import safe_dot

from .solve import _inverse


# A compact autotune set.
_GENERAL_SOLVE_FORWARD_CONFIGS = [
    triton.Config({"BK": bk, "BV": bv}, num_warps=warps, num_stages=stages)
    for bk, bv, warps, stages in (
        (16, 32, 1, 2),
        (64, 64, 1, 1),
        (64, 64, 1, 2),
        (32, 32, 4, 2),
        (32, 32, 4, 3),
        (64, 32, 4, 2),
        (64, 32, 4, 3),
        (64, 64, 8, 2),
        (64, 64, 8, 3),
    )
]
_GENERAL_SOLVE_BACKWARD_CONFIGS = [
    triton.Config({"BK": bk, "BV": bv}, num_warps=warps, num_stages=stages)
    for bk, bv, warps, stages in (
        (64, 64, 1, 2),
        (64, 64, 2, 2),
        (64, 64, 4, 2),
        (32, 32, 4, 2),
        (32, 32, 4, 3),
        (64, 32, 4, 2),
        (64, 32, 4, 3),
        (64, 64, 8, 2),
        (64, 64, 8, 3),
    )
]


@triton.autotune(
    configs=_GENERAL_SOLVE_FORWARD_CONFIGS,
    key=["C", "K", "V"],
)
@triton.jit
def _general_solve_fwd_kernel(
    MATRIX,
    WRITE,
    A1,
    A2,
    INVERSE,
    U0,
    W1,
    W2,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
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
    for value_block in range(tl.cdiv(V, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < V
        offsets = value_base + rows[:, None] * V + values[None, :]
        write = tl.load(WRITE + offsets, mask=value_mask[None, :], other=0.0)
        solution = safe_dot(
            inverse_bf16, write.to(tl.bfloat16), allow_tf32=False
        )
        tl.store(U0 + offsets, solution, mask=value_mask[None, :])

    key_base = pid * C * K
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        offsets = key_base + rows[:, None] * K + keys[None, :]
        a1 = tl.load(A1 + offsets, mask=key_mask[None, :], other=0.0)
        a2 = tl.load(A2 + offsets, mask=key_mask[None, :], other=0.0)
        solution1 = safe_dot(
            inverse_bf16, a1.to(tl.bfloat16), allow_tf32=False
        )
        solution2 = safe_dot(
            inverse_bf16, a2.to(tl.bfloat16), allow_tf32=False
        )
        tl.store(W1 + offsets, solution1, mask=key_mask[None, :])
        tl.store(W2 + offsets, solution2, mask=key_mask[None, :])


@triton.autotune(
    configs=_GENERAL_SOLVE_BACKWARD_CONFIGS,
    key=["C", "K", "V"],
)
@triton.jit
def _general_solve_bwd_kernel(
    INVERSE,
    U0,
    W1,
    W2,
    DU0,
    DW1,
    DW2,
    DMATRIX,
    DWRITE,
    DA1,
    DA2,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, C)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    matrix_offsets = pid * C * C + row_ids * C + col_ids
    inverse = tl.load(INVERSE + matrix_offsets)
    inverse_t = tl.trans(inverse)
    grad_matrix = tl.zeros((C, C), dtype=tl.float32)

    value_base = pid * C * V
    for value_block in range(tl.cdiv(V, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < V
        offsets = value_base + rows[:, None] * V + values[None, :]
        grad_solution = tl.load(
            DU0 + offsets, mask=value_mask[None, :], other=0.0
        )
        grad_rhs = safe_dot(inverse_t, grad_solution, allow_tf32=False)
        solution = tl.load(U0 + offsets, mask=value_mask[None, :], other=0.0)
        grad_matrix += safe_dot(
            grad_rhs.to(tl.bfloat16), tl.trans(solution), allow_tf32=False
        )
        tl.store(DWRITE + offsets, grad_rhs, mask=value_mask[None, :])

    key_base = pid * C * K
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        offsets = key_base + rows[:, None] * K + keys[None, :]
        grad_solution1 = tl.load(
            DW1 + offsets, mask=key_mask[None, :], other=0.0
        )
        grad_solution2 = tl.load(
            DW2 + offsets, mask=key_mask[None, :], other=0.0
        )
        grad_rhs1 = safe_dot(inverse_t, grad_solution1, allow_tf32=False)
        grad_rhs2 = safe_dot(inverse_t, grad_solution2, allow_tf32=False)
        solution1 = tl.load(W1 + offsets, mask=key_mask[None, :], other=0.0)
        solution2 = tl.load(W2 + offsets, mask=key_mask[None, :], other=0.0)
        grad_matrix += safe_dot(
            grad_rhs1.to(tl.bfloat16), tl.trans(solution1), allow_tf32=False
        )
        grad_matrix += safe_dot(
            grad_rhs2.to(tl.bfloat16), tl.trans(solution2), allow_tf32=False
        )
        tl.store(DA1 + offsets, grad_rhs1, mask=key_mask[None, :])
        tl.store(DA2 + offsets, grad_rhs2, mask=key_mask[None, :])

    grad_matrix = tl.where(row_ids > col_ids, -grad_matrix, 0.0)
    tl.store(DMATRIX + matrix_offsets, grad_matrix)


@triton.autotune(
    configs=[
        triton.Config(
            {"BK": bk, "BV": bv}, num_warps=warps, num_stages=stages
        )
        for bk, bv, warps in ((32, 32, 4), (64, 32, 4), (64, 64, 8))
        for stages in (2, 3)
    ],
    key=["C", "K", "V"],
)
@triton.jit
def _general_solve_apply_fwd_kernel(
    INVERSE,
    WRITE,
    A1,
    A2,
    U0,
    W1,
    W2,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Apply one FLA-produced inverse to all three RHS without concat/split."""
    pid = tl.program_id(0)
    rows = tl.arange(0, C)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    matrix_offsets = pid * C * C + row_ids * C + col_ids
    inverse = tl.load(INVERSE + matrix_offsets)

    value_base = pid * C * V
    for value_block in range(tl.cdiv(V, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < V
        offsets = value_base + rows[:, None] * V + values[None, :]
        write = tl.load(WRITE + offsets, mask=value_mask[None, :], other=0.0)
        solution = safe_dot(
            inverse, write.to(tl.bfloat16), allow_tf32=False
        )
        tl.store(U0 + offsets, solution, mask=value_mask[None, :])

    key_base = pid * C * K
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        offsets = key_base + rows[:, None] * K + keys[None, :]
        a1 = tl.load(A1 + offsets, mask=key_mask[None, :], other=0.0)
        a2 = tl.load(A2 + offsets, mask=key_mask[None, :], other=0.0)
        solution1 = safe_dot(
            inverse, a1.to(tl.bfloat16), allow_tf32=False
        )
        solution2 = safe_dot(
            inverse, a2.to(tl.bfloat16), allow_tf32=False
        )
        tl.store(W1 + offsets, solution1, mask=key_mask[None, :])
        tl.store(W2 + offsets, solution2, mask=key_mask[None, :])


class _GeneralSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, write, a1, a2):
        chunk_size = matrix.shape[-1]
        key_dim = a1.shape[-1]
        value_dim = write.shape[-1]
        systems = matrix.numel() // (chunk_size * chunk_size)
        matrix, write, a1, a2 = (
            tensor.contiguous() for tensor in (matrix, write, a1, a2)
        )
        inverse = torch.empty_like(matrix, dtype=torch.bfloat16)
        output_shape = matrix.shape[:-1]
        u0 = torch.empty(*output_shape, value_dim, device=write.device, dtype=write.dtype)
        w1 = torch.empty(*output_shape, key_dim, device=a1.device, dtype=torch.bfloat16)
        w2 = torch.empty_like(w1)
        _general_solve_fwd_kernel[(systems,)](
            matrix,
            write,
            a1,
            a2,
            inverse,
            u0,
            w1,
            w2,
            K=key_dim,
            V=value_dim,
            C=chunk_size,
        )
        ctx.save_for_backward(inverse, u0, w1, w2)
        ctx.input_dtypes = (matrix.dtype, write.dtype, a1.dtype, a2.dtype)
        ctx.dimensions = (systems, chunk_size, key_dim, value_dim)
        return u0, w1, w2

    @staticmethod
    def backward(ctx, du0, dw1, dw2):
        inverse, u0, w1, w2 = ctx.saved_tensors
        systems, chunk_size, key_dim, value_dim = ctx.dimensions
        dmatrix = torch.empty_like(inverse, dtype=torch.float32)
        dwrite = torch.empty_like(u0)
        da1 = torch.empty_like(w1, dtype=ctx.input_dtypes[2])
        da2 = torch.empty_like(w2, dtype=ctx.input_dtypes[3])
        _general_solve_bwd_kernel[(systems,)](
            inverse,
            u0,
            w1,
            w2,
            du0.contiguous(),
            dw1.contiguous(),
            dw2.contiguous(),
            dmatrix,
            dwrite,
            da1,
            da2,
            K=key_dim,
            V=value_dim,
            C=chunk_size,
        )
        grads = (dmatrix, dwrite, da1, da2)
        return tuple(
            grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)
        )


class _GeneralFLASolve(torch.autograd.Function):
    """FLA C32/C64 inverse with fused contiguous multi-RHS apply/backward."""

    @staticmethod
    def forward(ctx, matrix, write, a1, a2):
        chunk_size = matrix.shape[-1]
        key_dim = a1.shape[-1]
        value_dim = write.shape[-1]
        systems = matrix.numel() // (chunk_size * chunk_size)
        matrix, write, a1, a2 = (
            tensor.contiguous() for tensor in (matrix, write, a1, a2)
        )
        inverse = _inverse(
            matrix, matrix_is_strict_lower=True
        ).contiguous()
        output_shape = matrix.shape[:-1]
        u0 = torch.empty(
            *output_shape,
            value_dim,
            device=write.device,
            dtype=write.dtype,
        )
        w1 = torch.empty(
            *output_shape,
            key_dim,
            device=a1.device,
            dtype=torch.bfloat16,
        )
        w2 = torch.empty_like(w1)
        _general_solve_apply_fwd_kernel[(systems,)](
            inverse,
            write,
            a1,
            a2,
            u0,
            w1,
            w2,
            K=key_dim,
            V=value_dim,
            C=chunk_size,
        )
        ctx.save_for_backward(inverse, u0, w1, w2)
        ctx.input_dtypes = (matrix.dtype, write.dtype, a1.dtype, a2.dtype)
        ctx.dimensions = (systems, chunk_size, key_dim, value_dim)
        return u0, w1, w2

    @staticmethod
    def backward(ctx, du0, dw1, dw2):
        inverse, u0, w1, w2 = ctx.saved_tensors
        systems, chunk_size, key_dim, value_dim = ctx.dimensions
        dmatrix = torch.empty_like(inverse, dtype=torch.float32)
        dwrite = torch.empty_like(u0)
        da1 = torch.empty_like(w1, dtype=ctx.input_dtypes[2])
        da2 = torch.empty_like(w2, dtype=ctx.input_dtypes[3])
        _general_solve_bwd_kernel[(systems,)](
            inverse,
            u0,
            w1,
            w2,
            du0.contiguous(),
            dw1.contiguous(),
            dw2.contiguous(),
            dmatrix,
            dwrite,
            da1,
            da2,
            K=key_dim,
            V=value_dim,
            C=chunk_size,
        )
        grads = (dmatrix, dwrite, da1, da2)
        return tuple(
            grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)
        )


@torch.compiler.disable
def solve_general_operands(matrix, write, a1, a2):
    if not matrix.is_cuda or matrix.shape[-1] not in (16, 32, 64):
        raise ValueError("the general solve requires CUDA C16/C32/C64 operands")
    # One inverse-plus-three-RHS kernel for forward and backward; the
    # FLA-based class above remains for forward-only profiling.
    return _GeneralSolve.apply(matrix, write, a1, a2)


__all__ = ["solve_general_operands"]

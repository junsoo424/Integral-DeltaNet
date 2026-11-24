"""Fused residual-add + RMSNorm, forward and backward, for norm.RMSNorm.

Computes exactly what :class:`integral_deltanet.norm.RMSNorm` computes --
``s = x + residual`` in the promoted dtype, the norm in float32, the output in
the dtype of ``s`` -- in one pass over the row instead of the dozen
elementwise kernels PyTorch launches for it (and as many again in backward).
On a bandwidth-limited GPU those passes are not small.  Only the summation
order of the row reductions differs from the PyTorch path.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X, R, W, Y, S, RSTD,
    N: tl.constexpr, EPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr, ROUND_SUM: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    s = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        s += tl.load(R + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        if ROUND_SUM:
            # PyTorch adds two half-precision tensors in that precision.
            s = s.to(S.dtype.element_ty).to(tl.float32)
        tl.store(S + row * N + cols, s.to(S.dtype.element_ty), mask=mask)
    rstd = 1.0 / tl.sqrt_rn(tl.sum(s * s, axis=0) / N + EPS)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * N + cols, (s * rstd * w).to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + row, rstd)


@triton.jit
def _rmsnorm_bwd_kernel(
    S, W, RSTD, DY, DRES, DS, DW_PARTIAL,
    ROWS, N: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr,
    HAS_DRES: tl.constexpr, BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK,), dtype=tl.float32)
    for index in range(ROWS_PER_PROGRAM):
        row = (program * ROWS_PER_PROGRAM + index).to(tl.int64)
        if row < ROWS:
            s = tl.load(S + row * N + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + row * N + cols, mask=mask, other=0.0).to(tl.float32)
            rstd = tl.load(RSTD + row)
            xhat = s * rstd
            wdy = dy * w
            ds = (wdy - xhat * (tl.sum(xhat * wdy, axis=0) / N)) * rstd
            if HAS_DRES:
                ds += tl.load(DRES + row * N + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(DS + row * N + cols, ds.to(DS.dtype.element_ty), mask=mask)
            dw += dy * xhat
    tl.store(DW_PARTIAL + program * N + cols, dw, mask=mask)


class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, eps, prenorm):
        shape = x.shape
        n = shape[-1]
        x2 = x.reshape(-1, n).contiguous()
        rows = x2.shape[0]
        if residual is not None:
            sum_dtype = torch.promote_types(x.dtype, residual.dtype)
            r2 = residual.reshape(-1, n).contiguous()
            s2 = torch.empty(rows, n, device=x.device, dtype=sum_dtype)
        else:
            sum_dtype = x.dtype
            r2, s2 = x2, x2
        y2 = torch.empty(rows, n, device=x.device, dtype=sum_dtype)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(n)
        warps = min(max(block // 256, 1), 16)
        _rmsnorm_fwd_kernel[(rows,)](
            x2, r2, weight, y2, s2, rstd,
            N=n, EPS=float(eps), HAS_RESIDUAL=residual is not None,
            ROUND_SUM=sum_dtype in (torch.bfloat16, torch.float16),
            BLOCK=block, num_warps=warps,
        )
        ctx.save_for_backward(s2, weight, rstd)
        ctx.shape = shape
        ctx.input_dtypes = (x.dtype, None if residual is None else residual.dtype)
        ctx.has_residual = residual is not None
        y = y2.view(shape)
        if prenorm:
            return y, (s2 if residual is not None else x2).view(shape)
        return y

    @staticmethod
    def backward(ctx, dy, dres=None):
        s2, weight, rstd = ctx.saved_tensors
        rows, n = s2.shape
        dy2 = dy.reshape(-1, n).contiguous()
        dres2 = dres.reshape(-1, n).contiguous() if dres is not None else dy2
        ds = torch.empty(rows, n, device=s2.device, dtype=s2.dtype)
        block = triton.next_power_of_2(n)
        warps = min(max(block // 256, 1), 16)
        programs = min(rows, torch.cuda.get_device_properties(s2.device).multi_processor_count * 4)
        rows_per_program = triton.cdiv(rows, programs)
        programs = triton.cdiv(rows, rows_per_program)
        dw_partial = torch.empty(programs, n, device=s2.device, dtype=torch.float32)
        _rmsnorm_bwd_kernel[(programs,)](
            s2, weight, rstd, dy2, dres2, ds, dw_partial,
            rows, N=n, ROWS_PER_PROGRAM=rows_per_program,
            HAS_DRES=dres is not None, BLOCK=block, num_warps=warps,
        )
        dw = dw_partial.sum(0).to(weight.dtype)
        ds = ds.view(ctx.shape)
        x_dtype, residual_dtype = ctx.input_dtypes
        dx = ds.to(x_dtype)
        dresidual = ds.to(residual_dtype) if ctx.has_residual else None
        return dx, dresidual, dw, None, None


def fused_rmsnorm(x, residual, weight, eps, prenorm):
    return _FusedRMSNorm.apply(x, residual, weight, eps, prenorm)


__all__ = ["fused_rmsnorm"]

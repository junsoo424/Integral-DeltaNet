"""Register-bounded joint C16 autograd with FP32 gradient accumulation.

The prefix and source-relative recurrences keep their individually tuned
backward kernels.  The relative kernel first writes FP32 input gradients; the
prefix kernel then accumulates into those buffers in place.  This preserves the
speed and bounded compilation of the split recurrences while rounding their
sum only once when returning gradients to BF16 inputs.
"""
from __future__ import annotations

import os

import torch
import triton

from .arch import c16_fused
from .general_direct_config import needs_recomputed_segment_starts
from .general_direct_prefix_triton import (
    _PREFIX_BK,
    _PREFIX_WARPS,
    _prefix_fwd_kernel,
    _prefix_local_bwd_kernel,
)
from .general_direct_relative_triton import (
    _direct_c16_segment_bwd_kernel,
    _direct_c16_segment_bwd_reuse_kernel,
    _direct_c16_segment_bwd_scratch_kernel,
    _relative_backward_config,
    grouped_forward,
)
from .general_direct_reduce_triton import reduce_joint_scalar_grads
from .general_direct_relative_rows_triton import (
    ROWS_CHECKPOINTS,
    _direct_c16_bwd_rows_kernel,
    _direct_c16_fwd_rows_kernel,
    _rows_fwd_reduce_kernel,
)


_RELATIVE_VARIANTS = ("old", "reuse", "scratch3", "scratch1", "rows")


def relative_backward_variant() -> str:
    """Which relative backward kernel to launch (``INTEGRAL_RELATIVE_BWD``)."""
    variant = os.environ.get("INTEGRAL_RELATIVE_BWD")
    if variant is None:
        variant = "reuse" if os.environ.get("INTEGRAL_REUSE_BWD") == "1" else "old"
    if variant not in _RELATIVE_VARIANTS:
        raise ValueError(f"INTEGRAL_RELATIVE_BWD must be one of {_RELATIVE_VARIANTS}")
    return variant


class _StableDirectC16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, g, b, gamma, lam, scale):
        values = tuple(
            value.contiguous() if value.stride(-1) != 0 else value
            for value in (q, k, g, b, gamma, lam)
        )
        q, k, g, b, gamma, lam = values
        batch, length, heads, key_dim = q.shape
        chunks, chunk_size = length // 16, 16
        shape = (batch, heads, chunks, chunk_size, key_dim)
        prefix_outputs = tuple(
            torch.empty(shape, device=q.device, dtype=q.dtype) for _ in range(4)
        )
        transport = torch.empty(
            batch,
            heads,
            chunks,
            key_dim,
            4,
            device=q.device,
            dtype=torch.float32,
        )
        systems = batch * heads * chunks
        if c16_fused(q):
            rows_block = min(32, key_dim)
            rows_nk = triton.cdiv(key_dim, rows_block)
            tail1 = torch.empty(shape, device=q.device, dtype=q.dtype)
            tail2 = torch.empty_like(tail1)
            matrix = torch.empty(
                batch, heads, chunks, chunk_size, chunk_size,
                device=q.device, dtype=torch.float32,
            )
            causal = torch.empty_like(matrix, dtype=q.dtype)
            partial_shape = (systems, rows_nk, chunk_size, chunk_size)
            mpart = torch.empty(partial_shape, device=q.device, dtype=torch.float32)
            cpart = torch.empty_like(mpart)
            _direct_c16_fwd_rows_kernel[(systems * rows_nk,)](
                q, k, g, b, gamma, lam,
                *prefix_outputs, transport,
                tail1, tail2, mpart, cpart,
                FUSED_PREFIX=True,
                T=length, H=heads, N=chunks, K=key_dim, C=chunk_size,
                NK=rows_nk, SCALE=float(scale),
                SB0=b.stride(0), SB1=b.stride(1),
                SB2=b.stride(2), SB3=b.stride(3),
                BK=rows_block, num_warps=1, num_stages=1,
            )
            _rows_fwd_reduce_kernel[(systems,)](
                mpart, cpart, matrix, causal,
                C=chunk_size, NK=rows_nk,
                BLOCK_NK=triton.next_power_of_2(rows_nk),
                num_warps=1, num_stages=1,
            )
            relative_outputs = (tail1, tail2, matrix, causal)
        else:
            block = min(_PREFIX_BK, triton.next_power_of_2(key_dim))
            nk = triton.cdiv(key_dim, block)
            _prefix_fwd_kernel[(systems * nk,)](
                q,
                k,
                g,
                b,
                gamma,
                lam,
                *prefix_outputs,
                transport,
                T=length,
                H=heads,
                N=chunks,
                K=key_dim,
                C=chunk_size,
                SCALE=float(scale),
                SB0=b.stride(0),
                SB1=b.stride(1),
                SB2=b.stride(2),
                SB3=b.stride(3),
                BK=block,
                num_warps=_PREFIX_WARPS,
                num_stages=2,
            )
            relative_outputs = grouped_forward(q, k, g, b, gamma, lam, scale)
        ctx.save_for_backward(*values)
        ctx.dimensions = (batch, length, heads, key_dim, chunks, systems)
        ctx.scale = float(scale)
        ctx.input_dtypes = tuple(value.dtype for value in values)
        return (*prefix_outputs, transport, *relative_outputs)

    @staticmethod
    def backward(
        ctx,
        da1,
        da2,
        dq1,
        dq2,
        dtransport,
        dtail1,
        dtail2,
        dmatrix,
        dcausal,
    ):
        q, k, g, b, gamma, lam = ctx.saved_tensors
        batch, length, heads, key_dim, chunks, systems = ctx.dimensions
        chunk_size = 16
        vector_shape = (batch, heads, chunks, chunk_size, key_dim)
        square_shape = (batch, heads, chunks, chunk_size, chunk_size)
        transport_shape = (batch, heads, chunks, key_dim, 4)

        def grad_or_zero(value, shape, dtype):
            if value is None:
                return torch.zeros(shape, device=q.device, dtype=dtype)
            return value.contiguous()

        da1 = grad_or_zero(da1, vector_shape, q.dtype)
        da2 = grad_or_zero(da2, vector_shape, q.dtype)
        dq1 = grad_or_zero(dq1, vector_shape, q.dtype)
        dq2 = grad_or_zero(dq2, vector_shape, q.dtype)
        dtransport = grad_or_zero(dtransport, transport_shape, torch.float32)
        dtail1 = grad_or_zero(dtail1, vector_shape, q.dtype)
        dtail2 = grad_or_zero(dtail2, vector_shape, q.dtype)
        dmatrix = grad_or_zero(dmatrix, square_shape, torch.float32)
        dcausal = grad_or_zero(dcausal, square_shape, q.dtype)

        # The fused kernel forms every gradient sum in FP32 registers and
        # rounds once on store, so it can write the input dtype directly; the
        # split path accumulates across two kernels and needs FP32 buffers.
        grad_dtype = q.dtype if c16_fused(q) else torch.float32
        dq, dk, dg, db = (
            torch.empty_like(q, dtype=grad_dtype) for _ in range(4)
        )
        relative_block, relative_warps = _relative_backward_config(q, key_dim)
        relative_nk = triton.cdiv(key_dim, relative_block)
        relative_shape = (systems, relative_nk, chunk_size)
        relative_dgamma = torch.empty(
            relative_shape, device=q.device, dtype=torch.float32
        )
        relative_dlam = torch.empty_like(relative_dgamma)
        if c16_fused(q):
            scratch = torch.empty(
                systems * relative_nk * max(ROWS_CHECKPOINTS, 1)
                * (2 * chunk_size + 4) * relative_block,
                device=q.device, dtype=torch.float32,
            )
            _direct_c16_bwd_rows_kernel[(systems * relative_nk,)](
                q, k, g, b, gamma, lam,
                dmatrix, dcausal, dtail1, dtail2,
                da1, da2, dq1, dq2, dtransport,
                dq, dk, dg, db, relative_dgamma, relative_dlam,
                scratch,
                FUSED_PREFIX=True,
                T=length, H=heads, N=chunks, K=key_dim, C=chunk_size,
                NK=relative_nk, SCALE=ctx.scale,
                SB0=b.stride(0), SB1=b.stride(1),
                SB2=b.stride(2), SB3=b.stride(3),
                BK=relative_block, num_warps=relative_warps, num_stages=1,
            )
            # the prefix partials are already folded in
            def reduce_scalar(value):
                reduced = value.sum(dim=1).view(batch, heads, chunks, chunk_size)
                return reduced.reshape(batch, heads, length).transpose(1, 2)

            dgamma = reduce_scalar(relative_dgamma).to(gamma.dtype)
            dlam = reduce_scalar(relative_dlam).to(lam.dtype)
            return (
                dq, dk, dg, db, dgamma, dlam, None,
            )
        else:
            variant = relative_backward_variant()
            launch_extra = {}
            prefix_slots = ()       # the row kernel's fused-prefix operands
            if variant == "old":
                _relative_bwd = _direct_c16_segment_bwd_kernel
                launch_extra = dict(
                    RECOMPUTE_SEGMENT_STARTS=needs_recomputed_segment_starts(q),
                    ABL_DP=os.environ.get("INTEGRAL_ABL_DP") == "1",
                    ABL_DM=os.environ.get("INTEGRAL_ABL_DM") == "1",
                )
                extra_args = ()
            elif variant == "reuse":
                _relative_bwd = _direct_c16_segment_bwd_reuse_kernel
                launch_extra = dict(
                    RECOMPUTE_SEGMENT_STARTS=needs_recomputed_segment_starts(q),
                )
                extra_args = ()
            elif variant == "rows":
                _relative_bwd = _direct_c16_bwd_rows_kernel
                launch_extra = dict(FUSED_PREFIX=False)
                prefix_slots = (None,) * 5
                scratch = torch.empty(
                    systems * relative_nk * max(ROWS_CHECKPOINTS, 1) * 2
                    * chunk_size * relative_block,
                    device=q.device, dtype=torch.float32,
                )
                extra_args = (scratch,)
            else:
                ckpt = 3 if variant == "scratch3" else 1
                _relative_bwd = _direct_c16_segment_bwd_scratch_kernel
                launch_extra = dict(CKPT=ckpt)
                scratch = torch.empty(
                    systems * relative_nk * ckpt * 2 * chunk_size * relative_block,
                    device=q.device, dtype=torch.float32,
                )
                extra_args = (scratch,)
            _relative_bwd[(systems * relative_nk,)](
                q,
                k,
                g,
                b,
                gamma,
                lam,
                dmatrix,
                dcausal,
                dtail1,
                dtail2,
                *prefix_slots,
                dq,
                dk,
                dg,
                db,
                relative_dgamma,
                relative_dlam,
                *extra_args,
                T=length,
                H=heads,
                N=chunks,
                K=key_dim,
                C=chunk_size,
                NK=relative_nk,
                SCALE=ctx.scale,
                SB0=b.stride(0),
                SB1=b.stride(1),
                SB2=b.stride(2),
                SB3=b.stride(3),
                **launch_extra,
                BK=relative_block,
                num_warps=relative_warps,
                num_stages=1,
            )

            prefix_block = min(_PREFIX_BK, triton.next_power_of_2(key_dim))
            prefix_nk = triton.cdiv(key_dim, prefix_block)
            prefix_shape = (systems, prefix_nk, chunk_size)
            prefix_dgamma = torch.empty(
                prefix_shape, device=q.device, dtype=torch.float32
            )
            prefix_dlam = torch.empty_like(prefix_dgamma)
            _prefix_local_bwd_kernel[(systems * prefix_nk,)](
                q,
                k,
                g,
                b,
                gamma,
                lam,
                da1,
                da2,
                dq1,
                dq2,
                dtransport,
                dq,
                dk,
                dg,
                db,
                prefix_dgamma,
                prefix_dlam,
                T=length,
                H=heads,
                N=chunks,
                K=key_dim,
                C=chunk_size,
                NK=prefix_nk,
                SCALE=ctx.scale,
                SB0=b.stride(0),
                SB1=b.stride(1),
                SB2=b.stride(2),
                SB3=b.stride(3),
                ACCUMULATE=True,
                RECOMPUTE_SEGMENT_STARTS=needs_recomputed_segment_starts(q),
                BK=prefix_block,
                num_warps=_PREFIX_WARPS,
                num_stages=1,
            )

        dgamma, dlam = reduce_joint_scalar_grads(
            relative_dgamma,
            relative_dlam,
            prefix_dgamma,
            prefix_dlam,
            gamma,
            lam,
            batch=batch,
            length=length,
            heads=heads,
            chunks=chunks,
            chunk_size=chunk_size,
        )
        values = (dq, dk, dg, db, dgamma, dlam)
        return tuple(
            value.to(dtype) for value, dtype in zip(values, ctx.input_dtypes)
        ) + (None,)


stable_direct_c16 = _StableDirectC16.apply

__all__ = ["stable_direct_c16"]

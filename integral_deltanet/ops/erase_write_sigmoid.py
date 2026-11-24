"""GDN-2 erase/write gates from adjacent columns of the merged input GEMM.

``b = sigmoid(b_logits) * scale`` and ``w = sigmoid(w_logits)`` read the b|w
column block of the merged projection, which is strided (its row stride is
the GEMM's full width), so torch's two sigmoids fall to the non-vectorized
elementwise path.  One kernel reads each row's contiguous b|w span, keeps
the FP32 math of the unfused ops and writes both gates contiguous.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch.autograd.function import once_differentiable


@triton.jit
def _gates_fwd(X, B, W, SX, NB: tl.constexpr, NW: tl.constexpr, SCALE: tl.constexpr,
               BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * SX + cols, cols < NB + NW, 0).to(tl.float32)
    y = 1.0 / (1.0 + libdevice.exp(-x))
    tl.store(B + row * NB + cols, (y.to(B.dtype.element_ty) * SCALE), cols < NB)
    tl.store(W + row * NW + (cols - NB), y, (cols >= NB) & (cols < NB + NW))


@triton.jit
def _gates_bwd(B, W, DB, DW, DX, NB: tl.constexpr, NW: tl.constexpr, SCALE: tl.constexpr,
               BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    is_b = cols < NB
    is_w = (cols >= NB) & (cols < NB + NW)
    y = tl.where(is_b, tl.load(B + row * NB + cols, is_b, 0).to(tl.float32) / SCALE,
                 tl.load(W + row * NW + (cols - NB), is_w, 0).to(tl.float32))
    # The unfused graph rounds the scaled gradient to the gate dtype first.
    db = (tl.load(DB + row * NB + cols, is_b, 0).to(tl.float32) * SCALE).to(DB.dtype.element_ty)
    dy = tl.where(is_b, db.to(tl.float32), tl.load(DW + row * NW + (cols - NB), is_w, 0).to(tl.float32))
    tl.store(DX + row * (NB + NW) + cols, dy * (1.0 - y) * y, cols < NB + NW)


class _EraseWriteGates(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, erase_width, scale):
        rows = logits.numel() // logits.shape[-1]
        width = logits.shape[-1]
        b = logits.new_empty(*logits.shape[:-1], erase_width)
        w = logits.new_empty(*logits.shape[:-1], width - erase_width)
        flat = logits.reshape(rows, width)
        grid = (rows, triton.cdiv(width, 1024))
        _gates_fwd[grid](flat, b, w, flat.stride(0), erase_width, width - erase_width, scale, BLOCK=1024)
        ctx.save_for_backward(b, w)
        ctx.scale = scale
        return b, w

    @staticmethod
    @once_differentiable
    def backward(ctx, db, dw):
        b, w = ctx.saved_tensors
        erase_width, write_width = b.shape[-1], w.shape[-1]
        rows = b.numel() // erase_width
        db = torch.zeros_like(b) if db is None else db.contiguous()
        dw = torch.zeros_like(w) if dw is None else dw.contiguous()
        dx = b.new_empty(*b.shape[:-1], erase_width + write_width)
        grid = (rows, triton.cdiv(erase_width + write_width, 1024))
        _gates_bwd[grid](b, w, db, dw, dx, erase_width, write_width, ctx.scale, BLOCK=1024)
        return dx, None, None


def erase_write_gates(logits: torch.Tensor, erase_width: int, scale: float):
    """``(sigmoid(logits[..., :erase_width]) * scale, sigmoid(logits[..., erase_width:]))``.

    ``logits`` may be a column view whose rows are contiguous; the gates come
    back contiguous in the logits' dtype.
    """
    if (not logits.is_cuda or logits.stride(-1) != 1
            or logits.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
        raise ValueError("expected a CUDA b|w logit block with contiguous rows")
    flat_ok = logits.ndim < 2 or all(
        logits.stride(i) == logits.stride(i + 1) * logits.shape[i + 1]
        for i in range(logits.ndim - 2)
    )
    if not flat_ok:
        raise ValueError("the leading dims of the logit block must flatten to rows")
    return _EraseWriteGates.apply(logits, erase_width, float(scale))


__all__ = ["erase_write_gates"]

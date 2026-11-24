"""Independent channel gates: fused FP32 reference operations, separate logits.

No sigmoid algebra rewrite: retain softplus(threshold=20), exp and expm1,
including the reference's rounded expm1 derivative at saturated lambda.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch.autograd.function import once_differentiable


@triton.jit
def _values(x):
    e = libdevice.exp(x)
    sp = tl.where(x > 20.0, x, libdevice.log1p(e))
    return sp, e


@triton.jit
def _pair_fwd(GX, LX, G, L, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(GX + i, i < N, 0).to(tl.float32)
    y = tl.load(LX + i, i < N, 0).to(tl.float32)
    sx, _ = _values(x)
    sy, _ = _values(y)
    tl.store(G + i, libdevice.exp(-sx), i < N)
    tl.store(L + i, -libdevice.expm1(-sy), i < N)


@triton.jit
def _pair_bwd(GX, LX, DG, DL, DX, DY, N: tl.constexpr,
              HAS_G: tl.constexpr, HAS_L: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(GX + i, i < N, 0).to(tl.float32)
    y = tl.load(LX + i, i < N, 0).to(tl.float32)
    sx, ex = _values(x)
    sy, ey = _values(y)
    if HAS_G:
        dg = tl.load(DG + i, i < N, 0).to(tl.float32)
        # ExpBackward -> NegBackward -> SoftplusBackward, FP32 at each step.
        dx = -(dg * libdevice.exp(-sx))
        dx = tl.where(x > 20.0, dx, dx * ex / (ex + 1.0))
    else:
        dx = tl.full((BLOCK,), 0, tl.float32)
    if HAS_L:
        dl = tl.load(DL + i, i < N, 0).to(tl.float32)
        dy = -((-dl) * (libdevice.expm1(-sy) + 1.0))
        dy = tl.where(y > 20.0, dy, dy * ey / (ey + 1.0))
    else:
        dy = tl.full((BLOCK,), 0, tl.float32)
    tl.store(DX + i, dx, i < N)
    tl.store(DY + i, dy, i < N)


class _GatePair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gx, lx):
        gx, lx = gx.contiguous(), lx.contiguous()
        g, l = torch.empty_like(gx, dtype=torch.float32), torch.empty_like(lx, dtype=torch.float32)
        n = gx.numel()
        if n:
            _pair_fwd[(triton.cdiv(n, 1024),)](gx, lx, g, l, n, BLOCK=1024, enable_fp_fusion=False)
        ctx.save_for_backward(gx, lx)
        ctx.set_materialize_grads(False)
        return g, l

    @staticmethod
    @once_differentiable
    def backward(ctx, dg, dl):
        gx, lx = ctx.saved_tensors
        dx, dy = torch.empty_like(gx), torch.empty_like(lx)
        n = gx.numel()
        if n:
            _pair_bwd[(triton.cdiv(n, 1024),)](
                gx, lx, gx if dg is None else dg.contiguous(), lx if dl is None else dl.contiguous(),
                dx, dy, n, HAS_G=dg is not None, HAS_L=dl is not None,
                BLOCK=1024, enable_fp_fusion=False)
        return dx, dy


def independent_gate_pair(gamma_logits, lambda_logits):
    if gamma_logits.shape != lambda_logits.shape:
        raise ValueError('Paired gates require equal shapes')
    if (gamma_logits.device != lambda_logits.device or not gamma_logits.is_cuda
            or gamma_logits.dtype != lambda_logits.dtype
            or gamma_logits.dtype not in (torch.float32, torch.bfloat16, torch.float16)):
        raise ValueError('Paired gates require equal CUDA device and FP32/BF16/FP16 dtype')
    return _GatePair.apply(gamma_logits, lambda_logits)

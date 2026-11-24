"""Fuse the current unbounded per-channel decay gate, retaining FP32 math.

Low-precision A_log/dt_bias (a mixer cast to bf16) are read as FP32, as the
unfused path upcasts them, and their gradients are cast back after the FP32
reductions."""
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from .channel_gate_pair import _values


@triton.jit
def _decay_fwd(RAW, A, BIAS, G, N: tl.constexpr, H: tl.constexpr,
               K: tl.constexpr, BLOCK: tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    c=i%(H*K);h=c//K
    raw=tl.load(RAW+i,i<N,0).to(tl.float32)
    bias=tl.load(BIAS+c,i<N,0).to(tl.float32)
    a=tl.load(A+h,i<N,0).to(tl.float32)
    sp,_=_values(raw+bias)
    tl.store(G+i,(-a)*sp,i<N)


@triton.jit
def _decay_bwd(RAW,A,BIAS,DG,DR,DZ,DA,N:tl.constexpr,H:tl.constexpr,
               K:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    c=i%(H*K);h=c//K
    raw=tl.load(RAW+i,i<N,0).to(tl.float32)
    bias=tl.load(BIAS+c,i<N,0).to(tl.float32)
    a=tl.load(A+h,i<N,0).to(tl.float32)
    dg=tl.load(DG+i,i<N,0).to(tl.float32)
    z=raw+bias;sp,ez=_values(z)
    dz=dg*(-a)
    dz=tl.where(z>20.,dz,dz*ez/(ez+1.))
    tl.store(DR+i,dz,i<N)
    tl.store(DZ+i,dz,i<N)
    # The original reduces dG*softplus before negation and ExpBackward.
    tl.store(DA+i,dg*sp,i<N)


class _DecayGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,raw,a_log,bias):
        raw=raw.contiguous();bias=bias.contiguous();a=a_log.float().exp()
        h,k=raw.shape[-2:];n=raw.numel();g=torch.empty_like(raw,dtype=torch.float32)
        if n:
            _decay_fwd[(triton.cdiv(n,1024),)](raw,a,bias,g,n,h,k,BLOCK=1024,enable_fp_fusion=False)
        ctx.save_for_backward(raw,a,bias);ctx.a_dtype=a_log.dtype
        return g

    @staticmethod
    @once_differentiable
    def backward(ctx,dg):
        raw,a,bias=ctx.saved_tensors;h,k=raw.shape[-2:];n=raw.numel()
        dr=torch.empty_like(raw);dz=torch.empty_like(raw,dtype=torch.float32);da=torch.empty_like(dz)
        if n:
            _decay_bwd[(triton.cdiv(n,1024),)](raw,a,bias,dg.contiguous(),dr,dz,da,n,h,k,BLOCK=1024,enable_fp_fusion=False)
        # Use the reference PyTorch reductions, rather than unordered atomics.
        d_bias=dz.sum(dim=(0,1)).reshape(bias.shape).to(bias.dtype)
        d_a=(-(da.sum(dim=(0,1,3))))*a
        return dr,d_a.to(ctx.a_dtype),d_bias


def channel_decay_gate(raw,a_log,bias):
    if (not raw.is_cuda or raw.ndim!=4 or a_log.shape!=(raw.shape[-2],)
            or bias.numel()!=raw.shape[-2]*raw.shape[-1]
            or raw.device!=a_log.device or raw.device!=bias.device
            or raw.dtype not in (torch.float32,torch.bfloat16,torch.float16)
            or a_log.dtype not in (torch.float32,torch.bfloat16,torch.float16)
            or bias.dtype not in (torch.float32,torch.bfloat16,torch.float16)):
        raise ValueError('Expected CUDA B,T,H,K raw and floating H decay/H*K bias parameters')
    return _DecayGate.apply(raw,a_log,bias)

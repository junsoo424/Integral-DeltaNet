"""Inverse-free scalar transports for Integral GDN.

The normalized paper factorization uses ``c_s F_s^{-1}``.  Released GDN
initialization can make that coordinate system singular in FP32 even though
every relative transition in the original recurrence is bounded.  These
kernels form exactly those relative products directly and differentiate them
without any determinant, reciprocal, or gate clamp.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .gdn_relative_direct_triton import (
    _gdn_relative_direct_bwd_kernel,
    _gdn_relative_direct_fwd_kernel,
)


_SUPPORTED_CHUNK_SIZES = (16, 32, 64)
_COEFFICIENT_VARIANTS = ("direct", "old")


def scalar_coefficient_variant() -> str:
    """Which scalar-coefficient kernels to launch (``INTEGRAL_SCALAR_COEFF``)."""
    variant = os.environ.get("INTEGRAL_SCALAR_COEFF", "direct")
    if variant not in _COEFFICIENT_VARIANTS:
        raise ValueError(f"INTEGRAL_SCALAR_COEFF must be one of {_COEFFICIENT_VARIANTS}")
    return variant


@triton.jit
def _gdn_relative_fwd_kernel(
    G,
    BETA,
    GAMMA,
    LAM,
    PI1,
    PI2,
    ETA1,
    ETA2,
    INTERACTION,
    CAUSAL,
    TAIL1,
    TAIL2,
    TRANSPORT,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk = pid % N
    head = (pid // N) % H
    batch = pid // (N * H)
    rows = tl.arange(0, C)
    input_base = (batch * T + chunk * C) * H + head
    alpha = tl.exp(tl.load(G + input_base + rows * H).to(tl.float32))
    beta = tl.load(BETA + input_base + rows * H).to(tl.float32)
    gamma = tl.load(GAMMA + input_base + rows * H).to(tl.float32)
    lam = tl.load(LAM + input_base + rows * H).to(tl.float32)
    p11 = gamma * alpha
    p12 = lam * p11
    p21 = (1.0 - gamma) * alpha
    p22 = (1.0 - lam) + lam * p21

    # Incoming-state prefixes.  Only multiplications of contractive matrices
    # are used; no prefix is inverted.
    f11, f12, f21, f22 = 1.0, 0.0, 0.0, 1.0
    for target in range(C):
        alpha_beta = tl.sum(
            tl.where(rows == target, alpha * beta, 0.0), axis=0
        )
        coefficient_offset = pid * C + target
        tl.store(PI1 + coefficient_offset, f11 * alpha_beta)
        tl.store(PI2 + coefficient_offset, f21 * alpha_beta)
        p11_t = tl.sum(tl.where(rows == target, p11, 0.0), axis=0)
        p12_t = tl.sum(tl.where(rows == target, p12, 0.0), axis=0)
        p21_t = tl.sum(tl.where(rows == target, p21, 0.0), axis=0)
        p22_t = tl.sum(tl.where(rows == target, p22, 0.0), axis=0)
        next11 = f11 * p11_t + f12 * p21_t
        next12 = f11 * p12_t + f12 * p22_t
        next21 = f21 * p11_t + f22 * p21_t
        next22 = f21 * p12_t + f22 * p22_t
        f11, f12, f21, f22 = next11, next12, next21, next22
        tl.store(ETA1 + coefficient_offset, f11)
        tl.store(ETA2 + coefficient_offset, f21)

    # One row state per edit source.  Before target r it is
    # [1, lambda_s] P_{s+1} ... P_{r-1}; after the update it also includes P_r.
    c1 = tl.full((C,), 1.0, tl.float32)
    c2 = lam
    for target in range(C):
        earlier = rows < target
        alpha_beta = tl.sum(
            tl.where(rows == target, alpha * beta, 0.0), axis=0
        )
        matrix_offsets = pid * C * C + target * C + rows
        tl.store(
            INTERACTION + matrix_offsets,
            tl.where(earlier, c1 * alpha_beta, 0.0),
        )
        p11_t = tl.sum(tl.where(rows == target, p11, 0.0), axis=0)
        p12_t = tl.sum(tl.where(rows == target, p12, 0.0), axis=0)
        p21_t = tl.sum(tl.where(rows == target, p21, 0.0), axis=0)
        p22_t = tl.sum(tl.where(rows == target, p22, 0.0), axis=0)
        next1 = c1 * p11_t + c2 * p21_t
        next2 = c1 * p12_t + c2 * p22_t
        c1 = tl.where(earlier, next1, c1)
        c2 = tl.where(earlier, next2, c2)
        tl.store(
            CAUSAL + matrix_offsets,
            tl.where(earlier, c1, tl.where(rows == target, 1.0, 0.0)),
        )

    offsets = pid * C + rows
    tl.store(TAIL1 + offsets, c1)
    tl.store(TAIL2 + offsets, c2)
    tl.store(TRANSPORT + pid * 4 + 0, f11)
    tl.store(TRANSPORT + pid * 4 + 1, f12)
    tl.store(TRANSPORT + pid * 4 + 2, f21)
    tl.store(TRANSPORT + pid * 4 + 3, f22)


@triton.jit
def _gdn_relative_history_kernel(
    G,
    GAMMA,
    LAM,
    F_HISTORY,
    C1_HISTORY,
    C2_HISTORY,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
):
    """Recompute small prefix histories for backward, FLA-style."""
    pid = tl.program_id(0)
    chunk = pid % N
    head = (pid // N) % H
    batch = pid // (N * H)
    rows = tl.arange(0, C)
    input_base = (batch * T + chunk * C) * H + head
    alpha = tl.exp(tl.load(G + input_base + rows * H).to(tl.float32))
    gamma = tl.load(GAMMA + input_base + rows * H).to(tl.float32)
    lam = tl.load(LAM + input_base + rows * H).to(tl.float32)
    p11 = gamma * alpha
    p12 = lam * p11
    p21 = (1.0 - gamma) * alpha
    p22 = (1.0 - lam) + lam * p21

    f11, f12, f21, f22 = 1.0, 0.0, 0.0, 1.0
    c1 = tl.full((C,), 1.0, tl.float32)
    c2 = lam
    for target in range(C):
        f_offset = pid * C * 4 + target * 4
        tl.store(F_HISTORY + f_offset + 0, f11)
        tl.store(F_HISTORY + f_offset + 1, f12)
        tl.store(F_HISTORY + f_offset + 2, f21)
        tl.store(F_HISTORY + f_offset + 3, f22)
        history_offsets = pid * C * C + target * C + rows
        tl.store(C1_HISTORY + history_offsets, c1)
        tl.store(C2_HISTORY + history_offsets, c2)

        p11_t = tl.sum(tl.where(rows == target, p11, 0.0), axis=0)
        p12_t = tl.sum(tl.where(rows == target, p12, 0.0), axis=0)
        p21_t = tl.sum(tl.where(rows == target, p21, 0.0), axis=0)
        p22_t = tl.sum(tl.where(rows == target, p22, 0.0), axis=0)
        next11 = f11 * p11_t + f12 * p21_t
        next12 = f11 * p12_t + f12 * p22_t
        next21 = f21 * p11_t + f22 * p21_t
        next22 = f21 * p12_t + f22 * p22_t
        f11, f12, f21, f22 = next11, next12, next21, next22
        earlier = rows < target
        next1 = c1 * p11_t + c2 * p21_t
        next2 = c1 * p12_t + c2 * p22_t
        c1 = tl.where(earlier, next1, c1)
        c2 = tl.where(earlier, next2, c2)


@triton.jit
def _gdn_relative_bwd_kernel(
    G,
    BETA,
    GAMMA,
    LAM,
    F_HISTORY,
    C1_HISTORY,
    C2_HISTORY,
    DPI1,
    DPI2,
    DETA1,
    DETA2,
    DINTERACTION,
    DCAUSAL,
    DTAIL1,
    DTAIL2,
    DTRANSPORT,
    DG,
    DBETA,
    DGAMMA,
    DLAM,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk = pid % N
    head = (pid // N) % H
    batch = pid // (N * H)
    rows = tl.arange(0, C)
    input_base = (batch * T + chunk * C) * H + head
    alpha = tl.exp(tl.load(G + input_base + rows * H).to(tl.float32))
    beta = tl.load(BETA + input_base + rows * H).to(tl.float32)
    gamma = tl.load(GAMMA + input_base + rows * H).to(tl.float32)
    lam = tl.load(LAM + input_base + rows * H).to(tl.float32)
    p11 = gamma * alpha
    p12 = lam * p11
    p21 = (1.0 - gamma) * alpha
    p22 = (1.0 - lam) + lam * p21

    grad_alpha = tl.zeros((C,), tl.float32)
    grad_beta = tl.zeros((C,), tl.float32)
    grad_lam = tl.zeros((C,), tl.float32)
    grad_p11 = tl.zeros((C,), tl.float32)
    grad_p12 = tl.zeros((C,), tl.float32)
    grad_p21 = tl.zeros((C,), tl.float32)
    grad_p22 = tl.zeros((C,), tl.float32)

    # Reverse all direct relative source transports.  Histories are recomputed
    # immediately before this kernel, rather than retained for every layer.
    offsets = pid * C + rows
    adjoint1 = tl.load(DTAIL1 + offsets).to(tl.float32)
    adjoint2 = tl.load(DTAIL2 + offsets).to(tl.float32)
    for reverse_index in range(C - 1):
        target = C - 1 - reverse_index
        active_sources = rows < target
        matrix_offsets = pid * C * C + target * C + rows
        grad_interaction = tl.load(DINTERACTION + matrix_offsets).to(tl.float32)
        grad_causal = tl.load(DCAUSAL + matrix_offsets).to(tl.float32)
        cprev1 = tl.load(C1_HISTORY + matrix_offsets).to(tl.float32)
        cprev2 = tl.load(C2_HISTORY + matrix_offsets).to(tl.float32)
        after1 = adjoint1 + tl.where(active_sources, grad_causal, 0.0)
        after2 = adjoint2

        dp11 = tl.sum(
            tl.where(active_sources, cprev1 * after1, 0.0), axis=0
        )
        dp12 = tl.sum(
            tl.where(active_sources, cprev1 * after2, 0.0), axis=0
        )
        dp21 = tl.sum(
            tl.where(active_sources, cprev2 * after1, 0.0), axis=0
        )
        dp22 = tl.sum(
            tl.where(active_sources, cprev2 * after2, 0.0), axis=0
        )
        grad_p11 = tl.where(rows == target, dp11, grad_p11)
        grad_p12 = tl.where(rows == target, dp12, grad_p12)
        grad_p21 = tl.where(rows == target, dp21, grad_p21)
        grad_p22 = tl.where(rows == target, dp22, grad_p22)

        p11_t = tl.sum(tl.where(rows == target, p11, 0.0), axis=0)
        p12_t = tl.sum(tl.where(rows == target, p12, 0.0), axis=0)
        p21_t = tl.sum(tl.where(rows == target, p21, 0.0), axis=0)
        p22_t = tl.sum(tl.where(rows == target, p22, 0.0), axis=0)
        previous1 = after1 * p11_t + after2 * p12_t
        previous2 = after1 * p21_t + after2 * p22_t
        alpha_beta_t = tl.sum(
            tl.where(rows == target, alpha * beta, 0.0), axis=0
        )
        previous1 += tl.where(
            active_sources, grad_interaction * alpha_beta_t, 0.0
        )
        grad_alpha_beta = tl.sum(
            tl.where(active_sources, grad_interaction * cprev1, 0.0),
            axis=0,
        )
        alpha_t = tl.sum(tl.where(rows == target, alpha, 0.0), axis=0)
        beta_t = tl.sum(tl.where(rows == target, beta, 0.0), axis=0)
        grad_alpha += tl.where(rows == target, grad_alpha_beta * beta_t, 0.0)
        grad_beta += tl.where(rows == target, grad_alpha_beta * alpha_t, 0.0)
        adjoint1 = tl.where(active_sources, previous1, adjoint1)
        adjoint2 = tl.where(active_sources, previous2, adjoint2)
    grad_lam += adjoint2

    # Reverse the absolute incoming-state prefix.
    d_f11 = tl.load(DTRANSPORT + pid * 4 + 0).to(tl.float32)
    d_f12 = tl.load(DTRANSPORT + pid * 4 + 1).to(tl.float32)
    d_f21 = tl.load(DTRANSPORT + pid * 4 + 2).to(tl.float32)
    d_f22 = tl.load(DTRANSPORT + pid * 4 + 3).to(tl.float32)
    dpi1 = tl.load(DPI1 + offsets).to(tl.float32)
    dpi2 = tl.load(DPI2 + offsets).to(tl.float32)
    deta1 = tl.load(DETA1 + offsets).to(tl.float32)
    deta2 = tl.load(DETA2 + offsets).to(tl.float32)
    for reverse_index in range(C):
        target = C - 1 - reverse_index
        d_f11 += tl.sum(tl.where(rows == target, deta1, 0.0), axis=0)
        d_f21 += tl.sum(tl.where(rows == target, deta2, 0.0), axis=0)
        history_offset = pid * C * 4 + target * 4
        f11 = tl.load(F_HISTORY + history_offset + 0).to(tl.float32)
        f12 = tl.load(F_HISTORY + history_offset + 1).to(tl.float32)
        f21 = tl.load(F_HISTORY + history_offset + 2).to(tl.float32)
        f22 = tl.load(F_HISTORY + history_offset + 3).to(tl.float32)
        p11_t = tl.sum(tl.where(rows == target, p11, 0.0), axis=0)
        p12_t = tl.sum(tl.where(rows == target, p12, 0.0), axis=0)
        p21_t = tl.sum(tl.where(rows == target, p21, 0.0), axis=0)
        p22_t = tl.sum(tl.where(rows == target, p22, 0.0), axis=0)

        add11 = f11 * d_f11 + f21 * d_f21
        add12 = f11 * d_f12 + f21 * d_f22
        add21 = f12 * d_f11 + f22 * d_f21
        add22 = f12 * d_f12 + f22 * d_f22
        old11 = tl.sum(tl.where(rows == target, grad_p11, 0.0), axis=0)
        old12 = tl.sum(tl.where(rows == target, grad_p12, 0.0), axis=0)
        old21 = tl.sum(tl.where(rows == target, grad_p21, 0.0), axis=0)
        old22 = tl.sum(tl.where(rows == target, grad_p22, 0.0), axis=0)
        grad_p11 = tl.where(rows == target, old11 + add11, grad_p11)
        grad_p12 = tl.where(rows == target, old12 + add12, grad_p12)
        grad_p21 = tl.where(rows == target, old21 + add21, grad_p21)
        grad_p22 = tl.where(rows == target, old22 + add22, grad_p22)

        previous_d_f11 = d_f11 * p11_t + d_f12 * p12_t
        previous_d_f12 = d_f11 * p21_t + d_f12 * p22_t
        previous_d_f21 = d_f21 * p11_t + d_f22 * p12_t
        previous_d_f22 = d_f21 * p21_t + d_f22 * p22_t
        dpi1_t = tl.sum(tl.where(rows == target, dpi1, 0.0), axis=0)
        dpi2_t = tl.sum(tl.where(rows == target, dpi2, 0.0), axis=0)
        alpha_t = tl.sum(tl.where(rows == target, alpha, 0.0), axis=0)
        beta_t = tl.sum(tl.where(rows == target, beta, 0.0), axis=0)
        alpha_beta_t = alpha_t * beta_t
        previous_d_f11 += dpi1_t * alpha_beta_t
        previous_d_f21 += dpi2_t * alpha_beta_t
        grad_alpha_beta = dpi1_t * f11 + dpi2_t * f21
        grad_alpha += tl.where(rows == target, grad_alpha_beta * beta_t, 0.0)
        grad_beta += tl.where(rows == target, grad_alpha_beta * alpha_t, 0.0)
        d_f11, d_f12 = previous_d_f11, previous_d_f12
        d_f21, d_f22 = previous_d_f21, previous_d_f22

    active = gamma * alpha
    passive = (1.0 - gamma) * alpha
    grad_active = grad_p11 + lam * grad_p12
    grad_passive = grad_p21 + lam * grad_p22
    grad_alpha += grad_active * gamma + grad_passive * (1.0 - gamma)
    grad_gamma = alpha * (grad_active - grad_passive)
    grad_lam += grad_p12 * active + grad_p22 * (passive - 1.0)

    output_offsets = input_base + rows * H
    tl.store(DG + output_offsets, grad_alpha * alpha)
    tl.store(DBETA + output_offsets, grad_beta)
    tl.store(DGAMMA + output_offsets, grad_gamma)
    tl.store(DLAM + output_offsets, grad_lam)


class _GDNRelativeCoefficients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, beta, gamma, lam, chunk_size):
        batch, length, heads = beta.shape
        if chunk_size not in _SUPPORTED_CHUNK_SIZES or length % chunk_size:
            raise ValueError("length must be divisible by a 16/32/64 chunk")
        chunks = length // chunk_size
        g, beta, gamma, lam = (
            tensor.contiguous() for tensor in (g, beta, gamma, lam)
        )
        vector_shape = (batch, heads, chunks, chunk_size)
        matrix_shape = (*vector_shape, chunk_size)
        vectors = tuple(
            torch.empty(vector_shape, dtype=torch.float32, device=g.device)
            for _ in range(6)
        )
        interaction = torch.empty(matrix_shape, dtype=torch.float32, device=g.device)
        causal = torch.empty_like(interaction)
        transport = torch.empty(
            batch, heads, chunks, 4, dtype=torch.float32, device=g.device
        )
        pi1, pi2, eta1, eta2, tail1, tail2 = vectors
        # Keep the tuple allocation compact while presenting outputs in their
        # semantic order below.
        forward_kernel = (
            _gdn_relative_direct_fwd_kernel
            if scalar_coefficient_variant() == "direct"
            else _gdn_relative_fwd_kernel
        )
        forward_kernel[(batch * heads * chunks,)](
            g, beta, gamma, lam,
            pi1, pi2, eta1, eta2,
            interaction, causal, tail1, tail2, transport,
            T=length, H=heads, N=chunks, C=chunk_size,
            num_warps=1,
        )
        ctx.save_for_backward(g, beta, gamma, lam)
        ctx.dimensions = (batch, length, heads, chunks, chunk_size)
        return pi1, pi2, eta1, eta2, interaction, causal, tail1, tail2, transport

    @staticmethod
    def backward(ctx, *output_grads):
        g, beta, gamma, lam = ctx.saved_tensors
        batch, length, heads, chunks, chunk_size = ctx.dimensions
        vector_shape = (batch, heads, chunks, chunk_size)
        matrix_shape = (*vector_shape, chunk_size)
        output_shapes = (
            vector_shape, vector_shape, vector_shape, vector_shape,
            matrix_shape, matrix_shape,
            vector_shape, vector_shape,
            (batch, heads, chunks, 4),
        )
        output_grads = tuple(
            grad.contiguous()
            if grad is not None
            else torch.zeros(shape, dtype=torch.float32, device=g.device)
            for grad, shape in zip(output_grads, output_shapes)
        )
        systems = batch * heads * chunks
        input_grads = tuple(
            torch.empty_like(tensor) for tensor in (g, beta, gamma, lam)
        )
        if scalar_coefficient_variant() == "direct":
            # One launch; per-target scalars are loaded by index and the two
            # prefix histories go through a small scratch buffer instead of
            # chunk-wide select-reductions on registers.
            f_scratch = torch.empty(
                systems, chunk_size, 4, dtype=torch.float32, device=g.device
            )
            c1_scratch = torch.empty(
                systems, chunk_size, chunk_size,
                dtype=torch.float32, device=g.device,
            )
            c2_scratch = torch.empty_like(c1_scratch)
            _gdn_relative_direct_bwd_kernel[(systems,)](
                g, beta, gamma, lam,
                f_scratch, c1_scratch, c2_scratch,
                *output_grads,
                *input_grads,
                T=length, H=heads, N=chunks, C=chunk_size,
                num_warps=1,
            )
            return (*input_grads, None)
        # C32 and C64 recompute in registers, avoiding the large FP32 history
        # tensors. C16 keeps the smaller two-launch materialized path.
        if chunk_size in (32, 64):
            from .gdn_relative_recompute_triton import (
                _gdn_relative_recompute_bwd_kernel,
            )

            block = 8 if chunk_size == 32 else 16
            _gdn_relative_recompute_bwd_kernel[(systems,)](
                g,
                beta,
                gamma,
                lam,
                *output_grads,
                *input_grads,
                T=length,
                H=heads,
                N=chunks,
                C=chunk_size,
                B=block,
                num_warps=1,
            )
            return (*input_grads, None)

        f_history = torch.empty(
            systems, chunk_size, 4, dtype=torch.float32, device=g.device
        )
        c1_history = torch.empty(
            systems, chunk_size, chunk_size,
            dtype=torch.float32, device=g.device,
        )
        c2_history = torch.empty_like(c1_history)
        _gdn_relative_history_kernel[(systems,)](
            g, gamma, lam, f_history, c1_history, c2_history,
            T=length, H=heads, N=chunks, C=chunk_size,
            num_warps=1,
        )
        _gdn_relative_bwd_kernel[(systems,)](
            g, beta, gamma, lam,
            f_history, c1_history, c2_history,
            *output_grads,
            *input_grads,
            T=length, H=heads, N=chunks, C=chunk_size,
            num_warps=1,
        )
        return (*input_grads, None)


def gdn_relative_coefficients_triton(g, beta, gamma, lam, chunk_size=16):
    return _GDNRelativeCoefficients.apply(g, beta, gamma, lam, chunk_size)


__all__ = ["gdn_relative_coefficients_triton"]

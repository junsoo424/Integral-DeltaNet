"""FLA-style operand builder for the integral DeltaNet specialization.

The scalar Delta recurrence has rank-two token coefficients around one raw
Q/K Gram.  This module keeps those coefficients small and builds the two
within-chunk score matrices directly from the original sequence layout.  It
then applies the solved triangular inverse to the value RHS and both state
RHSs in one Triton launch, mirroring FLA's WY recompute organization.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import safe_dot

from .solve import _inverse
from .tuning import by_arch


_CHUNK_SIZE = 16


# Pinned (BK, BV, warps, stages) per architecture: ops/tuning.
_PINNED_DELTA_SOLVE_CONFIGS = by_arch("DELTA_SOLVE")
_DELTA_SOLVE_FORWARD_CONFIGS = [
    triton.Config({"BK": bk, "BV": bv}, num_warps=warps, num_stages=stages)
    for bk, bv, warps, stages in (
        (64, 64, 1, 1),
        (32, 32, 1, 3),
        (64, 64, 2, 2),
        (64, 64, 2, 3),
        (64, 64, 2, 4),
        (64, 64, 4, 2),
        (64, 64, 4, 3),
        (64, 64, 4, 4),
        (64, 64, 8, 2),
        (64, 64, 8, 3),
        (64, 64, 8, 4),
    )
]
_DELTA_SOLVE_BACKWARD_CONFIGS = [
    triton.Config({"BK": bk, "BV": bv}, num_warps=warps, num_stages=stages)
    for bk, bv, warps, stages in (
        (64, 64, 1, 2),
        (32, 64, 2, 2),
        (32, 32, 4, 3),
        (64, 64, 4, 2),
        (64, 64, 4, 3),
        (64, 64, 8, 2),
        (64, 64, 8, 3),
    )
]


def _chunk_vector(tensor: torch.Tensor, chunk_size: int = _CHUNK_SIZE):
    batch, length, heads, width = tensor.shape
    return tensor.transpose(1, 2).reshape(
        batch, heads, length // chunk_size, chunk_size, width
    )


def _chunk_scalar(tensor: torch.Tensor, chunk_size: int = _CHUNK_SIZE):
    batch, length, heads = tensor.shape
    return tensor.transpose(1, 2).reshape(
        batch, heads, length // chunk_size, chunk_size
    )


def _unchunk_vector(tensor: torch.Tensor):
    batch, heads, chunks, chunk_size, width = tensor.shape
    return tensor.reshape(batch, heads, chunks * chunk_size, width).transpose(1, 2).contiguous()


def _unchunk_scalar(tensor: torch.Tensor):
    batch, heads, chunks, chunk_size = tensor.shape
    return tensor.reshape(batch, heads, chunks * chunk_size).transpose(1, 2).contiguous()


@triton.jit
def _delta_coefficients_fwd_kernel(
    BETA,
    GAMMA,
    LAM,
    PI1,
    PI2,
    SIGMA1,
    SIGMA2,
    ETA1,
    ETA2,
    TAU1,
    TAU2,
    QUERY_COEFFICIENTS,
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
    beta = tl.load(BETA + input_base + rows * H).to(tl.float32)
    gamma = tl.load(GAMMA + input_base + rows * H).to(tl.float32)

    g_previous = tl.zeros((C,), dtype=tl.float32)
    h_previous = tl.zeros((C,), dtype=tl.float32)
    current_g = 1.0
    current_h = 0.0
    for row in range(C):
        g_previous = tl.where(rows == row, current_g, g_previous)
        h_previous = tl.where(rows == row, current_h, h_previous)
        gamma_row = tl.load(GAMMA + input_base + row * H).to(tl.float32)
        lambda_row = tl.load(LAM + input_base + row * H).to(tl.float32)
        denominator_row = gamma_row * current_g
        current_h += lambda_row * denominator_row
        current_g = (1.0 - lambda_row) * denominator_row

    denominator = gamma * g_previous
    inverse_denominator = 1.0 / denominator
    pi2 = beta * (h_previous + g_previous)
    sigma1 = -h_previous * inverse_denominator
    sigma2 = inverse_denominator
    eta2 = h_previous + denominator
    tau1 = (current_h - h_previous) * inverse_denominator
    tau2 = current_g * inverse_denominator
    offsets = pid * C + rows
    tl.store(PI1 + offsets, beta)
    tl.store(PI2 + offsets, pi2)
    tl.store(SIGMA1 + offsets, sigma1)
    tl.store(SIGMA2 + offsets, sigma2)
    tl.store(ETA1 + offsets, 1.0)
    tl.store(ETA2 + offsets, eta2)
    tl.store(TAU1 + offsets, tau1)
    tl.store(TAU2 + offsets, tau2)
    query_offsets = pid * C * 2 + rows * 2
    tl.store(QUERY_COEFFICIENTS + query_offsets + 0, 1.0)
    tl.store(QUERY_COEFFICIENTS + query_offsets + 1, eta2)
    tl.store(TRANSPORT + pid * 4 + 0, 1.0)
    tl.store(TRANSPORT + pid * 4 + 1, 0.0)
    tl.store(TRANSPORT + pid * 4 + 2, current_h)
    tl.store(TRANSPORT + pid * 4 + 3, current_g)


@triton.jit
def _delta_coefficients_bwd_kernel(
    BETA,
    GAMMA,
    LAM,
    DPI1,
    DPI2,
    DSIGMA1,
    DSIGMA2,
    DETA1,
    DETA2,
    DTAU1,
    DTAU2,
    DQUERY_COEFFICIENTS,
    DTRANSPORT,
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
    beta = tl.load(BETA + input_base + rows * H).to(tl.float32)
    gamma = tl.load(GAMMA + input_base + rows * H).to(tl.float32)
    lam = tl.load(LAM + input_base + rows * H).to(tl.float32)

    g_previous = tl.zeros((C,), dtype=tl.float32)
    h_previous = tl.zeros((C,), dtype=tl.float32)
    current_g = 1.0
    current_h = 0.0
    for row in range(C):
        g_previous = tl.where(rows == row, current_g, g_previous)
        h_previous = tl.where(rows == row, current_h, h_previous)
        gamma_row = tl.load(GAMMA + input_base + row * H).to(tl.float32)
        lambda_row = tl.load(LAM + input_base + row * H).to(tl.float32)
        denominator_row = gamma_row * current_g
        current_h += lambda_row * denominator_row
        current_g = (1.0 - lambda_row) * denominator_row

    offsets = pid * C + rows
    dpi1 = tl.load(DPI1 + offsets).to(tl.float32)
    dpi2 = tl.load(DPI2 + offsets).to(tl.float32)
    dsigma1 = tl.load(DSIGMA1 + offsets).to(tl.float32)
    dsigma2 = tl.load(DSIGMA2 + offsets).to(tl.float32)
    # ETA1 is constant; load its gradient only to keep the custom-function
    # output contract explicit while deliberately contributing zero.
    _ = tl.load(DETA1 + offsets)
    deta2 = tl.load(DETA2 + offsets).to(tl.float32)
    dtau1 = tl.load(DTAU1 + offsets).to(tl.float32)
    dtau2 = tl.load(DTAU2 + offsets).to(tl.float32)
    query_offsets = pid * C * 2 + rows * 2
    _ = tl.load(DQUERY_COEFFICIENTS + query_offsets + 0)
    deta2 += tl.load(DQUERY_COEFFICIENTS + query_offsets + 1).to(tl.float32)
    dtransport_h = tl.load(DTRANSPORT + pid * 4 + 2).to(tl.float32)
    dtransport_g = tl.load(DTRANSPORT + pid * 4 + 3).to(tl.float32)

    denominator = gamma * g_previous
    inverse_denominator = 1.0 / denominator
    inverse_denominator2 = inverse_denominator * inverse_denominator
    grad_beta = dpi1 + dpi2 * (h_previous + g_previous)
    grad_h_direct = (
        dpi2 * beta
        - dsigma1 * inverse_denominator
        + deta2
        - dtau1 * inverse_denominator
    )
    grad_denominator = (
        -dsigma2 * inverse_denominator2
        + dsigma1 * h_previous * inverse_denominator2
        + deta2
        - dtau1
        * (current_h - h_previous)
        * inverse_denominator2
        - dtau2 * current_g * inverse_denominator2
    )
    grad_g_direct = dpi2 * beta + grad_denominator * gamma
    grad_gamma_direct = grad_denominator * g_previous
    grad_final_h = dtransport_h + tl.sum(dtau1 * inverse_denominator, axis=0)
    grad_final_g = dtransport_g + tl.sum(dtau2 * inverse_denominator, axis=0)

    grad_gamma = tl.zeros((C,), dtype=tl.float32)
    grad_lam = tl.zeros((C,), dtype=tl.float32)
    adjoint_g = grad_final_g
    adjoint_h = grad_final_h
    for reverse_index in range(C):
        row = C - 1 - reverse_index
        gp = tl.sum(tl.where(rows == row, g_previous, 0.0), axis=0)
        gamma_row = tl.sum(tl.where(rows == row, gamma, 0.0), axis=0)
        lambda_row = tl.sum(tl.where(rows == row, lam, 0.0), axis=0)
        denominator_row = gamma_row * gp
        grad_denominator_recurrence = (
            adjoint_g * (1.0 - lambda_row) + adjoint_h * lambda_row
        )
        direct_gamma_row = tl.sum(
            tl.where(rows == row, grad_gamma_direct, 0.0), axis=0
        )
        gamma_gradient = direct_gamma_row + grad_denominator_recurrence * gp
        lambda_gradient = (adjoint_h - adjoint_g) * denominator_row
        grad_gamma = tl.where(rows == row, gamma_gradient, grad_gamma)
        grad_lam = tl.where(rows == row, lambda_gradient, grad_lam)
        direct_g_row = tl.sum(
            tl.where(rows == row, grad_g_direct, 0.0), axis=0
        )
        direct_h_row = tl.sum(
            tl.where(rows == row, grad_h_direct, 0.0), axis=0
        )
        adjoint_g = direct_g_row + grad_denominator_recurrence * gamma_row
        adjoint_h = direct_h_row + adjoint_h

    output_offsets = input_base + rows * H
    tl.store(DBETA + output_offsets, grad_beta)
    tl.store(DGAMMA + output_offsets, grad_gamma)
    tl.store(DLAM + output_offsets, grad_lam)


class _DeltaCoefficients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, beta, gamma, lam, chunk_size):
        batch, length, heads = beta.shape
        if chunk_size not in (16, 32, 64) or length % chunk_size:
            raise ValueError("length must be divisible by a 16/32/64 chunk")
        chunks = length // chunk_size
        beta, gamma, lam = (
            tensor.contiguous() for tensor in (beta, gamma, lam)
        )
        shape = (batch, heads, chunks, chunk_size)
        coefficient_tensors = tuple(
            torch.empty(*shape, dtype=torch.float32, device=beta.device)
            for _ in range(8)
        )
        query_coefficients = torch.empty(
            *shape, 1, 2, dtype=torch.float32, device=beta.device
        )
        transport = torch.empty(
            batch, heads, chunks, 4, dtype=torch.float32, device=beta.device
        )
        _delta_coefficients_fwd_kernel[(batch * heads * chunks,)](
            beta,
            gamma,
            lam,
            *coefficient_tensors,
            query_coefficients,
            transport,
            T=length,
            H=heads,
            N=chunks,
            C=chunk_size,
            num_warps=1,
        )
        ctx.save_for_backward(beta, gamma, lam)
        ctx.dimensions = (batch, length, heads, chunks, chunk_size)
        return (*coefficient_tensors, query_coefficients, transport)

    @staticmethod
    def backward(ctx, *output_grads):
        beta, gamma, lam = ctx.saved_tensors
        batch, length, heads, chunks, chunk_size = ctx.dimensions
        coefficient_shape = (batch, heads, chunks, chunk_size)
        output_shapes = (
            *(coefficient_shape for _ in range(8)),
            (*coefficient_shape, 1, 2),
            (batch, heads, chunks, 4),
        )
        output_grads = tuple(
            grad.contiguous()
            if grad is not None
            else torch.zeros(shape, dtype=torch.float32, device=beta.device)
            for grad, shape in zip(output_grads, output_shapes)
        )
        grad_beta = torch.empty_like(beta)
        grad_gamma = torch.empty_like(gamma)
        grad_lam = torch.empty_like(lam)
        _delta_coefficients_bwd_kernel[(batch * heads * chunks,)](
            beta,
            gamma,
            lam,
            *output_grads,
            grad_beta,
            grad_gamma,
            grad_lam,
            T=length,
            H=heads,
            N=chunks,
            C=chunk_size,
            num_warps=1,
        )
        return grad_beta, grad_gamma, grad_lam, None


def _delta_coefficients(beta, gamma, lam):
    """Small scalar tables for the [slow, active-slow] state basis."""
    beta, gamma, lam = (tensor.float() for tensor in (beta, gamma, lam))
    g_prefix = ((1.0 - lam) * gamma).cumprod(dim=-1)
    one = torch.ones_like(g_prefix[..., :1])
    g_previous = torch.cat((one, g_prefix[..., :-1]), dim=-1)
    h_prefix = (g_previous * lam * gamma).cumsum(dim=-1)
    zero = torch.zeros_like(h_prefix[..., :1])
    h_previous = torch.cat((zero, h_prefix[..., :-1]), dim=-1)

    pi1 = beta
    pi2 = beta * (h_previous + g_previous)
    # Since g_i=(1-lambda_i)*gamma_i*g_{i-1} and
    # h_i=h_{i-1}+lambda_i*gamma_i*g_{i-1}, all inverse-prefix
    # coefficients share one denominator.  These reduced expressions remove
    # several full coefficient-tensor operations from both forward/backward.
    denominator = (gamma * g_previous).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    sigma2 = denominator.reciprocal()
    sigma1 = -h_previous * sigma2
    eta1 = torch.ones_like(h_prefix)
    eta2 = h_previous + denominator
    final_h, final_g = h_prefix[..., -1], g_prefix[..., -1]
    tau1 = (final_h.unsqueeze(-1) - h_previous) * sigma2
    tau2 = final_g.unsqueeze(-1) * sigma2
    query_coefficients = torch.stack((eta1, eta2), dim=-1).unsqueeze(-2)
    transport = torch.stack(
        (
            torch.ones_like(final_h),
            torch.zeros_like(final_h),
            final_h,
            final_g,
        ),
        dim=-1,
    )
    return (
        pi1,
        pi2,
        sigma1,
        sigma2,
        eta1,
        eta2,
        tau1,
        tau2,
        query_coefficients,
        transport,
    )


_COMPILED_COEFFICIENTS = None


def _delta_coefficients_fast(*args):
    global _COMPILED_COEFFICIENTS
    if _COMPILED_COEFFICIENTS is None:
        try:
            _COMPILED_COEFFICIENTS = torch.compile(
                _delta_coefficients,
                dynamic=False,
                mode="max-autotune-no-cudagraphs",
            )
        except Exception:  # pragma: no cover - environment-specific fallback
            _COMPILED_COEFFICIENTS = _delta_coefficients
    try:
        return _COMPILED_COEFFICIENTS(*args)
    except Exception:  # pragma: no cover - safe eager fallback
        _COMPILED_COEFFICIENTS = _delta_coefficients
        return _delta_coefficients(*args)


def _prefix_2x2_scalar(p11, p12, p21, p22):
    """Inclusive noncommutative prefix for head-shared 2x2 transitions."""
    size = p11.shape[-1]
    f11, f12, f21, f22 = p11, p12, p21, p22
    offset = 1
    while offset < size:
        one = torch.ones_like(f11[..., :offset])
        zero = torch.zeros_like(one)
        l11 = torch.cat((one, f11[..., :-offset]), dim=-1)
        l12 = torch.cat((zero, f12[..., :-offset]), dim=-1)
        l21 = torch.cat((zero, f21[..., :-offset]), dim=-1)
        l22 = torch.cat((one, f22[..., :-offset]), dim=-1)
        f11, f12, f21, f22 = (
            l11 * f11 + l12 * f21,
            l11 * f12 + l12 * f22,
            l21 * f11 + l22 * f21,
            l21 * f12 + l22 * f22,
        )
        offset *= 2
    return f11, f12, f21, f22


def _gdn_coefficients(g, beta, gamma, lam):
    g, beta, gamma, lam = (
        tensor.float() for tensor in (g, beta, gamma, lam)
    )
    decay = g.exp()
    p11 = gamma * decay
    p12 = lam * p11
    p21 = (1.0 - gamma) * decay
    p22 = (1.0 - lam) + lam * p21
    f11, f12, f21, f22 = _prefix_2x2_scalar(p11, p12, p21, p22)

    one = torch.ones_like(f11[..., :1])
    zero = torch.zeros_like(one)
    previous11 = torch.cat((one, f11[..., :-1]), dim=-1)
    previous21 = torch.cat((zero, f21[..., :-1]), dim=-1)
    pi1 = previous11 * decay * beta
    pi2 = previous21 * decay * beta
    determinant = (gamma * decay * (1.0 - lam)).cumprod(dim=-1)
    determinant = determinant.clamp_min(torch.finfo(torch.float32).tiny)
    sigma1 = (f22 - lam * f21) / determinant
    sigma2 = (-f12 + lam * f11) / determinant
    eta1, eta2 = f11, f21

    final11, final12 = f11[..., -1], f12[..., -1]
    final21, final22 = f21[..., -1], f22[..., -1]
    tau1 = sigma1 * final11.unsqueeze(-1) + sigma2 * final21.unsqueeze(-1)
    tau2 = sigma1 * final12.unsqueeze(-1) + sigma2 * final22.unsqueeze(-1)
    query_coefficients = torch.stack((eta1, eta2), dim=-1).unsqueeze(-2)
    transport = torch.stack(
        (final11, final12, final21, final22), dim=-1
    )
    return (
        pi1,
        pi2,
        sigma1,
        sigma2,
        eta1,
        eta2,
        tau1,
        tau2,
        query_coefficients,
        transport,
    )


_COMPILED_GDN_COEFFICIENTS = None


def _gdn_coefficients_fast(*args):
    global _COMPILED_GDN_COEFFICIENTS
    if _COMPILED_GDN_COEFFICIENTS is None:
        try:
            _COMPILED_GDN_COEFFICIENTS = torch.compile(
                _gdn_coefficients,
                dynamic=False,
                mode="max-autotune-no-cudagraphs",
            )
        except Exception:  # pragma: no cover - environment-specific fallback
            _COMPILED_GDN_COEFFICIENTS = _gdn_coefficients
    try:
        return _COMPILED_GDN_COEFFICIENTS(*args)
    except Exception:  # pragma: no cover - safe eager fallback
        _COMPILED_GDN_COEFFICIENTS = _gdn_coefficients
        return _gdn_coefficients(*args)


@triton.autotune(
    configs=[
        triton.Config({"BK": block}, num_warps=warps, num_stages=stages)
        for block, warps in ((32, 2), (64, 4), (128, 8))
        for stages in (2, 3, 4)
    ],
    key=["K", "C"],
)
@triton.jit
def _delta_intra_fwd_kernel(
    Q,
    K_IN,
    PI1,
    PI2,
    SIGMA1,
    SIGMA2,
    ETA1,
    ETA2,
    TAU1,
    TAU2,
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
    SCALE: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk = pid % N
    head = (pid // N) % H
    batch = pid // (N * H)
    rows = tl.arange(0, C)

    pi1 = tl.load(PI1 + pid * C + rows).to(tl.float32)
    pi2 = tl.load(PI2 + pid * C + rows).to(tl.float32)
    sigma1 = tl.load(SIGMA1 + pid * C + rows).to(tl.float32)
    sigma2 = tl.load(SIGMA2 + pid * C + rows).to(tl.float32)
    eta1 = tl.load(ETA1 + pid * C + rows).to(tl.float32)
    eta2 = tl.load(ETA2 + pid * C + rows).to(tl.float32)
    tau1 = tl.load(TAU1 + pid * C + rows).to(tl.float32)
    tau2 = tl.load(TAU2 + pid * C + rows).to(tl.float32)

    gram_kk = tl.zeros((C, C), dtype=tl.float32)
    gram_qk = tl.zeros((C, C), dtype=tl.float32)
    sequence_base = ((batch * T + chunk * C) * H + head) * K
    output_base = pid * C * K
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        offsets = sequence_base + rows[:, None] * H * K + keys[None, :]
        mask = key_mask[None, :]
        key = tl.load(K_IN + offsets, mask=mask, other=0.0)
        query = tl.load(Q + offsets, mask=mask, other=0.0)
        query_scaled = (query * SCALE).to(tl.bfloat16)
        gram_kk += safe_dot(key, tl.trans(key), allow_tf32=False)
        gram_qk += safe_dot(query_scaled, tl.trans(key), allow_tf32=False)

        output_offsets = output_base + rows[:, None] * K + keys[None, :]
        tl.store(Q_OUT + output_offsets, query_scaled, mask=mask)
        tl.store(K1 + output_offsets, tau1[:, None] * key, mask=mask)
        tl.store(K2 + output_offsets, tau2[:, None] * key, mask=mask)

    interaction_coeff = (
        pi1[:, None] * sigma1[None, :]
        + pi2[:, None] * sigma2[None, :]
    )
    output_coeff = (
        eta1[:, None] * sigma1[None, :]
        + eta2[:, None] * sigma2[None, :]
    )
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    matrix = tl.where(row_ids > col_ids, interaction_coeff * gram_kk, 0.0)
    causal = tl.where(row_ids >= col_ids, output_coeff * gram_qk, 0.0)
    matrix_offsets = pid * C * C + row_ids * C + col_ids
    tl.store(MATRIX + matrix_offsets, matrix)
    tl.store(CAUSAL + matrix_offsets, causal)


@triton.autotune(
    configs=[
        triton.Config({"BK": block}, num_warps=warps, num_stages=stages)
        for block, warps in ((32, 4), (64, 4), (128, 8))
        for stages in (2, 3)
    ],
    key=["K", "C"],
)
@triton.jit
def _delta_intra_bwd_kernel(
    Q_OUT,
    K_IN,
    PI1,
    PI2,
    SIGMA1,
    SIGMA2,
    ETA1,
    ETA2,
    TAU1,
    TAU2,
    DMATRIX,
    DCAUSAL,
    DQ_DIRECT,
    DK1,
    DK2,
    DQ_OUT,
    DK_OUT,
    DPI1,
    DPI2,
    DSIGMA1,
    DSIGMA2,
    DETA1,
    DETA2,
    DTAU1,
    DTAU2,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
    BK: tl.constexpr,
):
    """Differentiate both Gram paths and transported vectors in one launch."""
    pid = tl.program_id(0)
    chunk = pid % N
    head = (pid // N) % H
    batch = pid // (N * H)
    rows = tl.arange(0, C)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    matrix_offsets = pid * C * C + row_ids * C + col_ids
    grad_matrix = tl.load(DMATRIX + matrix_offsets).to(tl.float32)
    grad_causal = tl.load(DCAUSAL + matrix_offsets).to(tl.float32)
    grad_matrix = tl.where(row_ids > col_ids, grad_matrix, 0.0)
    grad_causal = tl.where(row_ids >= col_ids, grad_causal, 0.0)

    pi1 = tl.load(PI1 + pid * C + rows).to(tl.float32)
    pi2 = tl.load(PI2 + pid * C + rows).to(tl.float32)
    sigma1 = tl.load(SIGMA1 + pid * C + rows).to(tl.float32)
    sigma2 = tl.load(SIGMA2 + pid * C + rows).to(tl.float32)
    eta1 = tl.load(ETA1 + pid * C + rows).to(tl.float32)
    eta2 = tl.load(ETA2 + pid * C + rows).to(tl.float32)
    tau1 = tl.load(TAU1 + pid * C + rows).to(tl.float32)
    tau2 = tl.load(TAU2 + pid * C + rows).to(tl.float32)
    interaction_coeff = (
        pi1[:, None] * sigma1[None, :]
        + pi2[:, None] * sigma2[None, :]
    )
    output_coeff = (
        eta1[:, None] * sigma1[None, :]
        + eta2[:, None] * sigma2[None, :]
    )
    grad_gram_kk = (grad_matrix * interaction_coeff).to(tl.bfloat16)
    grad_gram_qk = (grad_causal * output_coeff).to(tl.bfloat16)
    grad_gram_kk_symmetric = (
        grad_gram_kk.to(tl.float32) + tl.trans(grad_gram_kk).to(tl.float32)
    ).to(tl.bfloat16)
    gram_kk = tl.zeros((C, C), dtype=tl.float32)
    gram_qk = tl.zeros((C, C), dtype=tl.float32)
    grad_tau1 = tl.zeros((C,), dtype=tl.float32)
    grad_tau2 = tl.zeros((C,), dtype=tl.float32)

    sequence_base = ((batch * T + chunk * C) * H + head) * K
    chunk_base = pid * C * K
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        sequence_offsets = (
            sequence_base + rows[:, None] * H * K + keys[None, :]
        )
        chunk_offsets = chunk_base + rows[:, None] * K + keys[None, :]
        key = tl.load(K_IN + sequence_offsets, mask=key_mask[None, :], other=0.0)
        query = tl.load(Q_OUT + chunk_offsets, mask=key_mask[None, :], other=0.0)
        grad_query_direct = tl.load(
            DQ_DIRECT + chunk_offsets, mask=key_mask[None, :], other=0.0
        )
        grad_k1 = tl.load(DK1 + chunk_offsets, mask=key_mask[None, :], other=0.0)
        grad_k2 = tl.load(DK2 + chunk_offsets, mask=key_mask[None, :], other=0.0)
        gram_kk += safe_dot(key, tl.trans(key), allow_tf32=False)
        gram_qk += safe_dot(query, tl.trans(key), allow_tf32=False)
        grad_query = safe_dot(grad_gram_qk, key, allow_tf32=False)
        grad_query = (grad_query + grad_query_direct) * SCALE
        grad_key = safe_dot(grad_gram_kk_symmetric, key, allow_tf32=False)
        grad_key += safe_dot(tl.trans(grad_gram_qk), query, allow_tf32=False)
        grad_key += tau1[:, None] * grad_k1 + tau2[:, None] * grad_k2
        tl.store(
            DQ_OUT + sequence_offsets, grad_query, mask=key_mask[None, :]
        )
        tl.store(DK_OUT + sequence_offsets, grad_key, mask=key_mask[None, :])
        grad_tau1 += tl.sum(
            grad_k1.to(tl.float32) * key.to(tl.float32), axis=1
        )
        grad_tau2 += tl.sum(
            grad_k2.to(tl.float32) * key.to(tl.float32), axis=1
        )

    grad_interaction = grad_matrix * gram_kk
    grad_output = grad_causal * gram_qk
    grad_pi1 = tl.sum(grad_interaction * sigma1[None, :], axis=1)
    grad_pi2 = tl.sum(grad_interaction * sigma2[None, :], axis=1)
    grad_sigma1 = tl.sum(
        grad_interaction * pi1[:, None] + grad_output * eta1[:, None],
        axis=0,
    )
    grad_sigma2 = tl.sum(
        grad_interaction * pi2[:, None] + grad_output * eta2[:, None],
        axis=0,
    )
    grad_eta1 = tl.sum(grad_output * sigma1[None, :], axis=1)
    grad_eta2 = tl.sum(grad_output * sigma2[None, :], axis=1)
    coefficient_offsets = pid * C + rows
    tl.store(DPI1 + coefficient_offsets, grad_pi1)
    tl.store(DPI2 + coefficient_offsets, grad_pi2)
    tl.store(DSIGMA1 + coefficient_offsets, grad_sigma1)
    tl.store(DSIGMA2 + coefficient_offsets, grad_sigma2)
    tl.store(DETA1 + coefficient_offsets, grad_eta1)
    tl.store(DETA2 + coefficient_offsets, grad_eta2)
    tl.store(DTAU1 + coefficient_offsets, grad_tau1)
    tl.store(DTAU2 + coefficient_offsets, grad_tau2)


class _DeltaIntra(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        pi1,
        pi2,
        sigma1,
        sigma2,
        eta1,
        eta2,
        tau1,
        tau2,
        scale,
    ):
        batch, length, heads, key_dim = q.shape
        chunk_size = pi1.shape[-1]
        chunks = length // chunk_size
        coefficient_tensors = tuple(
            tensor.contiguous()
            for tensor in (pi1, pi2, sigma1, sigma2, eta1, eta2, tau1, tau2)
        )
        pi1, pi2, sigma1, sigma2, eta1, eta2, tau1, tau2 = coefficient_tensors
        q = q.contiguous()
        k = k.contiguous()
        shape = (batch, heads, chunks, chunk_size)
        matrix = torch.empty(*shape, chunk_size, device=q.device, dtype=torch.float32)
        causal = torch.empty(*shape, chunk_size, device=q.device, dtype=q.dtype)
        q_out = torch.empty(*shape, key_dim, device=q.device, dtype=q.dtype)
        k1 = torch.empty_like(q_out)
        k2 = torch.empty_like(q_out)
        _delta_intra_fwd_kernel[(batch * heads * chunks,)](
            q,
            k,
            pi1,
            pi2,
            sigma1,
            sigma2,
            eta1,
            eta2,
            tau1,
            tau2,
            matrix,
            causal,
            q_out,
            k1,
            k2,
            T=length,
            H=heads,
            N=chunks,
            K=key_dim,
            C=chunk_size,
            SCALE=float(scale),
        )
        ctx.save_for_backward(
            q, k, q_out, pi1, pi2, sigma1, sigma2, eta1, eta2, tau1, tau2
        )
        ctx.scale = float(scale)
        ctx.input_dtypes = (
            q.dtype,
            k.dtype,
            *(tensor.dtype for tensor in coefficient_tensors),
        )
        return matrix, causal, q_out, k1, k2

    @staticmethod
    def backward(ctx, grad_matrix, grad_causal, grad_q_out, grad_k1, grad_k2):
        (
            q,
            k,
            q_out,
            pi1,
            pi2,
            sigma1,
            sigma2,
            eta1,
            eta2,
            tau1,
            tau2,
        ) = ctx.saved_tensors
        chunk_size = pi1.shape[-1]
        if chunk_size in (16, 32) and q.is_cuda:
            batch, length, heads, key_dim = q.shape
            chunks = length // chunk_size
            grad_q = torch.empty_like(q)
            grad_k = torch.empty_like(k)
            coefficient_grads = tuple(
                torch.empty_like(tensor)
                for tensor in (
                    pi1,
                    pi2,
                    sigma1,
                    sigma2,
                    eta1,
                    eta2,
                    tau1,
                    tau2,
                )
            )
            _delta_intra_bwd_kernel[(batch * heads * chunks,)](
                q_out,
                k,
                pi1,
                pi2,
                sigma1,
                sigma2,
                eta1,
                eta2,
                tau1,
                tau2,
                grad_matrix.contiguous(),
                grad_causal.contiguous(),
                grad_q_out.contiguous(),
                grad_k1.contiguous(),
                grad_k2.contiguous(),
                grad_q,
                grad_k,
                *coefficient_grads,
                T=length,
                H=heads,
                N=chunks,
                K=key_dim,
                C=chunk_size,
                SCALE=ctx.scale,
            )
            return (grad_q, grad_k, *coefficient_grads, None)
        qc = _chunk_vector(q, chunk_size)
        kc = _chunk_vector(k, chunk_size)
        grad_matrix = torch.tril(grad_matrix.float(), diagonal=-1)
        grad_causal = torch.tril(grad_causal.float())
        gram_kk = kc.float() @ kc.float().transpose(-1, -2)
        gram_qk = q_out.float() @ kc.float().transpose(-1, -2)
        interaction_coeff = (
            pi1.unsqueeze(-1) * sigma1.unsqueeze(-2)
            + pi2.unsqueeze(-1) * sigma2.unsqueeze(-2)
        )
        output_coeff = (
            eta1.unsqueeze(-1) * sigma1.unsqueeze(-2)
            + eta2.unsqueeze(-1) * sigma2.unsqueeze(-2)
        )
        grad_gram_kk = grad_matrix * interaction_coeff
        grad_gram_qk = grad_causal * output_coeff

        grad_qc = (grad_gram_qk @ kc.float()) * ctx.scale
        grad_kc = (
            grad_gram_kk @ kc.float()
            + grad_gram_kk.transpose(-1, -2) @ kc.float()
            + grad_gram_qk.transpose(-1, -2) @ q_out.float()
        )
        grad_qc += grad_q_out.float() * ctx.scale
        grad_kc += (
            grad_k1.float() * tau1.unsqueeze(-1)
            + grad_k2.float() * tau2.unsqueeze(-1)
        )

        grad_interaction_coeff = grad_matrix * gram_kk
        grad_output_coeff = grad_causal * gram_qk
        grad_pi1 = (grad_interaction_coeff * sigma1.unsqueeze(-2)).sum(-1)
        grad_pi2 = (grad_interaction_coeff * sigma2.unsqueeze(-2)).sum(-1)
        grad_sigma1 = (
            grad_interaction_coeff * pi1.unsqueeze(-1)
            + grad_output_coeff * eta1.unsqueeze(-1)
        ).sum(-2)
        grad_sigma2 = (
            grad_interaction_coeff * pi2.unsqueeze(-1)
            + grad_output_coeff * eta2.unsqueeze(-1)
        ).sum(-2)
        grad_eta1 = (grad_output_coeff * sigma1.unsqueeze(-2)).sum(-1)
        grad_eta2 = (grad_output_coeff * sigma2.unsqueeze(-2)).sum(-1)
        grad_tau1 = (grad_k1.float() * kc.float()).sum(-1)
        grad_tau2 = (grad_k2.float() * kc.float()).sum(-1)
        grads = (
            _unchunk_vector(grad_qc),
            _unchunk_vector(grad_kc),
            grad_pi1,
            grad_pi2,
            grad_sigma1,
            grad_sigma2,
            grad_eta1,
            grad_eta2,
            grad_tau1,
            grad_tau2,
        )
        return tuple(
            grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)
        ) + (None,)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (2, 4, 8)
        for stages in (2, 3, 4)
    ],
    key=["K", "V", "C"],
)
@triton.jit
def _delta_recompute_fwd_kernel(
    INVERSE,
    K_IN,
    V_IN,
    BETA,
    PI1,
    PI2,
    U0,
    W1,
    W2,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk = pid % N
    head = (pid // N) % H
    batch = pid // (N * H)
    rows = tl.arange(0, C)
    # FLA's solve writes BF16 for tensor-core consumers.  Keep the explicit
    # conversion here as well: the local fallback can return FP32, and Triton
    # requires both ``safe_dot`` operands to have the same dtype.
    inverse = tl.load(
        INVERSE + pid * C * C + rows[:, None] * C + rows[None, :]
    ).to(tl.bfloat16)
    beta_base = (batch * T + chunk * C) * H + head
    beta = tl.load(BETA + beta_base + rows * H).to(tl.float32)
    pi1 = tl.load(PI1 + pid * C + rows).to(tl.float32)
    pi2 = tl.load(PI2 + pid * C + rows).to(tl.float32)

    value_base = ((batch * T + chunk * C) * H + head) * V
    value_output_base = pid * C * V
    for value_block in range(tl.cdiv(V, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < V
        input_offsets = value_base + rows[:, None] * H * V + values[None, :]
        value = tl.load(V_IN + input_offsets, mask=value_mask[None, :], other=0.0)
        rhs = (beta[:, None] * value).to(tl.bfloat16)
        solution = safe_dot(inverse, rhs, allow_tf32=False)
        output_offsets = value_output_base + rows[:, None] * V + values[None, :]
        tl.store(U0 + output_offsets, solution, mask=value_mask[None, :])

    key_base = ((batch * T + chunk * C) * H + head) * K
    key_output_base = pid * C * K
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        input_offsets = key_base + rows[:, None] * H * K + keys[None, :]
        key = tl.load(K_IN + input_offsets, mask=key_mask[None, :], other=0.0)
        rhs1 = (pi1[:, None] * key).to(tl.bfloat16)
        rhs2 = (pi2[:, None] * key).to(tl.bfloat16)
        solution1 = safe_dot(inverse, rhs1, allow_tf32=False)
        solution2 = safe_dot(inverse, rhs2, allow_tf32=False)
        output_offsets = key_output_base + rows[:, None] * K + keys[None, :]
        tl.store(W1 + output_offsets, solution1, mask=key_mask[None, :])
        tl.store(W2 + output_offsets, solution2, mask=key_mask[None, :])


@triton.autotune(
    configs=_DELTA_SOLVE_FORWARD_CONFIGS,
    key=["K", "V", "C"],
)
@triton.jit
def _delta_solve_recompute_fwd_kernel(
    MATRIX,
    K_IN,
    V_IN,
    BETA,
    PI1,
    PI2,
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
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Invert one supported unit-lower system and apply all three RHS groups."""
    pid = tl.program_id(0)
    chunk = pid % N
    head = (pid // N) % H
    batch = pid // (N * H)
    rows = tl.arange(0, C)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    matrix_offsets = pid * C * C + row_ids * C + col_ids
    matrix = tl.load(MATRIX + matrix_offsets).to(tl.float32)
    lower = row_ids > col_ids
    inverse = -tl.where(lower, matrix, 0.0)
    for row in range(2, C):
        matrix_row = tl.load(MATRIX + pid * C * C + row * C + rows)
        current = -tl.where(rows < row, matrix_row, 0.0)
        current += tl.sum(current[:, None] * inverse, axis=0)
        inverse = tl.where(row_ids == row, current[None, :], inverse)
    inverse += row_ids == col_ids
    inverse_bf16 = inverse.to(tl.bfloat16)
    tl.store(INVERSE + matrix_offsets, inverse_bf16)

    beta_base = (batch * T + chunk * C) * H + head
    beta = tl.load(BETA + beta_base + rows * H).to(tl.float32)
    pi1 = tl.load(PI1 + pid * C + rows).to(tl.float32)
    pi2 = tl.load(PI2 + pid * C + rows).to(tl.float32)

    value_base = ((batch * T + chunk * C) * H + head) * V
    value_output_base = pid * C * V
    for value_block in range(tl.cdiv(V, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < V
        input_offsets = value_base + rows[:, None] * H * V + values[None, :]
        value = tl.load(V_IN + input_offsets, mask=value_mask[None, :], other=0.0)
        rhs = (beta[:, None] * value).to(tl.bfloat16)
        solution = safe_dot(inverse_bf16, rhs, allow_tf32=False)
        output_offsets = value_output_base + rows[:, None] * V + values[None, :]
        tl.store(U0 + output_offsets, solution, mask=value_mask[None, :])

    key_base = ((batch * T + chunk * C) * H + head) * K
    key_output_base = pid * C * K
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        input_offsets = key_base + rows[:, None] * H * K + keys[None, :]
        key = tl.load(K_IN + input_offsets, mask=key_mask[None, :], other=0.0)
        rhs1 = (pi1[:, None] * key).to(tl.bfloat16)
        rhs2 = (pi2[:, None] * key).to(tl.bfloat16)
        solution1 = safe_dot(inverse_bf16, rhs1, allow_tf32=False)
        solution2 = safe_dot(inverse_bf16, rhs2, allow_tf32=False)
        output_offsets = key_output_base + rows[:, None] * K + keys[None, :]
        tl.store(W1 + output_offsets, solution1, mask=key_mask[None, :])
        tl.store(W2 + output_offsets, solution2, mask=key_mask[None, :])


@triton.autotune(
    configs=_DELTA_SOLVE_BACKWARD_CONFIGS,
    key=["K", "V", "C"],
)
@triton.jit
def _delta_solve_bwd_kernel(
    INVERSE,
    K_IN,
    V_IN,
    BETA,
    PI1,
    PI2,
    U0,
    W1,
    W2,
    DU0,
    DW1,
    DW2,
    DMATRIX,
    DK_OUT,
    DV_OUT,
    DBETA,
    DPI1,
    DPI2,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Differentiate all solved RHSs in one launch."""
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
    grad_beta = tl.zeros((C,), dtype=tl.float32)
    grad_pi1 = tl.zeros((C,), dtype=tl.float32)
    grad_pi2 = tl.zeros((C,), dtype=tl.float32)

    beta_base = (batch * T + chunk * C) * H + head
    beta = tl.load(BETA + beta_base + rows * H).to(tl.float32)
    pi1 = tl.load(PI1 + pid * C + rows).to(tl.float32)
    pi2 = tl.load(PI2 + pid * C + rows).to(tl.float32)

    value_base = ((batch * T + chunk * C) * H + head) * V
    chunk_value_base = pid * C * V
    for value_block in range(tl.cdiv(V, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < V
        chunk_offsets = chunk_value_base + rows[:, None] * V + values[None, :]
        grad_solution = tl.load(
            DU0 + chunk_offsets, mask=value_mask[None, :], other=0.0
        )
        grad_rhs = safe_dot(inverse_t, grad_solution, allow_tf32=False)
        solution = tl.load(U0 + chunk_offsets, mask=value_mask[None, :], other=0.0)
        grad_matrix += safe_dot(
            grad_rhs.to(tl.bfloat16), tl.trans(solution), allow_tf32=False
        )
        input_offsets = value_base + rows[:, None] * H * V + values[None, :]
        value = tl.load(V_IN + input_offsets, mask=value_mask[None, :], other=0.0)
        tl.store(
            DV_OUT + input_offsets,
            grad_rhs * beta[:, None],
            mask=value_mask[None, :],
        )
        grad_beta += tl.sum(grad_rhs.to(tl.float32) * value.to(tl.float32), axis=1)

    key_base = ((batch * T + chunk * C) * H + head) * K
    chunk_key_base = pid * C * K
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        chunk_offsets = chunk_key_base + rows[:, None] * K + keys[None, :]
        grad_solution1 = tl.load(
            DW1 + chunk_offsets, mask=key_mask[None, :], other=0.0
        )
        grad_solution2 = tl.load(
            DW2 + chunk_offsets, mask=key_mask[None, :], other=0.0
        )
        grad_rhs1 = safe_dot(inverse_t, grad_solution1, allow_tf32=False)
        grad_rhs2 = safe_dot(inverse_t, grad_solution2, allow_tf32=False)
        solution1 = tl.load(W1 + chunk_offsets, mask=key_mask[None, :], other=0.0)
        solution2 = tl.load(W2 + chunk_offsets, mask=key_mask[None, :], other=0.0)
        grad_matrix += safe_dot(
            grad_rhs1.to(tl.bfloat16), tl.trans(solution1), allow_tf32=False
        )
        grad_matrix += safe_dot(
            grad_rhs2.to(tl.bfloat16), tl.trans(solution2), allow_tf32=False
        )
        input_offsets = key_base + rows[:, None] * H * K + keys[None, :]
        key = tl.load(K_IN + input_offsets, mask=key_mask[None, :], other=0.0)
        tl.store(
            DK_OUT + input_offsets,
            pi1[:, None] * grad_rhs1 + pi2[:, None] * grad_rhs2,
            mask=key_mask[None, :],
        )
        grad_pi1 += tl.sum(grad_rhs1.to(tl.float32) * key.to(tl.float32), axis=1)
        grad_pi2 += tl.sum(grad_rhs2.to(tl.float32) * key.to(tl.float32), axis=1)

    grad_matrix = tl.where(row_ids > col_ids, -grad_matrix, 0.0)
    tl.store(DMATRIX + matrix_offsets, grad_matrix)
    tl.store(DBETA + beta_base + rows * H, grad_beta)
    tl.store(DPI1 + pid * C + rows, grad_pi1)
    tl.store(DPI2 + pid * C + rows, grad_pi2)


def _pinned_delta_solve_config(tensor, chunk_size, key_dim, value_dim, index):
    """Pinned (BK, BV, warps, stages) for this architecture, else None."""
    if not tensor.is_cuda or key_dim != 128 or value_dim != 128:
        return None
    tables = _PINNED_DELTA_SOLVE_CONFIGS.get(
        torch.cuda.get_device_capability(tensor.device)
    )
    if tables is None:
        return None
    return tables[index].get(chunk_size)


def _delta_solve_launch(kernel, config):
    launch = {}
    if config is not None:
        block_key, block_value, warps, stages = config
        kernel = getattr(kernel, "fn", kernel)
        launch.update(
            BK=block_key,
            BV=block_value,
            num_warps=warps,
            num_stages=stages,
        )
    return kernel, launch


def _delta_solve_forward_launch(tensor, chunk_size, key_dim, value_dim):
    return _delta_solve_launch(
        _delta_solve_recompute_fwd_kernel,
        _pinned_delta_solve_config(tensor, chunk_size, key_dim, value_dim, 0),
    )


def _delta_solve_backward_launch(tensor, chunk_size, key_dim, value_dim):
    return _delta_solve_launch(
        _delta_solve_bwd_kernel,
        _pinned_delta_solve_config(tensor, chunk_size, key_dim, value_dim, 1),
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BK": bk, "BV": bv}, num_warps=warps, num_stages=stages
        )
        for bk, bv, warps in ((32, 32, 4), (64, 32, 4), (64, 64, 8))
        for stages in (2, 3)
    ],
    key=["K", "V"],
)
@triton.jit
def _delta_builder_fwd_kernel(
    Q,
    K_IN,
    V_IN,
    BETA,
    PI1,
    PI2,
    SIGMA1,
    SIGMA2,
    ETA1,
    ETA2,
    TAU1,
    TAU2,
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
    """Keep the C16 interaction matrix in registers through the solve."""
    pid = tl.program_id(0)
    chunk = pid % N
    head = (pid // N) % H
    batch = pid // (N * H)
    rows = tl.arange(0, C)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    pi1 = tl.load(PI1 + pid * C + rows).to(tl.float32)
    pi2 = tl.load(PI2 + pid * C + rows).to(tl.float32)
    sigma1 = tl.load(SIGMA1 + pid * C + rows).to(tl.float32)
    sigma2 = tl.load(SIGMA2 + pid * C + rows).to(tl.float32)
    eta1 = tl.load(ETA1 + pid * C + rows).to(tl.float32)
    eta2 = tl.load(ETA2 + pid * C + rows).to(tl.float32)
    tau1 = tl.load(TAU1 + pid * C + rows).to(tl.float32)
    tau2 = tl.load(TAU2 + pid * C + rows).to(tl.float32)

    gram_kk = tl.zeros((C, C), dtype=tl.float32)
    gram_qk = tl.zeros((C, C), dtype=tl.float32)
    sequence_key_base = ((batch * T + chunk * C) * H + head) * K
    chunk_key_base = pid * C * K
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        sequence_offsets = (
            sequence_key_base + rows[:, None] * H * K + keys[None, :]
        )
        key = tl.load(K_IN + sequence_offsets, mask=key_mask[None, :], other=0.0)
        query = tl.load(Q + sequence_offsets, mask=key_mask[None, :], other=0.0)
        query = (query * SCALE).to(tl.bfloat16)
        gram_kk += safe_dot(key, tl.trans(key), allow_tf32=False)
        gram_qk += safe_dot(query, tl.trans(key), allow_tf32=False)
        chunk_offsets = chunk_key_base + rows[:, None] * K + keys[None, :]
        tl.store(Q_OUT + chunk_offsets, query, mask=key_mask[None, :])
        tl.store(K1 + chunk_offsets, tau1[:, None] * key, mask=key_mask[None, :])
        tl.store(K2 + chunk_offsets, tau2[:, None] * key, mask=key_mask[None, :])

    interaction_coeff = (
        pi1[:, None] * sigma1[None, :]
        + pi2[:, None] * sigma2[None, :]
    )
    output_coeff = (
        eta1[:, None] * sigma1[None, :]
        + eta2[:, None] * sigma2[None, :]
    )
    matrix = tl.where(
        row_ids > col_ids, interaction_coeff * gram_kk, 0.0
    )
    causal = tl.where(row_ids >= col_ids, output_coeff * gram_qk, 0.0)
    matrix_offsets = pid * C * C + row_ids * C + col_ids
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
    sequence_value_base = ((batch * T + chunk * C) * H + head) * V
    chunk_value_base = pid * C * V
    for value_block in range(tl.cdiv(V, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < V
        input_offsets = (
            sequence_value_base + rows[:, None] * H * V + values[None, :]
        )
        value = tl.load(V_IN + input_offsets, mask=value_mask[None, :], other=0.0)
        solution = safe_dot(
            inverse_bf16,
            (beta[:, None] * value).to(tl.bfloat16),
            allow_tf32=False,
        )
        output_offsets = chunk_value_base + rows[:, None] * V + values[None, :]
        tl.store(U0 + output_offsets, solution, mask=value_mask[None, :])

    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        input_offsets = (
            sequence_key_base + rows[:, None] * H * K + keys[None, :]
        )
        key = tl.load(K_IN + input_offsets, mask=key_mask[None, :], other=0.0)
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


class _DeltaBuild(torch.autograd.Function):
    """Fused C16 forward with the two production backward kernels."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        beta,
        pi1,
        pi2,
        sigma1,
        sigma2,
        eta1,
        eta2,
        tau1,
        tau2,
        scale,
    ):
        batch, length, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        chunk_size = pi1.shape[-1]
        chunks = length // chunk_size
        tensors = tuple(
            tensor.contiguous()
            for tensor in (
                q,
                k,
                v,
                beta,
                pi1,
                pi2,
                sigma1,
                sigma2,
                eta1,
                eta2,
                tau1,
                tau2,
            )
        )
        (
            q,
            k,
            v,
            beta,
            pi1,
            pi2,
            sigma1,
            sigma2,
            eta1,
            eta2,
            tau1,
            tau2,
        ) = tensors
        shape = (batch, heads, chunks, chunk_size)
        inverse = torch.empty(
            *shape, chunk_size, device=q.device, dtype=torch.bfloat16
        )
        u0 = torch.empty(*shape, value_dim, device=q.device, dtype=v.dtype)
        w1 = torch.empty(*shape, key_dim, device=q.device, dtype=k.dtype)
        w2 = torch.empty_like(w1)
        q_out = torch.empty_like(w1)
        k1 = torch.empty_like(w1)
        k2 = torch.empty_like(w1)
        causal = torch.empty(
            *shape, chunk_size, device=q.device, dtype=q.dtype
        )
        _delta_builder_fwd_kernel[(batch * heads * chunks,)](
            q,
            k,
            v,
            beta,
            pi1,
            pi2,
            sigma1,
            sigma2,
            eta1,
            eta2,
            tau1,
            tau2,
            inverse,
            u0,
            w1,
            w2,
            q_out,
            k1,
            k2,
            causal,
            T=length,
            H=heads,
            N=chunks,
            K=key_dim,
            V=value_dim,
            C=chunk_size,
            SCALE=float(scale),
        )
        ctx.save_for_backward(
            inverse,
            k,
            v,
            beta,
            pi1,
            pi2,
            sigma1,
            sigma2,
            eta1,
            eta2,
            tau1,
            tau2,
            u0,
            w1,
            w2,
            q_out,
        )
        ctx.dimensions = (batch, length, heads, key_dim, value_dim, chunk_size)
        ctx.scale = float(scale)
        ctx.input_dtypes = tuple(tensor.dtype for tensor in tensors)
        return u0, w1, w2, q_out, k1, k2, causal

    @staticmethod
    def backward(ctx, du0, dw1, dw2, dq_direct, dk1, dk2, dcausal):
        (
            inverse,
            k,
            v,
            beta,
            pi1,
            pi2,
            sigma1,
            sigma2,
            eta1,
            eta2,
            tau1,
            tau2,
            u0,
            w1,
            w2,
            q_out,
        ) = ctx.saved_tensors
        batch, length, heads, key_dim, value_dim, chunk_size = ctx.dimensions
        chunks = length // chunk_size
        dmatrix = torch.empty(
            batch,
            heads,
            chunks,
            chunk_size,
            chunk_size,
            dtype=torch.float32,
            device=k.device,
        )
        dk_solve = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbeta = torch.empty_like(beta)
        dpi1_solve = torch.empty_like(pi1)
        dpi2_solve = torch.empty_like(pi2)
        solve_kernel, solve_launch = _delta_solve_backward_launch(
            k, chunk_size, key_dim, value_dim
        )
        solve_kernel[(batch * heads * chunks,)](
            inverse,
            k,
            v,
            beta,
            pi1,
            pi2,
            u0,
            w1,
            w2,
            du0.contiguous(),
            dw1.contiguous(),
            dw2.contiguous(),
            dmatrix,
            dk_solve,
            dv,
            dbeta,
            dpi1_solve,
            dpi2_solve,
            T=length,
            H=heads,
            N=chunks,
            K=key_dim,
            V=value_dim,
            C=chunk_size,
            **solve_launch,
        )
        dq = torch.empty(
            batch, length, heads, key_dim, dtype=q_out.dtype, device=k.device
        )
        dk_intra = torch.empty_like(k)
        coefficient_grads = tuple(
            torch.empty_like(tensor)
            for tensor in (
                pi1,
                pi2,
                sigma1,
                sigma2,
                eta1,
                eta2,
                tau1,
                tau2,
            )
        )
        _delta_intra_bwd_kernel[(batch * heads * chunks,)](
            q_out,
            k,
            pi1,
            pi2,
            sigma1,
            sigma2,
            eta1,
            eta2,
            tau1,
            tau2,
            dmatrix,
            dcausal.contiguous(),
            dq_direct.contiguous(),
            dk1.contiguous(),
            dk2.contiguous(),
            dq,
            dk_intra,
            *coefficient_grads,
            T=length,
            H=heads,
            N=chunks,
            K=key_dim,
            C=chunk_size,
            SCALE=ctx.scale,
        )
        dpi1, dpi2, *other_coefficient_grads = coefficient_grads
        grads = (
            dq,
            dk_solve + dk_intra,
            dv,
            dbeta,
            dpi1_solve + dpi1,
            dpi2_solve + dpi2,
            *other_coefficient_grads,
        )
        return tuple(
            grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)
        ) + (None,)


class _DeltaSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, k, v, beta, pi1, pi2):
        batch, length, heads, key_dim = k.shape
        value_dim = v.shape[-1]
        chunk_size = matrix.shape[-1]
        chunks = length // chunk_size
        matrix = matrix.contiguous()
        k, v, beta, pi1, pi2 = (
            tensor.contiguous() for tensor in (k, v, beta, pi1, pi2)
        )
        shape = (batch, heads, chunks, chunk_size)
        inverse = torch.empty(
            *shape, chunk_size, device=matrix.device, dtype=torch.bfloat16
        )
        u0 = torch.empty(*shape, value_dim, device=v.device, dtype=v.dtype)
        w1 = torch.empty(*shape, key_dim, device=k.device, dtype=k.dtype)
        w2 = torch.empty_like(w1)
        if chunk_size in (16, 32, 64):
            solve_kernel, solve_launch = _delta_solve_forward_launch(
                matrix, chunk_size, key_dim, value_dim
            )
            solve_kernel[(batch * heads * chunks,)](
                matrix,
                k,
                v,
                beta,
                pi1,
                pi2,
                inverse,
                u0,
                w1,
                w2,
                T=length,
                H=heads,
                N=chunks,
                K=key_dim,
                V=value_dim,
                C=chunk_size,
                **solve_launch,
            )
        else:
            inverse = _inverse(
                matrix, matrix_is_strict_lower=True
            ).contiguous()
            _delta_recompute_fwd_kernel[(batch * heads * chunks,)](
                inverse,
                k,
                v,
                beta,
                pi1,
                pi2,
                u0,
                w1,
                w2,
                T=length,
                H=heads,
                N=chunks,
                K=key_dim,
                V=value_dim,
                C=chunk_size,
                BK=64,
                BV=64,
            )
        ctx.save_for_backward(inverse, k, v, beta, pi1, pi2, u0, w1, w2)
        ctx.input_dtypes = (
            matrix.dtype, k.dtype, v.dtype, beta.dtype, pi1.dtype, pi2.dtype
        )
        return u0, w1, w2

    @staticmethod
    def backward(ctx, grad_u0, grad_w1, grad_w2):
        inverse, k, v, beta, pi1, pi2, u0, w1, w2 = ctx.saved_tensors
        batch, length, heads, key_dim = k.shape
        chunks, chunk_size = u0.shape[2:4]
        if chunk_size in (16, 32, 64) and k.is_cuda:
            grad_matrix = torch.empty_like(inverse, dtype=torch.float32)
            grad_k = torch.empty_like(k)
            grad_v = torch.empty_like(v)
            grad_beta = torch.empty_like(beta)
            grad_pi1 = torch.empty_like(pi1)
            grad_pi2 = torch.empty_like(pi2)
            solve_kernel, solve_launch = _delta_solve_backward_launch(
                k, chunk_size, key_dim, v.shape[-1]
            )
            solve_kernel[(batch * heads * chunks,)](
                inverse,
                k,
                v,
                beta,
                pi1,
                pi2,
                u0,
                w1,
                w2,
                grad_u0.contiguous(),
                grad_w1.contiguous(),
                grad_w2.contiguous(),
                grad_matrix,
                grad_k,
                grad_v,
                grad_beta,
                grad_pi1,
                grad_pi2,
                T=length,
                H=heads,
                N=chunks,
                K=key_dim,
                V=v.shape[-1],
                C=chunk_size,
                **solve_launch,
            )
            return (
                grad_matrix,
                grad_k,
                grad_v,
                grad_beta,
                grad_pi1,
                grad_pi2,
            )
        flat_inverse = inverse.reshape(-1, chunk_size, chunk_size)
        grad_u0_b = grad_u0.reshape(-1, chunk_size, v.shape[-1]).to(inverse.dtype)
        grad_w1_b = grad_w1.reshape(-1, chunk_size, key_dim).to(inverse.dtype)
        grad_w2_b = grad_w2.reshape(-1, chunk_size, key_dim).to(inverse.dtype)
        inverse_t = flat_inverse.transpose(-1, -2)
        grad_rhs0 = inverse_t @ grad_u0_b
        grad_rhs1 = inverse_t @ grad_w1_b
        grad_rhs2 = inverse_t @ grad_w2_b
        grad_matrix = -(
            grad_rhs0 @ u0.reshape(-1, chunk_size, v.shape[-1]).transpose(-1, -2)
            + grad_rhs1 @ w1.reshape(-1, chunk_size, key_dim).transpose(-1, -2)
            + grad_rhs2 @ w2.reshape(-1, chunk_size, key_dim).transpose(-1, -2)
        ).reshape_as(inverse).float()

        kc = _chunk_vector(k, chunk_size).float()
        vc = _chunk_vector(v, chunk_size).float()
        bc = _chunk_scalar(beta, chunk_size).float()
        rhs0 = grad_rhs0.reshape_as(u0).float()
        rhs1 = grad_rhs1.reshape_as(w1).float()
        rhs2 = grad_rhs2.reshape_as(w2).float()
        grad_v = bc.unsqueeze(-1) * rhs0
        grad_k = pi1.unsqueeze(-1) * rhs1 + pi2.unsqueeze(-1) * rhs2
        grad_beta = (rhs0 * vc).sum(-1)
        grad_pi1 = (rhs1 * kc).sum(-1)
        grad_pi2 = (rhs2 * kc).sum(-1)
        grads = (
            grad_matrix,
            _unchunk_vector(grad_k),
            _unchunk_vector(grad_v),
            _unchunk_scalar(grad_beta),
            grad_pi1,
            grad_pi2,
        )
        return tuple(
            grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)
        )


def build_deltanet_operands_triton(
    q, k, v, beta, gamma, lam, scale, chunk_size=_CHUNK_SIZE
):
    """Build inverse-free DeltaNet operands in the active/slow state basis.

    DeltaNet is exactly scalar GDN with zero log-decay. The former
    specialization recovered relative transports by dividing by a cumulative
    ``gamma`` product. Learned gates can drive that product to zero in FP32,
    producing inf/NaN while the recurrent equation itself is still finite.
    The direct GDN builder forms the same relative transports without an
    inverse and is valid for C16, C32, and C64.
    """
    if not (q.is_cuda and q.dtype == torch.bfloat16):
        raise ValueError("the Delta Triton builder requires CUDA BF16 tensors")
    if chunk_size not in (16, 32, 64) or q.shape[1] % chunk_size:
        raise ValueError("sequence length must be divisible by a 16/32/64 chunk")
    from .gdn_direct_triton import build_gdn_relative_operands_triton

    return build_gdn_relative_operands_triton(
        q,
        k,
        v,
        torch.zeros_like(beta),
        beta,
        gamma,
        lam,
        scale,
        chunk_size=chunk_size,
    )


def build_gdn_operands_triton(
    q, k, v, g, beta, gamma, lam, scale, chunk_size=_CHUNK_SIZE
):
    """FLA-style, inverse-free operands for integral head-scalar GDN."""
    if not (q.is_cuda and q.dtype == torch.bfloat16):
        raise ValueError("the GDN Triton builder requires CUDA BF16 tensors")
    if chunk_size not in (16, 32, 64) or q.shape[1] % chunk_size:
        raise ValueError("sequence length must be divisible by a 16/32/64 chunk")
    from .gdn_direct_triton import build_gdn_relative_operands_triton

    return build_gdn_relative_operands_triton(
        q, k, v, g, beta, gamma, lam, scale, chunk_size=chunk_size
    )


__all__ = [
    "build_deltanet_operands_triton",
    "build_gdn_operands_triton",
]

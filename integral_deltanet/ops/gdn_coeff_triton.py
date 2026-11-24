"""Single-warp coefficient scan for integral head-scalar GDN."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


_SUPPORTED_CHUNK_SIZES = (16, 32, 64)


@triton.jit
def _gdn_coeff_fwd_kernel(
    G,
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
    alpha = tl.exp(tl.load(G + input_base + rows * H).to(tl.float32))
    beta = tl.load(BETA + input_base + rows * H).to(tl.float32)
    gamma = tl.load(GAMMA + input_base + rows * H).to(tl.float32)
    lam = tl.load(LAM + input_base + rows * H).to(tl.float32)

    f11_values = tl.zeros((C,), dtype=tl.float32)
    f12_values = tl.zeros((C,), dtype=tl.float32)
    f21_values = tl.zeros((C,), dtype=tl.float32)
    f22_values = tl.zeros((C,), dtype=tl.float32)
    previous11_values = tl.zeros((C,), dtype=tl.float32)
    previous21_values = tl.zeros((C,), dtype=tl.float32)
    determinant_values = tl.zeros((C,), dtype=tl.float32)
    f11, f12, f21, f22 = 1.0, 0.0, 0.0, 1.0
    determinant = 1.0
    for row in range(C):
        previous11_values = tl.where(rows == row, f11, previous11_values)
        previous21_values = tl.where(rows == row, f21, previous21_values)
        alpha_row = tl.load(G + input_base + row * H).to(tl.float32)
        alpha_row = tl.exp(alpha_row)
        gamma_row = tl.load(GAMMA + input_base + row * H).to(tl.float32)
        lambda_row = tl.load(LAM + input_base + row * H).to(tl.float32)
        p11 = gamma_row * alpha_row
        p12 = lambda_row * p11
        p21 = (1.0 - gamma_row) * alpha_row
        p22 = (1.0 - lambda_row) + lambda_row * p21
        next11 = f11 * p11 + f12 * p21
        next12 = f11 * p12 + f12 * p22
        next21 = f21 * p11 + f22 * p21
        next22 = f21 * p12 + f22 * p22
        f11, f12, f21, f22 = next11, next12, next21, next22
        determinant *= gamma_row * alpha_row * (1.0 - lambda_row)
        f11_values = tl.where(rows == row, f11, f11_values)
        f12_values = tl.where(rows == row, f12, f12_values)
        f21_values = tl.where(rows == row, f21, f21_values)
        f22_values = tl.where(rows == row, f22, f22_values)
        determinant_values = tl.where(
            rows == row, determinant, determinant_values
        )

    pi1 = previous11_values * alpha * beta
    pi2 = previous21_values * alpha * beta
    sigma1 = (f22_values - lam * f21_values) / determinant_values
    sigma2 = (-f12_values + lam * f11_values) / determinant_values
    tau1 = sigma1 * f11 + sigma2 * f21
    tau2 = sigma1 * f12 + sigma2 * f22
    offsets = pid * C + rows
    tl.store(PI1 + offsets, pi1)
    tl.store(PI2 + offsets, pi2)
    tl.store(SIGMA1 + offsets, sigma1)
    tl.store(SIGMA2 + offsets, sigma2)
    tl.store(ETA1 + offsets, f11_values)
    tl.store(ETA2 + offsets, f21_values)
    tl.store(TAU1 + offsets, tau1)
    tl.store(TAU2 + offsets, tau2)
    query_offsets = pid * C * 2 + rows * 2
    tl.store(QUERY_COEFFICIENTS + query_offsets + 0, f11_values)
    tl.store(QUERY_COEFFICIENTS + query_offsets + 1, f21_values)
    tl.store(TRANSPORT + pid * 4 + 0, f11)
    tl.store(TRANSPORT + pid * 4 + 1, f12)
    tl.store(TRANSPORT + pid * 4 + 2, f21)
    tl.store(TRANSPORT + pid * 4 + 3, f22)


@triton.jit
def _gdn_coeff_bwd_kernel(
    G,
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

    f11_values = tl.zeros((C,), dtype=tl.float32)
    f12_values = tl.zeros((C,), dtype=tl.float32)
    f21_values = tl.zeros((C,), dtype=tl.float32)
    f22_values = tl.zeros((C,), dtype=tl.float32)
    previous11_values = tl.zeros((C,), dtype=tl.float32)
    previous12_values = tl.zeros((C,), dtype=tl.float32)
    previous21_values = tl.zeros((C,), dtype=tl.float32)
    previous22_values = tl.zeros((C,), dtype=tl.float32)
    determinant_values = tl.zeros((C,), dtype=tl.float32)
    determinant_previous_values = tl.zeros((C,), dtype=tl.float32)
    f11, f12, f21, f22 = 1.0, 0.0, 0.0, 1.0
    determinant = 1.0
    for row in range(C):
        previous11_values = tl.where(rows == row, f11, previous11_values)
        previous12_values = tl.where(rows == row, f12, previous12_values)
        previous21_values = tl.where(rows == row, f21, previous21_values)
        previous22_values = tl.where(rows == row, f22, previous22_values)
        determinant_previous_values = tl.where(
            rows == row, determinant, determinant_previous_values
        )
        alpha_row = tl.exp(
            tl.load(G + input_base + row * H).to(tl.float32)
        )
        gamma_row = tl.load(GAMMA + input_base + row * H).to(tl.float32)
        lambda_row = tl.load(LAM + input_base + row * H).to(tl.float32)
        p11 = gamma_row * alpha_row
        p12 = lambda_row * p11
        p21 = (1.0 - gamma_row) * alpha_row
        p22 = (1.0 - lambda_row) + lambda_row * p21
        next11 = f11 * p11 + f12 * p21
        next12 = f11 * p12 + f12 * p22
        next21 = f21 * p11 + f22 * p21
        next22 = f21 * p12 + f22 * p22
        f11, f12, f21, f22 = next11, next12, next21, next22
        determinant *= gamma_row * alpha_row * (1.0 - lambda_row)
        f11_values = tl.where(rows == row, f11, f11_values)
        f12_values = tl.where(rows == row, f12, f12_values)
        f21_values = tl.where(rows == row, f21, f21_values)
        f22_values = tl.where(rows == row, f22, f22_values)
        determinant_values = tl.where(
            rows == row, determinant, determinant_values
        )

    offsets = pid * C + rows
    dpi1 = tl.load(DPI1 + offsets).to(tl.float32)
    dpi2 = tl.load(DPI2 + offsets).to(tl.float32)
    dsigma1 = tl.load(DSIGMA1 + offsets).to(tl.float32)
    dsigma2 = tl.load(DSIGMA2 + offsets).to(tl.float32)
    deta1 = tl.load(DETA1 + offsets).to(tl.float32)
    deta2 = tl.load(DETA2 + offsets).to(tl.float32)
    dtau1 = tl.load(DTAU1 + offsets).to(tl.float32)
    dtau2 = tl.load(DTAU2 + offsets).to(tl.float32)
    query_offsets = pid * C * 2 + rows * 2
    deta1 += tl.load(DQUERY_COEFFICIENTS + query_offsets + 0).to(tl.float32)
    deta2 += tl.load(DQUERY_COEFFICIENTS + query_offsets + 1).to(tl.float32)

    sigma1 = (f22_values - lam * f21_values) / determinant_values
    sigma2 = (-f12_values + lam * f11_values) / determinant_values
    dsigma1_total = dsigma1 + dtau1 * f11 + dtau2 * f12
    dsigma2_total = dsigma2 + dtau1 * f21 + dtau2 * f22
    numerator1 = f22_values - lam * f21_values
    numerator2 = -f12_values + lam * f11_values
    grad_numerator1 = dsigma1_total / determinant_values
    grad_numerator2 = dsigma2_total / determinant_values
    grad_determinant_direct = -(
        dsigma1_total * numerator1 + dsigma2_total * numerator2
    ) / (determinant_values * determinant_values)
    direct_f11 = deta1 + lam * grad_numerator2
    direct_f12 = -grad_numerator2
    direct_f21 = deta2 - lam * grad_numerator1
    direct_f22 = grad_numerator1
    direct_lam = (
        -f21_values * grad_numerator1
        + f11_values * grad_numerator2
    )
    direct_alpha = (
        dpi1 * previous11_values * beta
        + dpi2 * previous21_values * beta
    )
    grad_beta = alpha * (
        dpi1 * previous11_values + dpi2 * previous21_values
    )
    direct_previous11 = dpi1 * alpha * beta
    direct_previous21 = dpi2 * alpha * beta

    grad_f11 = tl.load(DTRANSPORT + pid * 4 + 0).to(tl.float32)
    grad_f12 = tl.load(DTRANSPORT + pid * 4 + 1).to(tl.float32)
    grad_f21 = tl.load(DTRANSPORT + pid * 4 + 2).to(tl.float32)
    grad_f22 = tl.load(DTRANSPORT + pid * 4 + 3).to(tl.float32)
    grad_f11 += tl.sum(dtau1 * sigma1, axis=0)
    grad_f21 += tl.sum(dtau1 * sigma2, axis=0)
    grad_f12 += tl.sum(dtau2 * sigma1, axis=0)
    grad_f22 += tl.sum(dtau2 * sigma2, axis=0)
    grad_determinant = 0.0
    grad_g_values = tl.zeros((C,), dtype=tl.float32)
    grad_gamma_values = tl.zeros((C,), dtype=tl.float32)
    grad_lam_values = tl.zeros((C,), dtype=tl.float32)

    for reverse_index in range(C):
        row = C - 1 - reverse_index
        grad_f11 += tl.sum(
            tl.where(rows == row, direct_f11, 0.0), axis=0
        )
        grad_f12 += tl.sum(
            tl.where(rows == row, direct_f12, 0.0), axis=0
        )
        grad_f21 += tl.sum(
            tl.where(rows == row, direct_f21, 0.0), axis=0
        )
        grad_f22 += tl.sum(
            tl.where(rows == row, direct_f22, 0.0), axis=0
        )
        grad_determinant += tl.sum(
            tl.where(rows == row, grad_determinant_direct, 0.0), axis=0
        )

        previous11 = tl.sum(
            tl.where(rows == row, previous11_values, 0.0), axis=0
        )
        previous12 = tl.sum(
            tl.where(rows == row, previous12_values, 0.0), axis=0
        )
        previous21 = tl.sum(
            tl.where(rows == row, previous21_values, 0.0), axis=0
        )
        previous22 = tl.sum(
            tl.where(rows == row, previous22_values, 0.0), axis=0
        )
        alpha_row = tl.sum(tl.where(rows == row, alpha, 0.0), axis=0)
        gamma_row = tl.sum(tl.where(rows == row, gamma, 0.0), axis=0)
        lambda_row = tl.sum(tl.where(rows == row, lam, 0.0), axis=0)
        p11 = gamma_row * alpha_row
        p12 = lambda_row * p11
        p21 = (1.0 - gamma_row) * alpha_row
        p22 = (1.0 - lambda_row) + lambda_row * p21

        grad_p11 = previous11 * grad_f11 + previous21 * grad_f21
        grad_p12 = previous11 * grad_f12 + previous21 * grad_f22
        grad_p21 = previous12 * grad_f11 + previous22 * grad_f21
        grad_p22 = previous12 * grad_f12 + previous22 * grad_f22
        next_grad_f11 = grad_f11 * p11 + grad_f12 * p12
        next_grad_f12 = grad_f11 * p21 + grad_f12 * p22
        next_grad_f21 = grad_f21 * p11 + grad_f22 * p12
        next_grad_f22 = grad_f21 * p21 + grad_f22 * p22
        next_grad_f11 += tl.sum(
            tl.where(rows == row, direct_previous11, 0.0), axis=0
        )
        next_grad_f21 += tl.sum(
            tl.where(rows == row, direct_previous21, 0.0), axis=0
        )

        determinant_previous = tl.sum(
            tl.where(rows == row, determinant_previous_values, 0.0), axis=0
        )
        determinant_step = gamma_row * alpha_row * (1.0 - lambda_row)
        grad_determinant_step = grad_determinant * determinant_previous
        next_grad_determinant = grad_determinant * determinant_step

        active = gamma_row * alpha_row
        passive = (1.0 - gamma_row) * alpha_row
        grad_active = grad_p11 + lambda_row * grad_p12
        grad_passive = grad_p21 + lambda_row * grad_p22
        grad_alpha = tl.sum(
            tl.where(rows == row, direct_alpha, 0.0), axis=0
        )
        grad_alpha += (
            grad_active * gamma_row
            + grad_passive * (1.0 - gamma_row)
            + grad_determinant_step * gamma_row * (1.0 - lambda_row)
        )
        grad_gamma = (
            grad_active * alpha_row
            - grad_passive * alpha_row
            + grad_determinant_step * alpha_row * (1.0 - lambda_row)
        )
        grad_lambda = tl.sum(
            tl.where(rows == row, direct_lam, 0.0), axis=0
        )
        grad_lambda += (
            grad_p12 * active
            + grad_p22 * (passive - 1.0)
            - grad_determinant_step * gamma_row * alpha_row
        )
        grad_g_values = tl.where(
            rows == row, grad_alpha * alpha_row, grad_g_values
        )
        grad_gamma_values = tl.where(
            rows == row, grad_gamma, grad_gamma_values
        )
        grad_lam_values = tl.where(
            rows == row, grad_lambda, grad_lam_values
        )
        grad_f11, grad_f12 = next_grad_f11, next_grad_f12
        grad_f21, grad_f22 = next_grad_f21, next_grad_f22
        grad_determinant = next_grad_determinant

    output_offsets = input_base + rows * H
    tl.store(DG + output_offsets, grad_g_values)
    tl.store(DBETA + output_offsets, grad_beta)
    tl.store(DGAMMA + output_offsets, grad_gamma_values)
    tl.store(DLAM + output_offsets, grad_lam_values)


class _GDNCoefficients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, beta, gamma, lam, chunk_size):
        batch, length, heads = beta.shape
        if chunk_size not in _SUPPORTED_CHUNK_SIZES or length % chunk_size:
            raise ValueError("length must be divisible by a 16/32/64 chunk")
        chunks = length // chunk_size
        g, beta, gamma, lam = (
            tensor.contiguous() for tensor in (g, beta, gamma, lam)
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
        _gdn_coeff_fwd_kernel[(batch * heads * chunks,)](
            g,
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
        ctx.save_for_backward(g, beta, gamma, lam)
        ctx.dimensions = (batch, length, heads, chunks, chunk_size)
        return (*coefficient_tensors, query_coefficients, transport)

    @staticmethod
    def backward(ctx, *output_grads):
        g, beta, gamma, lam = ctx.saved_tensors
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
            else torch.zeros(
                shape, dtype=torch.float32, device=beta.device
            )
            for grad, shape in zip(output_grads, output_shapes)
        )
        grad_g = torch.empty_like(g)
        grad_beta = torch.empty_like(beta)
        grad_gamma = torch.empty_like(gamma)
        grad_lam = torch.empty_like(lam)
        _gdn_coeff_bwd_kernel[(batch * heads * chunks,)](
            g,
            beta,
            gamma,
            lam,
            *output_grads,
            grad_g,
            grad_beta,
            grad_gamma,
            grad_lam,
            T=length,
            H=heads,
            N=chunks,
            C=chunk_size,
            num_warps=1,
        )
        return grad_g, grad_beta, grad_gamma, grad_lam, None


def gdn_coefficients_triton(g, beta, gamma, lam, chunk_size=16):
    return _GDNCoefficients.apply(g, beta, gamma, lam, chunk_size)


__all__ = ["gdn_coefficients_triton"]

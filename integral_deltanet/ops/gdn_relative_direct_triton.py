"""Direct-load scalar GDN coefficient kernels.

The original ``_gdn_relative_fwd_kernel`` and its backward walk the chunk
serially and fetch every per-target scalar with a masked ``tl.sum`` over the
whole chunk.  At C=32 that is roughly fifteen chunk-wide reductions per step
on a single warp, and the register-checkpointed backward additionally rebuilds
the source prefix from scratch for every block.  These kernels compute exactly
the same quantities (same FP32 operations in the same order) but read each
per-target scalar with a scalar load, keep the two small histories in a scratch
buffer instead of registers, and leave only the five genuine dot-product
reductions per backward step.  Outputs are bitwise identical to the original
kernels.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _target_transition(G, BETA, GAMMA, LAM, offset):
    """Per-token scalars (alpha, beta, p11, p12, p21, p22) via scalar loads."""
    alpha_t = tl.exp(tl.load(G + offset).to(tl.float32))
    beta_t = tl.load(BETA + offset).to(tl.float32)
    gamma_t = tl.load(GAMMA + offset).to(tl.float32)
    lam_t = tl.load(LAM + offset).to(tl.float32)
    p11_t = gamma_t * alpha_t
    p12_t = lam_t * p11_t
    p21_t = (1.0 - gamma_t) * alpha_t
    p22_t = (1.0 - lam_t) + lam_t * p21_t
    return alpha_t, beta_t, p11_t, p12_t, p21_t, p22_t


@triton.jit
def _gdn_relative_direct_fwd_kernel(
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
    lam = tl.load(LAM + input_base + rows * H).to(tl.float32)

    # Incoming-state prefixes.  Only multiplications of contractive matrices
    # are used; no prefix is inverted.
    f11, f12, f21, f22 = 1.0, 0.0, 0.0, 1.0
    for target in range(C):
        alpha_t, beta_t, p11_t, p12_t, p21_t, p22_t = _target_transition(
            G, BETA, GAMMA, LAM, input_base + target * H
        )
        alpha_beta = alpha_t * beta_t
        coefficient_offset = pid * C + target
        tl.store(PI1 + coefficient_offset, f11 * alpha_beta)
        tl.store(PI2 + coefficient_offset, f21 * alpha_beta)
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
        alpha_t, beta_t, p11_t, p12_t, p21_t, p22_t = _target_transition(
            G, BETA, GAMMA, LAM, input_base + target * H
        )
        alpha_beta = alpha_t * beta_t
        matrix_offsets = pid * C * C + target * C + rows
        tl.store(
            INTERACTION + matrix_offsets,
            tl.where(earlier, c1 * alpha_beta, 0.0),
        )
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
def _gdn_relative_direct_bwd_kernel(
    G,
    BETA,
    GAMMA,
    LAM,
    F_SCRATCH,
    C1_SCRATCH,
    C2_SCRATCH,
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
    gamma = tl.load(GAMMA + input_base + rows * H).to(tl.float32)
    lam = tl.load(LAM + input_base + rows * H).to(tl.float32)
    p11 = gamma * alpha
    p12 = lam * p11
    p21 = (1.0 - gamma) * alpha
    p22 = (1.0 - lam) + lam * p21

    # Pass 1: replay the forward prefixes into scratch (FLA-style recompute,
    # but written once and read back by index instead of selected from
    # registers).
    f11, f12, f21, f22 = 1.0, 0.0, 0.0, 1.0
    c1 = tl.full((C,), 1.0, tl.float32)
    c2 = lam
    for target in range(C):
        f_offset = pid * C * 4 + target * 4
        tl.store(F_SCRATCH + f_offset + 0, f11)
        tl.store(F_SCRATCH + f_offset + 1, f12)
        tl.store(F_SCRATCH + f_offset + 2, f21)
        tl.store(F_SCRATCH + f_offset + 3, f22)
        history_offsets = pid * C * C + target * C + rows
        tl.store(C1_SCRATCH + history_offsets, c1)
        tl.store(C2_SCRATCH + history_offsets, c2)
        alpha_t, beta_t, p11_t, p12_t, p21_t, p22_t = _target_transition(
            G, BETA, GAMMA, LAM, input_base + target * H
        )
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

    grad_alpha = tl.zeros((C,), tl.float32)
    grad_beta = tl.zeros((C,), tl.float32)
    grad_p11 = tl.zeros((C,), tl.float32)
    grad_p12 = tl.zeros((C,), tl.float32)
    grad_p21 = tl.zeros((C,), tl.float32)
    grad_p22 = tl.zeros((C,), tl.float32)

    # Pass 2: reverse all direct relative source transports.
    offsets = pid * C + rows
    adjoint1 = tl.load(DTAIL1 + offsets).to(tl.float32)
    adjoint2 = tl.load(DTAIL2 + offsets).to(tl.float32)
    for reverse_index in range(C - 1):
        target = C - 1 - reverse_index
        target_mask = rows == target
        active_sources = rows < target
        matrix_offsets = pid * C * C + target * C + rows
        grad_interaction = tl.load(DINTERACTION + matrix_offsets).to(tl.float32)
        grad_causal = tl.load(DCAUSAL + matrix_offsets).to(tl.float32)
        cprev1 = tl.load(C1_SCRATCH + matrix_offsets)
        cprev2 = tl.load(C2_SCRATCH + matrix_offsets)
        after1 = adjoint1 + tl.where(active_sources, grad_causal, 0.0)
        after2 = adjoint2
        dp11 = tl.sum(tl.where(active_sources, cprev1 * after1, 0.0), axis=0)
        dp12 = tl.sum(tl.where(active_sources, cprev1 * after2, 0.0), axis=0)
        dp21 = tl.sum(tl.where(active_sources, cprev2 * after1, 0.0), axis=0)
        dp22 = tl.sum(tl.where(active_sources, cprev2 * after2, 0.0), axis=0)
        grad_p11 = tl.where(target_mask, dp11, grad_p11)
        grad_p12 = tl.where(target_mask, dp12, grad_p12)
        grad_p21 = tl.where(target_mask, dp21, grad_p21)
        grad_p22 = tl.where(target_mask, dp22, grad_p22)
        alpha_t, beta_t, p11_t, p12_t, p21_t, p22_t = _target_transition(
            G, BETA, GAMMA, LAM, input_base + target * H
        )
        previous1 = after1 * p11_t + after2 * p12_t
        previous2 = after1 * p21_t + after2 * p22_t
        alpha_beta_t = alpha_t * beta_t
        previous1 += tl.where(
            active_sources, grad_interaction * alpha_beta_t, 0.0
        )
        grad_alpha_beta = tl.sum(
            tl.where(active_sources, grad_interaction * cprev1, 0.0), axis=0
        )
        grad_alpha += tl.where(target_mask, grad_alpha_beta * beta_t, 0.0)
        grad_beta += tl.where(target_mask, grad_alpha_beta * alpha_t, 0.0)
        adjoint1 = tl.where(active_sources, previous1, adjoint1)
        adjoint2 = tl.where(active_sources, previous2, adjoint2)
    grad_lam = adjoint2

    # Pass 3: reverse the absolute incoming-state prefix.
    d_f11 = tl.load(DTRANSPORT + pid * 4 + 0).to(tl.float32)
    d_f12 = tl.load(DTRANSPORT + pid * 4 + 1).to(tl.float32)
    d_f21 = tl.load(DTRANSPORT + pid * 4 + 2).to(tl.float32)
    d_f22 = tl.load(DTRANSPORT + pid * 4 + 3).to(tl.float32)
    for reverse_index in range(C):
        target = C - 1 - reverse_index
        target_mask = rows == target
        coefficient_offset = pid * C + target
        d_f11 += tl.load(DETA1 + coefficient_offset).to(tl.float32)
        d_f21 += tl.load(DETA2 + coefficient_offset).to(tl.float32)
        f_offset = pid * C * 4 + target * 4
        f11 = tl.load(F_SCRATCH + f_offset + 0)
        f12 = tl.load(F_SCRATCH + f_offset + 1)
        f21 = tl.load(F_SCRATCH + f_offset + 2)
        f22 = tl.load(F_SCRATCH + f_offset + 3)
        alpha_t, beta_t, p11_t, p12_t, p21_t, p22_t = _target_transition(
            G, BETA, GAMMA, LAM, input_base + target * H
        )
        add11 = f11 * d_f11 + f21 * d_f21
        add12 = f11 * d_f12 + f21 * d_f22
        add21 = f12 * d_f11 + f22 * d_f21
        add22 = f12 * d_f12 + f22 * d_f22
        grad_p11 += tl.where(target_mask, add11, 0.0)
        grad_p12 += tl.where(target_mask, add12, 0.0)
        grad_p21 += tl.where(target_mask, add21, 0.0)
        grad_p22 += tl.where(target_mask, add22, 0.0)
        previous_d_f11 = d_f11 * p11_t + d_f12 * p12_t
        previous_d_f12 = d_f11 * p21_t + d_f12 * p22_t
        previous_d_f21 = d_f21 * p11_t + d_f22 * p12_t
        previous_d_f22 = d_f21 * p21_t + d_f22 * p22_t
        dpi1_t = tl.load(DPI1 + coefficient_offset).to(tl.float32)
        dpi2_t = tl.load(DPI2 + coefficient_offset).to(tl.float32)
        alpha_beta_t = alpha_t * beta_t
        previous_d_f11 += dpi1_t * alpha_beta_t
        previous_d_f21 += dpi2_t * alpha_beta_t
        grad_alpha_beta = dpi1_t * f11 + dpi2_t * f21
        grad_alpha += tl.where(target_mask, grad_alpha_beta * beta_t, 0.0)
        grad_beta += tl.where(target_mask, grad_alpha_beta * alpha_t, 0.0)
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


__all__ = [
    "_gdn_relative_direct_fwd_kernel",
    "_gdn_relative_direct_bwd_kernel",
]

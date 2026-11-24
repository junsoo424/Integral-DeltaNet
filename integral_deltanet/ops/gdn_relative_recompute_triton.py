"""Register-checkpointed backward for inverse-free scalar GDN coefficients."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _gdn_relative_recompute_bwd_kernel(
    G,
    BETA,
    GAMMA,
    LAM,
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
    B: tl.constexpr,
):
    """Differentiate C16/C32 without materializing any history in HBM."""
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

    # The four absolute-prefix histories are only O(C). Keep them in registers
    # for the later reverse prefix pass.
    f_history11 = tl.zeros((C,), tl.float32)
    f_history12 = tl.zeros((C,), tl.float32)
    f_history21 = tl.zeros((C,), tl.float32)
    f_history22 = tl.zeros((C,), tl.float32)
    f11, f12, f21, f22 = 1.0, 0.0, 0.0, 1.0
    for target in range(C):
        target_mask = rows == target
        f_history11 = tl.where(target_mask, f11, f_history11)
        f_history12 = tl.where(target_mask, f12, f_history12)
        f_history21 = tl.where(target_mask, f21, f_history21)
        f_history22 = tl.where(target_mask, f22, f_history22)
        p11_t = tl.sum(tl.where(target_mask, p11, 0.0), axis=0)
        p12_t = tl.sum(tl.where(target_mask, p12, 0.0), axis=0)
        p21_t = tl.sum(tl.where(target_mask, p21, 0.0), axis=0)
        p22_t = tl.sum(tl.where(target_mask, p22, 0.0), axis=0)
        next11 = f11 * p11_t + f12 * p21_t
        next12 = f11 * p12_t + f12 * p22_t
        next21 = f21 * p11_t + f22 * p21_t
        next22 = f21 * p12_t + f22 * p22_t
        f11, f12, f21, f22 = next11, next12, next21, next22

    offsets = pid * C + rows
    adjoint1 = tl.load(DTAIL1 + offsets).to(tl.float32)
    adjoint2 = tl.load(DTAIL2 + offsets).to(tl.float32)
    local_rows = tl.arange(0, B)

    # Recompute one 16-token checkpoint at a time. The largest live history is
    # 2*B*C rather than two global C*C arrays; no ill-conditioned inverse is
    # introduced and no history is written to HBM.
    for reverse_block in range(C // B):
        block_start = C - (reverse_block + 1) * B
        c1 = tl.full((C,), 1.0, tl.float32)
        c2 = lam
        for prefix_target in range(block_start):
            prefix_mask = rows == prefix_target
            p11_t = tl.sum(tl.where(prefix_mask, p11, 0.0), axis=0)
            p12_t = tl.sum(tl.where(prefix_mask, p12, 0.0), axis=0)
            p21_t = tl.sum(tl.where(prefix_mask, p21, 0.0), axis=0)
            p22_t = tl.sum(tl.where(prefix_mask, p22, 0.0), axis=0)
            earlier = rows < prefix_target
            next1 = c1 * p11_t + c2 * p21_t
            next2 = c1 * p12_t + c2 * p22_t
            c1 = tl.where(earlier, next1, c1)
            c2 = tl.where(earlier, next2, c2)

        c1_history = tl.zeros((B, C), tl.float32)
        c2_history = tl.zeros((B, C), tl.float32)
        for local_index in range(B):
            target = block_start + local_index
            history_mask = local_rows[:, None] == local_index
            c1_history = tl.where(history_mask, c1[None, :], c1_history)
            c2_history = tl.where(history_mask, c2[None, :], c2_history)
            target_mask = rows == target
            p11_t = tl.sum(tl.where(target_mask, p11, 0.0), axis=0)
            p12_t = tl.sum(tl.where(target_mask, p12, 0.0), axis=0)
            p21_t = tl.sum(tl.where(target_mask, p21, 0.0), axis=0)
            p22_t = tl.sum(tl.where(target_mask, p22, 0.0), axis=0)
            earlier = rows < target
            next1 = c1 * p11_t + c2 * p21_t
            next2 = c1 * p12_t + c2 * p22_t
            c1 = tl.where(earlier, next1, c1)
            c2 = tl.where(earlier, next2, c2)

        for reverse_local in range(B):
            local_index = B - 1 - reverse_local
            target = block_start + local_index
            target_mask = rows == target
            active_sources = rows < target
            history_mask = local_rows[:, None] == local_index
            cprev1 = tl.sum(
                tl.where(history_mask, c1_history, 0.0), axis=0
            )
            cprev2 = tl.sum(
                tl.where(history_mask, c2_history, 0.0), axis=0
            )
            matrix_offsets = pid * C * C + target * C + rows
            grad_interaction = tl.load(
                DINTERACTION + matrix_offsets
            ).to(tl.float32)
            grad_causal = tl.load(DCAUSAL + matrix_offsets).to(tl.float32)
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
            grad_p11 = tl.where(target_mask, dp11, grad_p11)
            grad_p12 = tl.where(target_mask, dp12, grad_p12)
            grad_p21 = tl.where(target_mask, dp21, grad_p21)
            grad_p22 = tl.where(target_mask, dp22, grad_p22)

            p11_t = tl.sum(tl.where(target_mask, p11, 0.0), axis=0)
            p12_t = tl.sum(tl.where(target_mask, p12, 0.0), axis=0)
            p21_t = tl.sum(tl.where(target_mask, p21, 0.0), axis=0)
            p22_t = tl.sum(tl.where(target_mask, p22, 0.0), axis=0)
            previous1 = after1 * p11_t + after2 * p12_t
            previous2 = after1 * p21_t + after2 * p22_t
            alpha_beta_t = tl.sum(
                tl.where(target_mask, alpha * beta, 0.0), axis=0
            )
            previous1 += tl.where(
                active_sources, grad_interaction * alpha_beta_t, 0.0
            )
            grad_alpha_beta = tl.sum(
                tl.where(active_sources, grad_interaction * cprev1, 0.0),
                axis=0,
            )
            alpha_t = tl.sum(tl.where(target_mask, alpha, 0.0), axis=0)
            beta_t = tl.sum(tl.where(target_mask, beta, 0.0), axis=0)
            grad_alpha += tl.where(
                target_mask, grad_alpha_beta * beta_t, 0.0
            )
            grad_beta += tl.where(
                target_mask, grad_alpha_beta * alpha_t, 0.0
            )
            adjoint1 = tl.where(active_sources, previous1, adjoint1)
            adjoint2 = tl.where(active_sources, previous2, adjoint2)
    grad_lam += adjoint2

    # Reverse the absolute incoming-state prefix from its register history.
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
        target_mask = rows == target
        d_f11 += tl.sum(tl.where(target_mask, deta1, 0.0), axis=0)
        d_f21 += tl.sum(tl.where(target_mask, deta2, 0.0), axis=0)
        f11 = tl.sum(tl.where(target_mask, f_history11, 0.0), axis=0)
        f12 = tl.sum(tl.where(target_mask, f_history12, 0.0), axis=0)
        f21 = tl.sum(tl.where(target_mask, f_history21, 0.0), axis=0)
        f22 = tl.sum(tl.where(target_mask, f_history22, 0.0), axis=0)
        p11_t = tl.sum(tl.where(target_mask, p11, 0.0), axis=0)
        p12_t = tl.sum(tl.where(target_mask, p12, 0.0), axis=0)
        p21_t = tl.sum(tl.where(target_mask, p21, 0.0), axis=0)
        p22_t = tl.sum(tl.where(target_mask, p22, 0.0), axis=0)

        add11 = f11 * d_f11 + f21 * d_f21
        add12 = f11 * d_f12 + f21 * d_f22
        add21 = f12 * d_f11 + f22 * d_f21
        add22 = f12 * d_f12 + f22 * d_f22
        old11 = tl.sum(tl.where(target_mask, grad_p11, 0.0), axis=0)
        old12 = tl.sum(tl.where(target_mask, grad_p12, 0.0), axis=0)
        old21 = tl.sum(tl.where(target_mask, grad_p21, 0.0), axis=0)
        old22 = tl.sum(tl.where(target_mask, grad_p22, 0.0), axis=0)
        grad_p11 = tl.where(target_mask, old11 + add11, grad_p11)
        grad_p12 = tl.where(target_mask, old12 + add12, grad_p12)
        grad_p21 = tl.where(target_mask, old21 + add21, grad_p21)
        grad_p22 = tl.where(target_mask, old22 + add22, grad_p22)

        previous_d_f11 = d_f11 * p11_t + d_f12 * p12_t
        previous_d_f12 = d_f11 * p21_t + d_f12 * p22_t
        previous_d_f21 = d_f21 * p11_t + d_f22 * p12_t
        previous_d_f22 = d_f21 * p21_t + d_f22 * p22_t
        dpi1_t = tl.sum(tl.where(target_mask, dpi1, 0.0), axis=0)
        dpi2_t = tl.sum(tl.where(target_mask, dpi2, 0.0), axis=0)
        alpha_t = tl.sum(tl.where(target_mask, alpha, 0.0), axis=0)
        beta_t = tl.sum(tl.where(target_mask, beta, 0.0), axis=0)
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


__all__ = ["_gdn_relative_recompute_bwd_kernel"]

"""Fused Q/K normalization and integration-gate preprocessing.

Unfused, the training wrappers launch FLA L2 normalization twice and then
materialize every sigmoid/softplus/exp expression as separate framework
operations.  These kernels keep the FP32 gate arithmetic, but process one
``(batch, token, head)`` row per Triton program so Q, K, erase gates, and the
ordered integration gates share one forward and one backward launch.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_SCALAR_WARPS = 2
_CHANNEL_WARPS = 2


@triton.jit
def _stable_softplus(value):
    return tl.maximum(value, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(value)))


@triton.jit
def _scalar_preprocess_fwd_kernel(
    Q,
    K_IN,
    BETA_LOGITS,
    GATE_LOGITS,
    Q_OUT,
    K_OUT,
    BETA,
    GAMMA,
    LAM,
    RSTD_Q,
    RSTD_K,
    N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BD: tl.constexpr,
    LOG_CHI_MIN: tl.constexpr,
    BETA_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BD)
    mask = cols < D
    offsets = row * D + cols

    q = tl.load(Q + offsets, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(K_IN + offsets, mask=mask, other=0.0).to(tl.float32)
    rq = tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6)
    rk = tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
    tl.store(Q_OUT + offsets, q * rq, mask=mask)
    tl.store(K_OUT + offsets, k * rk, mask=mask)
    tl.store(RSTD_Q + row, rq)
    tl.store(RSTD_K + row, rk)

    token = row // H
    head = row - token * H
    gate_offset = token * (2 * H) + head
    gamma_logit = tl.load(GATE_LOGITS + gate_offset).to(tl.float32)
    ratio_logit = tl.load(GATE_LOGITS + gate_offset + H).to(tl.float32)
    rho_gamma = _stable_softplus(gamma_logit)
    log_chi = LOG_CHI_MIN + _stable_softplus(ratio_logit)
    rho_lam = rho_gamma * tl.exp(-log_chi)

    beta = tl.sigmoid(
        tl.load(BETA_LOGITS + row).to(tl.float32)
    ) * BETA_SCALE
    tl.store(BETA + row, beta)
    tl.store(GAMMA + row, tl.exp(-rho_gamma))
    tl.store(LAM + row, 1.0 - tl.exp(-rho_lam))


@triton.jit
def _scalar_preprocess_bwd_kernel(
    Q_OUT,
    K_OUT,
    BETA,
    GAMMA,
    LAM,
    GATE_LOGITS,
    RSTD_Q,
    RSTD_K,
    DQ_OUT,
    DK_OUT,
    DBETA,
    DGAMMA,
    DLAM,
    DQ,
    DK,
    DBETA_LOGITS,
    DGATE_LOGITS,
    N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BD: tl.constexpr,
    LOG_CHI_MIN: tl.constexpr,
    BETA_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BD)
    mask = cols < D
    offsets = row * D + cols

    q = tl.load(Q_OUT + offsets, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(K_OUT + offsets, mask=mask, other=0.0).to(tl.float32)
    dq = tl.load(DQ_OUT + offsets, mask=mask, other=0.0).to(tl.float32)
    dk = tl.load(DK_OUT + offsets, mask=mask, other=0.0).to(tl.float32)
    rq = tl.load(RSTD_Q + row).to(tl.float32)
    rk = tl.load(RSTD_K + row).to(tl.float32)
    dq = (dq - tl.sum(dq * q, axis=0) * q) * rq
    dk = (dk - tl.sum(dk * k, axis=0) * k) * rk
    tl.store(DQ + offsets, dq, mask=mask)
    tl.store(DK + offsets, dk, mask=mask)

    beta = tl.load(BETA + row).to(tl.float32) / BETA_SCALE
    dbeta = tl.load(DBETA + row).to(tl.float32)
    tl.store(
        DBETA_LOGITS + row,
        dbeta * BETA_SCALE * beta * (1.0 - beta),
    )

    token = row // H
    head = row - token * H
    gate_offset = token * (2 * H) + head
    gamma_logit = tl.load(GATE_LOGITS + gate_offset).to(tl.float32)
    ratio_logit = tl.load(GATE_LOGITS + gate_offset + H).to(tl.float32)
    rho_gamma = _stable_softplus(gamma_logit)
    log_chi = LOG_CHI_MIN + _stable_softplus(ratio_logit)
    inv_chi = tl.exp(-log_chi)
    rho_lam = rho_gamma * inv_chi
    gamma = tl.load(GAMMA + row).to(tl.float32)
    lam = tl.load(LAM + row).to(tl.float32)
    dgamma = tl.load(DGAMMA + row).to(tl.float32)
    dlam = tl.load(DLAM + row).to(tl.float32)
    d_rho_lam = dlam * (1.0 - lam)
    d_rho_gamma = -dgamma * gamma + d_rho_lam * inv_chi
    d_log_chi = -d_rho_lam * rho_lam
    tl.store(DGATE_LOGITS + gate_offset, d_rho_gamma * tl.sigmoid(gamma_logit))
    tl.store(
        DGATE_LOGITS + gate_offset + H,
        d_log_chi * tl.sigmoid(ratio_logit),
    )


@triton.jit
def _channel_preprocess_fwd_kernel(
    Q,
    K_IN,
    B_LOGITS,
    W_LOGITS,
    GATE_LOGITS,
    Q_OUT,
    K_OUT,
    B_OUT,
    W_OUT,
    GAMMA,
    LAM,
    RSTD_Q,
    RSTD_K,
    N: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    H: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
    LOG_CHI_MIN: tl.constexpr,
    B_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    dcols = tl.arange(0, BD)
    dmask = dcols < D
    doffsets = row * D + dcols
    vcols = tl.arange(0, BV)
    vmask = vcols < V
    voffsets = row * V + vcols

    q = tl.load(Q + doffsets, mask=dmask, other=0.0).to(tl.float32)
    k = tl.load(K_IN + doffsets, mask=dmask, other=0.0).to(tl.float32)
    rq = tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6)
    rk = tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
    tl.store(Q_OUT + doffsets, q * rq, mask=dmask)
    tl.store(K_OUT + doffsets, k * rk, mask=dmask)
    tl.store(RSTD_Q + row, rq)
    tl.store(RSTD_K + row, rk)

    b = tl.sigmoid(
        tl.load(B_LOGITS + doffsets, mask=dmask, other=0.0).to(tl.float32)
    ) * B_SCALE
    w = tl.sigmoid(
        tl.load(W_LOGITS + voffsets, mask=vmask, other=0.0).to(tl.float32)
    )
    tl.store(B_OUT + doffsets, b, mask=dmask)
    tl.store(W_OUT + voffsets, w, mask=vmask)

    token = row // H
    head = row - token * H
    gate_offset = token * (2 * H) + head
    gamma_logit = tl.load(GATE_LOGITS + gate_offset).to(tl.float32)
    ratio_logit = tl.load(GATE_LOGITS + gate_offset + H).to(tl.float32)
    rho_gamma = _stable_softplus(gamma_logit)
    log_chi = LOG_CHI_MIN + _stable_softplus(ratio_logit)
    rho_lam = rho_gamma * tl.exp(-log_chi)
    tl.store(GAMMA + row, tl.exp(-rho_gamma))
    tl.store(LAM + row, 1.0 - tl.exp(-rho_lam))


@triton.jit
def _channel_preprocess_bwd_kernel(
    Q_OUT,
    K_OUT,
    B_OUT,
    W_OUT,
    GAMMA,
    LAM,
    GATE_LOGITS,
    RSTD_Q,
    RSTD_K,
    DQ_OUT,
    DK_OUT,
    DB_OUT,
    DW_OUT,
    DGAMMA,
    DLAM,
    DQ,
    DK,
    DB_LOGITS,
    DW_LOGITS,
    DGATE_LOGITS,
    N: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    H: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
    LOG_CHI_MIN: tl.constexpr,
    B_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    dcols = tl.arange(0, BD)
    dmask = dcols < D
    doffsets = row * D + dcols
    vcols = tl.arange(0, BV)
    vmask = vcols < V
    voffsets = row * V + vcols

    q = tl.load(Q_OUT + doffsets, mask=dmask, other=0.0).to(tl.float32)
    k = tl.load(K_OUT + doffsets, mask=dmask, other=0.0).to(tl.float32)
    dq = tl.load(DQ_OUT + doffsets, mask=dmask, other=0.0).to(tl.float32)
    dk = tl.load(DK_OUT + doffsets, mask=dmask, other=0.0).to(tl.float32)
    rq = tl.load(RSTD_Q + row).to(tl.float32)
    rk = tl.load(RSTD_K + row).to(tl.float32)
    tl.store(
        DQ + doffsets,
        (dq - tl.sum(dq * q, axis=0) * q) * rq,
        mask=dmask,
    )
    tl.store(
        DK + doffsets,
        (dk - tl.sum(dk * k, axis=0) * k) * rk,
        mask=dmask,
    )

    b = tl.load(B_OUT + doffsets, mask=dmask, other=0.0).to(tl.float32) / B_SCALE
    db = tl.load(DB_OUT + doffsets, mask=dmask, other=0.0).to(tl.float32)
    w = tl.load(W_OUT + voffsets, mask=vmask, other=0.0).to(tl.float32)
    dw = tl.load(DW_OUT + voffsets, mask=vmask, other=0.0).to(tl.float32)
    tl.store(DB_LOGITS + doffsets, db * B_SCALE * b * (1.0 - b), mask=dmask)
    tl.store(DW_LOGITS + voffsets, dw * w * (1.0 - w), mask=vmask)

    token = row // H
    head = row - token * H
    gate_offset = token * (2 * H) + head
    gamma_logit = tl.load(GATE_LOGITS + gate_offset).to(tl.float32)
    ratio_logit = tl.load(GATE_LOGITS + gate_offset + H).to(tl.float32)
    rho_gamma = _stable_softplus(gamma_logit)
    log_chi = LOG_CHI_MIN + _stable_softplus(ratio_logit)
    inv_chi = tl.exp(-log_chi)
    rho_lam = rho_gamma * inv_chi
    gamma = tl.load(GAMMA + row).to(tl.float32)
    lam = tl.load(LAM + row).to(tl.float32)
    dgamma = tl.load(DGAMMA + row).to(tl.float32)
    dlam = tl.load(DLAM + row).to(tl.float32)
    d_rho_lam = dlam * (1.0 - lam)
    d_rho_gamma = -dgamma * gamma + d_rho_lam * inv_chi
    d_log_chi = -d_rho_lam * rho_lam
    tl.store(DGATE_LOGITS + gate_offset, d_rho_gamma * tl.sigmoid(gamma_logit))
    tl.store(
        DGATE_LOGITS + gate_offset + H,
        d_log_chi * tl.sigmoid(ratio_logit),
    )


def _empty_grad(grad: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    return grad.contiguous() if grad is not None else torch.zeros_like(like)


class _ScalarPreprocess(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, beta_logits, gate_logits, chi_min, beta_scale, num_warps
    ):
        if num_warps not in (2, 4, 8):
            raise ValueError("scalar preprocess num_warps must be 2, 4, or 8")
        q, k, beta_logits, gate_logits = (
            tensor.contiguous() for tensor in (q, k, beta_logits, gate_logits)
        )
        if q.shape != k.shape or q.shape[:-1] != beta_logits.shape:
            raise ValueError("scalar preprocess shape mismatch")
        *prefix, heads, key_dim = q.shape
        if gate_logits.shape != (*prefix, 2 * heads):
            raise ValueError("ordered gate logits must have 2H channels")
        rows = q.numel() // key_dim
        q_out, k_out = torch.empty_like(q), torch.empty_like(k)
        beta = torch.empty_like(beta_logits)
        gamma = torch.empty_like(beta_logits, dtype=torch.float32)
        lam = torch.empty_like(beta_logits, dtype=torch.float32)
        rstd_q = torch.empty(rows, device=q.device, dtype=torch.float32)
        rstd_k = torch.empty_like(rstd_q)
        block = triton.next_power_of_2(key_dim)
        _scalar_preprocess_fwd_kernel[(rows,)](
            q,
            k,
            beta_logits,
            gate_logits,
            q_out,
            k_out,
            beta,
            gamma,
            lam,
            rstd_q,
            rstd_k,
            N=rows,
            D=key_dim,
            H=heads,
            BD=block,
            LOG_CHI_MIN=math.log(float(chi_min)),
            BETA_SCALE=float(beta_scale),
            num_warps=num_warps,
        )
        ctx.save_for_backward(
            q_out, k_out, beta, gamma, lam, gate_logits, rstd_q, rstd_k
        )
        ctx.constants = (
            rows,
            key_dim,
            heads,
            float(chi_min),
            float(beta_scale),
            num_warps,
        )
        return q_out, k_out, beta, gamma, lam

    @staticmethod
    def backward(ctx, dq_out, dk_out, dbeta, dgamma, dlam):
        q_out, k_out, beta, gamma, lam, gate_logits, rstd_q, rstd_k = ctx.saved_tensors
        rows, key_dim, heads, chi_min, beta_scale, num_warps = ctx.constants
        dq_out = _empty_grad(dq_out, q_out)
        dk_out = _empty_grad(dk_out, k_out)
        dbeta = _empty_grad(dbeta, beta)
        dgamma = _empty_grad(dgamma, gamma)
        dlam = _empty_grad(dlam, lam)
        dq, dk = torch.empty_like(q_out), torch.empty_like(k_out)
        dbeta_logits = torch.empty_like(beta)
        dgate_logits = torch.empty_like(gate_logits)
        block = triton.next_power_of_2(key_dim)
        _scalar_preprocess_bwd_kernel[(rows,)](
            q_out,
            k_out,
            beta,
            gamma,
            lam,
            gate_logits,
            rstd_q,
            rstd_k,
            dq_out,
            dk_out,
            dbeta,
            dgamma,
            dlam,
            dq,
            dk,
            dbeta_logits,
            dgate_logits,
            N=rows,
            D=key_dim,
            H=heads,
            BD=block,
            LOG_CHI_MIN=math.log(chi_min),
            BETA_SCALE=beta_scale,
            num_warps=num_warps,
        )
        return dq, dk, dbeta_logits, dgate_logits, None, None, None


class _ChannelPreprocess(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        b_logits,
        w_logits,
        gate_logits,
        chi_min,
        b_scale,
        num_warps,
    ):
        if num_warps not in (2, 4, 8):
            raise ValueError("channel preprocess num_warps must be 2, 4, or 8")
        q, k, b_logits, w_logits, gate_logits = (
            tensor.contiguous()
            for tensor in (q, k, b_logits, w_logits, gate_logits)
        )
        if q.shape != k.shape or q.shape != b_logits.shape:
            raise ValueError("channel preprocess Q/K/B shape mismatch")
        if q.shape[:-1] != w_logits.shape[:-1]:
            raise ValueError("channel preprocess W prefix mismatch")
        *prefix, heads, key_dim = q.shape
        value_dim = w_logits.shape[-1]
        if gate_logits.shape != (*prefix, 2 * heads):
            raise ValueError("ordered gate logits must have 2H channels")
        rows = q.numel() // key_dim
        q_out, k_out = torch.empty_like(q), torch.empty_like(k)
        b_out, w_out = torch.empty_like(b_logits), torch.empty_like(w_logits)
        gamma = torch.empty((*prefix, heads), device=q.device, dtype=torch.float32)
        lam = torch.empty_like(gamma)
        rstd_q = torch.empty(rows, device=q.device, dtype=torch.float32)
        rstd_k = torch.empty_like(rstd_q)
        block_d = triton.next_power_of_2(key_dim)
        block_v = triton.next_power_of_2(value_dim)
        _channel_preprocess_fwd_kernel[(rows,)](
            q,
            k,
            b_logits,
            w_logits,
            gate_logits,
            q_out,
            k_out,
            b_out,
            w_out,
            gamma,
            lam,
            rstd_q,
            rstd_k,
            N=rows,
            D=key_dim,
            V=value_dim,
            H=heads,
            BD=block_d,
            BV=block_v,
            LOG_CHI_MIN=math.log(float(chi_min)),
            B_SCALE=float(b_scale),
            num_warps=num_warps,
        )
        ctx.save_for_backward(
            q_out,
            k_out,
            b_out,
            w_out,
            gamma,
            lam,
            gate_logits,
            rstd_q,
            rstd_k,
        )
        ctx.constants = (
            rows,
            key_dim,
            value_dim,
            heads,
            float(chi_min),
            float(b_scale),
            num_warps,
        )
        return q_out, k_out, b_out, w_out, gamma, lam

    @staticmethod
    def backward(ctx, dq_out, dk_out, db_out, dw_out, dgamma, dlam):
        (
            q_out,
            k_out,
            b_out,
            w_out,
            gamma,
            lam,
            gate_logits,
            rstd_q,
            rstd_k,
        ) = ctx.saved_tensors
        (
            rows,
            key_dim,
            value_dim,
            heads,
            chi_min,
            b_scale,
            num_warps,
        ) = ctx.constants
        dq_out = _empty_grad(dq_out, q_out)
        dk_out = _empty_grad(dk_out, k_out)
        db_out = _empty_grad(db_out, b_out)
        dw_out = _empty_grad(dw_out, w_out)
        dgamma = _empty_grad(dgamma, gamma)
        dlam = _empty_grad(dlam, lam)
        dq, dk = torch.empty_like(q_out), torch.empty_like(k_out)
        db_logits, dw_logits = torch.empty_like(b_out), torch.empty_like(w_out)
        dgate_logits = torch.empty_like(gate_logits)
        block_d = triton.next_power_of_2(key_dim)
        block_v = triton.next_power_of_2(value_dim)
        _channel_preprocess_bwd_kernel[(rows,)](
            q_out,
            k_out,
            b_out,
            w_out,
            gamma,
            lam,
            gate_logits,
            rstd_q,
            rstd_k,
            dq_out,
            dk_out,
            db_out,
            dw_out,
            dgamma,
            dlam,
            dq,
            dk,
            db_logits,
            dw_logits,
            dgate_logits,
            N=rows,
            D=key_dim,
            V=value_dim,
            H=heads,
            BD=block_d,
            BV=block_v,
            LOG_CHI_MIN=math.log(chi_min),
            B_SCALE=b_scale,
            num_warps=num_warps,
        )
        return dq, dk, db_logits, dw_logits, dgate_logits, None, None, None


def _ordered_rates(gate_logits: torch.Tensor, chi_min: float):
    gamma_logits, ratio_logits = gate_logits.float().chunk(2, dim=-1)
    rho_gamma = F.softplus(gamma_logits)
    log_chi = math.log(chi_min) + F.softplus(ratio_logits)
    rho_lam = rho_gamma * torch.exp(-log_chi)
    return torch.exp(-rho_gamma), -torch.expm1(-rho_lam)


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    rstd = torch.rsqrt(tensor.float().square().sum(dim=-1, keepdim=True) + 1e-6)
    return (tensor.float() * rstd).to(tensor.dtype)


def fused_scalar_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    beta_logits: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    chi_min: float,
    beta_scale: float = 1.0,
):
    """Normalize Q/K and activate scalar erase + ordered state gates."""

    if q.is_cuda and q.shape[-1] <= 512:
        return _ScalarPreprocess.apply(
            q,
            k,
            beta_logits,
            gate_logits,
            chi_min,
            beta_scale,
            _SCALAR_WARPS,
        )
    gamma, lam = _ordered_rates(gate_logits, chi_min)
    beta = torch.sigmoid(beta_logits) * beta_scale
    return _normalize(q), _normalize(k), beta, gamma, lam


def fused_channel_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    b_logits: torch.Tensor,
    w_logits: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    chi_min: float,
    b_scale: float = 1.0,
):
    """Normalize Q/K and activate channel erase/write + ordered gates."""

    if q.is_cuda and max(q.shape[-1], w_logits.shape[-1]) <= 512:
        return _ChannelPreprocess.apply(
            q,
            k,
            b_logits,
            w_logits,
            gate_logits,
            chi_min,
            b_scale,
            _CHANNEL_WARPS,
        )
    gamma, lam = _ordered_rates(gate_logits, chi_min)
    b = torch.sigmoid(b_logits) * b_scale
    w = torch.sigmoid(w_logits)
    return _normalize(q), _normalize(k), b, w, gamma, lam


__all__ = ["fused_channel_preprocess", "fused_scalar_preprocess"]

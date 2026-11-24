"""Exact chunkwise Integral DeltaNet implementation.

This is equations (augmented recurrence, triangular system, boundary update,
and post-update output) from ``integral_deltanet.tex``.  The implementation
uses the compressed two-``K`` representation: it never materializes vectors of
width ``2*K`` and delegates only the sequential inter-chunk boundary scan to a
single Triton launch.
"""
from __future__ import annotations

import contextlib

import torch

from .arch import optimized_readout
from .reference import _validate_inputs
from .solve import solve_unit_lower


def _prefix_2x2(p11, p12, p21, p22):
    """Inclusive non-commutative prefix product in log2(C) parallel stages."""
    chunk_size = p11.shape[-2]
    f11, f12, f21, f22 = p11, p12, p21, p22
    offset = 1
    while offset < chunk_size:
        prefix_shape = (*f11.shape[:-2], offset, f11.shape[-1])
        one = torch.ones(prefix_shape, dtype=f11.dtype, device=f11.device)
        zero = torch.zeros_like(one)
        left11 = torch.cat((one, f11[..., :-offset, :]), dim=-2)
        left12 = torch.cat((zero, f12[..., :-offset, :]), dim=-2)
        left21 = torch.cat((zero, f21[..., :-offset, :]), dim=-2)
        left22 = torch.cat((one, f22[..., :-offset, :]), dim=-2)
        next11 = left11 * f11 + left12 * f21
        next12 = left11 * f12 + left12 * f22
        next21 = left21 * f11 + left22 * f21
        next22 = left21 * f12 + left22 * f22
        f11, f12, f21, f22 = next11, next12, next21, next22
        offset *= 2
    return f11, f12, f21, f22


def _chunk(tensor: torch.Tensor, chunks: int, chunk_size: int) -> torch.Tensor:
    batch, _, heads, width = tensor.shape
    return tensor.transpose(1, 2).reshape(batch, heads, chunks, chunk_size, width)


def _build_operands(qc, kc, vc, gc, bc, wc, gamma, lam, scale):
    """Build all chunk-independent operands from the paper's exact formulas."""
    acc_dtype = torch.float64 if qc.dtype == torch.float64 else torch.float32
    gamma, lam = (tensor.to(acc_dtype) for tensor in (gamma, lam))
    log2_e = 1.4426950408889634 if acc_dtype == torch.float64 else 1.4426950216
    alpha = torch.exp2(gc.to(acc_dtype) * log2_e)
    erase = bc * kc
    write = wc * vc
    qc = qc * scale
    chunk_size = qc.shape[-2]
    value_dim, key_dim = vc.shape[-1], kc.shape[-1]
    gamma_key = gamma.unsqueeze(-1)
    lambda_key = lam.unsqueeze(-1)

    p11 = gamma_key * alpha
    p12 = lambda_key * p11
    p21 = (1.0 - gamma_key) * alpha
    p22 = (1.0 - lambda_key) + lambda_key * p21
    f11, f12, f21, f22 = _prefix_2x2(p11, p12, p21, p22)

    one = torch.ones_like(f11[..., :1, :])
    zero = torch.zeros_like(one)
    previous11 = torch.cat((one, f11[..., :-1, :]), dim=-2)
    previous21 = torch.cat((zero, f21[..., :-1, :]), dim=-2)

    alpha_erase = alpha * erase
    abar1 = previous11 * alpha_erase
    abar2 = previous21 * alpha_erase

    # det(F_r) is a product of known positive per-token determinants.  This is
    # more accurate than f11*f22-f12*f21 near a poorly conditioned boundary.
    det_step = gamma_key * alpha * (1.0 - lambda_key)
    determinant = det_step.cumprod(dim=-2)
    tiny = torch.finfo(acc_dtype).tiny
    determinant = determinant.clamp_min(tiny)
    sigma1 = (f22 - lambda_key * f21) / determinant
    sigma2 = (-f12 + lambda_key * f11) / determinant
    cbar1, cbar2 = sigma1 * kc, sigma2 * kc
    qbar1, qbar2 = f11 * qc, f21 * qc

    final11, final12 = f11[..., -1, :], f12[..., -1, :]
    final21, final22 = f21[..., -1, :], f22[..., -1, :]
    tau1 = sigma1 * final11.unsqueeze(-2) + sigma2 * final21.unsqueeze(-2)
    tau2 = sigma1 * final12.unsqueeze(-2) + sigma2 * final22.unsqueeze(-2)
    tail1, tail2 = tau1 * kc, tau2 * kc

    data_dtype = torch.bfloat16 if qc.is_cuda and acc_dtype == torch.float32 else acc_dtype
    a1d, a2d = abar1.to(data_dtype), abar2.to(data_dtype)
    c1d, c2d = cbar1.to(data_dtype), cbar2.to(data_dtype)
    q1d, q2d = qbar1.to(data_dtype), qbar2.to(data_dtype)
    interaction = (a1d @ c1d.transpose(-1, -2)).to(acc_dtype)
    interaction += (a2d @ c2d.transpose(-1, -2)).to(acc_dtype)
    causal_read = (q1d @ c1d.transpose(-1, -2)).to(acc_dtype)
    causal_read += (q2d @ c2d.transpose(-1, -2)).to(acc_dtype)

    strict_lower = torch.tril(interaction, diagonal=-1)
    if qc.is_cuda and chunk_size == 16 and acc_dtype == torch.float32:
        from .general_triton import solve_general_operands

        u0, state_weight1, state_weight2 = solve_general_operands(
            strict_lower, write, abar1, abar2
        )
    else:
        rhs = torch.cat((write, abar1, abar2), dim=-1)
        solution = solve_unit_lower(
            strict_lower,
            rhs,
            matrix_is_strict_lower=True,
        )
        u0, state_weight1, state_weight2 = solution.split(
            (value_dim, key_dim, key_dim), dim=-1
        )
    causal_read = torch.tril(causal_read)
    final_transport = torch.stack((final11, final12, final21, final22), dim=-1)
    if data_dtype == torch.bfloat16:
        qbar1, qbar2 = qbar1.to(data_dtype), qbar2.to(data_dtype)
        tail1, tail2 = tail1.to(data_dtype), tail2.to(data_dtype)
        causal_read = causal_read.to(data_dtype)
    return (
        u0,
        state_weight1,
        state_weight2,
        qbar1,
        qbar2,
        tail1,
        tail2,
        causal_read,
        final_transport,
    )


def _build_shared_operands(qc, kc, vc, decay, beta, gamma, lam, scale):
    """Compressed DeltaNet/GDN operands with one passive prefix per head.

    ``decay`` and ``beta`` have shape ``[..., C]``. Unlike the general
    diagonal path, the four entries of every 2x2 prefix are computed once and
    shared by all key channels. The two interaction matrices use the paper's
    Hadamard-factorized form, so transformed vectors of width ``2 * K`` are
    never materialized.
    """
    acc_dtype = torch.float64 if qc.dtype == torch.float64 else torch.float32
    data_dtype = torch.bfloat16 if qc.is_cuda and acc_dtype == torch.float32 else acc_dtype
    decay, beta, gamma, lam = (
        tensor.to(acc_dtype) for tensor in (decay, beta, gamma, lam)
    )
    qc = qc * scale
    value_dim, key_dim = vc.shape[-1], kc.shape[-1]

    p11 = gamma * decay
    p12 = lam * p11
    p21 = (1.0 - gamma) * decay
    p22 = (1.0 - lam) + lam * p21
    f11, f12, f21, f22 = (
        tensor.squeeze(-1)
        for tensor in _prefix_2x2(
            p11.unsqueeze(-1),
            p12.unsqueeze(-1),
            p21.unsqueeze(-1),
            p22.unsqueeze(-1),
        )
    )

    one = torch.ones_like(f11[..., :1])
    zero = torch.zeros_like(one)
    previous11 = torch.cat((one, f11[..., :-1]), dim=-1)
    previous21 = torch.cat((zero, f21[..., :-1]), dim=-1)
    pi1 = previous11 * decay * beta
    pi2 = previous21 * decay * beta

    determinant = (gamma * decay * (1.0 - lam)).cumprod(dim=-1)
    determinant = determinant.clamp_min(torch.finfo(acc_dtype).tiny)
    sigma1 = (f22 - lam * f21) / determinant
    sigma2 = (-f12 + lam * f11) / determinant
    eta1, eta2 = f11, f21

    final11, final12 = f11[..., -1], f12[..., -1]
    final21, final22 = f21[..., -1], f22[..., -1]
    tau1 = sigma1 * final11.unsqueeze(-1) + sigma2 * final21.unsqueeze(-1)
    tau2 = sigma1 * final12.unsqueeze(-1) + sigma2 * final22.unsqueeze(-1)

    kd = kc.to(data_dtype)
    qd = qc.to(data_dtype)
    key_gram = (kd @ kd.transpose(-1, -2)).to(acc_dtype)
    query_key = (qd @ kd.transpose(-1, -2)).to(acc_dtype)
    interaction_coeff = (
        pi1.unsqueeze(-1) * sigma1.unsqueeze(-2)
        + pi2.unsqueeze(-1) * sigma2.unsqueeze(-2)
    )
    output_coeff = (
        eta1.unsqueeze(-1) * sigma1.unsqueeze(-2)
        + eta2.unsqueeze(-1) * sigma2.unsqueeze(-2)
    )
    strict_lower = torch.tril(interaction_coeff * key_gram, diagonal=-1)
    causal_read = torch.tril(output_coeff * query_key)

    abar1 = pi1.unsqueeze(-1) * kc
    abar2 = pi2.unsqueeze(-1) * kc
    write = beta.unsqueeze(-1) * vc
    solution = solve_unit_lower(
        strict_lower,
        torch.cat((write, abar1, abar2), dim=-1),
        matrix_is_strict_lower=True,
    )
    u0, state_weight1, state_weight2 = solution.split(
        (value_dim, key_dim, key_dim), dim=-1
    )
    # Both boundary reads use the same query.  Keep one Q data plane and two
    # scalar row coefficients instead of materializing two [C, K] tensors.
    # The extra singleton dimension is an unambiguous internal tag consumed
    # by the eager/Triton scans (and remains valid even when K == 2).
    qbar1 = qc
    qbar2 = torch.stack((eta1, eta2), dim=-1).unsqueeze(-2)
    tail1 = tau1.unsqueeze(-1) * kc
    tail2 = tau2.unsqueeze(-1) * kc
    final_transport = torch.stack(
        (final11, final12, final21, final22), dim=-1
    )
    if data_dtype == torch.bfloat16:
        qbar1 = qbar1.to(data_dtype)
        tail1, tail2 = tail1.to(data_dtype), tail2.to(data_dtype)
        causal_read = causal_read.to(data_dtype)
    return (
        u0,
        state_weight1,
        state_weight2,
        qbar1,
        qbar2,
        tail1,
        tail2,
        causal_read,
        final_transport,
    )


def _build_gdn_operands(qc, kc, vc, gc, beta, gamma, lam, scale):
    """Build scalar-GDN operands without an inverse prefix.

    The normalized derivation factors every relative transport through
    ``F_s^{-1}``. That factorization is algebraically exact but can be
    catastrophically ill-conditioned for the released GDN initialization,
    even while the original two-state recurrence remains contractive. Build
    the two scalar relative transports directly instead. This is the eager
    correctness path; CUDA BF16 uses the equivalent Triton implementation.
    """
    acc_dtype = torch.float64 if qc.dtype == torch.float64 else torch.float32
    log2_e = 1.4426950408889634 if acc_dtype == torch.float64 else 1.4426950216
    decay = torch.exp2(gc.to(acc_dtype) * log2_e)
    beta, gamma, lam = (
        tensor.to(acc_dtype) for tensor in (beta, gamma, lam)
    )
    qc = qc * scale
    chunk_size = qc.shape[-2]
    value_dim, key_dim = vc.shape[-1], kc.shape[-1]

    p11 = gamma * decay
    p12 = lam * p11
    p21 = (1.0 - gamma) * decay
    p22 = (1.0 - lam) + lam * p21

    # Absolute prefixes are only multiplied, never inverted. They are needed
    # for the chunk-boundary response to the incoming active/slow states.
    f11 = torch.ones_like(decay[..., 0])
    f12 = torch.zeros_like(f11)
    f21 = torch.zeros_like(f11)
    f22 = torch.ones_like(f11)
    pi1_rows, pi2_rows = [], []
    eta1_rows, eta2_rows = [], []
    for target in range(chunk_size):
        alpha_beta = decay[..., target] * beta[..., target]
        pi1_rows.append(f11 * alpha_beta)
        pi2_rows.append(f21 * alpha_beta)
        next11 = f11 * p11[..., target] + f12 * p21[..., target]
        next12 = f11 * p12[..., target] + f12 * p22[..., target]
        next21 = f21 * p11[..., target] + f22 * p21[..., target]
        next22 = f21 * p12[..., target] + f22 * p22[..., target]
        f11, f12, f21, f22 = next11, next12, next21, next22
        eta1_rows.append(f11)
        eta2_rows.append(f21)
    pi1 = torch.stack(pi1_rows, dim=-1)
    pi2 = torch.stack(pi2_rows, dim=-1)
    eta1 = torch.stack(eta1_rows, dim=-1)
    eta2 = torch.stack(eta2_rows, dim=-1)

    # For an edit at source s, c=[1, lambda_s] is transported only by
    # P_{s+1}, ..., P_r. Keeping these contractive relative products avoids
    # the enormous cancelling factors produced by c F_s^{-1}.
    sources = torch.arange(chunk_size, device=qc.device)
    c1 = torch.ones_like(lam)
    c2 = lam
    interaction_rows = []
    causal_rows = []
    for target in range(chunk_size):
        earlier = sources < target
        interaction_rows.append(
            torch.where(
                earlier,
                c1
                * (decay[..., target] * beta[..., target]).unsqueeze(-1),
                torch.zeros_like(c1),
            )
        )
        next1 = (
            c1 * p11[..., target].unsqueeze(-1)
            + c2 * p21[..., target].unsqueeze(-1)
        )
        next2 = (
            c1 * p12[..., target].unsqueeze(-1)
            + c2 * p22[..., target].unsqueeze(-1)
        )
        c1 = torch.where(earlier, next1, c1)
        c2 = torch.where(earlier, next2, c2)
        causal_rows.append(
            torch.where(
                earlier,
                c1,
                torch.where(
                    sources == target,
                    torch.ones_like(c1),
                    torch.zeros_like(c1),
                ),
            )
        )
    interaction_coeff = torch.stack(interaction_rows, dim=-2)
    output_coeff = torch.stack(causal_rows, dim=-2)
    tail1, tail2 = c1, c2

    data_dtype = (
        torch.bfloat16
        if qc.is_cuda and acc_dtype == torch.float32
        else acc_dtype
    )
    kd = kc.to(data_dtype)
    qd = qc.to(data_dtype)
    key_gram = (kd @ kd.transpose(-1, -2)).to(acc_dtype)
    query_key = (qd @ kd.transpose(-1, -2)).to(acc_dtype)
    interaction = interaction_coeff * key_gram
    causal_read = output_coeff * query_key
    abar1 = pi1.unsqueeze(-1) * kc
    abar2 = pi2.unsqueeze(-1) * kc
    write = beta.unsqueeze(-1) * vc
    solution = solve_unit_lower(
        interaction,
        torch.cat((write, abar1, abar2), dim=-1),
        matrix_is_strict_lower=True,
    )
    u0, state_weight1, state_weight2 = solution.split(
        (value_dim, key_dim, key_dim), dim=-1
    )
    qbar1 = qc
    qbar2 = torch.stack((eta1, eta2), dim=-1).unsqueeze(-2)
    tail1 = tail1.unsqueeze(-1) * kc
    tail2 = tail2.unsqueeze(-1) * kc
    final_transport = torch.stack((f11, f12, f21, f22), dim=-1)
    if data_dtype == torch.bfloat16:
        qbar1 = qbar1.to(data_dtype)
        tail1 = tail1.to(data_dtype)
        tail2 = tail2.to(data_dtype)
        causal_read = causal_read.to(data_dtype)
    return (
        u0,
        state_weight1,
        state_weight2,
        qbar1,
        qbar2,
        tail1,
        tail2,
        causal_read,
        final_transport,
    )


def _build_deltanet_operands(qc, kc, vc, beta, gamma, lam, scale):
    """DeltaNet slow/difference-basis specialization from the appendix.

    The passive prefix ``[[1, 0], [h, g]]`` is evaluated with one cumulative
    product and one cumulative sum; no general 2x2 prefix is formed.
    """
    acc_dtype = torch.float64 if qc.dtype == torch.float64 else torch.float32
    data_dtype = torch.bfloat16 if qc.is_cuda and acc_dtype == torch.float32 else acc_dtype
    beta, gamma, lam = (tensor.to(acc_dtype) for tensor in (beta, gamma, lam))
    qc = qc * scale
    value_dim, key_dim = vc.shape[-1], kc.shape[-1]

    g_prefix = ((1.0 - lam) * gamma).cumprod(dim=-1)
    one = torch.ones_like(g_prefix[..., :1])
    g_previous = torch.cat((one, g_prefix[..., :-1]), dim=-1)
    h_prefix = (g_previous * lam * gamma).cumsum(dim=-1)
    zero = torch.zeros_like(h_prefix[..., :1])
    h_previous = torch.cat((zero, h_prefix[..., :-1]), dim=-1)

    pi1 = beta
    pi2 = beta * (h_previous + g_previous)
    safe_g = g_prefix.clamp_min(torch.finfo(acc_dtype).tiny)
    sigma2 = (1.0 - lam) / safe_g
    sigma1 = lam - h_prefix * sigma2
    eta1 = torch.ones_like(h_prefix)
    eta2 = h_prefix + g_prefix
    final_h, final_g = h_prefix[..., -1], g_prefix[..., -1]
    tau1 = sigma1 + final_h.unsqueeze(-1) * sigma2
    tau2 = final_g.unsqueeze(-1) * sigma2

    kd = kc.to(data_dtype)
    qd = qc.to(data_dtype)
    key_gram = (kd @ kd.transpose(-1, -2)).to(acc_dtype)
    query_key = (qd @ kd.transpose(-1, -2)).to(acc_dtype)
    interaction_coeff = (
        pi1.unsqueeze(-1) * sigma1.unsqueeze(-2)
        + pi2.unsqueeze(-1) * sigma2.unsqueeze(-2)
    )
    output_coeff = (
        eta1.unsqueeze(-1) * sigma1.unsqueeze(-2)
        + eta2.unsqueeze(-1) * sigma2.unsqueeze(-2)
    )
    strict_lower = torch.tril(interaction_coeff * key_gram, diagonal=-1)
    causal_read = torch.tril(output_coeff * query_key)

    abar1 = pi1.unsqueeze(-1) * kc
    abar2 = pi2.unsqueeze(-1) * kc
    write = beta.unsqueeze(-1) * vc
    solution = solve_unit_lower(
        strict_lower,
        torch.cat((write, abar1, abar2), dim=-1),
        matrix_is_strict_lower=True,
    )
    u0, state_weight1, state_weight2 = solution.split(
        (value_dim, key_dim, key_dim), dim=-1
    )
    # Shared-query encoding; see _build_shared_operands above.
    qbar1 = qc
    qbar2 = torch.stack((eta1, eta2), dim=-1).unsqueeze(-2)
    tail1 = tau1.unsqueeze(-1) * kc
    tail2 = tau2.unsqueeze(-1) * kc
    final_transport = torch.stack(
        (
            torch.ones_like(final_h),
            torch.zeros_like(final_h),
            final_h,
            final_g,
        ),
        dim=-1,
    )
    if data_dtype == torch.bfloat16:
        qbar1 = qbar1.to(data_dtype)
        tail1, tail2 = tail1.to(data_dtype), tail2.to(data_dtype)
        causal_read = causal_read.to(data_dtype)
    return (
        u0,
        state_weight1,
        state_weight2,
        qbar1,
        qbar2,
        tail1,
        tail2,
        causal_read,
        final_transport,
    )


def _build_shared_operands_blocked(
    qc, kc, vc, decay, beta, gamma, lam, scale
):
    """Stable C64 builder specialized for head-shared scalar gates.

    This is the shared-gate counterpart of :func:`_build_operands_blocked`.
    It keeps FLA's 16-token numerical centering, but represents every local
    transformed key by a scalar coefficient times the original key.  Each
    block pair therefore needs one raw key Gram GEMM instead of one GEMM per
    integral state, and the 2x2 prefixes are computed once per head.
    """
    chunk_size = qc.shape[-2]
    block_size = 16
    if chunk_size % block_size:
        raise ValueError("blocked chunks must be a multiple of 16")
    blocks = chunk_size // block_size
    acc_dtype = torch.float64 if qc.dtype == torch.float64 else torch.float32
    data_dtype = (
        torch.bfloat16
        if qc.is_cuda and acc_dtype == torch.float32
        else acc_dtype
    )
    outer_shape = qc.shape[:-2]
    key_dim, value_dim = kc.shape[-1], vc.shape[-1]
    decay, beta, gamma, lam = (
        tensor.to(acc_dtype) for tensor in (decay, beta, gamma, lam)
    )
    qc = qc * scale

    def split_vectors(tensor):
        return tensor.reshape(
            *outer_shape, blocks, block_size, tensor.shape[-1]
        )

    def split_scalars(tensor):
        return tensor.reshape(*outer_shape, blocks, block_size)

    q = split_vectors(qc)
    k = split_vectors(kc)
    decay = split_scalars(decay)
    beta = split_scalars(beta)
    gamma = split_scalars(gamma)
    lam = split_scalars(lam)

    p11 = gamma * decay
    p12 = lam * p11
    p21 = (1.0 - gamma) * decay
    p22 = (1.0 - lam) + lam * p21
    l11, l12, l21, l22 = (
        tensor.squeeze(-1)
        for tensor in _prefix_2x2(
            p11.unsqueeze(-1),
            p12.unsqueeze(-1),
            p21.unsqueeze(-1),
            p22.unsqueeze(-1),
        )
    )

    one_token = torch.ones_like(l11[..., :1])
    zero_token = torch.zeros_like(one_token)
    prev11 = torch.cat((one_token, l11[..., :-1]), dim=-1)
    prev21 = torch.cat((zero_token, l21[..., :-1]), dim=-1)
    pi1 = prev11 * decay * beta
    pi2 = prev21 * decay * beta
    eta1, eta2 = l11, l21

    determinant = (gamma * decay * (1.0 - lam)).cumprod(dim=-1)
    determinant = determinant.clamp_min(torch.finfo(acc_dtype).tiny)
    sigma1 = (l22 - lam * l21) / determinant
    sigma2 = (-l12 + lam * l11) / determinant

    block11, block12 = l11[..., -1], l12[..., -1]
    block21, block22 = l21[..., -1], l22[..., -1]
    tail_local1 = sigma1 * block11.unsqueeze(-1) + sigma2 * block21.unsqueeze(-1)
    tail_local2 = sigma1 * block12.unsqueeze(-1) + sigma2 * block22.unsqueeze(-1)

    prefix11, prefix12, prefix21, prefix22 = (
        tensor.squeeze(-1)
        for tensor in _prefix_2x2(
            block11.unsqueeze(-1),
            block12.unsqueeze(-1),
            block21.unsqueeze(-1),
            block22.unsqueeze(-1),
        )
    )
    one_block = torch.ones_like(prefix11[..., :1])
    zero_block = torch.zeros_like(one_block)
    before11 = torch.cat((one_block, prefix11[..., :-1]), dim=-1).unsqueeze(-1)
    before12 = torch.cat((zero_block, prefix12[..., :-1]), dim=-1).unsqueeze(-1)
    before21 = torch.cat((zero_block, prefix21[..., :-1]), dim=-1).unsqueeze(-1)
    before22 = torch.cat((one_block, prefix22[..., :-1]), dim=-1).unsqueeze(-1)

    global_pi1 = before11 * pi1 + before12 * pi2
    global_pi2 = before21 * pi1 + before22 * pi2
    global_eta1 = before11 * eta1 + before12 * eta2
    global_eta2 = before21 * eta1 + before22 * eta2

    transported_sources = {}
    for source in range(blocks):
        c1 = tail_local1[..., source, :]
        c2 = tail_local2[..., source, :]
        for target in range(source + 1, blocks + 1):
            if target > source + 1:
                middle = target - 1
                c1, c2 = (
                    c1 * block11[..., middle].unsqueeze(-1)
                    + c2 * block21[..., middle].unsqueeze(-1),
                    c1 * block12[..., middle].unsqueeze(-1)
                    + c2 * block22[..., middle].unsqueeze(-1),
                )
            transported_sources[source, target] = (c1, c2)

    # The numerical centering above must remain block-local, but Q/K are not
    # transformed.  Form their raw Gram matrices once at the full outer chunk
    # width so C64 maps to two TensorCore GEMMs instead of twenty 16x16 GEMMs.
    k_flat = k.reshape(*outer_shape, chunk_size, key_dim)
    q_flat = q.reshape(*outer_shape, chunk_size, key_dim)
    kd = k_flat.to(data_dtype)
    qd = q_flat.to(data_dtype)
    key_gram = (kd @ kd.transpose(-1, -2)).to(acc_dtype)
    query_key = (qd @ kd.transpose(-1, -2)).to(acc_dtype)

    zero_matrix = torch.zeros(
        *outer_shape,
        block_size,
        block_size,
        dtype=acc_dtype,
        device=qc.device,
    )
    interaction_coefficient_rows = []
    output_coefficient_rows = []
    for target in range(blocks):
        interaction_coefficient_columns = []
        output_coefficient_columns = []
        for source in range(blocks):
            if source > target:
                interaction_coefficient_columns.append(zero_matrix)
                output_coefficient_columns.append(zero_matrix)
                continue
            if source == target:
                c1 = sigma1[..., source, :]
                c2 = sigma2[..., source, :]
            else:
                c1, c2 = transported_sources[source, target]
            interaction_coeff = (
                pi1[..., target, :].unsqueeze(-1) * c1.unsqueeze(-2)
                + pi2[..., target, :].unsqueeze(-1) * c2.unsqueeze(-2)
            )
            output_coeff = (
                eta1[..., target, :].unsqueeze(-1) * c1.unsqueeze(-2)
                + eta2[..., target, :].unsqueeze(-1) * c2.unsqueeze(-2)
            )
            interaction_coefficient_columns.append(interaction_coeff)
            output_coefficient_columns.append(output_coeff)
        interaction_coefficient_rows.append(
            torch.cat(interaction_coefficient_columns, dim=-1)
        )
        output_coefficient_rows.append(
            torch.cat(output_coefficient_columns, dim=-1)
        )
    interaction_coefficients = torch.cat(
        interaction_coefficient_rows, dim=-2
    )
    output_coefficients = torch.cat(output_coefficient_rows, dim=-2)
    interaction = torch.tril(
        interaction_coefficients * key_gram,
        diagonal=-1,
    )
    causal_read = torch.tril(output_coefficients * query_key)

    global_pi1 = global_pi1.reshape(*outer_shape, chunk_size)
    global_pi2 = global_pi2.reshape(*outer_shape, chunk_size)
    global_eta1 = global_eta1.reshape(*outer_shape, chunk_size)
    global_eta2 = global_eta2.reshape(*outer_shape, chunk_size)
    beta_flat = beta.reshape(*outer_shape, chunk_size)
    write = beta_flat.unsqueeze(-1) * vc
    abar1 = global_pi1.unsqueeze(-1) * k_flat
    abar2 = global_pi2.unsqueeze(-1) * k_flat
    solution = solve_unit_lower(
        interaction,
        torch.cat((write, abar1, abar2), dim=-1),
        matrix_is_strict_lower=True,
    )
    u0, state_weight1, state_weight2 = solution.split(
        (value_dim, key_dim, key_dim), dim=-1
    )
    qbar1 = global_eta1.unsqueeze(-1) * q_flat
    qbar2 = global_eta2.unsqueeze(-1) * q_flat
    tail1 = torch.cat(
        [
            transported_sources[source, blocks][0].unsqueeze(-1)
            * k[..., source, :, :]
            for source in range(blocks)
        ],
        dim=-2,
    )
    tail2 = torch.cat(
        [
            transported_sources[source, blocks][1].unsqueeze(-1)
            * k[..., source, :, :]
            for source in range(blocks)
        ],
        dim=-2,
    )
    final_transport = torch.stack(
        (
            prefix11[..., -1],
            prefix12[..., -1],
            prefix21[..., -1],
            prefix22[..., -1],
        ),
        dim=-1,
    )
    if data_dtype == torch.bfloat16:
        qbar1 = qbar1.to(data_dtype)
        qbar2 = qbar2.to(data_dtype)
        tail1 = tail1.to(data_dtype)
        tail2 = tail2.to(data_dtype)
        causal_read = causal_read.to(data_dtype)
    return (
        u0,
        state_weight1,
        state_weight2,
        qbar1,
        qbar2,
        tail1,
        tail2,
        causal_read,
        final_transport,
    )


def _build_deltanet_operands_blocked(qc, kc, vc, beta, gamma, lam, scale):
    return _build_shared_operands_blocked(
        qc,
        kc,
        vc,
        torch.ones_like(beta),
        beta,
        gamma,
        lam,
        scale,
    )


def _build_gdn_operands_blocked(qc, kc, vc, gc, beta, gamma, lam, scale):
    acc_dtype = torch.float64 if qc.dtype == torch.float64 else torch.float32
    log2_e = 1.4426950408889634 if acc_dtype == torch.float64 else 1.4426950216
    decay = torch.exp2(gc.to(acc_dtype) * log2_e)
    return _build_shared_operands_blocked(
        qc, kc, vc, decay, beta, gamma, lam, scale
    )


def _build_operands_blocked(qc, kc, vc, gc, bc, wc, gamma, lam, scale):
    """Stable outer-chunk builder with FLA-style 16-token subchunks.

    Absolute inverse prefixes are formed only inside each secondary block.
    Off-diagonal blocks use relative 2x2 transports, the non-commutative
    counterpart of FLA's centered ``exp2(g_i - g_j)`` construction.  This
    permits an outer chunk of 32 or 64 without the ill-conditioning of one
    long normalized prefix.
    """
    chunk_size = qc.shape[-2]
    block_size = 16
    if chunk_size % block_size:
        raise ValueError("blocked chunks must be a multiple of 16")
    blocks = chunk_size // block_size
    acc_dtype = torch.float64 if qc.dtype == torch.float64 else torch.float32
    data_dtype = torch.bfloat16 if qc.is_cuda and acc_dtype == torch.float32 else acc_dtype
    outer_shape = qc.shape[:-2]
    key_dim, value_dim = kc.shape[-1], vc.shape[-1]
    gamma, lam = (tensor.to(acc_dtype) for tensor in (gamma, lam))
    log2_e = 1.4426950408889634 if acc_dtype == torch.float64 else 1.4426950216
    alpha = torch.exp2(gc.to(acc_dtype) * log2_e)
    qc = qc * scale

    def split_vectors(tensor):
        return tensor.reshape(*outer_shape, blocks, block_size, tensor.shape[-1])

    q = split_vectors(qc)
    k = split_vectors(kc)
    alpha = split_vectors(alpha)
    erase = split_vectors(bc * kc)
    write = split_vectors(wc * vc)
    gamma_key = gamma.reshape(*outer_shape, blocks, block_size).unsqueeze(-1)
    lambda_key = lam.reshape(*outer_shape, blocks, block_size).unsqueeze(-1)

    p11 = gamma_key * alpha
    p12 = lambda_key * p11
    p21 = (1.0 - gamma_key) * alpha
    p22 = (1.0 - lambda_key) + lambda_key * p21
    l11, l12, l21, l22 = _prefix_2x2(p11, p12, p21, p22)

    one_token = torch.ones_like(l11[..., :1, :])
    zero_token = torch.zeros_like(one_token)
    prev11 = torch.cat((one_token, l11[..., :-1, :]), dim=-2)
    prev21 = torch.cat((zero_token, l21[..., :-1, :]), dim=-2)

    alpha_erase = alpha * erase
    local_a1 = prev11 * alpha_erase
    local_a2 = prev21 * alpha_erase
    local_q1 = l11 * q
    local_q2 = l21 * q

    det_step = gamma_key * alpha * (1.0 - lambda_key)
    determinant = det_step.cumprod(dim=-2).clamp_min(torch.finfo(acc_dtype).tiny)
    sigma1 = (l22 - lambda_key * l21) / determinant
    sigma2 = (-l12 + lambda_key * l11) / determinant
    local_c1 = sigma1 * k
    local_c2 = sigma2 * k

    block11, block12 = l11[..., -1, :], l12[..., -1, :]
    block21, block22 = l21[..., -1, :], l22[..., -1, :]
    tail_local1 = (sigma1 * block11.unsqueeze(-2) + sigma2 * block21.unsqueeze(-2)) * k
    tail_local2 = (sigma1 * block12.unsqueeze(-2) + sigma2 * block22.unsqueeze(-2)) * k

    prefix11, prefix12, prefix21, prefix22 = _prefix_2x2(
        block11, block12, block21, block22
    )
    one_block = torch.ones_like(prefix11[..., :1, :])
    zero_block = torch.zeros_like(one_block)
    before11 = torch.cat((one_block, prefix11[..., :-1, :]), dim=-2).unsqueeze(-2)
    before12 = torch.cat((zero_block, prefix12[..., :-1, :]), dim=-2).unsqueeze(-2)
    before21 = torch.cat((zero_block, prefix21[..., :-1, :]), dim=-2).unsqueeze(-2)
    before22 = torch.cat((one_block, prefix22[..., :-1, :]), dim=-2).unsqueeze(-2)

    global_a1 = before11 * local_a1 + before12 * local_a2
    global_a2 = before21 * local_a1 + before22 * local_a2
    global_q1 = before11 * local_q1 + before12 * local_q2
    global_q2 = before21 * local_q1 + before22 * local_q2

    transported_sources = {}
    for source in range(blocks):
        c1 = tail_local1[..., source, :, :]
        c2 = tail_local2[..., source, :, :]
        for target in range(source + 1, blocks + 1):
            if target > source + 1:
                middle = target - 1
                c1, c2 = (
                    c1 * block11[..., middle, :].unsqueeze(-2)
                    + c2 * block21[..., middle, :].unsqueeze(-2),
                    c1 * block12[..., middle, :].unsqueeze(-2)
                    + c2 * block22[..., middle, :].unsqueeze(-2),
                )
            transported_sources[source, target] = (c1, c2)

    zero_matrix = torch.zeros(
        *outer_shape, block_size, block_size, dtype=acc_dtype, device=qc.device
    )
    interaction_rows = []
    causal_rows = []
    for target in range(blocks):
        interaction_columns = []
        causal_columns = []
        target_a1 = local_a1[..., target, :, :].to(data_dtype)
        target_a2 = local_a2[..., target, :, :].to(data_dtype)
        target_q1 = local_q1[..., target, :, :].to(data_dtype)
        target_q2 = local_q2[..., target, :, :].to(data_dtype)
        for source in range(blocks):
            if source > target:
                interaction_columns.append(zero_matrix)
                causal_columns.append(zero_matrix)
                continue
            if source == target:
                source_c1 = local_c1[..., source, :, :].to(data_dtype)
                source_c2 = local_c2[..., source, :, :].to(data_dtype)
                interaction_block = (target_a1 @ source_c1.transpose(-1, -2)).to(acc_dtype)
                interaction_block += (target_a2 @ source_c2.transpose(-1, -2)).to(acc_dtype)
                causal_block = (target_q1 @ source_c1.transpose(-1, -2)).to(acc_dtype)
                causal_block += (target_q2 @ source_c2.transpose(-1, -2)).to(acc_dtype)
                interaction_columns.append(torch.tril(interaction_block, diagonal=-1))
                causal_columns.append(torch.tril(causal_block))
                continue
            source_c1, source_c2 = transported_sources[source, target]
            source_c1 = source_c1.to(data_dtype)
            source_c2 = source_c2.to(data_dtype)
            interaction_block = (target_a1 @ source_c1.transpose(-1, -2)).to(acc_dtype)
            interaction_block += (target_a2 @ source_c2.transpose(-1, -2)).to(acc_dtype)
            causal_block = (target_q1 @ source_c1.transpose(-1, -2)).to(acc_dtype)
            causal_block += (target_q2 @ source_c2.transpose(-1, -2)).to(acc_dtype)
            interaction_columns.append(interaction_block)
            causal_columns.append(causal_block)
        interaction_rows.append(torch.cat(interaction_columns, dim=-1))
        causal_rows.append(torch.cat(causal_columns, dim=-1))
    interaction = torch.cat(interaction_rows, dim=-2)
    causal_read = torch.cat(causal_rows, dim=-2)

    tail1_blocks = []
    tail2_blocks = []
    for source in range(blocks):
        c1, c2 = transported_sources[source, blocks]
        tail1_blocks.append(c1)
        tail2_blocks.append(c2)
    tail1 = torch.cat(tail1_blocks, dim=-2)
    tail2 = torch.cat(tail2_blocks, dim=-2)

    global_a1 = global_a1.reshape(*outer_shape, chunk_size, key_dim)
    global_a2 = global_a2.reshape(*outer_shape, chunk_size, key_dim)
    global_q1 = global_q1.reshape(*outer_shape, chunk_size, key_dim)
    global_q2 = global_q2.reshape(*outer_shape, chunk_size, key_dim)
    write = write.reshape(*outer_shape, chunk_size, value_dim)
    rhs = torch.cat((write, global_a1, global_a2), dim=-1)
    solution = solve_unit_lower(
        interaction,
        rhs,
        matrix_is_strict_lower=True,
    )
    u0, state_weight1, state_weight2 = solution.split(
        (value_dim, key_dim, key_dim), dim=-1
    )
    final_transport = torch.stack(
        (
            prefix11[..., -1, :], prefix12[..., -1, :],
            prefix21[..., -1, :], prefix22[..., -1, :],
        ),
        dim=-1,
    )
    if data_dtype == torch.bfloat16:
        global_q1 = global_q1.to(data_dtype)
        global_q2 = global_q2.to(data_dtype)
        tail1 = tail1.to(data_dtype)
        tail2 = tail2.to(data_dtype)
        causal_read = causal_read.to(data_dtype)
    return (
        u0,
        state_weight1,
        state_weight2,
        global_q1,
        global_q2,
        tail1,
        tail2,
        causal_read,
        final_transport,
    )


_COMPILED_BUILD = None
_COMPILED_BLOCKED_BUILD = None
_COMPILED_GDN_BUILD = None
_COMPILED_DELTANET_BUILD = None
_COMPILED_BLOCKED_GDN_BUILD = None
_COMPILED_BLOCKED_DELTANET_BUILD = None
_COMPILE_OPTIONS = {
    "dynamic": False,
    "mode": "max-autotune-no-cudagraphs",
}


def _build_operands_fast(*args):
    global _COMPILED_BUILD
    if _COMPILED_BUILD is None:
        try:
            _COMPILED_BUILD = torch.compile(_build_operands, **_COMPILE_OPTIONS)
        except Exception:  # pragma: no cover - environment-specific fallback
            _COMPILED_BUILD = _build_operands
    try:
        return _COMPILED_BUILD(*args)
    except Exception:  # pragma: no cover - safe eager fallback
        _COMPILED_BUILD = _build_operands
        return _build_operands(*args)


def _build_operands_blocked_fast(*args):
    global _COMPILED_BLOCKED_BUILD
    if _COMPILED_BLOCKED_BUILD is None:
        try:
            _COMPILED_BLOCKED_BUILD = torch.compile(
                _build_operands_blocked,
                **_COMPILE_OPTIONS,
            )
        except Exception:  # pragma: no cover - environment-specific fallback
            _COMPILED_BLOCKED_BUILD = _build_operands_blocked
    try:
        return _COMPILED_BLOCKED_BUILD(*args)
    except Exception:  # pragma: no cover - safe eager fallback
        _COMPILED_BLOCKED_BUILD = _build_operands_blocked
        return _build_operands_blocked(*args)


def _build_gdn_operands_fast(*args):
    global _COMPILED_GDN_BUILD
    if _COMPILED_GDN_BUILD is None:
        try:
            _COMPILED_GDN_BUILD = torch.compile(_build_gdn_operands, **_COMPILE_OPTIONS)
        except Exception:  # pragma: no cover - environment-specific fallback
            _COMPILED_GDN_BUILD = _build_gdn_operands
    try:
        return _COMPILED_GDN_BUILD(*args)
    except Exception:  # pragma: no cover - safe eager fallback
        _COMPILED_GDN_BUILD = _build_gdn_operands
        return _build_gdn_operands(*args)


def _build_deltanet_operands_fast(*args):
    global _COMPILED_DELTANET_BUILD
    if _COMPILED_DELTANET_BUILD is None:
        try:
            _COMPILED_DELTANET_BUILD = torch.compile(
                _build_deltanet_operands, **_COMPILE_OPTIONS
            )
        except Exception:  # pragma: no cover - environment-specific fallback
            _COMPILED_DELTANET_BUILD = _build_deltanet_operands
    try:
        return _COMPILED_DELTANET_BUILD(*args)
    except Exception:  # pragma: no cover - safe eager fallback
        _COMPILED_DELTANET_BUILD = _build_deltanet_operands
        return _build_deltanet_operands(*args)


def _build_gdn_operands_blocked_fast(*args):
    global _COMPILED_BLOCKED_GDN_BUILD
    if _COMPILED_BLOCKED_GDN_BUILD is None:
        try:
            _COMPILED_BLOCKED_GDN_BUILD = torch.compile(
                _build_gdn_operands_blocked,
                **_COMPILE_OPTIONS,
            )
        except Exception:  # pragma: no cover - environment-specific fallback
            _COMPILED_BLOCKED_GDN_BUILD = _build_gdn_operands_blocked
    try:
        return _COMPILED_BLOCKED_GDN_BUILD(*args)
    except Exception:  # pragma: no cover - safe eager fallback
        _COMPILED_BLOCKED_GDN_BUILD = _build_gdn_operands_blocked
        return _build_gdn_operands_blocked(*args)


def _build_deltanet_operands_blocked_fast(*args):
    global _COMPILED_BLOCKED_DELTANET_BUILD
    if _COMPILED_BLOCKED_DELTANET_BUILD is None:
        try:
            _COMPILED_BLOCKED_DELTANET_BUILD = torch.compile(
                _build_deltanet_operands_blocked,
                **_COMPILE_OPTIONS,
            )
        except Exception:  # pragma: no cover - environment-specific fallback
            _COMPILED_BLOCKED_DELTANET_BUILD = _build_deltanet_operands_blocked
    try:
        return _COMPILED_BLOCKED_DELTANET_BUILD(*args)
    except Exception:  # pragma: no cover - safe eager fallback
        _COMPILED_BLOCKED_DELTANET_BUILD = _build_deltanet_operands_blocked
        return _build_deltanet_operands_blocked(*args)


def _scan_eager(u0, w1, w2, q1, q2, k1, k2, causal, transport, active, slow):
    chunks = u0.shape[2]
    acc_dtype = torch.float64 if active.dtype == torch.float64 else torch.float32
    outputs = []
    for chunk_index in range(chunks):
        residual = (
            u0[:, :, chunk_index].to(acc_dtype)
            - (w1[:, :, chunk_index] @ active.to(w1.dtype)).to(acc_dtype)
            - (w2[:, :, chunk_index] @ slow.to(w2.dtype)).to(acc_dtype)
        )
        output = (causal[:, :, chunk_index] @ residual.to(causal.dtype)).to(acc_dtype)
        if q2.ndim == q1.ndim + 1:
            coefficients = q2[:, :, chunk_index].squeeze(-2).to(acc_dtype)
            query = q1[:, :, chunk_index]
            active_read = (query @ active.to(query.dtype)).to(acc_dtype)
            slow_read = (query @ slow.to(query.dtype)).to(acc_dtype)
            output = (
                output
                + coefficients[..., 0, None] * active_read
                + coefficients[..., 1, None] * slow_read
            )
        else:
            output = (
                output
                + (q1[:, :, chunk_index] @ active.to(q1.dtype)).to(acc_dtype)
                + (q2[:, :, chunk_index] @ slow.to(q2.dtype)).to(acc_dtype)
            )
        outputs.append(output)
        f = transport[:, :, chunk_index]
        coefficients = []
        for index in range(4):
            coefficient = f[..., index]
            while coefficient.ndim < active.ndim:
                coefficient = coefficient.unsqueeze(-1)
            coefficients.append(coefficient)
        f11, f12, f21, f22 = coefficients
        previous_active, previous_slow = active, slow
        active = (
            f11 * previous_active
            + f21 * previous_slow
            + (k1[:, :, chunk_index].transpose(-1, -2) @ residual.to(k1.dtype)).to(acc_dtype)
        )
        slow = (
            f12 * previous_active
            + f22 * previous_slow
            + (k2[:, :, chunk_index].transpose(-1, -2) @ residual.to(k2.dtype)).to(acc_dtype)
        )
    return torch.stack(outputs, dim=2), active, slow


def _scan_triton(
    u0,
    w1,
    w2,
    q1,
    q2,
    k1,
    k2,
    causal,
    transport,
    active,
    slow,
    *,
    zero_initial=False,
    readout_gate=False,
):
    from .scan_triton import fused_integral_scan

    batch, heads = u0.shape[:2]
    # Published GDN shapes include K=200. Keep their parameter/state geometry
    # intact and pad only the tensor-core scan's private key axis.
    original_key_dim = w1.shape[-1]
    padded_key_dim = 1 << (original_key_dim - 1).bit_length()
    pad_keys = padded_key_dim - original_key_dim
    if pad_keys:
        pad = torch.nn.functional.pad
        w1, q1, k1, k2 = (pad(x, (0, pad_keys)) for x in (w1, q1, k1, k2))
        w2 = w1 if readout_gate else pad(w2, (0, pad_keys))
        if q2.ndim == q1.ndim:
            q2 = pad(q2, (0, pad_keys))
        if transport.ndim == 5:
            transport = pad(transport, (0, 0, 0, pad_keys))
        active, slow = (pad(x, (0, 0, 0, pad_keys)) for x in (active, slow))

    def flatten(tensor):
        return tensor.reshape(batch * heads, *tensor.shape[2:]).contiguous()

    output, final_active, final_slow = fused_integral_scan(
        flatten(u0),
        flatten(w1),
        flatten(w1) if readout_gate else flatten(w2),
        flatten(q1),
        flatten(q2),
        flatten(k1),
        flatten(k2),
        flatten(causal),
        flatten(transport),
        flatten(active),
        flatten(slow),
        zero_initial=zero_initial,
        readout_gate=readout_gate,
        **({"output_heads": heads} if (
            readout_gate and original_key_dim == 128 and u0.shape[-1] == 128
            and optimized_readout(u0, u0.shape[2] * u0.shape[3])
        ) else {}),
    )
    return (
        output.view(batch, heads, *u0.shape[2:]),
        final_active.view(batch, heads, *active.shape[2:])[..., :original_key_dim, :],
        final_slow.view(batch, heads, *slow.shape[2:])[..., :original_key_dim, :],
    )


def _validate_runtime(q, chunk_size, implementation, *, allow_key_padding=False):
    if chunk_size not in (16, 32, 64):
        raise ValueError("chunk_size must be one of 16, 32, or 64")
    if q.shape[1] % chunk_size:
        raise ValueError("sequence length must be divisible by chunk_size")
    if implementation not in ("auto", "eager", "triton"):
        raise ValueError("implementation must be auto, eager, or triton")
    if implementation == "triton" and not q.is_cuda:
        raise ValueError("the Triton scan requires CUDA")
    if q.is_cuda and not allow_key_padding and (q.shape[-1] & (q.shape[-1] - 1)):
        raise ValueError("the Triton scan requires a power-of-two key dimension")


def _validate_scalar_inputs(q, k, v, scalars):
    if q.ndim != 4 or q.shape != k.shape:
        raise ValueError("q and k must have identical [B, T, H, K] shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("v must have shape [B, T, H, V]")
    expected = q.shape[:3]
    if any(tensor.shape != expected for tensor in scalars):
        raise ValueError("all scalar gates must have shape [B, T, H]")
    if any(tensor.device != q.device for tensor in (q, k, v, *scalars)):
        raise ValueError("all inputs must be on the same device")


def _execute_operands(
    operands,
    q,
    v,
    initial_state,
    output_final_state,
    implementation,
    *,
    state_basis="active_slow",
    readout_gate=False,
):
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    zero_initial_state = initial_state is None
    selected = "triton" if implementation == "auto" and q.is_cuda else implementation
    selected = "eager" if selected == "auto" else selected
    if zero_initial_state:
        factory = torch.empty if selected == "triton" else torch.zeros
        active = factory(
            batch,
            heads,
            key_dim,
            value_dim,
            dtype=acc_dtype,
            device=q.device,
        )
        slow = torch.empty_like(active) if selected == "triton" else torch.zeros_like(active)
    else:
        active, slow = (state.to(acc_dtype) for state in initial_state)
        expected = (batch, heads, key_dim, value_dim)
        if active.shape != expected or slow.shape != expected:
            raise ValueError(f"initial states must both have shape {expected}")

    if state_basis == "slow_difference":
        first = slow
        second = active if zero_initial_state else active - slow
    else:
        first, second = active, slow
    if selected == "triton":
        output, first, second = _scan_triton(
            *operands,
            first,
            second,
            zero_initial=zero_initial_state,
            readout_gate=readout_gate,
        )
    else:
        output, first, second = _scan_eager(*operands, first, second)
    output = output.reshape(batch, heads, length, value_dim).transpose(1, 2).contiguous()
    output = output.to(v.dtype)
    if not output_final_state:
        return output, None
    if state_basis == "slow_difference":
        active, slow = first + second, first
    else:
        active, slow = first, second
    final_state = (active, slow)
    return output, final_state


def chunk_integral_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    chunk_size: int = 16,
    implementation: str = "auto",
    compile_precompute: bool = True,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    r"""Exact training kernel for the paper's general integral GDN-2 form.

    Args follow :func:`integral_delta_reference`.  ``implementation='auto'``
    selects the fused Triton boundary scan on CUDA and the eager oracle scan on
    CPU.  CUDA BF16 uses direct-relative C16 recurrences without inverse or
    centered coordinates.  Chunk sizes 32 and 64 compose those exact local
    blocks using forward-only 2x2 transports.
    """
    _validate_inputs(q, k, v, g, b, w, gamma, lam)
    _validate_runtime(q, chunk_size, implementation)
    if scale is None:
        scale = q.shape[-1] ** -0.5

    autocast_context = (
        torch.amp.autocast(device_type="cuda", enabled=False)
        if q.is_cuda
        else contextlib.nullcontext()
    )
    with autocast_context:
        if (
            compile_precompute
            and q.is_cuda
            and q.dtype == torch.bfloat16
            and chunk_size in (16, 32, 64)
        ):
            from .general_direct_triton import (
                build_general_direct_operands_triton,
            )

            operands = build_general_direct_operands_triton(
                q, k, v, g, b, w, gamma, lam, scale,
                chunk_size=chunk_size,
            )
            return _execute_operands(
                operands,
                q,
                v,
                initial_state,
                output_final_state,
                implementation,
            )
        batch, length, heads, key_dim = q.shape
        chunks = length // chunk_size
        qc = _chunk(q, chunks, chunk_size)
        kc = _chunk(k, chunks, chunk_size)
        vc = _chunk(v, chunks, chunk_size)
        gc = _chunk(g, chunks, chunk_size)
        bc = _chunk(b, chunks, chunk_size)
        wc = _chunk(w, chunks, chunk_size)
        gamma_chunks = gamma.transpose(1, 2).reshape(batch, heads, chunks, chunk_size)
        lambda_chunks = lam.transpose(1, 2).reshape(batch, heads, chunks, chunk_size)
        if chunk_size == 16:
            build = _build_operands_fast if compile_precompute and q.is_cuda else _build_operands
        else:
            build = (
                _build_operands_blocked_fast
                if compile_precompute and q.is_cuda
                else _build_operands_blocked
            )
        operands = build(
            qc, kc, vc, gc, bc, wc, gamma_chunks, lambda_chunks, scale
        )
        return _execute_operands(
            operands,
            q,
            v,
            initial_state,
            output_final_state,
            implementation,
        )


def chunk_integral_kda(q, k, v, g, beta, gamma, lam, **kwargs):
    """Exact integral KDA specialization of the diagonal kernel."""
    _validate_scalar_inputs(q, k, v, (beta, gamma, lam))
    return chunk_integral_delta(
        q,
        k,
        v,
        g,
        beta.unsqueeze(-1).expand_as(k),
        beta.unsqueeze(-1).expand_as(v),
        gamma,
        lam,
        **kwargs,
    )


def chunk_integral_gdn2(q, k, v, g, b, w, gamma, lam, **kwargs):
    """Named entry point for the paper's general integral GDN-2 form."""
    return chunk_integral_delta(q, k, v, g, b, w, gamma, lam, **kwargs)


__all__ = [
    "chunk_integral_delta",
    "chunk_integral_kda",
    "chunk_integral_gdn2",
]

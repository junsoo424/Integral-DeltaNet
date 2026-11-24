"""Fused coefficient tables for integral KDA and GDN-2.

The diagonal variants carry one 2x2 passive prefix per key channel.  Keeping
that prefix in a Triton program avoids the many full-sequence temporaries made
by a generic parallel prefix.  The backward kernel reconstructs each previous
prefix from the saved final transport and a stable per-token 2x2 inverse.  It
therefore keeps O(BK), not O(C*BK), values live; only the two scalar
integration-gate gradients require a reduction across key tiles.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .general_triton import solve_general_operands


_BLOCK_KEY = 128
_BLOCK_WARPS = 1


@triton.jit
def _general_coeff_fwd_kernel(
    Q,
    K_IN,
    G,
    B_GATE,
    GAMMA,
    LAM,
    A1,
    A2,
    C1,
    C2,
    Q1,
    Q2,
    TAIL1,
    TAIL2,
    TRANSPORT,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
    SB0: tl.constexpr,
    SB1: tl.constexpr,
    SB2: tl.constexpr,
    SB3: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    key_block = pid % tl.cdiv(K, BK)
    system = pid // tl.cdiv(K, BK)
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)

    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K

    current11 = tl.full((BK,), 1.0, tl.float32)
    current12 = tl.zeros((BK,), dtype=tl.float32)
    current21 = tl.zeros((BK,), dtype=tl.float32)
    current22 = tl.full((BK,), 1.0, tl.float32)
    current_det = tl.full((BK,), 1.0, tl.float32)
    for row in range(C):
        token = chunk * C + row
        offsets = ((batch * T + token) * H + head) * K + keys
        q = tl.load(Q + offsets, mask=key_mask, other=0.0).to(tl.float32)
        key = tl.load(K_IN + offsets, mask=key_mask, other=0.0).to(tl.float32)
        alpha = tl.exp(
            tl.load(G + offsets, mask=key_mask, other=0.0).to(tl.float32)
        )
        b_offsets = batch * SB0 + token * SB1 + head * SB2 + keys * SB3
        b = tl.load(B_GATE + b_offsets, mask=key_mask, other=0.0).to(tl.float32)
        scalar_offset = (batch * T + token) * H + head
        gamma = tl.load(GAMMA + scalar_offset).to(tl.float32)
        lam = tl.load(LAM + scalar_offset).to(tl.float32)
        p11 = gamma * alpha
        p12 = lam * p11
        p21 = (1.0 - gamma) * alpha
        p22 = (1.0 - lam) + lam * p21
        next11 = current11 * p11 + current12 * p21
        next12 = current11 * p12 + current12 * p22
        next21 = current21 * p11 + current22 * p21
        next22 = current21 * p12 + current22 * p22
        current_det *= gamma * alpha * (1.0 - lam)
        sigma1 = (next22 - lam * next21) / current_det
        sigma2 = (-next12 + lam * next11) / current_det
        erase = b * key
        output_offsets = system * C * K + row * K + keys
        tl.store(A1 + output_offsets, current11 * alpha * erase, mask=key_mask)
        tl.store(A2 + output_offsets, current21 * alpha * erase, mask=key_mask)
        tl.store(C1 + output_offsets, sigma1 * key, mask=key_mask)
        tl.store(C2 + output_offsets, sigma2 * key, mask=key_mask)
        tl.store(Q1 + output_offsets, next11 * q * SCALE, mask=key_mask)
        tl.store(Q2 + output_offsets, next21 * q * SCALE, mask=key_mask)
        current11, current12 = next11, next12
        current21, current22 = next21, next22

    final11, final12 = current11, current12
    final21, final22 = current21, current22

    # [1, lambda_r] F_r^-1 F_C equals [1, lambda_r] times the suffix
    # P_{r+1} ... P_C.  A reverse suffix scan therefore writes the same tail
    # without a second determinant/division-heavy absolute-prefix pass.
    suffix11 = tl.full((BK,), 1.0, tl.float32)
    suffix12 = tl.zeros((BK,), dtype=tl.float32)
    suffix21 = tl.zeros((BK,), dtype=tl.float32)
    suffix22 = tl.full((BK,), 1.0, tl.float32)
    for reverse_index in range(C):
        row = C - 1 - reverse_index
        token = chunk * C + row
        offsets = ((batch * T + token) * H + head) * K + keys
        key = tl.load(K_IN + offsets, mask=key_mask, other=0.0).to(tl.float32)
        alpha = tl.exp(
            tl.load(G + offsets, mask=key_mask, other=0.0).to(tl.float32)
        )
        scalar_offset = (batch * T + token) * H + head
        gamma = tl.load(GAMMA + scalar_offset).to(tl.float32)
        lam = tl.load(LAM + scalar_offset).to(tl.float32)
        tau1 = suffix11 + lam * suffix21
        tau2 = suffix12 + lam * suffix22
        output_offsets = system * C * K + row * K + keys
        tl.store(TAIL1 + output_offsets, tau1 * key, mask=key_mask)
        tl.store(TAIL2 + output_offsets, tau2 * key, mask=key_mask)

        p11 = gamma * alpha
        p12 = lam * p11
        p21 = (1.0 - gamma) * alpha
        p22 = (1.0 - lam) + lam * p21
        next11 = p11 * suffix11 + p12 * suffix21
        next12 = p11 * suffix12 + p12 * suffix22
        next21 = p21 * suffix11 + p22 * suffix21
        next22 = p21 * suffix12 + p22 * suffix22
        suffix11, suffix12 = next11, next12
        suffix21, suffix22 = next21, next22

    transport_offsets = system * K * 4 + keys * 4
    tl.store(TRANSPORT + transport_offsets + 0, final11, mask=key_mask)
    tl.store(TRANSPORT + transport_offsets + 1, final12, mask=key_mask)
    tl.store(TRANSPORT + transport_offsets + 2, final21, mask=key_mask)
    tl.store(TRANSPORT + transport_offsets + 3, final22, mask=key_mask)


@triton.jit
def _general_coeff_bwd_kernel(
    Q,
    K_IN,
    G,
    B_GATE,
    GAMMA,
    LAM,
    DA1,
    DA2,
    DC1,
    DC2,
    DQ1,
    DQ2,
    DTAIL1,
    DTAIL2,
    TRANSPORT,
    DTRANSPORT,
    DQ,
    DK,
    DG,
    DB,
    DGAMMA_PARTIAL,
    DLAM_PARTIAL,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
    SB0: tl.constexpr,
    SB1: tl.constexpr,
    SB2: tl.constexpr,
    SB3: tl.constexpr,
    DIRECT_SCALAR: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    nk = tl.cdiv(K, BK)
    key_block = pid % nk
    system = pid // nk
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    transport_offsets = system * K * 4 + keys * 4
    final11 = tl.load(
        TRANSPORT + transport_offsets + 0, mask=key_mask, other=0.0
    ).to(tl.float32)
    final12 = tl.load(
        TRANSPORT + transport_offsets + 1, mask=key_mask, other=0.0
    ).to(tl.float32)
    final21 = tl.load(
        TRANSPORT + transport_offsets + 2, mask=key_mask, other=0.0
    ).to(tl.float32)
    final22 = tl.load(
        TRANSPORT + transport_offsets + 3, mask=key_mask, other=0.0
    ).to(tl.float32)
    adj11 = tl.load(DTRANSPORT + transport_offsets + 0, mask=key_mask, other=0.0).to(tl.float32)
    adj12 = tl.load(DTRANSPORT + transport_offsets + 1, mask=key_mask, other=0.0).to(tl.float32)
    adj21 = tl.load(DTRANSPORT + transport_offsets + 2, mask=key_mask, other=0.0).to(tl.float32)
    adj22 = tl.load(DTRANSPORT + transport_offsets + 3, mask=key_mask, other=0.0).to(tl.float32)

    # det(F_C) is cheap to recompute and considerably more stable than the
    # cancellation-prone expression f11*f22-f12*f21.
    final_det = tl.full((BK,), 1.0, tl.float32)
    for row in range(C):
        token = chunk * C + row
        input_offsets = ((batch * T + token) * H + head) * K + keys
        alpha = tl.exp(
            tl.load(G + input_offsets, mask=key_mask, other=0.0).to(tl.float32)
        )
        scalar_offset = (batch * T + token) * H + head
        gamma = tl.load(GAMMA + scalar_offset).to(tl.float32)
        lam = tl.load(LAM + scalar_offset).to(tl.float32)
        final_det *= gamma * alpha * (1.0 - lam)

    # Tail reads depend on the final prefix from every token.  Accumulate that
    # contribution first while walking F_r -> F_{r-1} with a stable 2x2
    # inverse.  This keeps only O(BK) values live instead of O(C*BK).
    current11, current12 = final11, final12
    current21, current22 = final21, final22
    current_det = final_det
    for reverse_index in range(C):
        row = C - 1 - reverse_index
        token = chunk * C + row
        input_offsets = ((batch * T + token) * H + head) * K + keys
        output_offsets = system * C * K + row * K + keys
        alpha = tl.exp(
            tl.load(G + input_offsets, mask=key_mask, other=0.0).to(tl.float32)
        )
        scalar_offset = (batch * T + token) * H + head
        gamma = tl.load(GAMMA + scalar_offset).to(tl.float32)
        lam = tl.load(LAM + scalar_offset).to(tl.float32)
        p11 = gamma * alpha
        p12 = lam * p11
        p21 = (1.0 - gamma) * alpha
        p22 = (1.0 - lam) + lam * p21
        step_det = gamma * alpha * (1.0 - lam)

        numerator1 = current22 - lam * current21
        numerator2 = -current12 + lam * current11
        sigma1 = numerator1 / current_det
        sigma2 = numerator2 / current_det
        key = tl.load(K_IN + input_offsets, mask=key_mask, other=0.0).to(tl.float32)
        dt1 = tl.load(
            DTAIL1 + output_offsets, mask=key_mask, other=0.0
        ).to(tl.float32)
        dt2 = tl.load(
            DTAIL2 + output_offsets, mask=key_mask, other=0.0
        ).to(tl.float32)
        dtau1 = key * dt1
        dtau2 = key * dt2
        adj11 += dtau1 * sigma1
        adj12 += dtau2 * sigma1
        adj21 += dtau1 * sigma2
        adj22 += dtau2 * sigma2

        inverse_step = 1.0 / step_det
        previous11 = (current11 * p22 - current12 * p21) * inverse_step
        previous12 = (-current11 * p12 + current12 * p11) * inverse_step
        previous21 = (current21 * p22 - current22 * p21) * inverse_step
        previous22 = (-current21 * p12 + current22 * p11) * inverse_step
        current11, current12 = previous11, previous12
        current21, current22 = previous21, previous22
        current_det *= inverse_step

    current11, current12 = final11, final12
    current21, current22 = final21, final22
    current_det = final_det
    adj_det = tl.zeros((BK,), dtype=tl.float32)
    for reverse_index in range(C):
        row = C - 1 - reverse_index
        token = chunk * C + row
        input_offsets = ((batch * T + token) * H + head) * K + keys
        output_offsets = system * C * K + row * K + keys
        scalar_offset = (batch * T + token) * H + head
        alpha = tl.exp(
            tl.load(G + input_offsets, mask=key_mask, other=0.0).to(tl.float32)
        )
        gamma = tl.load(GAMMA + scalar_offset).to(tl.float32)
        lam = tl.load(LAM + scalar_offset).to(tl.float32)
        p11 = gamma * alpha
        p12 = lam * p11
        p21 = (1.0 - gamma) * alpha
        p22 = (1.0 - lam) + lam * p21
        step_det = gamma * alpha * (1.0 - lam)
        inverse_step = 1.0 / step_det
        previous11 = (current11 * p22 - current12 * p21) * inverse_step
        previous12 = (-current11 * p12 + current12 * p11) * inverse_step
        previous21 = (current21 * p22 - current22 * p21) * inverse_step
        previous22 = (-current21 * p12 + current22 * p11) * inverse_step
        previous_det = current_det * inverse_step

        q = tl.load(Q + input_offsets, mask=key_mask, other=0.0).to(tl.float32)
        key = tl.load(K_IN + input_offsets, mask=key_mask, other=0.0).to(tl.float32)
        b_offsets = batch * SB0 + token * SB1 + head * SB2 + keys * SB3
        b = tl.load(B_GATE + b_offsets, mask=key_mask, other=0.0).to(tl.float32)
        da1 = tl.load(DA1 + output_offsets, mask=key_mask, other=0.0).to(tl.float32)
        da2 = tl.load(DA2 + output_offsets, mask=key_mask, other=0.0).to(tl.float32)
        dc1 = tl.load(DC1 + output_offsets, mask=key_mask, other=0.0).to(tl.float32)
        dc2 = tl.load(DC2 + output_offsets, mask=key_mask, other=0.0).to(tl.float32)
        dq1 = tl.load(DQ1 + output_offsets, mask=key_mask, other=0.0).to(tl.float32)
        dq2 = tl.load(DQ2 + output_offsets, mask=key_mask, other=0.0).to(tl.float32)
        dt1 = tl.load(DTAIL1 + output_offsets, mask=key_mask, other=0.0).to(tl.float32)
        dt2 = tl.load(DTAIL2 + output_offsets, mask=key_mask, other=0.0).to(tl.float32)

        numerator1 = current22 - lam * current21
        numerator2 = -current12 + lam * current11
        inverse_det = 1.0 / current_det
        sigma1 = numerator1 * inverse_det
        sigma2 = numerator2 * inverse_det
        tau1 = sigma1 * final11 + sigma2 * final21
        tau2 = sigma1 * final12 + sigma2 * final22

        erase = b * key
        scaled_erase = alpha * erase
        dprevious11_direct = scaled_erase * da1
        dprevious21_direct = scaled_erase * da2
        dscaled_erase = previous11 * da1 + previous21 * da2
        dalpha_direct = erase * dscaled_erase
        derase = alpha * dscaled_erase
        db = key * derase
        dk = b * derase

        dq = SCALE * (current11 * dq1 + current21 * dq2)
        df11 = SCALE * q * dq1
        df12 = tl.zeros((BK,), dtype=tl.float32)
        df21 = SCALE * q * dq2
        df22 = tl.zeros((BK,), dtype=tl.float32)
        dk += sigma1 * dc1 + sigma2 * dc2 + tau1 * dt1 + tau2 * dt2
        dtau1 = key * dt1
        dtau2 = key * dt2
        dsigma1 = key * dc1 + dtau1 * final11 + dtau2 * final12
        dsigma2 = key * dc2 + dtau1 * final21 + dtau2 * final22

        dn1 = dsigma1 * inverse_det
        dn2 = dsigma2 * inverse_det
        df22 += dn1
        df21 -= lam * dn1
        df12 -= dn2
        df11 += lam * dn2
        dlam_direct = -current21 * dn1 + current11 * dn2
        ddet_direct = -(
            dsigma1 * numerator1 + dsigma2 * numerator2
        ) * inverse_det * inverse_det

        adj11 += df11
        adj12 += df12
        adj21 += df21
        adj22 += df22
        adj_det += ddet_direct

        dprevious11 = adj11 * p11 + adj12 * p12
        dprevious12 = adj11 * p21 + adj12 * p22
        dprevious21 = adj21 * p11 + adj22 * p12
        dprevious22 = adj21 * p21 + adj22 * p22
        dp11 = previous11 * adj11 + previous21 * adj21
        dp12 = previous11 * adj12 + previous21 * adj22
        dp21 = previous12 * adj11 + previous22 * adj21
        dp22 = previous12 * adj12 + previous22 * adj22
        dprevious11 += dprevious11_direct
        dprevious21 += dprevious21_direct

        ddet_step = adj_det * previous_det
        next_adj_det = adj_det * step_det
        dp11 += lam * dp12
        dlambda = p11 * dp12
        dp21 += lam * dp22
        dlambda += (p21 - 1.0) * dp22
        dgamma = alpha * (dp11 - dp21)
        dalpha = gamma * dp11 + (1.0 - gamma) * dp21
        dgamma += ddet_step * alpha * (1.0 - lam)
        dalpha += ddet_step * gamma * (1.0 - lam)
        dlambda -= ddet_step * gamma * alpha
        dlambda += dlam_direct
        dalpha += dalpha_direct
        dg = dalpha * alpha

        tl.store(DQ + input_offsets, dq, mask=key_mask)
        tl.store(DK + input_offsets, dk, mask=key_mask)
        tl.store(DG + input_offsets, dg, mask=key_mask)
        tl.store(DB + input_offsets, db, mask=key_mask)
        dgamma_scalar = tl.sum(tl.where(key_mask, dgamma, 0.0), axis=0)
        dlam_scalar = tl.sum(tl.where(key_mask, dlambda, 0.0), axis=0)
        if DIRECT_SCALAR:
            # BK covers the full key head, so this program is the unique
            # writer. Store in the original [B,T,H] layout and avoid a
            # partial buffer plus a separate reduction/transpose kernel.
            tl.store(DGAMMA_PARTIAL + scalar_offset, dgamma_scalar)
            tl.store(DLAM_PARTIAL + scalar_offset, dlam_scalar)
        else:
            partial_offset = (system * nk + key_block) * C + row
            tl.store(DGAMMA_PARTIAL + partial_offset, dgamma_scalar)
            tl.store(DLAM_PARTIAL + partial_offset, dlam_scalar)

        adj11, adj12 = dprevious11, dprevious12
        adj21, adj22 = dprevious21, dprevious22
        adj_det = next_adj_det
        current11, current12 = previous11, previous12
        current21, current22 = previous21, previous22
        current_det = previous_det


class _GeneralCoefficients(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, g, b, gamma, lam, scale, chunk_size, block_key, block_warps
    ):
        if block_key not in (32, 64, 128):
            raise ValueError("general coefficient block_key must be 32, 64, or 128")
        if block_warps not in (1, 2, 4, 8):
            raise ValueError("general coefficient block_warps must be 1, 2, 4, or 8")
        batch, length, heads, key_dim = q.shape
        chunks = length // chunk_size
        systems = batch * heads * chunks
        q, k, g, gamma, lam = (
            tensor.contiguous() for tensor in (q, k, g, gamma, lam)
        )
        shape = (batch, heads, chunks, chunk_size, key_dim)
        outputs = tuple(
            torch.empty(*shape, device=q.device, dtype=torch.bfloat16)
            for _ in range(8)
        )
        transport = torch.empty(
            batch,
            heads,
            chunks,
            key_dim,
            4,
            device=q.device,
            dtype=torch.float32,
        )
        nk = triton.cdiv(key_dim, block_key)
        _general_coeff_fwd_kernel[(systems * nk,)](
            q,
            k,
            g,
            b,
            gamma,
            lam,
            *outputs,
            transport,
            T=length,
            H=heads,
            N=chunks,
            K=key_dim,
            C=chunk_size,
            SCALE=float(scale),
            SB0=b.stride(0),
            SB1=b.stride(1),
            SB2=b.stride(2),
            SB3=b.stride(3),
            BK=block_key,
            num_warps=block_warps,
            num_stages=2,
        )
        ctx.save_for_backward(q, k, g, b, gamma, lam, transport)
        ctx.dimensions = (
            batch, length, heads, key_dim, chunks, systems, nk, chunk_size
        )
        ctx.scale = float(scale)
        ctx.block_key = block_key
        ctx.block_warps = block_warps
        ctx.input_dtypes = tuple(x.dtype for x in (q, k, g, b, gamma, lam))
        return (*outputs, transport)

    @staticmethod
    def backward(ctx, *grads):
        q, k, g, b, gamma, lam, transport = ctx.saved_tensors
        (
            batch, length, heads, key_dim, chunks, systems, nk, chunk_size
        ) = ctx.dimensions
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dg = torch.empty_like(g)
        db = torch.empty_like(q, dtype=b.dtype)
        direct_scalar = nk == 1
        if direct_scalar:
            dgamma_partial = torch.empty_like(gamma)
            dlam_partial = torch.empty_like(lam)
        else:
            dgamma_partial = torch.empty(
                systems, nk, chunk_size, device=q.device, dtype=torch.float32
            )
            dlam_partial = torch.empty_like(dgamma_partial)
        _general_coeff_bwd_kernel[(systems * nk,)](
            q,
            k,
            g,
            b,
            gamma,
            lam,
            *(grad.contiguous() for grad in grads[:8]),
            transport,
            grads[8].contiguous(),
            dq,
            dk,
            dg,
            db,
            dgamma_partial,
            dlam_partial,
            T=length,
            H=heads,
            N=chunks,
            K=key_dim,
            C=chunk_size,
            SCALE=ctx.scale,
            SB0=b.stride(0),
            SB1=b.stride(1),
            SB2=b.stride(2),
            SB3=b.stride(3),
            DIRECT_SCALAR=direct_scalar,
            BK=ctx.block_key,
            num_warps=ctx.block_warps,
            num_stages=2,
        )

        def reduce_scalar(partial):
            reduced = partial.sum(dim=1).view(
                batch, heads, chunks, chunk_size
            )
            return reduced.reshape(batch, heads, length).transpose(1, 2).contiguous()

        if direct_scalar:
            dgamma, dlam = dgamma_partial, dlam_partial
        else:
            dgamma = reduce_scalar(dgamma_partial)
            dlam = reduce_scalar(dlam_partial)
        values = (dq, dk, dg, db, dgamma, dlam)
        return tuple(
            value.to(dtype) for value, dtype in zip(values, ctx.input_dtypes)
        ) + (None, None, None, None)


@torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
def _products_and_write(a1, a2, c1, c2, q1, q2, v, w, chunk_size):
    matrix = (a1 @ c1.transpose(-1, -2)).float()
    matrix = matrix + (a2 @ c2.transpose(-1, -2)).float()
    causal = (q1 @ c1.transpose(-1, -2)).float()
    causal = causal + (q2 @ c2.transpose(-1, -2)).float()
    matrix = torch.tril(matrix, diagonal=-1)
    causal = torch.tril(causal).to(torch.bfloat16)
    batch, length, heads, value_dim = v.shape
    chunks = length // chunk_size
    write = (v * w).transpose(1, 2).reshape(
        batch, heads, chunks, chunk_size, value_dim
    ).contiguous()
    return matrix, causal, write


@torch.compiler.disable
def build_general_operands_triton(
    q, k, v, g, b, w, gamma, lam, scale, chunk_size=16
):
    """Build C16 diagonal operands with streaming prefixes."""
    if not (q.is_cuda and q.dtype == torch.bfloat16):
        raise ValueError("the general coefficient kernel requires CUDA BF16")
    if chunk_size != 16 or q.shape[1] % chunk_size:
        raise ValueError("direct general builder requires a C16 sequence")
    a1, a2, c1, c2, q1, q2, tail1, tail2, transport = (
        _GeneralCoefficients.apply(
            q,
            k,
            g,
            b,
            gamma,
            lam,
            scale,
            chunk_size,
            _BLOCK_KEY,
            _BLOCK_WARPS,
        )
    )
    matrix, causal, write = _products_and_write(
        a1, a2, c1, c2, q1, q2, v, w, chunk_size
    )
    u0, state_weight1, state_weight2 = solve_general_operands(
        matrix, write, a1, a2
    )
    return (
        u0,
        state_weight1,
        state_weight2,
        q1,
        q2,
        tail1,
        tail2,
        causal,
        transport,
    )


__all__ = ["build_general_operands_triton"]

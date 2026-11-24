"""Inverse-free absolute C16 prefixes for channel-wise integral models.

The backward saves four local FP32 segment starts and reconstructs each
pre-token prefix using multiplication only.  It never divides by a decay or
forms an inverse prefix, including for KDA/GDN-2 model-scale gate ranges.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .general_direct_config import needs_recomputed_segment_starts


@triton.jit
def _load_transition(
    G, GAMMA, LAM,
    batch, chunk, head, keys, key_mask,
    target: tl.constexpr,
    T: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr,
):
    """Load one transition directly instead of selecting it by reduction."""
    token = chunk * C + target
    offsets = ((batch * T + token) * H + head) * K + keys
    alpha = tl.exp(
        tl.load(G + offsets, mask=key_mask, other=0.0).to(tl.float32)
    )
    scalar_offset = (batch * T + token) * H + head
    gamma = tl.load(GAMMA + scalar_offset).to(tl.float32)
    lam = tl.load(LAM + scalar_offset).to(tl.float32)
    return alpha, gamma, lam


@triton.jit
def _prefix_step(
    f11, f12, f21, f22, G, GAMMA, LAM,
    batch, chunk, head, keys, key_mask,
    target: tl.constexpr,
    T: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr,
):
    alpha, gamma, lam = _load_transition(
        G, GAMMA, LAM, batch, chunk, head, keys, key_mask, target,
        T, H, K, C,
    )
    p11 = gamma * alpha
    p12 = lam * p11
    p21 = (1.0 - gamma) * alpha
    p22 = (1.0 - lam) + lam * p21
    return (
        f11 * p11 + f12 * p21,
        f11 * p12 + f12 * p22,
        f21 * p11 + f22 * p21,
        f21 * p12 + f22 * p22,
    )


@triton.jit
def _prefix_fwd_kernel(
    Q, K_IN, G, B_GATE, GAMMA, LAM,
    A1, A2, Q1, Q2, TRANSPORT,
    T: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr, SCALE: tl.constexpr,
    SB0: tl.constexpr, SB1: tl.constexpr,
    SB2: tl.constexpr, SB3: tl.constexpr,
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
    f11 = tl.full((BK,), 1.0, tl.float32)
    f12 = tl.zeros((BK,), tl.float32)
    f21 = tl.zeros((BK,), tl.float32)
    f22 = tl.full((BK,), 1.0, tl.float32)
    for row in range(C):
        token = chunk * C + row
        offsets = ((batch * T + token) * H + head) * K + keys
        q = tl.load(Q + offsets, mask=key_mask, other=0.0).to(tl.float32)
        key = tl.load(K_IN + offsets, mask=key_mask, other=0.0).to(tl.float32)
        alpha = tl.exp(tl.load(
            G + offsets, mask=key_mask, other=0.0
        ).to(tl.float32))
        b_offsets = batch * SB0 + token * SB1 + head * SB2 + keys * SB3
        b = tl.load(B_GATE + b_offsets, mask=key_mask, other=0.0).to(tl.float32)
        erase = alpha * b * key
        output_offsets = system * C * K + row * K + keys
        tl.store(A1 + output_offsets, f11 * erase, mask=key_mask)
        tl.store(A2 + output_offsets, f21 * erase, mask=key_mask)
        scalar_offset = (batch * T + token) * H + head
        gamma = tl.load(GAMMA + scalar_offset).to(tl.float32)
        lam = tl.load(LAM + scalar_offset).to(tl.float32)
        p11 = gamma * alpha
        p12 = lam * p11
        p21 = (1.0 - gamma) * alpha
        p22 = (1.0 - lam) + lam * p21
        n11 = f11 * p11 + f12 * p21
        n12 = f11 * p12 + f12 * p22
        n21 = f21 * p11 + f22 * p21
        n22 = f21 * p12 + f22 * p22
        f11, f12, f21, f22 = n11, n12, n21, n22
        tl.store(Q1 + output_offsets, f11 * q * SCALE, mask=key_mask)
        tl.store(Q2 + output_offsets, f21 * q * SCALE, mask=key_mask)
    transport_offsets = (system * K + keys) * 4
    tl.store(TRANSPORT + transport_offsets + 0, f11, mask=key_mask)
    tl.store(TRANSPORT + transport_offsets + 1, f12, mask=key_mask)
    tl.store(TRANSPORT + transport_offsets + 2, f21, mask=key_mask)
    tl.store(TRANSPORT + transport_offsets + 3, f22, mask=key_mask)


@triton.jit
def _prefix_local_bwd_kernel(
    Q, K_IN, G, B_GATE, GAMMA, LAM,
    DA1, DA2, DQ1, DQ2, DTRANSPORT,
    DQ, DK, DG, DB, DGAMMA_PARTIAL, DLAM_PARTIAL,
    T: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr, NK: tl.constexpr,
    SCALE: tl.constexpr,
    SB0: tl.constexpr, SB1: tl.constexpr,
    SB2: tl.constexpr, SB3: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    RECOMPUTE_SEGMENT_STARTS: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    key_block = pid % NK
    system = pid // NK
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K

    one = tl.full((BK,), 1.0, tl.float32)
    zero = tl.zeros((BK,), tl.float32)
    f11, f12, f21, f22 = one, zero, zero, one
    cp011, cp012, cp021, cp022 = f11, f12, f21, f22
    cp411, cp412, cp421, cp422 = f11, f12, f21, f22
    cp811, cp812, cp821, cp822 = f11, f12, f21, f22
    cp1211, cp1212, cp1221, cp1222 = f11, f12, f21, f22
    if not RECOMPUTE_SEGMENT_STARTS:
        for target in range(C):
            if target == 4:
                cp411, cp412, cp421, cp422 = f11, f12, f21, f22
            if target == 8:
                cp811, cp812, cp821, cp822 = f11, f12, f21, f22
            if target == 12:
                cp1211, cp1212, cp1221, cp1222 = f11, f12, f21, f22
            f11, f12, f21, f22 = _prefix_step(
                f11, f12, f21, f22, G, GAMMA, LAM,
                batch, chunk, head, keys, key_mask, target,
                T, H, K, C,
            )

    transport_offsets = (system * K + keys) * 4
    df11 = tl.load(
        DTRANSPORT + transport_offsets + 0, mask=key_mask, other=0.0
    ).to(tl.float32)
    df12 = tl.load(
        DTRANSPORT + transport_offsets + 1, mask=key_mask, other=0.0
    ).to(tl.float32)
    df21 = tl.load(
        DTRANSPORT + transport_offsets + 2, mask=key_mask, other=0.0
    ).to(tl.float32)
    df22 = tl.load(
        DTRANSPORT + transport_offsets + 3, mask=key_mask, other=0.0
    ).to(tl.float32)

    for reverse_segment in range(4):
        segment = 12 - reverse_segment * 4
        if RECOMPUTE_SEGMENT_STARTS:
            h011, h012, h021, h022 = one, zero, zero, one
            for replay in range(C):
                if replay < segment:
                    h011, h012, h021, h022 = _prefix_step(
                        h011, h012, h021, h022, G, GAMMA, LAM,
                        batch, chunk, head, keys, key_mask, replay,
                        T, H, K, C,
                    )
        else:
            if reverse_segment == 0:
                h011, h012 = cp1211, cp1212
                h021, h022 = cp1221, cp1222
            elif reverse_segment == 1:
                h011, h012 = cp811, cp812
                h021, h022 = cp821, cp822
            elif reverse_segment == 2:
                h011, h012 = cp411, cp412
                h021, h022 = cp421, cp422
            else:
                h011, h012 = cp011, cp012
                h021, h022 = cp021, cp022
        h111, h112, h121, h122 = _prefix_step(
            h011, h012, h021, h022, G, GAMMA, LAM,
            batch, chunk, head, keys, key_mask, segment,
            T, H, K, C,
        )
        h211, h212, h221, h222 = _prefix_step(
            h111, h112, h121, h122, G, GAMMA, LAM,
            batch, chunk, head, keys, key_mask, segment + 1,
            T, H, K, C,
        )
        h311, h312, h321, h322 = _prefix_step(
            h211, h212, h221, h222, G, GAMMA, LAM,
            batch, chunk, head, keys, key_mask, segment + 2,
            T, H, K, C,
        )
        for local_reverse in range(4):
            row = segment + 3 - local_reverse
            if local_reverse == 0:
                f11, f12, f21, f22 = h311, h312, h321, h322
            elif local_reverse == 1:
                f11, f12, f21, f22 = h211, h212, h221, h222
            elif local_reverse == 2:
                f11, f12, f21, f22 = h111, h112, h121, h122
            else:
                f11, f12, f21, f22 = h011, h012, h021, h022
            alpha, gamma, lam = _load_transition(
                G, GAMMA, LAM,
                batch, chunk, head, keys, key_mask, row,
                T, H, K, C,
            )
            p11 = gamma * alpha
            p12 = lam * p11
            p21 = (1.0 - gamma) * alpha
            p22 = (1.0 - lam) + lam * p21
            n11 = f11 * p11 + f12 * p21
            n12 = f11 * p12 + f12 * p22
            n21 = f21 * p11 + f22 * p21
            n22 = f21 * p12 + f22 * p22

            token = chunk * C + row
            offsets = ((batch * T + token) * H + head) * K + keys
            q = tl.load(Q + offsets, mask=key_mask, other=0.0).to(tl.float32)
            key = tl.load(K_IN + offsets, mask=key_mask, other=0.0).to(tl.float32)
            b_offsets = batch * SB0 + token * SB1 + head * SB2 + keys * SB3
            b = tl.load(B_GATE + b_offsets, mask=key_mask, other=0.0).to(tl.float32)
            output_offsets = system * C * K + row * K + keys
            da1 = tl.load(
                DA1 + output_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            da2 = tl.load(
                DA2 + output_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            dq1 = tl.load(
                DQ1 + output_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            dq2 = tl.load(
                DQ2 + output_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            dn11 = df11 + dq1 * q * SCALE
            dn12 = df12
            dn21 = df21 + dq2 * q * SCALE
            dn22 = df22
            dq = (dq1 * n11 + dq2 * n21) * SCALE

            dp11 = f11 * dn11 + f21 * dn21
            dp12 = f11 * dn12 + f21 * dn22
            dp21 = f12 * dn11 + f22 * dn21
            dp22 = f12 * dn12 + f22 * dn22
            prev11 = dn11 * p11 + dn12 * p12
            prev12 = dn11 * p21 + dn12 * p22
            prev21 = dn21 * p11 + dn22 * p12
            prev22 = dn21 * p21 + dn22 * p22

            erase = alpha * b * key
            prev11 += da1 * erase
            prev21 += da2 * erase
            derase = da1 * f11 + da2 * f21
            dactive = dp11 + lam * dp12
            dpassive = dp21 + lam * dp22
            dalpha = gamma * dactive + (1.0 - gamma) * dpassive
            dgamma = alpha * (dactive - dpassive)
            dlam = p11 * dp12 + (p21 - 1.0) * dp22
            dalpha += derase * b * key
            dk = derase * alpha * b
            db = derase * alpha * key
            dg = dalpha * alpha
            if ACCUMULATE:
                dq += tl.load(DQ + offsets, mask=key_mask, other=0.0).to(tl.float32)
                dk += tl.load(DK + offsets, mask=key_mask, other=0.0).to(tl.float32)
                dg += tl.load(DG + offsets, mask=key_mask, other=0.0).to(tl.float32)
                db += tl.load(DB + offsets, mask=key_mask, other=0.0).to(tl.float32)
            tl.store(DQ + offsets, dq, mask=key_mask)
            tl.store(DK + offsets, dk, mask=key_mask)
            tl.store(DG + offsets, dg, mask=key_mask)
            tl.store(DB + offsets, db, mask=key_mask)
            partial_offset = (system * NK + key_block) * C + row
            tl.store(
                DGAMMA_PARTIAL + partial_offset,
                tl.sum(tl.where(key_mask, dgamma, 0.0), axis=0),
            )
            tl.store(
                DLAM_PARTIAL + partial_offset,
                tl.sum(tl.where(key_mask, dlam, 0.0), axis=0),
            )
            df11, df12 = prev11, prev12
            df21, df22 = prev21, prev22


_PREFIX_BK = 128
_PREFIX_WARPS = 1


class _StableDirectPrefix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, g, b, gamma, lam, scale):
        q, k, g, gamma, lam = (
            x.contiguous() for x in (q, k, g, gamma, lam)
        )
        batch, length, heads, key_dim = q.shape
        chunks, chunk_size = length // 16, 16
        shape = (batch, heads, chunks, chunk_size, key_dim)
        outputs = tuple(
            torch.empty(shape, device=q.device, dtype=q.dtype)
            for _ in range(4)
        )
        transport = torch.empty(
            batch, heads, chunks, key_dim, 4,
            device=q.device, dtype=torch.float32,
        )
        block = min(_PREFIX_BK, triton.next_power_of_2(key_dim))
        nk = triton.cdiv(key_dim, block)
        systems = batch * heads * chunks
        _prefix_fwd_kernel[(systems * nk,)](
            q, k, g, b, gamma, lam, *outputs, transport,
            T=length, H=heads, N=chunks, K=key_dim, C=chunk_size,
            SCALE=float(scale),
            SB0=b.stride(0), SB1=b.stride(1),
            SB2=b.stride(2), SB3=b.stride(3),
            BK=block, num_warps=_PREFIX_WARPS, num_stages=2,
        )
        ctx.save_for_backward(q, k, g, b, gamma, lam)
        ctx.dimensions = (
            batch, length, heads, key_dim, chunks, systems, nk, block
        )
        ctx.scale = float(scale)
        ctx.input_dtypes = tuple(x.dtype for x in (q, k, g, b, gamma, lam))
        return *outputs, transport

    @staticmethod
    def backward(ctx, da1, da2, dq1, dq2, dtransport):
        q, k, g, b, gamma, lam = ctx.saved_tensors
        batch, length, heads, key_dim, chunks, systems, nk, block = ctx.dimensions
        grads = tuple(
            torch.empty_like(q, dtype=torch.float32) for _ in range(4)
        )
        partial_shape = (systems, nk, 16)
        dgamma_partial = torch.empty(
            partial_shape, device=q.device, dtype=torch.float32
        )
        dlam_partial = torch.empty_like(dgamma_partial)
        _prefix_local_bwd_kernel[(systems * nk,)](
            q, k, g, b, gamma, lam,
            da1.contiguous(), da2.contiguous(),
            dq1.contiguous(), dq2.contiguous(), dtransport.contiguous(),
            *grads, dgamma_partial, dlam_partial,
            T=length, H=heads, N=chunks, K=key_dim, C=16, NK=nk,
            SCALE=ctx.scale,
            SB0=b.stride(0), SB1=b.stride(1),
            SB2=b.stride(2), SB3=b.stride(3),
            ACCUMULATE=False,
            RECOMPUTE_SEGMENT_STARTS=needs_recomputed_segment_starts(q),
            BK=block, num_warps=_PREFIX_WARPS, num_stages=1,
        )

        def reduce(value):
            value = value.sum(1).view(batch, heads, chunks, 16)
            return value.reshape(
                batch, heads, length
            ).transpose(1, 2).contiguous()

        values = (*grads, reduce(dgamma_partial), reduce(dlam_partial))
        return tuple(
            value.to(dtype) for value, dtype in zip(values, ctx.input_dtypes)
        ) + (None,)



stable_direct_prefix = _StableDirectPrefix.apply

__all__ = ["stable_direct_prefix"]

"""Direct-relative C16 coefficients for channel-wise integral models.

Forward source groups share token loads.  Backward retains only four FP32
segment starts and recomputes short forward products, avoiding every inverse
or reciprocal decay while keeping register pressure bounded.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .general_direct_config import needs_recomputed_segment_starts


_CONFIGS = [
    triton.Config({"BS": bs, "BK": 128}, num_warps=warps, num_stages=stages)
    for bs, warps, stages in (
        (1, 4, 1),
        (2, 4, 1),
        (4, 4, 1),
        (4, 8, 1),
        (8, 8, 1),
        (4, 4, 2),
    )
]


@triton.autotune(configs=_CONFIGS, key=["K"])
@triton.jit
def _direct_c16_group_fwd_kernel(
    Q, K_IN, G, B_GATE, GAMMA, LAM,
    TAIL1, TAIL2, MATRIX, CAUSAL,
    T: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr, SCALE: tl.constexpr,
    SB0: tl.constexpr, SB1: tl.constexpr,
    SB2: tl.constexpr, SB3: tl.constexpr,
    BS: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    groups = tl.cdiv(C, BS)
    group = pid % groups
    system = pid // groups
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    sources = group * BS + tl.arange(0, BS)
    keys = tl.arange(0, BK)
    source_mask = sources < C
    key_mask = keys < K
    mask = source_mask[:, None] & key_mask[None, :]
    source_tokens = chunk * C + sources
    source_offsets = (
        ((batch * T + source_tokens[:, None]) * H + head) * K
        + keys[None, :]
    )
    source_keys = tl.load(
        K_IN + source_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    source_lam = tl.load(
        LAM + (batch * T + source_tokens) * H + head,
        mask=source_mask, other=0.0,
    ).to(tl.float32)
    c1 = tl.full((BS, BK), 1.0, tl.float32)
    c2 = source_lam[:, None] + tl.zeros((BS, BK), tl.float32)

    for target in range(C):
        token = chunk * C + target
        offsets = ((batch * T + token) * H + head) * K + keys
        q = tl.load(Q + offsets, mask=key_mask, other=0.0).to(tl.float32)
        key = tl.load(K_IN + offsets, mask=key_mask, other=0.0).to(tl.float32)
        g = tl.load(G + offsets, mask=key_mask, other=0.0).to(tl.float32)
        b_offsets = batch * SB0 + token * SB1 + head * SB2 + keys * SB3
        b = tl.load(B_GATE + b_offsets, mask=key_mask, other=0.0).to(tl.float32)
        alpha = tl.exp(g)
        erase = alpha * b * key
        interaction = tl.sum(
            tl.where(mask, c1 * source_keys * erase[None, :], 0.0), axis=1
        )
        active = sources < target
        matrix_offsets = system * C * C + target * C + sources
        tl.store(
            MATRIX + matrix_offsets,
            tl.where(active, interaction, 0.0),
            mask=source_mask,
        )

        scalar_offset = (batch * T + token) * H + head
        gamma = tl.load(GAMMA + scalar_offset).to(tl.float32)
        lam = tl.load(LAM + scalar_offset).to(tl.float32)
        p11 = gamma * alpha
        p12 = lam * p11
        p21 = (1.0 - gamma) * alpha
        p22 = (1.0 - lam) + lam * p21
        next1 = c1 * p11[None, :] + c2 * p21[None, :]
        next2 = c1 * p12[None, :] + c2 * p22[None, :]
        c1 = tl.where(active[:, None], next1, c1)
        c2 = tl.where(active[:, None], next2, c2)
        read = tl.sum(
            tl.where(mask, c1 * source_keys * (q * SCALE)[None, :], 0.0),
            axis=1,
        )
        tl.store(
            CAUSAL + matrix_offsets,
            tl.where(active | (sources == target), read, 0.0),
            mask=source_mask,
        )

    tail_offsets = system * C * K + sources[:, None] * K + keys[None, :]
    tl.store(TAIL1 + tail_offsets, c1 * source_keys, mask=mask)
    tl.store(TAIL2 + tail_offsets, c2 * source_keys, mask=mask)


def relative_forward_variant() -> str:
    variant = os.environ.get("INTEGRAL_RELATIVE_FWD", "tile")
    if variant not in ("tile", "rows"):
        raise ValueError("INTEGRAL_RELATIVE_FWD must be 'tile' or 'rows'")
    return variant


def grouped_forward(q, k, g, b, gamma, lam, scale):
    batch, length, heads, key_dim = q.shape
    chunks, chunk_size = length // 16, 16
    shape = (batch, heads, chunks, chunk_size, key_dim)
    tail1 = torch.empty(shape, device=q.device, dtype=q.dtype)
    tail2 = torch.empty_like(tail1)
    matrix = torch.empty(
        batch, heads, chunks, chunk_size, chunk_size,
        device=q.device, dtype=torch.float32,
    )
    causal = torch.empty_like(matrix, dtype=q.dtype)
    systems = batch * heads * chunks
    if relative_forward_variant() == "rows":
        from .general_direct_relative_rows_triton import (
            _direct_c16_fwd_rows_kernel,
            _rows_fwd_reduce_kernel,
        )
        block = min(32, key_dim)
        nk = triton.cdiv(key_dim, block)
        partial_shape = (systems, nk, chunk_size, chunk_size)
        mpart = torch.empty(partial_shape, device=q.device, dtype=torch.float32)
        cpart = torch.empty_like(mpart)
        _direct_c16_fwd_rows_kernel[(systems * nk,)](
            q, k, g, b, gamma, lam, None, None, None, None, None,  # no fused prefix
            tail1, tail2, mpart, cpart,
            FUSED_PREFIX=False,
            T=length, H=heads, N=chunks, K=key_dim, C=chunk_size, NK=nk,
            SCALE=float(scale),
            SB0=b.stride(0), SB1=b.stride(1),
            SB2=b.stride(2), SB3=b.stride(3),
            BK=block, num_warps=1, num_stages=1,
        )
        _rows_fwd_reduce_kernel[(systems,)](
            mpart, cpart, matrix, causal,
            C=chunk_size, NK=nk, BLOCK_NK=triton.next_power_of_2(nk),
            num_warps=1, num_stages=1,
        )
        return tail1, tail2, matrix, causal
    grid = lambda meta: (
        systems * triton.cdiv(chunk_size, meta["BS"]),
    )
    _direct_c16_group_fwd_kernel[grid](
        q, k, g, b, gamma, lam, tail1, tail2, matrix, causal,
        T=length, H=heads, N=chunks, K=key_dim, C=chunk_size,
        SCALE=float(scale),
        SB0=b.stride(0), SB1=b.stride(1),
        SB2=b.stride(2), SB3=b.stride(3),
    )
    return tail1, tail2, matrix, causal




def _relative_backward_config(tensor, key_dim):
    # (BK, num_warps).  One warp: the recurrence is serially dependent, so
    # more warps or pipeline stages have nothing to overlap.
    return (min(32, key_dim), 1)


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
def _state_step(
    c1, c2, G, GAMMA, LAM,
    batch, chunk, head, keys, key_mask, rows,
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
    next1 = c1 * p11[None, :] + c2 * p21[None, :]
    next2 = c1 * p12[None, :] + c2 * p22[None, :]
    active = rows < target
    return (
        tl.where(active[:, None], next1, c1),
        tl.where(active[:, None], next2, c2),
    )


@triton.jit
def _direct_c16_segment_bwd_kernel(
    Q, K_IN, G, B_GATE, GAMMA, LAM,
    DMATRIX, DCAUSAL, DTAIL1, DTAIL2,
    DQ, DK, DG, DB, DGAMMA_PARTIAL, DLAM_PARTIAL,
    T: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr, NK: tl.constexpr,
    SCALE: tl.constexpr,
    SB0: tl.constexpr, SB1: tl.constexpr,
    SB2: tl.constexpr, SB3: tl.constexpr,
    RECOMPUTE_SEGMENT_STARTS: tl.constexpr,
    ABL_DP: tl.constexpr,
    ABL_DM: tl.constexpr,
    BK: tl.constexpr,
):
    key_block = tl.program_id(0) % NK
    system = tl.program_id(0) // NK
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    rows = tl.arange(0, C)
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    mask = key_mask[None, :]
    tokens = chunk * C + rows
    offsets = (
        ((batch * T + tokens[:, None]) * H + head) * K
        + keys[None, :]
    )
    key = tl.load(K_IN + offsets, mask=mask, other=0.0).to(tl.float32)
    scalar_offsets = (batch * T + tokens) * H + head
    lam_rows = tl.load(LAM + scalar_offsets).to(tl.float32)

    # One forward pass saves only four segment starts.  Every saved tensor is
    # an exact FP32 recurrence state; no division or inverse prefix appears.
    c1 = tl.full((C, BK), 1.0, tl.float32)
    c2 = tl.broadcast_to(lam_rows[:, None], (C, BK))
    cp01, cp02 = c1, c2
    cp41, cp42 = c1, c2
    cp81, cp82 = c1, c2
    cp121, cp122 = c1, c2
    for target in range(C):
        if not RECOMPUTE_SEGMENT_STARTS:
            if target == 4:
                cp41, cp42 = c1, c2
            if target == 8:
                cp81, cp82 = c1, c2
            if target == 12:
                cp121, cp122 = c1, c2
        c1, c2 = _state_step(
            c1, c2, G, GAMMA, LAM,
            batch, chunk, head, keys, key_mask, rows, target,
            T, H, K, C,
        )

    tail_offsets = (system * C + rows[:, None]) * K + keys[None, :]
    dtail1 = tl.load(DTAIL1 + tail_offsets, mask=mask, other=0.0).to(tl.float32)
    dtail2 = tl.load(DTAIL2 + tail_offsets, mask=mask, other=0.0).to(tl.float32)
    dc1 = dtail1 * key
    dc2 = dtail2 * key
    dk = dtail1 * c1 + dtail2 * c2
    dlam_key = tl.zeros((C, BK), tl.float32)

    for reverse_segment in range(4):
        segment = 12 - reverse_segment * 4
        if RECOMPUTE_SEGMENT_STARTS:
            h01 = tl.full((C, BK), 1.0, tl.float32)
            h02 = tl.broadcast_to(lam_rows[:, None], (C, BK))
            for replay in range(C):
                if replay < segment:
                    h01, h02 = _state_step(
                        h01, h02, G, GAMMA, LAM,
                        batch, chunk, head, keys, key_mask, rows, replay,
                        T, H, K, C,
                    )
        else:
            if reverse_segment == 0:
                h01, h02 = cp121, cp122
            elif reverse_segment == 1:
                h01, h02 = cp81, cp82
            elif reverse_segment == 2:
                h01, h02 = cp41, cp42
            else:
                h01, h02 = cp01, cp02
        h11, h12 = _state_step(
            h01, h02, G, GAMMA, LAM,
            batch, chunk, head, keys, key_mask, rows, segment,
            T, H, K, C,
        )
        h21, h22 = _state_step(
            h11, h12, G, GAMMA, LAM,
            batch, chunk, head, keys, key_mask, rows, segment + 1,
            T, H, K, C,
        )
        h31, h32 = _state_step(
            h21, h22, G, GAMMA, LAM,
            batch, chunk, head, keys, key_mask, rows, segment + 2,
            T, H, K, C,
        )

        for local_reverse in range(4):
            target = segment + 3 - local_reverse
            if local_reverse == 0:
                cprev1, cprev2 = h31, h32
            elif local_reverse == 1:
                cprev1, cprev2 = h21, h22
            elif local_reverse == 2:
                cprev1, cprev2 = h11, h12
            else:
                cprev1, cprev2 = h01, h02
            target_mask = rows == target
            active = rows < target
            select_target = target_mask[:, None]
            alpha, gamma, lam = _load_transition(
                G, GAMMA, LAM,
                batch, chunk, head, keys, key_mask, target,
                T, H, K, C,
            )
            token = chunk * C + target
            target_offsets = ((batch * T + token) * H + head) * K + keys
            key_t = tl.load(
                K_IN + target_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            p11 = gamma * alpha
            p12 = lam * p11
            p21 = (1.0 - gamma) * alpha
            p22 = (1.0 - lam) + lam * p21
            # cafter is the state after `target`, which the segment expansion
            # above already produced for three of the four steps: applying
            # P_target to h01/h11/h21 is by definition h11/h21/h31.  Only the
            # newest one (past the end of the segment) has to be formed.
            # cafter2 is never read, so it is not formed at all.
            if local_reverse == 0:
                cafter1 = tl.where(
                    active[:, None],
                    cprev1 * p11[None, :] + cprev2 * p21[None, :],
                    cprev1,
                )
            elif local_reverse == 1:
                cafter1 = h31
            elif local_reverse == 2:
                cafter1 = h21
            else:
                cafter1 = h11

            q_t = tl.load(
                Q + target_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            b_offsets = batch * SB0 + token * SB1 + head * SB2 + keys * SB3
            b_t = tl.load(
                B_GATE + b_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            matrix_offsets = system * C * C + target * C + rows
            if ABL_DM:
                # TIMING ONLY: DMATRIX/DCAUSAL are reloaded once per key block
                # (4x redundant at BK=32); substitute already-resident data to
                # price those loads.
                dm = key_t[0] + tl.zeros((C,), tl.float32)
                dl = dm
            else:
                dm = tl.load(DMATRIX + matrix_offsets).to(tl.float32)
                dl = tl.load(DCAUSAL + matrix_offsets).to(tl.float32)

            causal_weight = tl.where(active, dl, 0.0)[:, None]
            diagonal_dl = tl.load(
                DCAUSAL + system * C * C + target * C + target
            ).to(tl.float32)
            after1 = dc1 + causal_weight * key * q_t[None, :] * SCALE
            after2 = dc2
            dq_t = (
                tl.sum(causal_weight * cafter1 * key, axis=0) * SCALE
                + diagonal_dl * key_t * SCALE
            )
            dk += causal_weight * cafter1 * q_t[None, :] * SCALE
            dk += tl.where(
                select_target, diagonal_dl * q_t[None, :] * SCALE, 0.0
            )

            dp11 = tl.sum(
                tl.where(active[:, None], cprev1 * after1, 0.0), axis=0
            )
            if ABL_DP:
                # TIMING ONLY: reuse one reduction for the other three, to
                # price the marginal cost of 48 of the 96 cross-lane sums.
                dp12 = dp11
                dp21 = dp11
                dp22 = dp11
            else:
                dp12 = tl.sum(
                    tl.where(active[:, None], cprev1 * after2, 0.0), axis=0
                )
                dp21 = tl.sum(
                    tl.where(active[:, None], cprev2 * after1, 0.0), axis=0
                )
                dp22 = tl.sum(
                    tl.where(active[:, None], cprev2 * after2, 0.0), axis=0
                )
            previous1 = after1 * p11[None, :] + after2 * p12[None, :]
            previous2 = after1 * p21[None, :] + after2 * p22[None, :]

            erase = alpha * b_t * key_t
            matrix_weight = tl.where(active, dm, 0.0)[:, None]
            previous1 += matrix_weight * key * erase[None, :]
            dk += matrix_weight * cprev1 * erase[None, :]
            derase = tl.sum(matrix_weight * cprev1 * key, axis=0)
            dc1 = tl.where(active[:, None], previous1, dc1)
            dc2 = tl.where(active[:, None], previous2, dc2)

            dactive = dp11 + lam * dp12
            dpassive = dp21 + lam * dp22
            dalpha = gamma * dactive + (1.0 - gamma) * dpassive
            dgamma_t = alpha * (dactive - dpassive)
            dlam_t = p11 * dp12 + (p21 - 1.0) * dp22
            dalpha += derase * b_t * key_t
            db_t = derase * alpha * key_t
            dk_t = derase * alpha * b_t
            dg_t = dalpha * alpha

            dk += tl.where(select_target, dk_t[None, :], 0.0)
            dlam_key += tl.where(select_target, dlam_t[None, :], 0.0)
            # Every target is visited exactly once across the two unrolled
            # loops, and dq/dg/db/dgamma are written only under select_target.
            # Writing the (BK,) row straight out avoids a masked add over the
            # whole (C, BK) tile per target: four accumulators x sixteen
            # targets of tile traffic removed, worth ~6% end to end.
            tl.store(DQ + target_offsets, dq_t, mask=key_mask)
            tl.store(DG + target_offsets, dg_t, mask=key_mask)
            tl.store(DB + target_offsets, db_t, mask=key_mask)
            tl.store(
                DGAMMA_PARTIAL + (system * NK + key_block) * C + target,
                tl.sum(tl.where(key_mask, dgamma_t, 0.0), axis=0),
            )

    dlam_key += dc2
    tl.store(DK + offsets, dk, mask=mask)
    partial_offsets = (system * NK + key_block) * C + rows
    tl.store(
        DLAM_PARTIAL + partial_offsets,
        tl.sum(tl.where(mask, dlam_key, 0.0), axis=1),
    )


@triton.jit
def _transition_matrix(
    G, GAMMA, LAM,
    batch, chunk, head, keys, key_mask,
    target: tl.constexpr,
    T: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr,
):
    """Load one transition and form its 2x2 block once, for all consumers."""
    alpha, gamma, lam = _load_transition(
        G, GAMMA, LAM, batch, chunk, head, keys, key_mask, target,
        T, H, K, C,
    )
    p11 = gamma * alpha
    p12 = lam * p11
    p21 = (1.0 - gamma) * alpha
    p22 = (1.0 - lam) + lam * p21
    return alpha, gamma, lam, p11, p12, p21, p22


@triton.jit
def _advance(c1, c2, p11, p12, p21, p22, rows, target: tl.constexpr):
    """Apply an already-formed transition; the arithmetic half of _state_step."""
    next1 = c1 * p11[None, :] + c2 * p21[None, :]
    next2 = c1 * p12[None, :] + c2 * p22[None, :]
    active = rows < target
    return (
        tl.where(active[:, None], next1, c1),
        tl.where(active[:, None], next2, c2),
    )


@triton.jit
def _direct_c16_segment_bwd_reuse_kernel(
    Q, K_IN, G, B_GATE, GAMMA, LAM,
    DMATRIX, DCAUSAL, DTAIL1, DTAIL2,
    DQ, DK, DG, DB, DGAMMA_PARTIAL, DLAM_PARTIAL,
    T: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr, NK: tl.constexpr,
    SCALE: tl.constexpr,
    SB0: tl.constexpr, SB1: tl.constexpr,
    SB2: tl.constexpr, SB3: tl.constexpr,
    RECOMPUTE_SEGMENT_STARTS: tl.constexpr,
    BK: tl.constexpr,
):
    """_direct_c16_segment_bwd_kernel without its repeated recurrence work.

    Three overlaps are removed; every value is produced by the same operations
    in the same order as before, so the gradients are bit-identical:

    1. The gradient loop reloaded a transition per target, but the segment
       expansion above it had already loaded three of every four.  The four
       transitions of a segment are now formed once and shared.
    2. The standalone 16-step forward pass existed only to produce the tail
       state.  Segment 12 replays 0..11 and expands 12..14 regardless, so the
       tail is one more application of a transition that segment already holds.
    3. `cafter` for the newest target of a segment is by definition the start
       state of the segment above it, which the previous iteration just built.
       It is carried down instead of re-applied.
    """
    key_block = tl.program_id(0) % NK
    system = tl.program_id(0) // NK
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    rows = tl.arange(0, C)
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    mask = key_mask[None, :]
    tokens = chunk * C + rows
    offsets = (
        ((batch * T + tokens[:, None]) * H + head) * K
        + keys[None, :]
    )
    key = tl.load(K_IN + offsets, mask=mask, other=0.0).to(tl.float32)
    scalar_offsets = (batch * T + tokens) * H + head
    lam_rows = tl.load(LAM + scalar_offsets).to(tl.float32)

    tail_offsets = (system * C + rows[:, None]) * K + keys[None, :]
    dtail1 = tl.load(DTAIL1 + tail_offsets, mask=mask, other=0.0).to(tl.float32)
    dtail2 = tl.load(DTAIL2 + tail_offsets, mask=mask, other=0.0).to(tl.float32)
    dc1 = dtail1 * key
    dc2 = dtail2 * key
    dlam_key = tl.zeros((C, BK), tl.float32)

    c1 = tl.full((C, BK), 1.0, tl.float32)
    c2 = tl.broadcast_to(lam_rows[:, None], (C, BK))
    cp01, cp02 = c1, c2
    cp41, cp42 = c1, c2
    cp81, cp82 = c1, c2
    cp121, cp122 = c1, c2
    if RECOMPUTE_SEGMENT_STARTS:
        # No forward pass: segment 12 reaches the tail on its own.
        dk = tl.zeros((C, BK), tl.float32)
    else:
        for target in range(C):
            if target == 4:
                cp41, cp42 = c1, c2
            if target == 8:
                cp81, cp82 = c1, c2
            if target == 12:
                cp121, cp122 = c1, c2
            c1, c2 = _state_step(
                c1, c2, G, GAMMA, LAM,
                batch, chunk, head, keys, key_mask, rows, target,
                T, H, K, C,
            )
        dk = dtail1 * c1 + dtail2 * c2

    # State after the newest target of the segment being processed.  The
    # checkpoint path already holds it as the forward pass's final state.
    carry1, carry2 = c1, c2

    for reverse_segment in range(4):
        segment = 12 - reverse_segment * 4
        if RECOMPUTE_SEGMENT_STARTS:
            h01 = tl.full((C, BK), 1.0, tl.float32)
            h02 = tl.broadcast_to(lam_rows[:, None], (C, BK))
            for replay in range(C):
                if replay < segment:
                    h01, h02 = _state_step(
                        h01, h02, G, GAMMA, LAM,
                        batch, chunk, head, keys, key_mask, rows, replay,
                        T, H, K, C,
                    )
        else:
            if reverse_segment == 0:
                h01, h02 = cp121, cp122
            elif reverse_segment == 1:
                h01, h02 = cp81, cp82
            elif reverse_segment == 2:
                h01, h02 = cp41, cp42
            else:
                h01, h02 = cp01, cp02

        a0, g0, l0, p110, p120, p210, p220 = _transition_matrix(
            G, GAMMA, LAM, batch, chunk, head, keys, key_mask,
            segment, T, H, K, C,
        )
        a1, g1, l1, p111, p121, p211, p221 = _transition_matrix(
            G, GAMMA, LAM, batch, chunk, head, keys, key_mask,
            segment + 1, T, H, K, C,
        )
        a2, g2, l2, p112, p122, p212, p222 = _transition_matrix(
            G, GAMMA, LAM, batch, chunk, head, keys, key_mask,
            segment + 2, T, H, K, C,
        )
        a3, g3, l3, p113, p123, p213, p223 = _transition_matrix(
            G, GAMMA, LAM, batch, chunk, head, keys, key_mask,
            segment + 3, T, H, K, C,
        )

        h11, h12 = _advance(h01, h02, p110, p120, p210, p220, rows, segment)
        h21, h22 = _advance(h11, h12, p111, p121, p211, p221, rows, segment + 1)
        h31, h32 = _advance(h21, h22, p112, p122, p212, p222, rows, segment + 2)

        if RECOMPUTE_SEGMENT_STARTS:
            if reverse_segment == 0:
                carry1, carry2 = _advance(
                    h31, h32, p113, p123, p213, p223, rows, segment + 3
                )
                dk += dtail1 * carry1 + dtail2 * carry2

        for local_reverse in range(4):
            target = segment + 3 - local_reverse
            if local_reverse == 0:
                cprev1, cprev2 = h31, h32
                cafter1 = carry1
                alpha, gamma, lam = a3, g3, l3
                p11, p12, p21, p22 = p113, p123, p213, p223
            elif local_reverse == 1:
                cprev1, cprev2 = h21, h22
                cafter1 = h31
                alpha, gamma, lam = a2, g2, l2
                p11, p12, p21, p22 = p112, p122, p212, p222
            elif local_reverse == 2:
                cprev1, cprev2 = h11, h12
                cafter1 = h21
                alpha, gamma, lam = a1, g1, l1
                p11, p12, p21, p22 = p111, p121, p211, p221
            else:
                cprev1, cprev2 = h01, h02
                cafter1 = h11
                alpha, gamma, lam = a0, g0, l0
                p11, p12, p21, p22 = p110, p120, p210, p220

            target_mask = rows == target
            active = rows < target
            select_target = target_mask[:, None]
            token = chunk * C + target
            target_offsets = ((batch * T + token) * H + head) * K + keys
            key_t = tl.load(
                K_IN + target_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)

            q_t = tl.load(
                Q + target_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            b_offsets = batch * SB0 + token * SB1 + head * SB2 + keys * SB3
            b_t = tl.load(
                B_GATE + b_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            matrix_offsets = system * C * C + target * C + rows
            dm = tl.load(DMATRIX + matrix_offsets).to(tl.float32)
            dl = tl.load(DCAUSAL + matrix_offsets).to(tl.float32)

            causal_weight = tl.where(active, dl, 0.0)[:, None]
            diagonal_dl = tl.load(
                DCAUSAL + system * C * C + target * C + target
            ).to(tl.float32)
            after1 = dc1 + causal_weight * key * q_t[None, :] * SCALE
            after2 = dc2
            dq_t = (
                tl.sum(causal_weight * cafter1 * key, axis=0) * SCALE
                + diagonal_dl * key_t * SCALE
            )
            dk += causal_weight * cafter1 * q_t[None, :] * SCALE
            dk += tl.where(
                select_target, diagonal_dl * q_t[None, :] * SCALE, 0.0
            )

            dp11 = tl.sum(
                tl.where(active[:, None], cprev1 * after1, 0.0), axis=0
            )
            dp12 = tl.sum(
                tl.where(active[:, None], cprev1 * after2, 0.0), axis=0
            )
            dp21 = tl.sum(
                tl.where(active[:, None], cprev2 * after1, 0.0), axis=0
            )
            dp22 = tl.sum(
                tl.where(active[:, None], cprev2 * after2, 0.0), axis=0
            )
            previous1 = after1 * p11[None, :] + after2 * p12[None, :]
            previous2 = after1 * p21[None, :] + after2 * p22[None, :]

            erase = alpha * b_t * key_t
            matrix_weight = tl.where(active, dm, 0.0)[:, None]
            previous1 += matrix_weight * key * erase[None, :]
            dk += matrix_weight * cprev1 * erase[None, :]
            derase = tl.sum(matrix_weight * cprev1 * key, axis=0)
            dc1 = tl.where(active[:, None], previous1, dc1)
            dc2 = tl.where(active[:, None], previous2, dc2)

            dactive = dp11 + lam * dp12
            dpassive = dp21 + lam * dp22
            dalpha = gamma * dactive + (1.0 - gamma) * dpassive
            dgamma_t = alpha * (dactive - dpassive)
            dlam_t = p11 * dp12 + (p21 - 1.0) * dp22
            dalpha += derase * b_t * key_t
            db_t = derase * alpha * key_t
            dk_t = derase * alpha * b_t
            dg_t = dalpha * alpha

            dk += tl.where(select_target, dk_t[None, :], 0.0)
            dlam_key += tl.where(select_target, dlam_t[None, :], 0.0)
            tl.store(DQ + target_offsets, dq_t, mask=key_mask)
            tl.store(DG + target_offsets, dg_t, mask=key_mask)
            tl.store(DB + target_offsets, db_t, mask=key_mask)
            tl.store(
                DGAMMA_PARTIAL + (system * NK + key_block) * C + target,
                tl.sum(tl.where(key_mask, dgamma_t, 0.0), axis=0),
            )

        carry1, carry2 = h01, h02

    dlam_key += dc2
    tl.store(DK + offsets, dk, mask=mask)
    partial_offsets = (system * NK + key_block) * C + rows
    tl.store(
        DLAM_PARTIAL + partial_offsets,
        tl.sum(tl.where(mask, dlam_key, 0.0), axis=1),
    )


@triton.jit
def _direct_c16_segment_bwd_scratch_kernel(
    Q, K_IN, G, B_GATE, GAMMA, LAM,
    DMATRIX, DCAUSAL, DTAIL1, DTAIL2,
    DQ, DK, DG, DB, DGAMMA_PARTIAL, DLAM_PARTIAL,
    SCRATCH,
    T: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr, NK: tl.constexpr,
    SCALE: tl.constexpr,
    SB0: tl.constexpr, SB1: tl.constexpr,
    SB2: tl.constexpr, SB3: tl.constexpr,
    CKPT: tl.constexpr,
    BK: tl.constexpr,
):
    """Reuse kernel with segment starts checkpointed through global memory.

    The SM103 fault only appears when a recurrence state stays live in
    registers across the segment loop.  Storing the segment starts to a
    per-program scratch slice during one forward sweep and loading each one
    back inside its segment keeps every tile short-lived, without the replay.

    CKPT == 3 stores the starts of segments 4, 8 and 12 (no replay at all).
    CKPT == 1 stores only the start of segment 8; segments 12 and 4 replay
    four steps each from the nearest stored or initial state.
    """
    key_block = tl.program_id(0) % NK
    system = tl.program_id(0) // NK
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    rows = tl.arange(0, C)
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    mask = key_mask[None, :]
    tokens = chunk * C + rows
    offsets = (
        ((batch * T + tokens[:, None]) * H + head) * K
        + keys[None, :]
    )
    key = tl.load(K_IN + offsets, mask=mask, other=0.0).to(tl.float32)
    scalar_offsets = (batch * T + tokens) * H + head
    lam_rows = tl.load(LAM + scalar_offsets).to(tl.float32)

    tail_offsets = (system * C + rows[:, None]) * K + keys[None, :]
    dtail1 = tl.load(DTAIL1 + tail_offsets, mask=mask, other=0.0).to(tl.float32)
    dtail2 = tl.load(DTAIL2 + tail_offsets, mask=mask, other=0.0).to(tl.float32)
    dc1 = dtail1 * key
    dc2 = dtail2 * key
    dlam_key = tl.zeros((C, BK), tl.float32)

    scratch = SCRATCH + tl.program_id(0).to(tl.int64) * (CKPT * 2 * C * BK)
    tile = rows[:, None] * BK + tl.arange(0, BK)[None, :]

    c1 = tl.full((C, BK), 1.0, tl.float32)
    c2 = tl.broadcast_to(lam_rows[:, None], (C, BK))
    for target in range(C):
        if CKPT == 3:
            if target == 4:
                tl.store(scratch + 0 * C * BK + tile, c1)
                tl.store(scratch + 1 * C * BK + tile, c2)
            if target == 8:
                tl.store(scratch + 2 * C * BK + tile, c1)
                tl.store(scratch + 3 * C * BK + tile, c2)
            if target == 12:
                tl.store(scratch + 4 * C * BK + tile, c1)
                tl.store(scratch + 5 * C * BK + tile, c2)
        else:
            if target == 8:
                tl.store(scratch + 0 * C * BK + tile, c1)
                tl.store(scratch + 1 * C * BK + tile, c2)
        c1, c2 = _state_step(
            c1, c2, G, GAMMA, LAM,
            batch, chunk, head, keys, key_mask, rows, target,
            T, H, K, C,
        )
    dk = dtail1 * c1 + dtail2 * c2
    carry1, carry2 = c1, c2

    for reverse_segment in range(4):
        segment = 12 - reverse_segment * 4
        if CKPT == 3:
            if reverse_segment == 0:
                h01 = tl.load(scratch + 4 * C * BK + tile)
                h02 = tl.load(scratch + 5 * C * BK + tile)
            elif reverse_segment == 1:
                h01 = tl.load(scratch + 2 * C * BK + tile)
                h02 = tl.load(scratch + 3 * C * BK + tile)
            elif reverse_segment == 2:
                h01 = tl.load(scratch + 0 * C * BK + tile)
                h02 = tl.load(scratch + 1 * C * BK + tile)
            else:
                h01 = tl.full((C, BK), 1.0, tl.float32)
                h02 = tl.broadcast_to(lam_rows[:, None], (C, BK))
        else:
            if reverse_segment <= 1:
                h01 = tl.load(scratch + 0 * C * BK + tile)
                h02 = tl.load(scratch + 1 * C * BK + tile)
            else:
                h01 = tl.full((C, BK), 1.0, tl.float32)
                h02 = tl.broadcast_to(lam_rows[:, None], (C, BK))
            if reverse_segment == 0 or reverse_segment == 2:
                for replay in range(C):
                    if replay >= segment - 4 and replay < segment:
                        h01, h02 = _state_step(
                            h01, h02, G, GAMMA, LAM,
                            batch, chunk, head, keys, key_mask, rows, replay,
                            T, H, K, C,
                        )

        a0, g0, l0, p110, p120, p210, p220 = _transition_matrix(
            G, GAMMA, LAM, batch, chunk, head, keys, key_mask,
            segment, T, H, K, C,
        )
        a1, g1, l1, p111, p121, p211, p221 = _transition_matrix(
            G, GAMMA, LAM, batch, chunk, head, keys, key_mask,
            segment + 1, T, H, K, C,
        )
        a2, g2, l2, p112, p122, p212, p222 = _transition_matrix(
            G, GAMMA, LAM, batch, chunk, head, keys, key_mask,
            segment + 2, T, H, K, C,
        )
        a3, g3, l3, p113, p123, p213, p223 = _transition_matrix(
            G, GAMMA, LAM, batch, chunk, head, keys, key_mask,
            segment + 3, T, H, K, C,
        )

        h11, h12 = _advance(h01, h02, p110, p120, p210, p220, rows, segment)
        h21, h22 = _advance(h11, h12, p111, p121, p211, p221, rows, segment + 1)
        h31, h32 = _advance(h21, h22, p112, p122, p212, p222, rows, segment + 2)

        for local_reverse in range(4):
            target = segment + 3 - local_reverse
            if local_reverse == 0:
                cprev1, cprev2 = h31, h32
                cafter1 = carry1
                alpha, gamma, lam = a3, g3, l3
                p11, p12, p21, p22 = p113, p123, p213, p223
            elif local_reverse == 1:
                cprev1, cprev2 = h21, h22
                cafter1 = h31
                alpha, gamma, lam = a2, g2, l2
                p11, p12, p21, p22 = p112, p122, p212, p222
            elif local_reverse == 2:
                cprev1, cprev2 = h11, h12
                cafter1 = h21
                alpha, gamma, lam = a1, g1, l1
                p11, p12, p21, p22 = p111, p121, p211, p221
            else:
                cprev1, cprev2 = h01, h02
                cafter1 = h11
                alpha, gamma, lam = a0, g0, l0
                p11, p12, p21, p22 = p110, p120, p210, p220

            target_mask = rows == target
            active = rows < target
            select_target = target_mask[:, None]
            token = chunk * C + target
            target_offsets = ((batch * T + token) * H + head) * K + keys
            key_t = tl.load(
                K_IN + target_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)

            q_t = tl.load(
                Q + target_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            b_offsets = batch * SB0 + token * SB1 + head * SB2 + keys * SB3
            b_t = tl.load(
                B_GATE + b_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            matrix_offsets = system * C * C + target * C + rows
            dm = tl.load(DMATRIX + matrix_offsets).to(tl.float32)
            dl = tl.load(DCAUSAL + matrix_offsets).to(tl.float32)

            causal_weight = tl.where(active, dl, 0.0)[:, None]
            diagonal_dl = tl.load(
                DCAUSAL + system * C * C + target * C + target
            ).to(tl.float32)
            after1 = dc1 + causal_weight * key * q_t[None, :] * SCALE
            after2 = dc2
            dq_t = (
                tl.sum(causal_weight * cafter1 * key, axis=0) * SCALE
                + diagonal_dl * key_t * SCALE
            )
            dk += causal_weight * cafter1 * q_t[None, :] * SCALE
            dk += tl.where(
                select_target, diagonal_dl * q_t[None, :] * SCALE, 0.0
            )

            dp11 = tl.sum(
                tl.where(active[:, None], cprev1 * after1, 0.0), axis=0
            )
            dp12 = tl.sum(
                tl.where(active[:, None], cprev1 * after2, 0.0), axis=0
            )
            dp21 = tl.sum(
                tl.where(active[:, None], cprev2 * after1, 0.0), axis=0
            )
            dp22 = tl.sum(
                tl.where(active[:, None], cprev2 * after2, 0.0), axis=0
            )
            previous1 = after1 * p11[None, :] + after2 * p12[None, :]
            previous2 = after1 * p21[None, :] + after2 * p22[None, :]

            erase = alpha * b_t * key_t
            matrix_weight = tl.where(active, dm, 0.0)[:, None]
            previous1 += matrix_weight * key * erase[None, :]
            dk += matrix_weight * cprev1 * erase[None, :]
            derase = tl.sum(matrix_weight * cprev1 * key, axis=0)
            dc1 = tl.where(active[:, None], previous1, dc1)
            dc2 = tl.where(active[:, None], previous2, dc2)

            dactive = dp11 + lam * dp12
            dpassive = dp21 + lam * dp22
            dalpha = gamma * dactive + (1.0 - gamma) * dpassive
            dgamma_t = alpha * (dactive - dpassive)
            dlam_t = p11 * dp12 + (p21 - 1.0) * dp22
            dalpha += derase * b_t * key_t
            db_t = derase * alpha * key_t
            dk_t = derase * alpha * b_t
            dg_t = dalpha * alpha

            dk += tl.where(select_target, dk_t[None, :], 0.0)
            dlam_key += tl.where(select_target, dlam_t[None, :], 0.0)
            tl.store(DQ + target_offsets, dq_t, mask=key_mask)
            tl.store(DG + target_offsets, dg_t, mask=key_mask)
            tl.store(DB + target_offsets, db_t, mask=key_mask)
            tl.store(
                DGAMMA_PARTIAL + (system * NK + key_block) * C + target,
                tl.sum(tl.where(key_mask, dgamma_t, 0.0), axis=0),
            )

        carry1, carry2 = h01, h02

    dlam_key += dc2
    tl.store(DK + offsets, dk, mask=mask)
    partial_offsets = (system * NK + key_block) * C + rows
    tl.store(
        DLAM_PARTIAL + partial_offsets,
        tl.sum(tl.where(mask, dlam_key, 0.0), axis=1),
    )


class _StableDirectRelative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, g, b, gamma, lam, scale):
        values = tuple(
            value.contiguous() if value.stride(-1) != 0 else value
            for value in (q, k, g, b, gamma, lam)
        )
        q, k, g, b, gamma, lam = values
        outputs = grouped_forward(q, k, g, b, gamma, lam, scale)
        ctx.save_for_backward(*values)
        batch, length, heads, key_dim = q.shape
        ctx.dimensions = (batch, length, heads, key_dim, length // 16)
        ctx.scale = float(scale)
        ctx.input_dtypes = tuple(x.dtype for x in values)
        return outputs

    @staticmethod
    def backward(ctx, dtail1, dtail2, dmatrix, dcausal):
        q, k, g, b, gamma, lam = ctx.saved_tensors
        batch, length, heads, key_dim, chunks = ctx.dimensions
        chunk_size = 16
        systems = batch * heads * chunks

        def grad_or_zero(value, shape, dtype):
            if value is None:
                return torch.zeros(shape, device=q.device, dtype=dtype)
            return value.contiguous()

        tail_shape = (batch, heads, chunks, chunk_size, key_dim)
        square_shape = (batch, heads, chunks, chunk_size, chunk_size)
        dtail1 = grad_or_zero(dtail1, tail_shape, q.dtype)
        dtail2 = grad_or_zero(dtail2, tail_shape, q.dtype)
        dmatrix = grad_or_zero(dmatrix, square_shape, torch.float32)
        dcausal = grad_or_zero(dcausal, square_shape, q.dtype)
        dq, dk, dg, db = (
            torch.empty_like(q, dtype=torch.float32) for _ in range(4)
        )
        bk, backward_warps = _relative_backward_config(q, key_dim)
        nk = triton.cdiv(key_dim, bk)
        dgamma_partial = torch.empty(
            systems, nk, chunk_size, device=q.device, dtype=torch.float32
        )
        dlam_partial = torch.empty_like(dgamma_partial)
        _direct_c16_segment_bwd_kernel[(systems * nk,)](
            q, k, g, b, gamma, lam,
            dmatrix, dcausal, dtail1, dtail2,
            dq, dk, dg, db, dgamma_partial, dlam_partial,
            T=length, H=heads, N=chunks, K=key_dim, C=chunk_size,
            NK=nk, SCALE=ctx.scale,
            SB0=b.stride(0), SB1=b.stride(1),
            SB2=b.stride(2), SB3=b.stride(3),
            RECOMPUTE_SEGMENT_STARTS=needs_recomputed_segment_starts(q),
            ABL_DP=False, ABL_DM=False,
            BK=bk, num_warps=backward_warps, num_stages=1,
        )

        def reduce_scalar(value):
            reduced = value.sum(dim=1).view(
                batch, heads, chunks, chunk_size
            )
            return reduced.reshape(
                batch, heads, length
            ).transpose(1, 2).contiguous()

        values = (
            dq, dk, dg, db,
            reduce_scalar(dgamma_partial), reduce_scalar(dlam_partial),
        )
        return tuple(
            value.to(dtype) for value, dtype in zip(values, ctx.input_dtypes)
        ) + (None,)



stable_direct_relative = _StableDirectRelative.apply

__all__ = [
    "stable_direct_relative",
    "_direct_c16_segment_bwd_reuse_kernel",
    "_direct_c16_segment_bwd_scratch_kernel",
]

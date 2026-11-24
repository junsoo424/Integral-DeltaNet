"""Direct C16 readout-side operand builder in Triton.

The torch builder in :mod:`readout` materializes three ``C x C x K`` tensors
per block (relative decay ``E``, slow admission ``P`` and ``q k``) through
torch.compile and pays for their backward with 16 saved recurrence steps.
This module computes the same operands row by row inside
one Triton program per (batch, head, chunk, key block): a ``[C, BK]`` tile of
keys and cumulative decays stays in registers, every row ``i`` forms its
relative decay ``E_i = exp(G_i - G_j)`` (``j <= i``) and advances the convex
admission ``P_i = (1 - lambda_i) P_{i-1} + lambda_i E_i``, and the two
``C x C`` reductions (``interaction``, ``causal``) are reduced over the key
block immediately.  Nothing ``C x C x K`` ever reaches memory.

Tiles are ``[BK, C]`` with the key axis first: Triton then splits the key
axis (not the rows) across warps, every reduction over rows is thread-local,
and one program per chunk (``BK = K``, four warps) amortizes the 16-row
serial chain over the whole head dimension.

The backward is one forward sweep over the rows as well.  The slow
admission is written with explicit weights ``P_i = sum_tau W[i, tau] E_tau``,
``W[i, tau] = lambda_tau prod_{tau < s <= i} (1 - lambda_s)`` (every factor
at most one), so its cotangents are contractions rather than a reverse
recurrence: ``dE_tau += sum_i W[i, tau] dP_i`` and ``dW[i, tau] = <dP_i,
E_tau>``.  ``dP_i = dcausal_ij (1 - gamma_i) q_i k_j`` is rank structured,
which turns both into ``16 x 16`` tensor-core dots per row; the scalar chain
rule from ``dW`` to ``lambda`` is a few ``16 x 16`` tables in the same kernel.  No ``1 / (1 - lambda)``, no
``1 / lambda`` and no ``exp(-G)`` appear anywhere, so the ``G_16 ~ -800``
decays of a trained KDA and ``lambda -> 1`` saturation are both safe.

``gamma`` is read through explicit strides as ``[B, T, H, K]``: a scalar
per-head gate is passed as a stride-0 expansion, a per-channel gate as a real
tensor.  The same kernels therefore serve both the scalar and the vector
readout mix.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .arch import (
    RELOAD_BWD_CONFIGS,
    TRANSPORT_READOUT_GATE_ARCHS,
    on_arch,
    readout_rows_backward_flags,
    readout_rows_forward_flags,
)
from .readout_rows_triton import _readout_rows_bwd_kernel, _readout_rows_fwd_kernel
from .general_fused_write_triton import solve_general_operands_fused_write
from .solve import solve_unit_lower
from .tuning import by_arch


_CHUNK = 16


@triton.jit
def _readout_fwd_reduce_kernel(
    MPART, CPART, MATRIX, CAUSAL,
    C: tl.constexpr, NK: tl.constexpr, BLOCK_NK: tl.constexpr,
):
    system = tl.program_id(0)
    cells = tl.arange(0, C * C)
    blocks = tl.arange(0, BLOCK_NK)
    offsets = (system * NK + blocks[:, None]) * C * C + cells[None, :]
    valid = (blocks < NK)[:, None]
    lower = (cells % C) < (cells // C)
    causal_lower = (cells % C) <= (cells // C)
    # Mask unwritten upper triangles at the load, avoiding two buffer memsets.
    m = tl.sum(tl.load(MPART + offsets, mask=valid & lower[None, :], other=0.), 0)
    c = tl.sum(tl.load(CPART + offsets, mask=valid & causal_lower[None, :], other=0.), 0)
    tl.store(MATRIX + system * C * C + cells, m)
    tl.store(CAUSAL + system * C * C + cells, c)


@triton.jit
def _row(tile, rows, i: tl.constexpr):
    """Row ``i`` of a ``[BK, C]`` register tile as a ``[BK]`` vector."""
    return tl.sum(tl.where(rows[None, :] == i, tile, 0.0), axis=1)


@triton.jit
def _col_sq(tile, rows, i: tl.constexpr):
    """Column ``i`` of a ``[C, C]`` register tile as a ``[C]`` vector."""
    return tl.sum(tl.where(rows[None, :] == i, tile, 0.0), axis=1)


@triton.jit
def _row_sq(tile, rows, i: tl.constexpr):
    """Row ``i`` of a ``[C, C]`` register tile as a ``[C]`` vector."""
    return tl.sum(tl.where(rows[:, None] == i, tile, 0.0), axis=0)


@triton.jit
def _relative_row(Gc, rows, key_mask, i: tl.constexpr):
    """``E_i[:, j] = exp(G_i - G_j)`` for ``j <= i`` (zero above the diagonal)."""
    g_i = _row(Gc, rows, i)
    e = tl.exp(tl.minimum(g_i[:, None] - Gc, 0.0))
    return tl.where((rows[None, :] <= i) & key_mask[:, None], e, 0.0)


@triton.jit
def _readout_c16_fwd_kernel(
    Q, K_IN, G, B_GATE, GAMMA, LAM,
    ABAR, QBAR1, QBAR2, TAIL1, TAIL2, TRANSPORT, MPART, CPART,
    T: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr, NK: tl.constexpr,
    SCALE: tl.constexpr,
    SB0: tl.constexpr, SB1: tl.constexpr, SB2: tl.constexpr, SB3: tl.constexpr,
    SG0: tl.constexpr, SG1: tl.constexpr, SG2: tl.constexpr, SG3: tl.constexpr,
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
    rows = tl.arange(0, C)
    token0 = chunk * C
    tile = ((batch * T + token0 + rows)[None, :] * H + head) * K + keys[:, None]
    tile_mask = key_mask[:, None] & (rows[None, :] < C)
    k_t = tl.load(K_IN + tile, mask=tile_mask, other=0.0).to(tl.float32)
    g_t = tl.load(G + tile, mask=tile_mask, other=0.0).to(tl.float32)
    Gc = tl.cumsum(g_t, axis=1)
    out_base = system * C * K
    part_base = (system * NK + key_block) * C * C

    P = tl.zeros((BK, C), tl.float32)
    E = tl.zeros((BK, C), tl.float32)
    Bnd = tl.zeros((BK,), tl.float32)
    a_i = tl.zeros((BK,), tl.float32)
    keep = tl.zeros((BK,), tl.float32) + 1.0
    for i in tl.static_range(C):
        token = token0 + i
        row_off = ((batch * T + token) * H + head) * K + keys
        q_i = tl.load(Q + row_off, mask=key_mask, other=0.0).to(tl.float32) * SCALE
        k_i = tl.load(K_IN + row_off, mask=key_mask, other=0.0).to(tl.float32)
        b_i = tl.load(
            B_GATE + batch * SB0 + token * SB1 + head * SB2 + keys * SB3,
            mask=key_mask, other=0.0,
        ).to(tl.float32)
        gm_i = tl.load(
            GAMMA + batch * SG0 + token * SG1 + head * SG2 + keys * SG3,
            mask=key_mask, other=0.0,
        ).to(tl.float32)
        lam_i = tl.load(LAM + (batch * T + token) * H + head).to(tl.float32)
        a_i = tl.exp(_row(Gc, rows, i))
        E = _relative_row(Gc, rows, key_mask, i)
        P = (1.0 - lam_i) * P + lam_i * E
        Bnd = (1.0 - lam_i) * Bnd + lam_i * a_i
        keep = keep * (1.0 - lam_i)
        q_active = gm_i * q_i
        q_slow = (1.0 - gm_i) * q_i
        erase = b_i * k_i
        inter = tl.sum(erase[:, None] * k_t * E, axis=0)
        caus = tl.sum(k_t * (q_active[:, None] * E + q_slow[:, None] * P), axis=0)
        tl.store(MPART + part_base + i * C + rows, tl.where(rows < i, inter, 0.0))
        tl.store(CPART + part_base + i * C + rows, tl.where(rows <= i, caus, 0.0))
        out_off = out_base + i * K + keys
        tl.store(ABAR + out_off, (a_i * erase).to(ABAR.dtype.element_ty), mask=key_mask)
        tl.store(
            QBAR1 + out_off,
            (q_active * a_i + q_slow * Bnd).to(QBAR1.dtype.element_ty),
            mask=key_mask,
        )
        tl.store(QBAR2 + out_off, (q_slow * keep).to(QBAR2.dtype.element_ty), mask=key_mask)

    tail_off = out_base + rows[None, :] * K + keys[:, None]
    tl.store(TAIL1 + tail_off, (k_t * E).to(TAIL1.dtype.element_ty), mask=tile_mask)
    tl.store(TAIL2 + tail_off, (k_t * P).to(TAIL2.dtype.element_ty), mask=tile_mask)
    transport = (system * K + keys) * 4
    tl.store(TRANSPORT + transport + 0, a_i, mask=key_mask)
    tl.store(TRANSPORT + transport + 1, Bnd, mask=key_mask)
    tl.store(TRANSPORT + transport + 2, tl.zeros((BK,), tl.float32), mask=key_mask)
    tl.store(TRANSPORT + transport + 3, keep, mask=key_mask)


@triton.jit
def _readout_c16_bwd_rows_kernel(
    Q, K_IN, G, B_GATE, GAMMA, LAM,
    DABAR, DQBAR1, DQBAR2, DTAIL1, DTAIL2, DTRANSPORT, DMATRIX, DCAUSAL,
    DQ, DK, DG, DB, DGAMMA,
    T: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr, NK: tl.constexpr,
    SCALE: tl.constexpr,
    SB0: tl.constexpr, SB1: tl.constexpr, SB2: tl.constexpr, SB3: tl.constexpr,
    SG0: tl.constexpr, SG1: tl.constexpr, SG2: tl.constexpr, SG3: tl.constexpr,
    BK: tl.constexpr,
):
    """Backward, part 1: everything that follows from the row recurrences.

    One forward sweep recomputes ``E_i``, ``P_i``, ``A_i``, ``Bnd_i`` and
    ``keep_i`` exactly as the forward did and emits the query, readout-mix,
    erase-gate and key gradients and the *direct* decay cotangent (``dE``
    from ``causal`` / ``interaction`` / ``tail1``, ``dA`` from ``abar`` /
    ``qbar1`` / ``f11``).  The slow-admission path (``dE``, ``dA`` through
    ``W``) and everything about ``lambda`` is part 2.  ``DG`` receives
    per-row values ``dG_i`` (not yet suffix-summed); part 2 finishes it.
    """
    pid = tl.program_id(0)
    key_block = pid % NK
    system = pid // NK
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    rows = tl.arange(0, C)
    token0 = chunk * C
    tile = ((batch * T + token0 + rows)[None, :] * H + head) * K + keys[:, None]
    tile_mask = key_mask[:, None] & (rows[None, :] < C)
    k_t = tl.load(K_IN + tile, mask=tile_mask, other=0.0).to(tl.float32)
    g_t = tl.load(G + tile, mask=tile_mask, other=0.0).to(tl.float32)
    Gc = tl.cumsum(g_t, axis=1)
    out_base = system * C * K
    square_base = system * C * C
    transport = (system * K + keys) * 4

    P = tl.zeros((BK, C), tl.float32)
    Bnd = tl.zeros((BK,), tl.float32)
    keep = tl.zeros((BK,), tl.float32) + 1.0
    dk_t = tl.zeros((BK, C), tl.float32)
    dGneg = tl.zeros((BK, C), tl.float32)
    for i in tl.static_range(C):
        token = token0 + i
        row_off = ((batch * T + token) * H + head) * K + keys
        q_i = tl.load(Q + row_off, mask=key_mask, other=0.0).to(tl.float32) * SCALE
        k_i = tl.load(K_IN + row_off, mask=key_mask, other=0.0).to(tl.float32)
        b_i = tl.load(
            B_GATE + batch * SB0 + token * SB1 + head * SB2 + keys * SB3,
            mask=key_mask, other=0.0,
        ).to(tl.float32)
        gm_i = tl.load(
            GAMMA + batch * SG0 + token * SG1 + head * SG2 + keys * SG3,
            mask=key_mask, other=0.0,
        ).to(tl.float32)
        lam_i = tl.load(LAM + (batch * T + token) * H + head).to(tl.float32)
        out_off = out_base + i * K + keys
        dabar_i = tl.load(DABAR + out_off, mask=key_mask, other=0.0).to(tl.float32)
        dq1_i = tl.load(DQBAR1 + out_off, mask=key_mask, other=0.0).to(tl.float32)
        dq2_i = tl.load(DQBAR2 + out_off, mask=key_mask, other=0.0).to(tl.float32)
        dcaus_i = tl.where(rows <= i, tl.load(DCAUSAL + square_base + i * C + rows).to(tl.float32), 0.0)
        # interaction is strict lower: E_ii = 1 must not leak the diagonal
        dmat_i = tl.where(rows < i, tl.load(DMATRIX + square_base + i * C + rows).to(tl.float32), 0.0)

        a_i = tl.exp(_row(Gc, rows, i))
        E = _relative_row(Gc, rows, key_mask, i)
        P = (1.0 - lam_i) * P + lam_i * E
        Bnd = (1.0 - lam_i) * Bnd + lam_i * a_i
        keep = keep * (1.0 - lam_i)
        q_active = gm_i * q_i
        q_slow = (1.0 - gm_i) * q_i
        erase = b_i * k_i

        read_t = k_t * (gm_i[:, None] * E + (1.0 - gm_i)[:, None] * P)
        dq_i = (
            dq1_i * (gm_i * a_i + (1.0 - gm_i) * Bnd)
            + dq2_i * (1.0 - gm_i) * keep
            + tl.sum(dcaus_i[None, :] * read_t, axis=1)
        )
        dgm_i = (
            dq1_i * q_i * (a_i - Bnd)
            - dq2_i * keep * q_i
            + q_i * tl.sum(dcaus_i[None, :] * (k_t * (E - P)), axis=1)
        )
        inter_k = tl.sum(dmat_i[None, :] * (k_t * E), axis=1)              # sum_j dmat_ij k_j E_ij
        db_i = dabar_i * a_i * k_i + k_i * inter_k
        dk_row = dabar_i * a_i * b_i + b_i * inter_k
        dk_t += dcaus_i[None, :] * (q_active[:, None] * E + q_slow[:, None] * P)
        dk_t += dmat_i[None, :] * (erase[:, None] * E)
        dk_t += tl.where(rows[None, :] == i, dk_row[:, None], 0.0)
        tl.store(DQ + row_off, (dq_i * SCALE).to(DQ.dtype.element_ty), mask=key_mask)
        tl.store(DGAMMA + row_off, dgm_i, mask=key_mask)
        tl.store(DB + row_off, db_i.to(DB.dtype.element_ty), mask=key_mask)

        dE = dcaus_i[None, :] * (q_active[:, None] * k_t) + dmat_i[None, :] * (erase[:, None] * k_t)
        dA = dabar_i * erase + dq1_i * q_active
        if i == C - 1:
            tail_off = out_base + rows[None, :] * K + keys[:, None]
            dt1 = tl.load(DTAIL1 + tail_off, mask=tile_mask, other=0.0).to(tl.float32)
            dt2 = tl.load(DTAIL2 + tail_off, mask=tile_mask, other=0.0).to(tl.float32)
            dE += dt1 * k_t
            dk_t += dt1 * E + dt2 * P
            dA += tl.load(DTRANSPORT + transport + 0, mask=key_mask, other=0.0).to(tl.float32)
        dEE = dE * E
        dGneg += dEE
        tl.store(DG + row_off, tl.sum(dEE, axis=1) + dA * a_i, mask=key_mask)   # provisional (fp32)

    # dG_j -= sum_i dE_ij E_ij (column part), still per row, not suffix-summed
    for s in tl.static_range(C):
        row_off = ((batch * T + token0 + s) * H + head) * K + keys
        value = tl.load(DG + row_off, mask=key_mask, other=0.0) - _row(dGneg, rows, s)
        tl.store(DG + row_off, value, mask=key_mask)
    tl.store(DK + tile, dk_t.to(DK.dtype.element_ty), mask=tile_mask)


@triton.jit
def _readout_c16_bwd_slow_kernel(
    Q, K_IN, G, GAMMA, LAM,
    DQBAR1, DQBAR2, DTAIL2, DTRANSPORT, DCAUSAL,
    DG, DLAM_PARTIAL,
    T: tl.constexpr, H: tl.constexpr, N: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr, NK: tl.constexpr,
    SCALE: tl.constexpr,
    SG0: tl.constexpr, SG1: tl.constexpr, SG2: tl.constexpr, SG3: tl.constexpr,
    BK: tl.constexpr, DOT_PRECISION: tl.constexpr,
):
    """Backward, part 2: the slow-admission path, ``lambda``, and the decay
    suffix sum.

    The admission is written with explicit weights ``P_i = sum_tau W[i, tau]
    E_tau`` and ``Bnd_i = sum_tau W[i, tau] A_tau`` with ``W[i, tau] =
    lambda_tau R[i, tau]``, ``R[i, tau] = prod_{tau < s <= i} (1 -
    lambda_s)`` (every factor at most one), so its cotangents are
    contractions rather than a reverse recurrence: ``dE_tau += sum_i W[i,
    tau] dP_i``, ``dA_tau += sum_i W[i, tau] dBnd_i`` and ``dW[i, tau] =
    <dP_i, E_tau> + <dBnd_i, A_tau>``.  ``dP_i = dcausal_ij (1 - gamma_i) q_i
    k_j`` is rank structured, which makes both contractions ``16 x 16`` dots
    per row.  The chain rule from ``dW`` (and from the retention ``keep_i =
    prod_{s <= i} (1 - lambda_s)``) to ``lambda`` uses only products of
    factors in ``[0, 1]``::

        dlambda_tau  = sum_{i >= tau} dW[i, tau] R[i, tau]
        dlambda_s   -= sum_{i >= s} R[i, s] sum_{tau < s} dW[i, tau] lambda_tau R[s - 1, tau]
        dlambda_s   -= keep_{s-1} sum_{i >= s} dkeep_i R[i, s]

    so ``lambda -> 1`` (saturation) and ``lambda -> 0`` (the
    initialization) are both exact.  Finally ``dg_s = sum_{i >= s} dG_i``
    over the rows written by part 1 plus this part's contributions.
    """
    pid = tl.program_id(0)
    key_block = pid % NK
    system = pid // NK
    chunk = system % N
    head = (system // N) % H
    batch = system // (N * H)
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    rows = tl.arange(0, C)
    token0 = chunk * C
    tile = ((batch * T + token0 + rows)[None, :] * H + head) * K + keys[:, None]
    tile_mask = key_mask[:, None] & (rows[None, :] < C)
    k_t = tl.load(K_IN + tile, mask=tile_mask, other=0.0).to(tl.float32)
    g_t = tl.load(G + tile, mask=tile_mask, other=0.0).to(tl.float32)
    Gc = tl.cumsum(g_t, axis=1)
    out_base = system * C * K
    square_base = system * C * C
    square = square_base + rows[:, None] * C + rows[None, :]
    dcaus_t = tl.load(DCAUSAL + square).to(tl.float32)                     # [m, j]
    dcaus_t = tl.where(rows[:, None] >= rows[None, :], dcaus_t, 0.0)

    # R[i, tau] = prod_{tau < s <= i} (1 - lambda_s) (i >= tau, else 0) and
    # Rp[s, tau] = R[s - 1, tau] (tau < s, else 0), built from bounded factors.
    ii = rows[:, None]
    tt = rows[None, :]
    R = tl.where(ii >= tt, 1.0, 0.0)
    Rp = tl.where(ii > tt, 1.0, 0.0)
    for s in tl.static_range(C):
        lam_s = tl.load(LAM + (batch * T + token0 + s) * H + head).to(tl.float32)
        R = tl.where((tt < s) & (s <= ii), R * (1.0 - lam_s), R)
        Rp = tl.where((tt < s) & (s < ii), Rp * (1.0 - lam_s), Rp)
    lam_t = tl.load(LAM + (batch * T + token0 + rows) * H + head).to(tl.float32)

    gamma_tile = batch * SG0 + (token0 + rows)[None, :] * SG1 + head * SG2 + keys[:, None] * SG3
    gm_t = tl.load(GAMMA + gamma_tile, mask=tile_mask, other=0.0).to(tl.float32)
    q_slow_t = (1.0 - gm_t) * tl.load(Q + tile, mask=tile_mask, other=0.0).to(tl.float32) * SCALE
    vec = out_base + rows[None, :] * K + keys[:, None]
    transport = (system * K + keys) * 4
    dtr1 = tl.load(DTRANSPORT + transport + 1, mask=key_mask, other=0.0).to(tl.float32)
    dtr3 = tl.load(DTRANSPORT + transport + 3, mask=key_mask, other=0.0).to(tl.float32)
    dbnd_t = tl.load(DQBAR1 + vec, mask=tile_mask, other=0.0).to(tl.float32) * q_slow_t
    dbnd_t += tl.where(rows[None, :] == C - 1, dtr1[:, None], 0.0)
    # retention cotangent dkeep_i = <dqbar2_i, (1 - gamma_i) q_i> (+ f22 on the last row)
    dkeep = tl.sum(tl.load(DQBAR2 + vec, mask=tile_mask, other=0.0).to(tl.float32) * q_slow_t, axis=0)
    dkeep += tl.where(rows == C - 1, tl.sum(dtr3, axis=0), 0.0)
    dw_t = tl.dot(tl.trans(dbnd_t), tl.exp(Gc), input_precision=DOT_PRECISION)   # [i, tau] = <dBnd_i, A_tau>
    dp_tail = tl.load(DTAIL2 + vec, mask=tile_mask, other=0.0).to(tl.float32) * k_t   # dP_end

    # Reverse sweep: the admission adjoints are the recurrences
    #   QP_i = dP_i + (1 - lambda_{i+1}) QP_{i+1},  dE_i += lambda_i QP_i
    #   QB_i = dBnd_i + (1 - lambda_{i+1}) QB_{i+1}, dA_i += lambda_i QB_i
    # (one FMA per row); only <dP_m, E_i> over m >= i needs a dot.
    QP = tl.zeros((BK, C), tl.float32)
    QB = tl.zeros((BK,), tl.float32)
    lam_next = tl.zeros((BK,), tl.float32)
    dGneg = tl.zeros((BK, C), tl.float32)
    for i in tl.static_range(C - 1, -1, -1):
        lam_i = tl.load(LAM + (batch * T + token0 + i) * H + head).to(tl.float32)
        dcaus_i = _row_sq(dcaus_t, rows, i)                                 # dcausal[i, :]
        q_slow_i = _row(q_slow_t, rows, i)
        E = _relative_row(Gc, rows, key_mask, i)
        ke = k_t * E
        z = tl.dot(tl.trans(q_slow_t), ke, input_precision=DOT_PRECISION)   # [m, j] = sum_k q_slow_mk k_jk E_ijk
        dw_col = tl.sum(dcaus_t * z, axis=1)                                # <dP_m, E_i> over m
        dw_col += tl.where(rows == C - 1, tl.sum(tl.sum(dp_tail * E, axis=1), axis=0), 0.0)
        dw_t += tl.where(rows[None, :] == i, dw_col[:, None], 0.0)
        dP = dcaus_i[None, :] * (q_slow_i[:, None] * k_t)
        if i == C - 1:
            dP += dp_tail
        QP = dP + (1.0 - lam_next)[:, None] * QP
        QB = _row(dbnd_t, rows, i) + (1.0 - lam_next) * QB
        lam_next = tl.zeros((BK,), tl.float32) + lam_i
        dEE = (lam_i * QP) * E
        dGneg += dEE
        row_off = ((batch * T + token0 + i) * H + head) * K + keys
        dG_i = tl.sum(dEE, axis=1) + (lam_i * QB) * tl.exp(_row(Gc, rows, i))
        tl.store(DG + row_off, tl.load(DG + row_off, mask=key_mask, other=0.0) + dG_i, mask=key_mask)

    # G_i = sum_{s <= i} g_s  ->  dg_s = sum_{i >= s} dG_i
    suffix = tl.zeros((BK,), tl.float32)
    for s in tl.static_range(C - 1, -1, -1):
        row_off = ((batch * T + token0 + s) * H + head) * K + keys
        suffix += tl.load(DG + row_off, mask=key_mask, other=0.0) - _row(dGneg, rows, s)
        tl.store(DG + row_off, suffix, mask=key_mask)

    # chain rule dW, dkeep -> dlambda (partial over this key block)
    dw_t = tl.where(ii >= tt, dw_t, 0.0)
    direct = tl.sum(dw_t * R, axis=0)                                       # [tau]
    a_sq = Rp * lam_t[None, :]                                              # A[s, tau] = lambda_tau R[s-1, tau]
    m_sq = tl.dot(dw_t, tl.trans(a_sq), input_precision="ieee")             # [i, s] = sum_tau dW[i, tau] A[s, tau]
    through = tl.sum(m_sq * R, axis=0)                                      # [s]
    lam_0 = tl.sum(tl.where(rows == 0, lam_t, 0.0), axis=0)
    keep_prev = tl.where(rows >= 1, _col_sq(Rp, rows, 0) * (1.0 - lam_0), 1.0)   # keep_{s-1}
    retention = keep_prev * tl.sum(dkeep[:, None] * R, axis=0)
    tl.store(DLAM_PARTIAL + (system * NK + key_block) * C + rows, direct - through - retention)


def _kernel_variant() -> str:
    """``rows`` (generated row-explicit kernels, default) or ``tile``.

    ``INTEGRAL_READOUT_KERNEL=tile`` selects the ``(BK, C)`` tile kernels;
    both produce the same operands and gradients.
    """
    value = os.environ.get("INTEGRAL_READOUT_KERNEL", "rows")
    if value not in ("rows", "tile"):
        raise ValueError("INTEGRAL_READOUT_KERNEL must be rows or tile")
    return value


def _dot_precision() -> str:
    """Tensor-core input precision of the backward's ``16 x 16`` dots.

    The dots feed only the ``lambda`` gradient (``<dP_m, E_i>`` and
    ``<dBnd_i, A_tau>``).  With ``tf32`` the ``dlam`` error stays at the
    floor set by the scan's bf16 operands, as with ``ieee``, which emulates
    the dot with fp32 FMAs and is much slower.
    ``INTEGRAL_READOUT_DOT=ieee|tf32x3|tf32`` overrides.
    """
    value = os.environ.get("INTEGRAL_READOUT_DOT", "tf32")
    if value not in ("ieee", "tf32", "tf32x3"):
        raise ValueError("INTEGRAL_READOUT_DOT must be ieee, tf32, or tf32x3")
    return value


def _launch_config(key_dim: int, direction: str, variant: str = "rows",
                   tensor: torch.Tensor | None = None) -> tuple[int, int]:
    """``(BK, num_warps)`` for the forward / backward kernels.

    Tile kernels: one warp per program (Triton splits additional warps across
    the rows of the ``(BK, C)`` tile, which turns every row reduction into a
    shared-memory exchange); BK32 forward, BK64 backward.  Row kernels: one
    element per thread whatever the block, so the block is the whole head
    dimension and the warps split it (``ROWS_CONFIG``, or ``_ROWS_CONFIGS``
    for the device of ``tensor``).  Overridable through
    ``INTEGRAL_READOUT_FWD`` / ``INTEGRAL_READOUT_BWD`` as ``"<BK>,<warps>"``.
    """
    override = os.environ.get(f"INTEGRAL_READOUT_{direction.upper()}")
    if override:
        block, warps = (int(v) for v in override.split(","))
        return max(16, min(block, triton.next_power_of_2(key_dim))), warps
    if variant == "rows":
        block, warps = _rows_config(tensor)[direction]
        block = min(block, triton.next_power_of_2(key_dim))
        return block, max(1, min(warps, block // 32))
    block = 32 if direction == "fwd" else 64
    return max(16, min(block, triton.next_power_of_2(key_dim))), 1


# (BK, num_warps) of the row kernels where ops/tuning has no ROWS table.
ROWS_CONFIG = {"fwd": (64, 1), "bwd": (32, 1)}
# Per-architecture row tiles and reload-backward (BK, num_warps): ops/tuning.
# Which implementation of the row kernels runs is ops/arch.py's choice.
_ROWS_CONFIGS = by_arch("ROWS")
_RELOAD_BWD_CONFIGS = RELOAD_BWD_CONFIGS


def _device_capability(tensor):
    if tensor is None or not tensor.is_cuda:
        return None
    return torch.cuda.get_device_capability(tensor.device)


def _rows_config(tensor):
    return _ROWS_CONFIGS.get(_device_capability(tensor), ROWS_CONFIG)


def _strides(tensor: torch.Tensor) -> tuple[int, int, int, int]:
    return tuple(int(s) for s in tensor.stride())


class _ReadoutDirectC16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, g, b, gamma, lam, scale, operand_dtype):
        values = tuple(
            value.contiguous() if value.stride(-1) != 0 else value
            for value in (q, k, g, b, gamma, lam)
        )
        q, k, g, b, gamma, lam = values
        if _kernel_variant() != "rows":
            g = g.contiguous()
            values = (q, k, g, b, gamma, lam)
        batch, length, heads, key_dim = q.shape
        chunk_size = _CHUNK
        chunks = length // chunk_size
        systems = batch * heads * chunks
        shape = (batch, heads, chunks, chunk_size, key_dim)
        abar, qbar1, qbar2, tail1, tail2 = (
            torch.empty(shape, device=q.device, dtype=operand_dtype) for _ in range(5)
        )
        transport = torch.empty(
            batch, heads, chunks, key_dim, 4, device=q.device, dtype=torch.float32
        )
        matrix = torch.empty(
            batch, heads, chunks, chunk_size, chunk_size, device=q.device, dtype=torch.float32
        )
        causal = torch.empty_like(matrix, dtype=operand_dtype)
        variant = _kernel_variant()
        block, warps = _launch_config(key_dim, "fwd", variant, q)
        nk = triton.cdiv(key_dim, block)
        if variant == "rows":
            # the row kernels store only the lower triangle of each partial
            mpart = torch.empty(systems, nk, chunk_size, chunk_size, device=q.device, dtype=torch.float32)
            cpart = torch.empty_like(mpart)
            fwd_kernel = _readout_rows_fwd_kernel
        else:
            mpart = torch.empty(systems, nk, chunk_size, chunk_size, device=q.device, dtype=torch.float32)
            cpart = torch.empty_like(mpart)
            fwd_kernel = _readout_c16_fwd_kernel
        sb = _strides(b)
        sg = _strides(gamma)
        lam_view = lam if lam.ndim == 4 else lam.unsqueeze(-1).expand(*lam.shape, key_dim)
        sl = _strides(lam_view)
        if variant != "rows" and lam.ndim == 4:
            raise NotImplementedError("the tile readout-side kernels have no per-channel lam path")
        fwd_kernel[(systems * nk,)](
            q, k, g, b, gamma, lam,
            abar, qbar1, qbar2, tail1, tail2, transport, mpart, cpart,
            T=length, H=heads, N=chunks, K=key_dim, C=chunk_size, NK=nk,
            SCALE=float(scale),
            SB0=sb[0], SB1=sb[1], SB2=sb[2], SB3=sb[3],
            SG0=sg[0], SG1=sg[1], SG2=sg[2], SG3=sg[3],
            **({"SL0": sl[0], "SL1": sl[1], "SL2": sl[2], "SL3": sl[3],
                "SD0": g.stride(0), "SD1": g.stride(1), "SD2": g.stride(2), "SD3": g.stride(3),
                **readout_rows_forward_flags()}
               if variant == "rows" else {}),
            BK=block, num_warps=warps, num_stages=1,
        )
        _readout_fwd_reduce_kernel[(systems,)](
            mpart, cpart, matrix, causal,
            C=chunk_size, NK=nk, BLOCK_NK=triton.next_power_of_2(nk),
            num_warps=1, num_stages=1,
        )
        ctx.save_for_backward(*values)
        ctx.dimensions = (batch, length, heads, key_dim, chunks, systems)
        ctx.scale = float(scale)
        ctx.operand_dtype = operand_dtype
        ctx.variant = variant
        return abar, qbar1, qbar2, tail1, tail2, transport, matrix, causal

    @staticmethod
    def backward(ctx, dabar, dqbar1, dqbar2, dtail1, dtail2, dtransport, dmatrix, dcausal):
        q, k, g, b, gamma, lam = ctx.saved_tensors
        batch, length, heads, key_dim, chunks, systems = ctx.dimensions
        chunk_size = _CHUNK
        vector_shape = (batch, heads, chunks, chunk_size, key_dim)
        square_shape = (batch, heads, chunks, chunk_size, chunk_size)

        def grad_or_zero(value, shape, dtype):
            if value is None:
                return torch.zeros(shape, device=q.device, dtype=dtype)
            return value.contiguous()

        dabar = grad_or_zero(dabar, vector_shape, ctx.operand_dtype)
        dqbar1 = grad_or_zero(dqbar1, vector_shape, ctx.operand_dtype)
        dqbar2 = grad_or_zero(dqbar2, vector_shape, ctx.operand_dtype)
        dtail1 = grad_or_zero(dtail1, vector_shape, ctx.operand_dtype)
        dtail2 = grad_or_zero(dtail2, vector_shape, ctx.operand_dtype)
        dtransport = grad_or_zero(dtransport, (batch, heads, chunks, key_dim, 4), torch.float32)
        dmatrix = grad_or_zero(dmatrix, square_shape, torch.float32)
        dcausal = grad_or_zero(dcausal, square_shape, ctx.operand_dtype)

        full = (batch, length, heads, key_dim)
        dq = torch.empty(full, device=q.device, dtype=q.dtype)
        dk = torch.empty(full, device=q.device, dtype=k.dtype)
        dg = torch.empty(full, device=q.device, dtype=torch.float32)   # provisional rows are re-read
        db = torch.empty(full, device=q.device, dtype=b.dtype)
        dgamma = torch.empty(full, device=q.device, dtype=torch.float32)
        block, warps = _launch_config(key_dim, "bwd", ctx.variant, q)
        flags = readout_rows_backward_flags(q, length, key_dim) if ctx.variant == "rows" else {}
        if flags.get("RELOAD") and "INTEGRAL_READOUT_BWD" not in os.environ:
            block, warps = _RELOAD_BWD_CONFIGS[_device_capability(q)]
        nk = triton.cdiv(key_dim, block)
        sb = _strides(b)
        sg = _strides(gamma)
        lam_view = lam if lam.ndim == 4 else lam.unsqueeze(-1).expand(*lam.shape, key_dim)
        sl = _strides(lam_view)
        if ctx.variant == "rows":
            dlam_full = torch.empty(full, device=q.device, dtype=torch.float32)
            _readout_rows_bwd_kernel[(systems * nk,)](
                q, k, g, b, gamma, lam,
                dabar, dqbar1, dqbar2, dtail1, dtail2, dtransport, dmatrix, dcausal,
                dq, dk, dg, db, dgamma, dlam_full,
                T=length, H=heads, N=chunks, K=key_dim, C=chunk_size, NK=nk,
                SCALE=ctx.scale,
                SB0=sb[0], SB1=sb[1], SB2=sb[2], SB3=sb[3],
                SG0=sg[0], SG1=sg[1], SG2=sg[2], SG3=sg[3],
                SL0=sl[0], SL1=sl[1], SL2=sl[2], SL3=sl[3],
                SD0=g.stride(0), SD1=g.stride(1), SD2=g.stride(2), SD3=g.stride(3),
                **flags,
                BK=block, num_warps=warps, num_stages=1,
            )
            # The row kernel writes token-major gradients directly.
            dlam = dlam_full
            if lam.ndim == 3:
                dlam = dlam.sum(-1)
            dlam = dlam.to(lam.dtype)
            return dq, dk, dg.to(g.dtype), db, dgamma.to(gamma.dtype), dlam, None, None
        _readout_c16_bwd_rows_kernel[(systems * nk,)](
            q, k, g, b, gamma, lam,
            dabar, dqbar1, dqbar2, dtail1, dtail2, dtransport, dmatrix, dcausal,
            dq, dk, dg, db, dgamma,
            T=length, H=heads, N=chunks, K=key_dim, C=chunk_size, NK=nk,
            SCALE=ctx.scale,
            SB0=sb[0], SB1=sb[1], SB2=sb[2], SB3=sb[3],
            SG0=sg[0], SG1=sg[1], SG2=sg[2], SG3=sg[3],
            BK=block, num_warps=warps, num_stages=1,
        )
        dlam_partial = torch.empty(systems, nk, chunk_size, device=q.device, dtype=torch.float32)
        _readout_c16_bwd_slow_kernel[(systems * nk,)](
            q, k, g, gamma, lam,
            dqbar1, dqbar2, dtail2, dtransport, dcausal,
            dg, dlam_partial,
            T=length, H=heads, N=chunks, K=key_dim, C=chunk_size, NK=nk,
            SCALE=ctx.scale,
            SG0=sg[0], SG1=sg[1], SG2=sg[2], SG3=sg[3],
            BK=block, DOT_PRECISION=_dot_precision(), num_warps=warps, num_stages=1,
        )
        dlam = (
            dlam_partial.sum(dim=1)
            .view(batch, heads, length)
            .transpose(1, 2)
            .to(lam.dtype)
        )
        return dq, dk, dg.to(g.dtype), db, dgamma.to(gamma.dtype), dlam, None, None


def readout_builder_available(q: torch.Tensor, chunk_size: int) -> bool:
    return (
        q.is_cuda
        and chunk_size in (16, 32, 64)
        and q.dtype != torch.float64
        and os.environ.get("INTEGRAL_READOUT_BUILDER", "triton") != "torch"
    )


@torch.compiler.disable
def build_readout_operands_triton(
    q, k, v, g, b, w, gamma, lam, scale, chunk_size, operand_dtype=None
):
    """Readout-side C16/C32/C64 operands for the fused two-state scan in Triton.

    Same contract as :func:`readout._build_readout_operands`.  ``gamma`` may
    be ``[B, T, H]`` (one readout mix per head) or ``[B, T, H, K]`` (one per
    key channel).  The residual solve uses the fused BF16 write solve when the
    value path is BF16, otherwise the fp32 unit-lower solve.
    """
    if chunk_size not in (16, 32, 64) or q.shape[1] % chunk_size:
        raise ValueError("the Triton readout-side builder requires C16/C32/C64 aligned sequences")
    if operand_dtype is None:
        operand_dtype = torch.bfloat16
    if gamma.ndim == 3:
        gamma = gamma.unsqueeze(-1).expand_as(q)
    abar, qbar1, qbar2, tail1, tail2, transport, matrix, causal = _ReadoutDirectC16.apply(
        q, k, g, b, gamma, lam, scale, operand_dtype
    )
    if chunk_size > _CHUNK:
        # Compose stable C16 recurrences without inverse decay factors or a
        # torch.compile graph. Diagonal products are reused exactly; only the
        # strict off-diagonal block pairs require TensorCore products.
        from .general_block_triton import _compose_general_blocks_triton
        from .general_direct_block_triton import _GeneralDirectBlockedIntra

        # The second erase operand of the readout path is the constant zero.
        # On TRANSPORT_READOUT_GATE_ARCHS the readout-gate kernels skip it and
        # abar only stands in for it (it is neither read nor differentiated).
        skip_zero = on_arch(abar, TRANSPORT_READOUT_GATE_ARCHS)
        zero = abar if skip_zero else torch.zeros_like(abar)
        empty_coordinates = abar.new_empty(0)
        global_a1, _, global_q1, global_q2, pair1, pair2, global_tail1, global_tail2, global_transport = (
            _compose_general_blocks_triton(
                abar, zero, empty_coordinates, empty_coordinates, qbar1, qbar2, tail1, tail2,
                transport, chunk_size=chunk_size, direct=True, readout_gate=skip_zero,
            )
        )
        matrix, causal = _GeneralDirectBlockedIntra.apply(
            abar, zero, qbar1, qbar2, pair1, pair2, matrix, causal,
            chunk_size, True,
        )
        abar, qbar1, qbar2 = global_a1, global_q1, global_q2
        tail1, tail2, transport = global_tail1, global_tail2, global_transport
    if (
        operand_dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
    ):
        u0, state_weight1, state_weight2 = solve_general_operands_fused_write(
            matrix, v, w, abar
        )
    else:
        batch, length, heads, key_dim = q.shape
        chunks = length // chunk_size
        acc_dtype = torch.float32

        def chunked(tensor):
            return tensor.reshape(batch, chunks, chunk_size, heads, -1).permute(0, 3, 1, 2, 4)

        write = (chunked(w).to(acc_dtype) * chunked(v).to(acc_dtype))
        rhs = torch.cat((write, abar.to(acc_dtype)), dim=-1)
        solution = solve_unit_lower(matrix, rhs, matrix_is_strict_lower=True)
        u0, state_weight1 = solution.split((write.shape[-1], key_dim), dim=-1)
        u0 = u0.to(operand_dtype)
        state_weight1 = state_weight1.to(operand_dtype)
        state_weight2 = torch.zeros_like(state_weight1)
    return (
        u0,
        state_weight1,
        state_weight2,
        qbar1,
        qbar2,
        tail1,
        tail2,
        causal,
        transport,
    )


__all__ = ["build_readout_operands_triton", "readout_builder_available"]

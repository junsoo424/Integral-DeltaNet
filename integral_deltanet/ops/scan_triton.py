"""FLA-style two-state inter-chunk scan for Integral DeltaNet.

Only the boundary recurrence is sequential.  As in FLA's chunk kernels, the
state scan stores chunk-boundary states and corrected values, then a separate
kernel evaluates every chunk's output in parallel.  Backward mirrors that
split: output-to-state terms are precomputed in parallel and the reverse scan
contains only recurrence-dependent matrix products.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import safe_dot
from fla.utils import autotune_cache_kwargs

from .arch import optimized_readout
from .tuning import by_arch


# State-scan autotune candidates.  num_stages=4 is left out: it collapses once
# the pipelined operand tiles exceed shared memory.  Keep the candidates within
# a small factor of each other so a noisy autotune pick costs little.
_STATE_CONFIGS = [
    triton.Config({"BV": value_tile}, num_warps=warps, num_stages=stages)
    for value_tile, warps in ((16, 2), (32, 4), (64, 4), (64, 8))
    for stages in (1, 2, 3)
]

_OUTPUT_CONFIGS = [
    triton.Config({"BK": bk, "BV": bv}, num_warps=warps, num_stages=stages)
    for bk, bv, warps in (
        (32, 32, 2),
        (32, 64, 4),
        (32, 128, 8),
        (64, 32, 4),
        (64, 64, 4),
        (64, 128, 8),
    )
    for stages in (2, 3)
]

_LEAF_CONFIGS = [
    triton.Config({"BK": bk, "BV": bv}, num_warps=warps, num_stages=stages)
    for bk, bv, warps, stages in (
        (64, 32, 4, 2),
        (64, 32, 4, 3),
        (64, 32, 8, 2),
        (64, 32, 8, 3),
        (64, 64, 4, 2),
        (64, 64, 4, 3),
        (64, 64, 8, 2),
        (64, 64, 8, 3),
        (128, 16, 8, 3),
        (128, 64, 8, 3),
    )
]

# Pinned launch tables per architecture: ops/tuning.  The readout scan's
# small/large-grid variants swap in the state tiles by grid size
# (_small_grid_tiles).
_PINNED_INTEGRAL_SCAN_CONFIGS = by_arch("INTEGRAL_SCAN")
_SMALL_GRID_INTEGRAL_SCAN_CONFIGS = by_arch("INTEGRAL_SCAN_SMALL_GRID")
_LARGE_GRID_INTEGRAL_SCAN_CONFIGS = by_arch("INTEGRAL_SCAN_LARGE_GRID")


def _small_grid_tiles(tables, device, chunk_size, batch_heads, value_dim):
    """The readout tables, with the state tiles swapped for the small-grid ones
    when the pinned state grid would occupy at most half of the SMs, or for
    the large-grid ones when a BV=64 grid would fill at least 3/4 of them."""
    capability = torch.cuda.get_device_capability(device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    large = _LARGE_GRID_INTEGRAL_SCAN_CONFIGS.get(capability)
    if (large is not None and chunk_size in large[0]
            and os.environ.get("INTEGRAL_SCAN_LARGE_GRID", "1") != "0"
            and 4 * batch_heads * triton.cdiv(value_dim, 64) >= 3 * sms):
        swapped = [dict(t) for t in tables]
        for index, tiles in enumerate(large):
            swapped[index][chunk_size] = tiles[chunk_size]
        return tuple(swapped)
    small = _SMALL_GRID_INTEGRAL_SCAN_CONFIGS.get(capability)
    if (small is None or os.environ.get("INTEGRAL_SCAN_SMALL_GRID", "1") == "0"
            or chunk_size not in small[0]):
        return tables
    programs = batch_heads * triton.cdiv(value_dim, tables[0][chunk_size][0])
    if 2 * programs > sms:
        return tables
    swapped = [dict(t) for t in tables]
    for index, tiles in enumerate(small):
        swapped[index][chunk_size] = tiles[chunk_size]
    return tuple(swapped)

_PINNED_GENERAL_SCAN_CONFIGS = by_arch("GENERAL_SCAN")
_PINNED_SCALAR_SCAN_CONFIGS = by_arch("SCALAR_SCAN")


def _pinned_general_scan_configs(
    tensor, chunk_size, key_dim, value_dim, shared_f, shared_q
):
    """Per-architecture launch tables for the non-shared K=V=128 scan."""
    if not tensor.is_cuda or shared_f or shared_q:
        return None
    if key_dim != 128 or value_dim != 128:
        return None
    tables = _PINNED_GENERAL_SCAN_CONFIGS.get(
        torch.cuda.get_device_capability(tensor.device)
    )
    if tables is None or chunk_size not in tables[0]:
        return None
    return tables


def _pinned_scalar_scan_configs(
    tensor, chunk_size, key_dim, value_dim, shared_f, shared_q
):
    """Per-architecture launch tables for the shared-gate K=V=128 scan."""
    if not tensor.is_cuda or not (shared_f and shared_q):
        return None
    if key_dim != 128 or value_dim != 128:
        return None
    tables = _PINNED_SCALAR_SCAN_CONFIGS.get(
        torch.cuda.get_device_capability(tensor.device)
    )
    if tables is None or chunk_size not in tables[0]:
        return None
    return tables


def _scan_tile(entry, batch_heads):
    """Resolve one table entry: a launch tuple, or ``((min_batch_heads, tuple), ...)``.

    The state scans run one program per (sequence, value tile) for the whole
    length, so how many sequences there are decides whether several programs
    share an SM (latency hidden by occupancy) or each has one to itself
    (latency hidden by software pipelining).  Bucketed entries list the
    largest ``batch * heads`` threshold first; the last bucket is the floor.
    """
    if isinstance(entry[0], tuple):
        for min_batch_heads, config in entry:
            if batch_heads >= min_batch_heads:
                return config
        return entry[-1][1]
    return entry


def _pinned_scan_configs(
    tensor, chunk_size, key_dim, value_dim, shared_f, shared_q
):
    """Launch tables for whichever scan regime the inputs select, else None."""
    pinned = _pinned_general_scan_configs(
        tensor, chunk_size, key_dim, value_dim, shared_f, shared_q
    )
    if pinned is None:
        pinned = _pinned_scalar_scan_configs(
            tensor, chunk_size, key_dim, value_dim, shared_f, shared_q
        )
    return pinned


_CAUSAL_GRAD_CONFIGS = [
    triton.Config({"BV": value_tile}, num_warps=warps, num_stages=stages)
    for value_tile, warps in ((32, 2), (64, 4), (128, 8))
    for stages in (2, 3)
]


@triton.autotune(
    configs=_STATE_CONFIGS,
    key=[
        "C",
        "DK",
        "DV",
        "SHARED_F",
        "SHARED_Q",
        "FUSE_OUTPUT",
        "ZERO_INITIAL",
        "READOUT_GATE",
    ],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["N"])
def _scan_state_fwd_kernel(
    U0,
    W1,
    W2,
    Q1,
    Q2,
    K1,
    K2,
    L,
    F,
    H0S,
    H0X,
    R,
    HS,
    HX,
    O,
    HTS,
    HTX,
    N,
    C: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    SHARED_F: tl.constexpr,
    SHARED_Q: tl.constexpr,
    FUSE_OUTPUT: tl.constexpr,
    ZERO_INITIAL: tl.constexpr,
    BV: tl.constexpr,
    READOUT_GATE: tl.constexpr = False,
    OH: tl.constexpr = 1, ON: tl.constexpr = 1,
    OS0: tl.constexpr = 0, OS1: tl.constexpr = 0, OS2: tl.constexpr = 0,
    OS3: tl.constexpr = 0, OS4: tl.constexpr = -1,
    STORE_STATES: tl.constexpr = True,
):
    # int64: the saved states hold B*H*N*K*V elements, past 2**31 already at
    # B8/T32768/H16/C16, and the per-program bases are formed from the pid.
    # STORE_STATES=False (no backward will run) skips the per-chunk states and
    # the residual, which only the backward reads once the output is fused.
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_v = tl.program_id(1)
    values = pid_v * BV + tl.arange(0, BV)
    value_mask = values < DV
    rows = tl.arange(0, C)
    keys = tl.arange(0, DK)

    base_u = U0 + pid_bh * N * C * DV
    base_w1 = W1 + pid_bh * N * C * DK
    base_w2 = W2 + pid_bh * N * C * DK
    base_q1 = Q1 + pid_bh * N * C * DK
    if SHARED_Q:
        base_q2 = Q2 + pid_bh * N * C * 2
    else:
        base_q2 = Q2 + pid_bh * N * C * DK
    base_k1 = K1 + pid_bh * N * C * DK
    base_k2 = K2 + pid_bh * N * C * DK
    base_l = L + pid_bh * N * C * C
    if SHARED_F:
        base_f = F + pid_bh * N * 4
    else:
        base_f = F + pid_bh * N * DK * 4
    base_r = R + pid_bh * N * C * DV
    base_hs = HS + pid_bh * N * DK * DV
    base_hx = HX + pid_bh * N * DK * DV
    base_o = O + pid_bh * N * C * DV

    state_offset = pid_bh * DK * DV + keys[:, None] * DV + values[None, :]
    if ZERO_INITIAL:
        hs = tl.zeros((DK, BV), dtype=tl.float32)
        hx = tl.zeros((DK, BV), dtype=tl.float32)
    else:
        hs = tl.load(
            H0S + state_offset, mask=value_mask[None, :], other=0.0
        ).to(tl.float32)
        hx = tl.load(
            H0X + state_offset, mask=value_mask[None, :], other=0.0
        ).to(tl.float32)

    for chunk in range(N):
        hsb = hs.to(tl.bfloat16)
        hxb = hx.to(tl.bfloat16)
        if STORE_STATES:
            chunk_state_offset = chunk * DK * DV + keys[:, None] * DV + values[None, :]
            tl.store(base_hs + chunk_state_offset, hsb, mask=value_mask[None, :])
            tl.store(base_hx + chunk_state_offset, hxb, mask=value_mask[None, :])

        residual = tl.load(
            base_u + chunk * C * DV + rows[:, None] * DV + values[None, :],
            mask=value_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w1 = tl.load(base_w1 + chunk * C * DK + rows[:, None] * DK + keys[None, :])
        residual -= safe_dot(w1, hsb)
        if not READOUT_GATE:
            w2 = tl.load(base_w2 + chunk * C * DK + rows[:, None] * DK + keys[None, :])
            residual -= safe_dot(w2, hxb)
        residual_bf16 = residual.to(tl.bfloat16)
        if STORE_STATES or not FUSE_OUTPUT:
            tl.store(
                base_r + chunk * C * DV + rows[:, None] * DV + values[None, :],
                residual_bf16,
                mask=value_mask[None, :],
            )

        if FUSE_OUTPUT:
            causal = tl.load(
                base_l + chunk * C * C + rows[:, None] * C + rows[None, :]
            )
            output = safe_dot(causal, residual_bf16)
            q1 = tl.load(
                base_q1 + chunk * C * DK + rows[:, None] * DK + keys[None, :]
            ).to(tl.bfloat16)
            if SHARED_Q:
                a1 = tl.load(base_q2 + chunk * C * 2 + rows * 2 + 0).to(tl.float32)
                a2 = tl.load(base_q2 + chunk * C * 2 + rows * 2 + 1).to(tl.float32)
                output += a1[:, None] * safe_dot(q1, hsb)
                output += a2[:, None] * safe_dot(q1, hxb)
            else:
                q2 = tl.load(
                    base_q2 + chunk * C * DK + rows[:, None] * DK + keys[None, :]
                ).to(tl.bfloat16)
                output += safe_dot(q1, hsb)
                output += safe_dot(q2, hxb)
            tl.store(
                O + _output_offset(pid_bh, chunk, rows, values, C, DV, OH, N, OS0, OS1, OS2, OS3, OS4),
                output,
                mask=value_mask[None, :],
            )

        if SHARED_F:
            f11 = tl.load(base_f + chunk * 4 + 0).to(tl.float32)
            f12 = tl.load(base_f + chunk * 4 + 1).to(tl.float32)
            f21 = tl.load(base_f + chunk * 4 + 2).to(tl.float32)
            f22 = tl.load(base_f + chunk * 4 + 3).to(tl.float32)
        else:
            f11 = tl.load(base_f + chunk * DK * 4 + keys * 4 + 0).to(tl.float32)
            f12 = tl.load(base_f + chunk * DK * 4 + keys * 4 + 1).to(tl.float32)
            f21 = tl.load(base_f + chunk * DK * 4 + keys * 4 + 2).to(tl.float32)
            f22 = tl.load(base_f + chunk * DK * 4 + keys * 4 + 3).to(tl.float32)
        k1 = tl.load(
            base_k1 + chunk * C * DK + rows[:, None] * DK + keys[None, :]
        )
        k2 = tl.load(
            base_k2 + chunk * C * DK + rows[:, None] * DK + keys[None, :]
        )
        next_hs = f11[:, None] * hs
        if not READOUT_GATE:
            next_hs += f21[:, None] * hx
        next_hs += safe_dot(tl.trans(k1), residual_bf16)
        next_hx = f12[:, None] * hs + f22[:, None] * hx
        next_hx += safe_dot(tl.trans(k2), residual_bf16)
        hs, hx = next_hs, next_hx

    tl.store(HTS + state_offset, hs, mask=value_mask[None, :])
    tl.store(HTX + state_offset, hx, mask=value_mask[None, :])


@triton.autotune(
    configs=_OUTPUT_CONFIGS,
    key=["C", "DK", "DV", "SHARED_Q"],
    **autotune_cache_kwargs,
)
@triton.jit
def _output_fwd_kernel(
    R,
    Q1,
    Q2,
    L,
    HS,
    HX,
    O,
    C: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    SHARED_Q: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    OH: tl.constexpr = 1, ON: tl.constexpr = 1,
    OS0: tl.constexpr = 0, OS1: tl.constexpr = 0, OS2: tl.constexpr = 0,
    OS3: tl.constexpr = 0, OS4: tl.constexpr = -1,
):
    pid_chunk = tl.program_id(0).to(tl.int64)
    pid_v = tl.program_id(1)
    rows = tl.arange(0, C)
    values = pid_v * BV + tl.arange(0, BV)
    value_mask = values < DV

    residual = tl.load(
        R + pid_chunk * C * DV + rows[:, None] * DV + values[None, :],
        mask=value_mask[None, :],
        other=0.0,
    )
    causal = tl.load(L + pid_chunk * C * C + rows[:, None] * C + rows[None, :])
    output = safe_dot(causal, residual)

    for key_block in range(tl.cdiv(DK, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < DK
        q1 = tl.load(
            Q1 + pid_chunk * C * DK + rows[:, None] * DK + keys[None, :],
            mask=key_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        if SHARED_Q:
            a1 = tl.load(Q2 + pid_chunk * C * 2 + rows * 2 + 0).to(tl.float32)
            a2 = tl.load(Q2 + pid_chunk * C * 2 + rows * 2 + 1).to(tl.float32)
        else:
            q2 = tl.load(
                Q2 + pid_chunk * C * DK + rows[:, None] * DK + keys[None, :],
                mask=key_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
        hs = tl.load(
            HS + pid_chunk * DK * DV + keys[:, None] * DV + values[None, :],
            mask=key_mask[:, None] & value_mask[None, :],
            other=0.0,
        )
        hx = tl.load(
            HX + pid_chunk * DK * DV + keys[:, None] * DV + values[None, :],
            mask=key_mask[:, None] & value_mask[None, :],
            other=0.0,
        )
        if SHARED_Q:
            output += a1[:, None] * safe_dot(q1, hs)
            output += a2[:, None] * safe_dot(q1, hx)
        else:
            output += safe_dot(q1, hs)
            output += safe_dot(q2, hx)

    tl.store(
        O + _output_offset(pid_chunk // ON, pid_chunk % ON, rows, values, C, DV, OH, ON, OS0, OS1, OS2, OS3, OS4),
        output,
        mask=value_mask[None, :],
    )


@triton.autotune(
    configs=_STATE_CONFIGS,
    key=["C", "DK", "DV"],
    **autotune_cache_kwargs,
)
@triton.jit
def _output_bwd_state_kernel(
    DO,
    Q1,
    Q2,
    L,
    DU0,
    DHS0,
    DHX0,
    C: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BV: tl.constexpr,
):
    pid_chunk = tl.program_id(0).to(tl.int64)
    pid_v = tl.program_id(1)
    rows = tl.arange(0, C)
    keys = tl.arange(0, DK)
    values = pid_v * BV + tl.arange(0, BV)
    value_mask = values < DV

    grad_output = tl.load(
        DO + pid_chunk * C * DV + rows[:, None] * DV + values[None, :],
        mask=value_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    causal = tl.load(L + pid_chunk * C * C + rows[:, None] * C + rows[None, :])
    q1 = tl.load(
        Q1 + pid_chunk * C * DK + rows[:, None] * DK + keys[None, :]
    ).to(tl.bfloat16)
    q2 = tl.load(
        Q2 + pid_chunk * C * DK + rows[:, None] * DK + keys[None, :]
    ).to(tl.bfloat16)

    grad_u0 = safe_dot(tl.trans(causal), grad_output)
    grad_hs0 = safe_dot(tl.trans(q1), grad_output)
    grad_hx0 = safe_dot(tl.trans(q2), grad_output)
    tl.store(
        DU0 + pid_chunk * C * DV + rows[:, None] * DV + values[None, :],
        grad_u0,
        mask=value_mask[None, :],
    )
    state_offset = pid_chunk * DK * DV + keys[:, None] * DV + values[None, :]
    tl.store(DHS0 + state_offset, grad_hs0, mask=value_mask[None, :])
    tl.store(DHX0 + state_offset, grad_hx0, mask=value_mask[None, :])


@triton.autotune(
    configs=_STATE_CONFIGS,
    key=["C", "DK", "DV", "SHARED_F", "SHARED_Q", "READOUT_GATE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["N"])
def _scan_state_bwd_kernel(
    DO,
    Q1,
    Q2,
    L,
    DHSF,
    DHXF,
    W1,
    W2,
    K1,
    K2,
    F,
    DU,
    DHS,
    DHX,
    DHI_S,
    DHI_X,
    DU0,
    DHS0,
    DHX0,
    N,
    C: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    SHARED_F: tl.constexpr,
    SHARED_Q: tl.constexpr,
    BV: tl.constexpr,
    FUSE_OUTPUT: tl.constexpr = True,
    PRE_STATE: tl.constexpr = False,
    READOUT_GATE: tl.constexpr = False,
    OH: tl.constexpr = 1, ON: tl.constexpr = 1,
    OS0: tl.constexpr = 0, OS1: tl.constexpr = 0, OS2: tl.constexpr = 0,
    OS3: tl.constexpr = 0, OS4: tl.constexpr = -1,
):
    """Reverse state scan.

    FUSE_OUTPUT: form causal^T @ dO on the chain; otherwise load it from DU0
    (written per chunk in parallel by `_output_bwd_state_kernel`).
    PRE_STATE: additionally load q^T @ dO from DHS0 / DHX0 instead of forming
    it on the chain.
    """
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_v = tl.program_id(1)
    values = pid_v * BV + tl.arange(0, BV)
    value_mask = values < DV
    rows = tl.arange(0, C)
    keys = tl.arange(0, DK)

    base_do = DO + pid_bh * N * C * DV
    base_q1 = Q1 + pid_bh * N * C * DK
    if SHARED_Q:
        base_q2 = Q2 + pid_bh * N * C * 2
    else:
        base_q2 = Q2 + pid_bh * N * C * DK
    base_l = L + pid_bh * N * C * C
    base_w1 = W1 + pid_bh * N * C * DK
    base_w2 = W2 + pid_bh * N * C * DK
    base_k1 = K1 + pid_bh * N * C * DK
    base_k2 = K2 + pid_bh * N * C * DK
    if SHARED_F:
        base_f = F + pid_bh * N * 4
    else:
        base_f = F + pid_bh * N * DK * 4
    base_du = DU + pid_bh * N * C * DV
    base_dhs = DHS + pid_bh * N * DK * DV
    base_dhx = DHX + pid_bh * N * DK * DV
    base_du0 = DU0 + pid_bh * N * C * DV
    base_dhs0 = DHS0 + pid_bh * N * DK * DV
    base_dhx0 = DHX0 + pid_bh * N * DK * DV

    state_offset = pid_bh * DK * DV + keys[:, None] * DV + values[None, :]
    dhs = tl.load(DHSF + state_offset, mask=value_mask[None, :], other=0.0).to(tl.float32)
    dhx = tl.load(DHXF + state_offset, mask=value_mask[None, :], other=0.0).to(tl.float32)

    for reverse_index in range(N):
        chunk = N - 1 - reverse_index
        dhs_bf16 = dhs.to(tl.bfloat16)
        dhx_bf16 = dhx.to(tl.bfloat16)
        chunk_state_offset = chunk * DK * DV + keys[:, None] * DV + values[None, :]
        tl.store(base_dhs + chunk_state_offset, dhs_bf16, mask=value_mask[None, :])
        tl.store(base_dhx + chunk_state_offset, dhx_bf16, mask=value_mask[None, :])

        if FUSE_OUTPUT or not PRE_STATE:
            grad_output = tl.load(
                DO + _output_offset(pid_bh, chunk, rows, values, C, DV, OH, N, OS0, OS1, OS2, OS3, OS4),
                mask=value_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
        if FUSE_OUTPUT:
            causal = tl.load(
                base_l + chunk * C * C + rows[:, None] * C + rows[None, :]
            )
            grad_residual = safe_dot(tl.trans(causal), grad_output)
        else:
            grad_residual = tl.load(
                base_du0 + chunk * C * DV + rows[:, None] * DV + values[None, :],
                mask=value_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        k1 = tl.load(
            base_k1 + chunk * C * DK + rows[:, None] * DK + keys[None, :]
        )
        k2 = tl.load(
            base_k2 + chunk * C * DK + rows[:, None] * DK + keys[None, :]
        )
        grad_residual += safe_dot(k1, dhs_bf16)
        grad_residual += safe_dot(k2, dhx_bf16)
        grad_residual_bf16 = grad_residual.to(tl.bfloat16)
        tl.store(
            base_du + chunk * C * DV + rows[:, None] * DV + values[None, :],
            grad_residual_bf16,
            mask=value_mask[None, :],
        )

        if SHARED_F:
            f11 = tl.load(base_f + chunk * 4 + 0).to(tl.float32)
            f12 = tl.load(base_f + chunk * 4 + 1).to(tl.float32)
            f21 = tl.load(base_f + chunk * 4 + 2).to(tl.float32)
            f22 = tl.load(base_f + chunk * 4 + 3).to(tl.float32)
        else:
            f11 = tl.load(base_f + chunk * DK * 4 + keys * 4 + 0).to(tl.float32)
            f12 = tl.load(base_f + chunk * DK * 4 + keys * 4 + 1).to(tl.float32)
            f21 = tl.load(base_f + chunk * DK * 4 + keys * 4 + 2).to(tl.float32)
            f22 = tl.load(base_f + chunk * DK * 4 + keys * 4 + 3).to(tl.float32)
        w1 = tl.load(base_w1 + chunk * C * DK + rows[:, None] * DK + keys[None, :])
        if not READOUT_GATE:
            w2 = tl.load(base_w2 + chunk * C * DK + rows[:, None] * DK + keys[None, :])
        if PRE_STATE and not FUSE_OUTPUT:
            next_dhs = tl.load(
                base_dhs0 + chunk_state_offset, mask=value_mask[None, :], other=0.0
            ).to(tl.float32)
            next_dhx = tl.load(
                base_dhx0 + chunk_state_offset, mask=value_mask[None, :], other=0.0
            ).to(tl.float32)
        else:
            q1 = tl.load(base_q1 + chunk * C * DK + rows[:, None] * DK + keys[None, :]).to(tl.bfloat16)
            if SHARED_Q:
                a1 = tl.load(base_q2 + chunk * C * 2 + rows * 2 + 0).to(tl.float32)
                a2 = tl.load(base_q2 + chunk * C * 2 + rows * 2 + 1).to(tl.float32)
                grad_output1 = (a1[:, None] * grad_output).to(tl.bfloat16)
                grad_output2 = (a2[:, None] * grad_output).to(tl.bfloat16)
                next_dhs = safe_dot(tl.trans(q1), grad_output1)
                next_dhx = safe_dot(tl.trans(q1), grad_output2)
            else:
                q2 = tl.load(base_q2 + chunk * C * DK + rows[:, None] * DK + keys[None, :]).to(tl.bfloat16)
                next_dhs = safe_dot(tl.trans(q1), grad_output)
                next_dhx = safe_dot(tl.trans(q2), grad_output)
        next_dhs -= safe_dot(tl.trans(w1), grad_residual_bf16)
        next_dhs += f11[:, None] * dhs + f12[:, None] * dhx
        if READOUT_GATE:
            next_dhx += f22[:, None] * dhx
        else:
            next_dhx -= safe_dot(tl.trans(w2), grad_residual_bf16)
            next_dhx += f21[:, None] * dhs + f22[:, None] * dhx
        dhs, dhx = next_dhs, next_dhx

    tl.store(DHI_S + state_offset, dhs, mask=value_mask[None, :])
    tl.store(DHI_X + state_offset, dhx, mask=value_mask[None, :])


@triton.autotune(
    configs=_LEAF_CONFIGS,
    key=["C", "DK", "DV", "SHARED_F", "F_PRIMARY", "F_OTHER"],
    **autotune_cache_kwargs,
)
@triton.jit
def _leaf_state_bwd_kernel(
    R,
    H,
    DH,
    DH_OTHER,
    DU,
    DO,
    DW,
    DQ,
    DK_OUT,
    DF,
    C: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    SHARED_F: tl.constexpr,
    F_PRIMARY: tl.constexpr,
    F_OTHER: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """FLA-style fused W/Q/K and transport gradients for one state."""
    pid_chunk = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1)
    rows = tl.arange(0, C)
    keys = pid_k * BK + tl.arange(0, BK)
    key_mask = keys < DK

    grad_w = tl.zeros((C, BK), dtype=tl.float32)
    grad_q = tl.zeros((C, BK), dtype=tl.float32)
    grad_k = tl.zeros((C, BK), dtype=tl.float32)
    grad_f_primary = tl.zeros((BK,), dtype=tl.float32)
    grad_f_other = tl.zeros((BK,), dtype=tl.float32)

    for value_block in range(tl.cdiv(DV, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < DV
        token_value_mask = value_mask[None, :]
        state_mask = key_mask[:, None] & value_mask[None, :]
        token_offset = (
            pid_chunk * C * DV + rows[:, None] * DV + values[None, :]
        )
        state_offset = (
            pid_chunk * DK * DV + keys[:, None] * DV + values[None, :]
        )
        residual = tl.load(R + token_offset, mask=token_value_mask, other=0.0)
        grad_u = tl.load(DU + token_offset, mask=token_value_mask, other=0.0)
        grad_output = tl.load(
            DO + token_offset, mask=token_value_mask, other=0.0
        )
        state = tl.load(H + state_offset, mask=state_mask, other=0.0)
        grad_state = tl.load(DH + state_offset, mask=state_mask, other=0.0)
        grad_state_other = tl.load(
            DH_OTHER + state_offset, mask=state_mask, other=0.0
        )

        grad_w -= safe_dot(grad_u, tl.trans(state))
        grad_q += safe_dot(grad_output, tl.trans(state))
        grad_k += safe_dot(residual, tl.trans(grad_state))
        grad_f_primary += tl.sum(
            grad_state.to(tl.float32) * state.to(tl.float32), axis=1
        )
        grad_f_other += tl.sum(
            grad_state_other.to(tl.float32) * state.to(tl.float32), axis=1
        )

    leaf_offset = pid_chunk * C * DK + rows[:, None] * DK + keys[None, :]
    leaf_mask = key_mask[None, :]
    tl.store(DW + leaf_offset, grad_w, mask=leaf_mask)
    tl.store(DQ + leaf_offset, grad_q, mask=leaf_mask)
    tl.store(DK_OUT + leaf_offset, grad_k, mask=leaf_mask)
    if SHARED_F:
        tl.atomic_add(
            DF + pid_chunk * 4 + F_PRIMARY,
            tl.sum(tl.where(key_mask, grad_f_primary, 0.0), axis=0),
        )
        tl.atomic_add(
            DF + pid_chunk * 4 + F_OTHER,
            tl.sum(tl.where(key_mask, grad_f_other, 0.0), axis=0),
        )
    else:
        tl.store(
            DF + pid_chunk * DK * 4 + keys * 4 + F_PRIMARY,
            grad_f_primary,
            mask=key_mask,
        )
        tl.store(
            DF + pid_chunk * DK * 4 + keys * 4 + F_OTHER,
            grad_f_other,
            mask=key_mask,
        )


@triton.autotune(
    configs=_LEAF_CONFIGS,
    key=["C", "DK", "DV", "READOUT_GATE"],
    **autotune_cache_kwargs,
)
@triton.jit
def _leaf_pair_bwd_kernel(
    R,
    HS,
    HX,
    DHS,
    DHX,
    DU,
    DO,
    DW1,
    DW2,
    DQ1,
    DQ2,
    DK1,
    DK2,
    DF,
    C: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    READOUT_GATE: tl.constexpr = False,
    OH: tl.constexpr = 1, ON: tl.constexpr = 1,
    OS0: tl.constexpr = 0, OS1: tl.constexpr = 0, OS2: tl.constexpr = 0,
    OS3: tl.constexpr = 0, OS4: tl.constexpr = -1,
):
    """Accumulate both states' leaf gradients while sharing token loads."""
    pid_chunk = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1)
    rows = tl.arange(0, C)
    keys = pid_k * BK + tl.arange(0, BK)
    key_mask = keys < DK

    grad_w1 = tl.zeros((C, BK), dtype=tl.float32)
    grad_w2 = tl.zeros((C, BK), dtype=tl.float32)
    grad_q1 = tl.zeros((C, BK), dtype=tl.float32)
    grad_q2 = tl.zeros((C, BK), dtype=tl.float32)
    grad_k1 = tl.zeros((C, BK), dtype=tl.float32)
    grad_k2 = tl.zeros((C, BK), dtype=tl.float32)
    grad_f11 = tl.zeros((BK,), dtype=tl.float32)
    grad_f12 = tl.zeros((BK,), dtype=tl.float32)
    grad_f21 = tl.zeros((BK,), dtype=tl.float32)
    grad_f22 = tl.zeros((BK,), dtype=tl.float32)

    for value_block in range(tl.cdiv(DV, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < DV
        token_mask = value_mask[None, :]
        state_mask = key_mask[:, None] & value_mask[None, :]
        token_offset = (
            pid_chunk * C * DV + rows[:, None] * DV + values[None, :]
        )
        state_offset = (
            pid_chunk * DK * DV + keys[:, None] * DV + values[None, :]
        )
        residual = tl.load(R + token_offset, mask=token_mask, other=0.0)
        grad_u = tl.load(DU + token_offset, mask=token_mask, other=0.0)
        grad_output = tl.load(DO + _output_offset(pid_chunk // ON, pid_chunk % ON, rows, values, C, DV, OH, ON, OS0, OS1, OS2, OS3, OS4), mask=token_mask, other=0.0)
        hs = tl.load(HS + state_offset, mask=state_mask, other=0.0)
        hx = tl.load(HX + state_offset, mask=state_mask, other=0.0)
        grad_hs = tl.load(DHS + state_offset, mask=state_mask, other=0.0)
        grad_hx = tl.load(DHX + state_offset, mask=state_mask, other=0.0)

        grad_w1 -= safe_dot(grad_u, tl.trans(hs))
        if not READOUT_GATE:
            grad_w2 -= safe_dot(grad_u, tl.trans(hx))
        grad_q1 += safe_dot(grad_output, tl.trans(hs))
        grad_q2 += safe_dot(grad_output, tl.trans(hx))
        grad_k1 += safe_dot(residual, tl.trans(grad_hs))
        grad_k2 += safe_dot(residual, tl.trans(grad_hx))
        grad_f11 += tl.sum(
            grad_hs.to(tl.float32) * hs.to(tl.float32), axis=1
        )
        grad_f12 += tl.sum(
            grad_hx.to(tl.float32) * hs.to(tl.float32), axis=1
        )
        if not READOUT_GATE:
            grad_f21 += tl.sum(
                grad_hs.to(tl.float32) * hx.to(tl.float32), axis=1
            )
        grad_f22 += tl.sum(
            grad_hx.to(tl.float32) * hx.to(tl.float32), axis=1
        )

    leaf_offset = pid_chunk * C * DK + rows[:, None] * DK + keys[None, :]
    leaf_mask = key_mask[None, :]
    tl.store(DW1 + leaf_offset, grad_w1, mask=leaf_mask)
    if not READOUT_GATE:
        tl.store(DW2 + leaf_offset, grad_w2, mask=leaf_mask)
    tl.store(DQ1 + leaf_offset, grad_q1, mask=leaf_mask)
    tl.store(DQ2 + leaf_offset, grad_q2, mask=leaf_mask)
    tl.store(DK1 + leaf_offset, grad_k1, mask=leaf_mask)
    tl.store(DK2 + leaf_offset, grad_k2, mask=leaf_mask)
    transport_offset = pid_chunk * DK * 4 + keys * 4
    tl.store(DF + transport_offset + 0, grad_f11, mask=key_mask)
    tl.store(DF + transport_offset + 1, grad_f12, mask=key_mask)
    tl.store(DF + transport_offset + 2, grad_f21, mask=key_mask)
    tl.store(DF + transport_offset + 3, grad_f22, mask=key_mask)


@triton.autotune(
    configs=_LEAF_CONFIGS,
    key=["C", "DK", "DV"],
    reset_to_zero=["DAV", "DF"],
    **autotune_cache_kwargs,
)
@triton.jit
def _scalar_leaf_pair_bwd_kernel(
    R,
    HS,
    HX,
    DHS,
    DHX,
    DU,
    DO,
    Q,
    AV,
    DW1,
    DW2,
    DQ,
    DAV,
    DK1,
    DK2,
    DF,
    C: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Fuse scalar query/transport reductions into the two-state leaf pass."""
    pid_chunk = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1)
    rows = tl.arange(0, C)
    keys = pid_k * BK + tl.arange(0, BK)
    key_mask = keys < DK

    grad_w1 = tl.zeros((C, BK), dtype=tl.float32)
    grad_w2 = tl.zeros((C, BK), dtype=tl.float32)
    grad_q1 = tl.zeros((C, BK), dtype=tl.float32)
    grad_q2 = tl.zeros((C, BK), dtype=tl.float32)
    grad_k1 = tl.zeros((C, BK), dtype=tl.float32)
    grad_k2 = tl.zeros((C, BK), dtype=tl.float32)
    grad_f11 = tl.zeros((BK,), dtype=tl.float32)
    grad_f12 = tl.zeros((BK,), dtype=tl.float32)
    grad_f21 = tl.zeros((BK,), dtype=tl.float32)
    grad_f22 = tl.zeros((BK,), dtype=tl.float32)

    for value_block in range(tl.cdiv(DV, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < DV
        token_mask = value_mask[None, :]
        state_mask = key_mask[:, None] & value_mask[None, :]
        token_offset = (
            pid_chunk * C * DV + rows[:, None] * DV + values[None, :]
        )
        state_offset = (
            pid_chunk * DK * DV + keys[:, None] * DV + values[None, :]
        )
        residual = tl.load(R + token_offset, mask=token_mask, other=0.0)
        grad_u = tl.load(DU + token_offset, mask=token_mask, other=0.0)
        grad_output = tl.load(DO + token_offset, mask=token_mask, other=0.0)
        hs = tl.load(HS + state_offset, mask=state_mask, other=0.0)
        hx = tl.load(HX + state_offset, mask=state_mask, other=0.0)
        grad_hs = tl.load(DHS + state_offset, mask=state_mask, other=0.0)
        grad_hx = tl.load(DHX + state_offset, mask=state_mask, other=0.0)

        grad_w1 -= safe_dot(grad_u, tl.trans(hs))
        grad_w2 -= safe_dot(grad_u, tl.trans(hx))
        grad_q1 += safe_dot(grad_output, tl.trans(hs))
        grad_q2 += safe_dot(grad_output, tl.trans(hx))
        grad_k1 += safe_dot(residual, tl.trans(grad_hs))
        grad_k2 += safe_dot(residual, tl.trans(grad_hx))
        grad_f11 += tl.sum(
            grad_hs.to(tl.float32) * hs.to(tl.float32), axis=1
        )
        grad_f12 += tl.sum(
            grad_hx.to(tl.float32) * hs.to(tl.float32), axis=1
        )
        grad_f21 += tl.sum(
            grad_hs.to(tl.float32) * hx.to(tl.float32), axis=1
        )
        grad_f22 += tl.sum(
            grad_hx.to(tl.float32) * hx.to(tl.float32), axis=1
        )

    leaf_offset = pid_chunk * C * DK + rows[:, None] * DK + keys[None, :]
    leaf_mask = key_mask[None, :]
    query = tl.load(Q + leaf_offset, mask=leaf_mask, other=0.0).to(tl.float32)
    a1 = tl.load(AV + pid_chunk * C * 2 + rows * 2 + 0).to(tl.float32)
    a2 = tl.load(AV + pid_chunk * C * 2 + rows * 2 + 1).to(tl.float32)
    grad_query = a1[:, None] * grad_q1 + a2[:, None] * grad_q2
    tl.store(DW1 + leaf_offset, grad_w1, mask=leaf_mask)
    tl.store(DW2 + leaf_offset, grad_w2, mask=leaf_mask)
    tl.store(DQ + leaf_offset, grad_query, mask=leaf_mask)
    tl.store(DK1 + leaf_offset, grad_k1, mask=leaf_mask)
    tl.store(DK2 + leaf_offset, grad_k2, mask=leaf_mask)

    grad_a1 = tl.sum(
        tl.where(leaf_mask, grad_q1 * query, 0.0), axis=1
    )
    grad_a2 = tl.sum(
        tl.where(leaf_mask, grad_q2 * query, 0.0), axis=1
    )
    tl.atomic_add(DAV + pid_chunk * C * 2 + rows * 2 + 0, grad_a1)
    tl.atomic_add(DAV + pid_chunk * C * 2 + rows * 2 + 1, grad_a2)
    transport_offset = pid_chunk * 4
    tl.atomic_add(
        DF + transport_offset + 0,
        tl.sum(tl.where(key_mask, grad_f11, 0.0), axis=0),
    )
    tl.atomic_add(
        DF + transport_offset + 1,
        tl.sum(tl.where(key_mask, grad_f12, 0.0), axis=0),
    )
    tl.atomic_add(
        DF + transport_offset + 2,
        tl.sum(tl.where(key_mask, grad_f21, 0.0), axis=0),
    )
    tl.atomic_add(
        DF + transport_offset + 3,
        tl.sum(tl.where(key_mask, grad_f22, 0.0), axis=0),
    )


@triton.autotune(
    configs=_CAUSAL_GRAD_CONFIGS,
    key=["C", "DV"],
    **autotune_cache_kwargs,
)
@triton.jit
def _causal_bwd_kernel(
    R,
    DO,
    DL,
    C: tl.constexpr,
    DV: tl.constexpr,
    BV: tl.constexpr,
    OH: tl.constexpr = 1, ON: tl.constexpr = 1,
    OS0: tl.constexpr = 0, OS1: tl.constexpr = 0, OS2: tl.constexpr = 0,
    OS3: tl.constexpr = 0, OS4: tl.constexpr = -1,
):
    pid_chunk = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, C)
    grad_causal = tl.zeros((C, C), dtype=tl.float32)
    for value_block in range(tl.cdiv(DV, BV)):
        values = value_block * BV + tl.arange(0, BV)
        value_mask = values < DV
        offset = pid_chunk * C * DV + rows[:, None] * DV + values[None, :]
        residual = tl.load(R + offset, mask=value_mask[None, :], other=0.0)
        grad_output = tl.load(DO + _output_offset(pid_chunk // ON, pid_chunk % ON, rows, values, C, DV, OH, ON, OS0, OS1, OS2, OS3, OS4), mask=value_mask[None, :], other=0.0)
        grad_causal += safe_dot(grad_output, tl.trans(residual))
    tl.store(DL + pid_chunk * C * C + rows[:, None] * C + rows[None, :], grad_causal)


@triton.jit
def _merge_shared_query_bwd_kernel(
    DQ1,
    DQ2,
    Q,
    AV,
    DQ,
    DAV,
    C: tl.constexpr,
    DK: tl.constexpr,
    BK: tl.constexpr,
):
    """Backpropagate q1=a1*q and q2=a2*q without materializing either."""
    pid_chunk = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, C)
    keys = tl.arange(0, BK)
    key_mask = keys < DK
    offset = pid_chunk * C * DK + rows[:, None] * DK + keys[None, :]
    mask = key_mask[None, :]
    dq1 = tl.load(DQ1 + offset, mask=mask, other=0.0).to(tl.float32)
    dq2 = tl.load(DQ2 + offset, mask=mask, other=0.0).to(tl.float32)
    query = tl.load(Q + offset, mask=mask, other=0.0).to(tl.float32)
    a1 = tl.load(AV + pid_chunk * C * 2 + rows * 2 + 0).to(tl.float32)
    a2 = tl.load(AV + pid_chunk * C * 2 + rows * 2 + 1).to(tl.float32)
    dq = a1[:, None] * dq1 + a2[:, None] * dq2
    tl.store(DQ + offset, dq, mask=mask)
    da1 = tl.sum(tl.where(mask, dq1 * query, 0.0), axis=1)
    da2 = tl.sum(tl.where(mask, dq2 * query, 0.0), axis=1)
    tl.store(DAV + pid_chunk * C * 2 + rows * 2 + 0, da1)
    tl.store(DAV + pid_chunk * C * 2 + rows * 2 + 1, da2)


@triton.jit
def _output_offset(bh, chunk, rows, values, C: tl.constexpr, DV: tl.constexpr,
                   OH: tl.constexpr, ON, OS0: tl.constexpr,
                   OS1: tl.constexpr, OS2: tl.constexpr, OS3: tl.constexpr, OS4: tl.constexpr):
    if OS4 < 0:
        return (bh * ON + chunk) * C * DV + rows[:, None] * DV + values[None, :]
    return ((bh // OH) * OS0 + (bh % OH) * OS1 + chunk * OS2
            + rows[:, None] * OS3 + values[None, :] * OS4)


class _IntegralScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, u0, w1, w2, q1, q2, k1, k2, causal, f, hs, hx, zero_initial, readout_gate, output_heads=1,
        store_states=True,
    ):
        batch_heads, chunks, chunk_size, value_dim = u0.shape
        key_dim = w1.shape[-1]
        ctx.readout_gate = readout_gate
        ctx.output_heads = output_heads if readout_gate else 1
        if ctx.output_heads < 1 or batch_heads % ctx.output_heads:
            raise ValueError("output_heads must be a positive divisor of batch_heads")
        shared_f = f.ndim == 3
        shared_q = q2.ndim == q1.ndim + 1
        expected_f_shape = (
            (batch_heads, chunks, 4)
            if shared_f
            else (batch_heads, chunks, key_dim, 4)
        )
        if f.shape != expected_f_shape:
            raise ValueError(f"transport must have shape {expected_f_shape}")
        ctx.input_dtypes = tuple(
            x.dtype for x in (u0, w1, w2, q1, q2, k1, k2, causal, f, hs, hx)
        )
        u0, w1, w2, k1, k2, causal = (
            x.contiguous().to(torch.bfloat16)
            for x in (u0, w1, w2, k1, k2, causal)
        )
        q1 = q1.contiguous()
        q2 = q2.contiguous().float() if shared_q else q2.contiguous()
        f, hs, hx = (x.contiguous().float() for x in (f, hs, hx))
        residual = torch.empty_like(u0)
        if ctx.output_heads > 1:
            output = torch.empty(batch_heads // ctx.output_heads, chunks, chunk_size,
                                 ctx.output_heads, value_dim, device=u0.device, dtype=u0.dtype).permute(0,3,1,2,4)
        else:
            output = torch.empty_like(u0)
        output_launch = {}
        if ctx.output_heads > 1:
            output_launch = dict(
                OH=ctx.output_heads, ON=chunks,
                **{f"OS{i}": stride for i, stride in enumerate(output.stride())},
            )
        fuse_output = chunk_size in (16, 32, 64)
        store_states = store_states or not fuse_output
        saved_hs = torch.empty(
            (batch_heads, chunks, key_dim, value_dim) if store_states else (0,),
            dtype=torch.bfloat16,
            device=u0.device,
        )
        saved_hx = torch.empty_like(saved_hs)
        final_hs = torch.empty_like(hs)
        final_hx = torch.empty_like(hx)

        def state_grid(meta):
            return batch_heads, triton.cdiv(value_dim, meta["BV"])

        pinned = _pinned_scan_configs(
            u0, chunk_size, key_dim, value_dim, shared_f, shared_q
        )
        integral_pinned = (
            _PINNED_INTEGRAL_SCAN_CONFIGS.get(torch.cuda.get_device_capability(u0.device))
            if u0.is_cuda else None
        )
        if (readout_gate and integral_pinned is not None
                and chunk_size in integral_pinned[0]
                and key_dim == 128 and value_dim == 128
                and optimized_readout(u0, chunks * chunk_size)):
            pinned = _small_grid_tiles(
                integral_pinned, u0.device, chunk_size, batch_heads, value_dim
            )
        use_pinned = pinned is not None
        state_kernel = _scan_state_fwd_kernel
        state_launch = {}
        if use_pinned:
            value_tile, warps, stages = _scan_tile(pinned[0][chunk_size], batch_heads)
            state_kernel = getattr(state_kernel, "fn", state_kernel)
            state_grid = (batch_heads, triton.cdiv(value_dim, value_tile))
            state_launch.update(
                BV=value_tile,
                num_warps=warps,
                num_stages=stages,
            )
        state_kernel[state_grid](
            u0, w1, w2, q1, q2, k1, k2, causal, f, hs, hx,
            residual, saved_hs, saved_hx, output, final_hs, final_hx,
            N=chunks,
            C=chunk_size,
            DK=key_dim,
            DV=value_dim,
            SHARED_F=shared_f,
            SHARED_Q=shared_q,
            FUSE_OUTPUT=fuse_output,
            ZERO_INITIAL=zero_initial,
            READOUT_GATE=readout_gate,
            STORE_STATES=store_states,
            **output_launch,
            **state_launch,
        )
        ctx.use_pinned = use_pinned
        ctx.pinned = pinned

        def output_grid(meta):
            return batch_heads * chunks, triton.cdiv(value_dim, meta["BV"])

        if not fuse_output:
            _output_fwd_kernel[output_grid](
                residual, q1, q2, causal, saved_hs, saved_hx, output,
                C=chunk_size,
                DK=key_dim,
                DV=value_dim,
                SHARED_Q=shared_q,
                **output_launch,
            )
        ctx.save_for_backward(
            residual, w1, w2, q1, q2, k1, k2, causal, f, saved_hs, saved_hx
        )
        ctx.dimensions = (batch_heads, chunks, chunk_size, key_dim, value_dim)
        ctx.shared_f = shared_f
        ctx.shared_q = shared_q
        return output, final_hs, final_hx

    @staticmethod
    def backward(ctx, grad_output, grad_final_hs, grad_final_hx):
        residual, w1, w2, q1, q2, k1, k2, causal, f, hs, hx = ctx.saved_tensors
        batch_heads, chunks, chunk_size, key_dim, value_dim = ctx.dimensions
        output_launch = {}
        if ctx.output_heads == 1:
            grad_output = grad_output.contiguous()
        else:
            output_launch = dict(
                OH=ctx.output_heads, ON=chunks,
                **{f"OS{i}": stride for i, stride in enumerate(grad_output.stride())},
            )
        if grad_final_hs is None:
            grad_final_hs = torch.zeros(
                batch_heads, key_dim, value_dim, device=grad_output.device, dtype=torch.float32
            )
        if grad_final_hx is None:
            grad_final_hx = torch.zeros_like(grad_final_hs)
        grad_final_hs = grad_final_hs.contiguous().float()
        grad_final_hx = grad_final_hx.contiguous().float()

        grad_u0 = torch.empty_like(residual)
        grad_state_hs = torch.empty_like(hs)
        grad_state_hx = torch.empty_like(hx)
        grad_initial_hs = torch.empty_like(grad_final_hs)
        grad_initial_hx = torch.empty_like(grad_final_hx)

        def state_bwd_grid(meta):
            return batch_heads, triton.cdiv(value_dim, meta["BV"])

        state_bwd_kernel = _scan_state_bwd_kernel
        state_bwd_launch = {}
        if ctx.use_pinned:
            value_tile, warps, stages = _scan_tile(ctx.pinned[1][chunk_size], batch_heads)
            state_bwd_kernel = getattr(state_bwd_kernel, "fn", state_bwd_kernel)
            state_bwd_grid = (
                batch_heads,
                triton.cdiv(value_dim, value_tile),
            )
            state_bwd_launch.update(
                BV=value_tile,
                num_warps=warps,
                num_stages=stages,
            )
        state_bwd_kernel[state_bwd_grid](
            grad_output, q1, q2, causal,
            grad_final_hs, grad_final_hx,
            w1, w2, k1, k2, f,
            grad_u0, grad_state_hs, grad_state_hx,
            grad_initial_hs, grad_initial_hx,
            grad_u0, grad_state_hs, grad_state_hx,
            N=chunks,
            C=chunk_size,
            DK=key_dim,
            DV=value_dim,
            SHARED_F=ctx.shared_f,
            SHARED_Q=ctx.shared_q,
            READOUT_GATE=ctx.readout_gate,
            **output_launch,
            **state_bwd_launch,
        )

        # Match FLA's backward organization: each launch accumulates W/Q/K
        # gradients over V in registers instead of materializing independent
        # batched matmuls and their FP32 conversion buffers. Scalar GDN and
        # DeltaNet additionally fuse the shared-query merge and shared 2x2
        # transport reduction, avoiding two CxK temporaries and two launches.
        grad_w1 = torch.empty_like(w1)
        grad_w2 = None if ctx.readout_gate else torch.empty_like(w2)
        grad_k1, grad_k2 = torch.empty_like(k1), torch.empty_like(k2)
        grad_causal = torch.empty_like(causal)

        def leaf_grid(meta):
            return batch_heads * chunks, triton.cdiv(key_dim, meta["BK"])

        if ctx.shared_f and ctx.shared_q:
            grad_q1 = torch.empty_like(q1)
            grad_q2 = torch.zeros_like(q2)
            grad_f = torch.zeros_like(f)
            scalar_leaf_kernel = _scalar_leaf_pair_bwd_kernel
            scalar_leaf_launch = {}
            if ctx.use_pinned:
                block_key, block_value, warps, stages = _scan_tile(
                    ctx.pinned[2][chunk_size], batch_heads
                )
                scalar_leaf_kernel = getattr(
                    scalar_leaf_kernel, "fn", scalar_leaf_kernel
                )
                leaf_grid = (
                    batch_heads * chunks,
                    triton.cdiv(key_dim, block_key),
                )
                scalar_leaf_launch.update(
                    BK=block_key,
                    BV=block_value,
                    num_warps=warps,
                    num_stages=stages,
                )
            scalar_leaf_kernel[leaf_grid](
                residual,
                hs,
                hx,
                grad_state_hs,
                grad_state_hx,
                grad_u0,
                grad_output,
                q1,
                q2,
                grad_w1,
                grad_w2,
                grad_q1,
                grad_q2,
                grad_k1,
                grad_k2,
                grad_f,
                C=chunk_size,
                DK=key_dim,
                DV=value_dim,
                **scalar_leaf_launch,
            )
        else:
            grad_q1_leaf = torch.empty_like(q1)
            grad_q2_leaf = torch.empty_like(q1 if ctx.shared_q else q2)
            # Autotuners may execute a candidate more than once, so write
            # deterministic per-key transport partials and reduce afterward.
            grad_f_leaf = (
                torch.empty(
                    batch_heads,
                    chunks,
                    key_dim,
                    4,
                    dtype=f.dtype,
                    device=f.device,
                )
                if ctx.shared_f
                else torch.empty_like(f)
            )
            leaf_kernel = _leaf_pair_bwd_kernel
            leaf_launch = {}
            if ctx.use_pinned:
                block_key, block_value, warps, stages = _scan_tile(
                    ctx.pinned[2][chunk_size], batch_heads
                )
                leaf_kernel = getattr(leaf_kernel, "fn", leaf_kernel)
                leaf_grid = (
                    batch_heads * chunks,
                    triton.cdiv(key_dim, block_key),
                )
                leaf_launch.update(
                    BK=block_key,
                    BV=block_value,
                    num_warps=warps,
                    num_stages=stages,
                )
            leaf_kernel[leaf_grid](
                residual,
                hs,
                hx,
                grad_state_hs,
                grad_state_hx,
                grad_u0,
                grad_output,
                grad_w1,
                grad_w1 if ctx.readout_gate else grad_w2,
                grad_q1_leaf,
                grad_q2_leaf,
                grad_k1,
                grad_k2,
                grad_f_leaf,
                C=chunk_size,
                DK=key_dim,
                DV=value_dim,
                READOUT_GATE=ctx.readout_gate,
                **output_launch,
                **leaf_launch,
            )
            if ctx.shared_q:
                grad_q1 = torch.empty_like(q1)
                grad_q2 = torch.empty_like(q2)
                _merge_shared_query_bwd_kernel[(batch_heads * chunks,)](
                    grad_q1_leaf,
                    grad_q2_leaf,
                    q1,
                    q2,
                    grad_q1,
                    grad_q2,
                    C=chunk_size,
                    DK=key_dim,
                    BK=triton.next_power_of_2(key_dim),
                    num_warps=4,
                )
            else:
                grad_q1, grad_q2 = grad_q1_leaf, grad_q2_leaf
            grad_f = grad_f_leaf.sum(dim=-2) if ctx.shared_f else grad_f_leaf

        causal_kernel = _causal_bwd_kernel
        causal_launch = {}
        if (
            ctx.use_pinned
            and chunk_size in ctx.pinned[3]
        ):
            value_tile, warps, stages = _scan_tile(ctx.pinned[3][chunk_size], batch_heads)
            causal_kernel = getattr(causal_kernel, "fn", causal_kernel)
            causal_launch.update(
                BV=value_tile,
                num_warps=warps,
                num_stages=stages,
            )
        causal_kernel[(batch_heads * chunks,)](
            residual,
            grad_output,
            grad_causal,
            C=chunk_size,
            DV=value_dim,
            **output_launch,
            **causal_launch,
        )
        grads = (
            grad_u0, grad_w1, grad_w2, grad_q1, grad_q2,
            grad_k1, grad_k2, grad_causal, grad_f,
            grad_initial_hs, grad_initial_hx,
        )
        return tuple(
            None if grad is None else grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)
        ) + (None,) * (len(ctx.needs_input_grad) - 11)


@torch.compiler.disable
def fused_integral_scan(
    u0,
    w1,
    w2,
    q1,
    q2,
    k1,
    k2,
    causal,
    f,
    hs,
    hx,
    *,
    zero_initial=False,
    readout_gate=False,
    output_heads=1,
):
    # Readout-side callers can expose B,H,N,C,V as a view of token-major storage;
    # the public chunk output then becomes contiguous without a transpose copy.
    # Without a backward to feed (no grad, as in memory-first's first pass),
    # the per-chunk states are not written.
    store_states = torch.is_grad_enabled() and any(
        x.requires_grad for x in (u0, w1, w2, q1, q2, k1, k2, causal, f, hs, hx)
    )
    return _IntegralScan.apply(
        u0, w1, w2, q1, q2, k1, k2, causal, f, hs, hx, zero_initial, readout_gate, output_heads,
        store_states,
    )
__all__ = ["fused_integral_scan"]

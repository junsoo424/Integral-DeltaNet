"""Bounded C16 block transport shared by general integral builders.

The direct path keeps exact inverse-free diagonal C16 recurrences and
composes only strict lower off-diagonal pairs for C32/C64.  The centered
builder shares the same bounded transport.  Custom Triton backward
avoids generated graphs and an unbounded first-backward compile at C64.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import safe_dot

from .general_coeff_triton import (
    _BLOCK_KEY,
    _BLOCK_WARPS,
    _GeneralCoefficients,
)
from .general_triton import solve_general_operands
from .tuning import by_arch


_LOCAL_BLOCK = 16
_TRANSPORT_CONFIGS = {
    32: (16, 1, 1),
    64: (64, 1, 2),
}
_TRANSPORT_BACKWARD_CONFIGS = {
    32: (64, 1, 3),
    64: (16, 1, 3),
}

# Pinned transport launches per architecture: ops/tuning.
_PINNED_TRANSPORT_CONFIGS = by_arch("TRANSPORT")


def _transport_configs(tensor):
    """Forward/backward transport launch tables for `tensor`'s device."""
    if tensor.is_cuda:
        pinned = _PINNED_TRANSPORT_CONFIGS.get(
            torch.cuda.get_device_capability(tensor.device)
        )
        if pinned is not None:
            return pinned
    return _TRANSPORT_CONFIGS, _TRANSPORT_BACKWARD_CONFIGS

# Pinned intra launches per architecture: ops/tuning.
_PINNED_INTRA_CONFIGS = by_arch("INTRA")


def _pinned_intra_configs(tensor, key_dim, chunk_size):
    """(forward, backward) intra launch tables for this device, or None."""
    if not tensor.is_cuda or key_dim != 128:
        return None
    tables = _PINNED_INTRA_CONFIGS.get(
        torch.cuda.get_device_capability(tensor.device)
    )
    if tables is None or chunk_size not in tables[0]:
        return None
    return tables


@torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
def _compose_general_blocks(
    local_a1,
    local_a2,
    local_c1,
    local_c2,
    local_q1,
    local_q2,
    local_tail1,
    local_tail2,
    local_transport,
    chunk_size,
):
    batch, heads, local_chunks, _, key_dim = local_a1.shape
    blocks = chunk_size // _LOCAL_BLOCK
    chunks = local_chunks // blocks

    def group_vectors(tensor):
        return tensor.reshape(
            batch, heads, chunks, blocks, _LOCAL_BLOCK, key_dim
        )

    a1 = group_vectors(local_a1)
    a2 = group_vectors(local_a2)
    c1 = group_vectors(local_c1)
    c2 = group_vectors(local_c2)
    q1 = group_vectors(local_q1)
    q2 = group_vectors(local_q2)
    local_tail1 = group_vectors(local_tail1)
    local_tail2 = group_vectors(local_tail2)
    transport = local_transport.reshape(
        batch, heads, chunks, blocks, key_dim, 4
    )

    identity = torch.ones_like(transport[..., 0, :, 0])
    zero = torch.zeros_like(identity)
    f11, f12, f21, f22 = identity, zero, zero, identity
    global_a1, global_a2, global_q1, global_q2 = [], [], [], []
    for block in range(blocks):
        before11, before12 = f11, f12
        before21, before22 = f21, f22
        global_a1.append(
            before11.unsqueeze(-2) * a1[..., block, :, :]
            + before12.unsqueeze(-2) * a2[..., block, :, :]
        )
        global_a2.append(
            before21.unsqueeze(-2) * a1[..., block, :, :]
            + before22.unsqueeze(-2) * a2[..., block, :, :]
        )
        global_q1.append(
            before11.unsqueeze(-2) * q1[..., block, :, :]
            + before12.unsqueeze(-2) * q2[..., block, :, :]
        )
        global_q2.append(
            before21.unsqueeze(-2) * q1[..., block, :, :]
            + before22.unsqueeze(-2) * q2[..., block, :, :]
        )
        block_transport = transport[..., block, :, :]
        b11, b12 = block_transport[..., 0], block_transport[..., 1]
        b21, b22 = block_transport[..., 2], block_transport[..., 3]
        next11 = f11 * b11 + f12 * b21
        next12 = f11 * b12 + f12 * b22
        next21 = f21 * b11 + f22 * b21
        next22 = f21 * b12 + f22 * b22
        f11, f12, f21, f22 = next11, next12, next21, next22

    pair_c1, pair_c2 = [], []
    for target in range(blocks):
        for source in range(target + 1):
            if source == target:
                relative1 = c1[..., source, :, :]
                relative2 = c2[..., source, :, :]
            else:
                relative1 = local_tail1[..., source, :, :]
                relative2 = local_tail2[..., source, :, :]
                for middle in range(source + 1, target):
                    middle_transport = transport[..., middle, :, :]
                    m11 = middle_transport[..., 0].unsqueeze(-2)
                    m12 = middle_transport[..., 1].unsqueeze(-2)
                    m21 = middle_transport[..., 2].unsqueeze(-2)
                    m22 = middle_transport[..., 3].unsqueeze(-2)
                    next1 = relative1 * m11 + relative2 * m21
                    next2 = relative1 * m12 + relative2 * m22
                    relative1, relative2 = next1, next2
            pair_c1.append(relative1)
            pair_c2.append(relative2)

    final_tail1, final_tail2 = [], []
    for source in range(blocks):
        relative1 = local_tail1[..., source, :, :]
        relative2 = local_tail2[..., source, :, :]
        for middle in range(source + 1, blocks):
            middle_transport = transport[..., middle, :, :]
            m11 = middle_transport[..., 0].unsqueeze(-2)
            m12 = middle_transport[..., 1].unsqueeze(-2)
            m21 = middle_transport[..., 2].unsqueeze(-2)
            m22 = middle_transport[..., 3].unsqueeze(-2)
            next1 = relative1 * m11 + relative2 * m21
            next2 = relative1 * m12 + relative2 * m22
            relative1, relative2 = next1, next2
        final_tail1.append(relative1)
        final_tail2.append(relative2)

    dtype = local_a1.dtype

    def merge_blocks(values):
        return torch.stack(values, dim=3).reshape(
            batch, heads, chunks, chunk_size, key_dim
        ).to(dtype).contiguous()

    global_vectors = tuple(
        merge_blocks(values)
        for values in (global_a1, global_a2, global_q1, global_q2)
    )
    pair_vectors = tuple(
        torch.stack(values, dim=3).to(dtype).contiguous()
        for values in (pair_c1, pair_c2)
    )
    tails = tuple(
        merge_blocks(values) for values in (final_tail1, final_tail2)
    )
    final_transport = torch.stack((f11, f12, f21, f22), dim=-1).contiguous()
    return (*global_vectors, *pair_vectors, *tails, final_transport)


@torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
def _compose_general_blocks_c64(
    local_a1,
    local_a2,
    local_c1,
    local_c2,
    local_q1,
    local_q2,
    local_tail1,
    local_tail2,
    local_transport,
):
    """C64 transport graph with shared prefix and relative chains.

    The generic graph independently rebuilt every off-diagonal pair and every
    final tail.  With four local blocks, six relative 2x2 applications are
    sufficient for all of those outputs; the generic loops performed ten.
    Keeping the chain explicit also gives Inductor a small fixed graph for its
    forward and generated backward.
    """

    batch, heads, local_chunks, _, key_dim = local_a1.shape
    chunks = local_chunks // 4

    def grouped(tensor):
        return tensor.reshape(batch, heads, chunks, 4, 16, key_dim)

    a1, a2, c1, c2, q1, q2, tail1, tail2 = (
        grouped(tensor)
        for tensor in (
            local_a1,
            local_a2,
            local_c1,
            local_c2,
            local_q1,
            local_q2,
            local_tail1,
            local_tail2,
        )
    )
    transport = local_transport.reshape(batch, heads, chunks, 4, key_dim, 4)

    def matrix(block):
        value = transport[..., block, :, :]
        return value[..., 0], value[..., 1], value[..., 2], value[..., 3]

    def compose(left, right):
        l11, l12, l21, l22 = left
        r11, r12, r21, r22 = right
        return (
            l11 * r11 + l12 * r21,
            l11 * r12 + l12 * r22,
            l21 * r11 + l22 * r21,
            l21 * r12 + l22 * r22,
        )

    def transform_columns(x1, x2, value):
        f11, f12, f21, f22 = value
        return (
            f11.unsqueeze(-2) * x1 + f12.unsqueeze(-2) * x2,
            f21.unsqueeze(-2) * x1 + f22.unsqueeze(-2) * x2,
        )

    def transform_rows(x1, x2, value):
        f11, f12, f21, f22 = value
        return (
            x1 * f11.unsqueeze(-2) + x2 * f21.unsqueeze(-2),
            x1 * f12.unsqueeze(-2) + x2 * f22.unsqueeze(-2),
        )

    t0, t1, t2, t3 = (matrix(block) for block in range(4))
    identity = torch.ones_like(t0[0])
    zero = torch.zeros_like(identity)
    p0 = (identity, zero, zero, identity)
    p1 = t0
    p2 = compose(p1, t1)
    p3 = compose(p2, t2)
    final = compose(p3, t3)
    prefixes = (p0, p1, p2, p3)

    global_a, global_q = [], []
    for block in range(4):
        global_a.append(
            transform_columns(a1[..., block, :, :], a2[..., block, :, :], prefixes[block])
        )
        global_q.append(
            transform_columns(q1[..., block, :, :], q2[..., block, :, :], prefixes[block])
        )

    # Pair order is triangular target-major:
    # (0,0), (1,0), (1,1), ..., (3,3).
    r20 = transform_rows(tail1[..., 0, :, :], tail2[..., 0, :, :], t1)
    r30 = transform_rows(*r20, t2)
    r31 = transform_rows(tail1[..., 1, :, :], tail2[..., 1, :, :], t2)
    pairs1 = (
        c1[..., 0, :, :],
        tail1[..., 0, :, :],
        c1[..., 1, :, :],
        r20[0],
        tail1[..., 1, :, :],
        c1[..., 2, :, :],
        r30[0],
        r31[0],
        tail1[..., 2, :, :],
        c1[..., 3, :, :],
    )
    pairs2 = (
        c2[..., 0, :, :],
        tail2[..., 0, :, :],
        c2[..., 1, :, :],
        r20[1],
        tail2[..., 1, :, :],
        c2[..., 2, :, :],
        r30[1],
        r31[1],
        tail2[..., 2, :, :],
        c2[..., 3, :, :],
    )

    final0 = transform_rows(*r30, t3)
    final1 = transform_rows(*r31, t3)
    final2 = transform_rows(tail1[..., 2, :, :], tail2[..., 2, :, :], t3)
    tails1 = (final0[0], final1[0], final2[0], tail1[..., 3, :, :])
    tails2 = (final0[1], final1[1], final2[1], tail2[..., 3, :, :])
    dtype = local_a1.dtype

    def merge_pairs(values):
        return torch.stack(values, dim=3).to(dtype).contiguous()

    def merge_blocks(values):
        return torch.stack(values, dim=3).reshape(
            batch, heads, chunks, 64, key_dim
        ).to(dtype).contiguous()

    global_a1, global_a2 = (
        merge_blocks(tuple(values[index] for values in global_a))
        for index in range(2)
    )
    global_q1, global_q2 = (
        merge_blocks(tuple(values[index] for values in global_q))
        for index in range(2)
    )
    final_tail1, final_tail2 = merge_blocks(tails1), merge_blocks(tails2)
    final_transport = torch.stack(final, dim=-1).contiguous()
    return (
        global_a1,
        global_a2,
        global_q1,
        global_q2,
        merge_pairs(pairs1),
        merge_pairs(pairs2),
        final_tail1,
        final_tail2,
        final_transport,
    )


@triton.jit
def _general_block_prefix_fwd_kernel(
    A1,
    A2,
    Q1,
    Q2,
    LOCAL_TRANSPORT,
    GLOBAL_A1,
    GLOBAL_A2,
    GLOBAL_Q1,
    GLOBAL_Q2,
    FINAL_TRANSPORT,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    READOUT_GATE: tl.constexpr = False,
):
    # READOUT_GATE: A2 is the readout path's constant zero, so A2 is not read
    # and GLOBAL_A2 (which the readout path discards) is not written.
    pid = tl.program_id(0)
    nk = tl.cdiv(K, BK)
    key_block = pid % nk
    outer = pid // nk
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    rows = tl.arange(0, 16)
    f11 = tl.full((BK,), 1.0, tl.float32)
    f12 = tl.zeros((BK,), dtype=tl.float32)
    f21 = tl.zeros((BK,), dtype=tl.float32)
    f22 = tl.full((BK,), 1.0, tl.float32)
    for block in range(BLOCKS):
        token = outer * C + block * 16 + rows
        offsets = token[:, None] * K + keys[None, :]
        a1 = tl.load(A1 + offsets, mask=key_mask[None, :], other=0.0)
        q1 = tl.load(Q1 + offsets, mask=key_mask[None, :], other=0.0)
        q2 = tl.load(Q2 + offsets, mask=key_mask[None, :], other=0.0)
        if READOUT_GATE:
            # f12 * 0 adds an exact zero: the same rounded product.
            tl.store(GLOBAL_A1 + offsets, f11[None, :] * a1, mask=key_mask[None, :])
        else:
            a2 = tl.load(A2 + offsets, mask=key_mask[None, :], other=0.0)
            tl.store(
                GLOBAL_A1 + offsets,
                f11[None, :] * a1 + f12[None, :] * a2,
                mask=key_mask[None, :],
            )
            tl.store(
                GLOBAL_A2 + offsets,
                f21[None, :] * a1 + f22[None, :] * a2,
                mask=key_mask[None, :],
            )
        tl.store(
            GLOBAL_Q1 + offsets,
            f11[None, :] * q1 + f12[None, :] * q2,
            mask=key_mask[None, :],
        )
        tl.store(
            GLOBAL_Q2 + offsets,
            f21[None, :] * q1 + f22[None, :] * q2,
            mask=key_mask[None, :],
        )
        transport = ((outer * BLOCKS + block) * K + keys) * 4
        b11 = tl.load(LOCAL_TRANSPORT + transport + 0, mask=key_mask, other=0.0)
        b12 = tl.load(LOCAL_TRANSPORT + transport + 1, mask=key_mask, other=0.0)
        b21 = tl.load(LOCAL_TRANSPORT + transport + 2, mask=key_mask, other=0.0)
        b22 = tl.load(LOCAL_TRANSPORT + transport + 3, mask=key_mask, other=0.0)
        next11 = f11 * b11 + f12 * b21
        next12 = f11 * b12 + f12 * b22
        next21 = f21 * b11 + f22 * b21
        next22 = f21 * b12 + f22 * b22
        f11, f12, f21, f22 = next11, next12, next21, next22
    final = (outer * K + keys) * 4
    tl.store(FINAL_TRANSPORT + final + 0, f11, mask=key_mask)
    tl.store(FINAL_TRANSPORT + final + 1, f12, mask=key_mask)
    tl.store(FINAL_TRANSPORT + final + 2, f21, mask=key_mask)
    tl.store(FINAL_TRANSPORT + final + 3, f22, mask=key_mask)


@triton.jit
def _general_block_relative_fwd_kernel(
    C1,
    C2,
    LOCAL_TAIL1,
    LOCAL_TAIL2,
    LOCAL_TRANSPORT,
    PAIR_C1,
    PAIR_C2,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAIRS: tl.constexpr,
    C: tl.constexpr,
    DIRECT: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    nk = tl.cdiv(K, BK)
    key_block = pid % nk
    pair_system = pid // nk
    pair = pair_system % PAIRS
    outer = pair_system // PAIRS
    if DIRECT:
        if BLOCKS == 2:
            target = 1
            source = 0
        else:
            target = tl.where(pair < 1, 1, tl.where(pair < 3, 2, 3))
            source = pair - target * (target - 1) // 2
    else:
        if BLOCKS == 2:
            target = tl.where(pair == 0, 0, 1)
            source = tl.where(pair == 2, 1, 0)
        else:
            target = tl.where(
                pair < 1, 0, tl.where(pair < 3, 1, tl.where(pair < 6, 2, 3))
            )
            source = pair - target * (target + 1) // 2
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    rows = tl.arange(0, 16)
    local_token = outer * C + source * 16 + rows
    local_offsets = local_token[:, None] * K + keys[None, :]
    diagonal = target == source
    tail1 = tl.load(
        LOCAL_TAIL1 + local_offsets, mask=key_mask[None, :], other=0.0
    )
    tail2 = tl.load(
        LOCAL_TAIL2 + local_offsets, mask=key_mask[None, :], other=0.0
    )
    if DIRECT:
        value1 = tail1.to(tl.float32)
        value2 = tail2.to(tl.float32)
    else:
        c1 = tl.load(C1 + local_offsets, mask=key_mask[None, :], other=0.0)
        c2 = tl.load(C2 + local_offsets, mask=key_mask[None, :], other=0.0)
        value1 = tl.where(diagonal, c1, tail1).to(tl.float32)
        value2 = tl.where(diagonal, c2, tail2).to(tl.float32)
    for middle in range(1, BLOCKS):
        transport = ((outer * BLOCKS + middle) * K + keys) * 4
        f11 = tl.load(LOCAL_TRANSPORT + transport + 0, mask=key_mask, other=0.0)
        f12 = tl.load(LOCAL_TRANSPORT + transport + 1, mask=key_mask, other=0.0)
        f21 = tl.load(LOCAL_TRANSPORT + transport + 2, mask=key_mask, other=0.0)
        f22 = tl.load(LOCAL_TRANSPORT + transport + 3, mask=key_mask, other=0.0)
        next1 = value1 * f11[None, :] + value2 * f21[None, :]
        next2 = value1 * f12[None, :] + value2 * f22[None, :]
        apply = (~diagonal) & (middle > source) & (middle < target)
        value1 = tl.where(apply, next1, value1)
        value2 = tl.where(apply, next2, value2)
    pair_token = (outer * PAIRS + pair) * 16 + rows
    pair_offsets = pair_token[:, None] * K + keys[None, :]
    tl.store(PAIR_C1 + pair_offsets, value1, mask=key_mask[None, :])
    tl.store(PAIR_C2 + pair_offsets, value2, mask=key_mask[None, :])


@triton.jit
def _general_block_tail_fwd_kernel(
    LOCAL_TAIL1,
    LOCAL_TAIL2,
    LOCAL_TRANSPORT,
    FINAL_TAIL1,
    FINAL_TAIL2,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    nk = tl.cdiv(K, BK)
    key_block = pid % nk
    source_system = pid // nk
    source = source_system % BLOCKS
    outer = source_system // BLOCKS
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    rows = tl.arange(0, 16)
    token = outer * C + source * 16 + rows
    offsets = token[:, None] * K + keys[None, :]
    value1 = tl.load(
        LOCAL_TAIL1 + offsets, mask=key_mask[None, :], other=0.0
    ).to(tl.float32)
    value2 = tl.load(
        LOCAL_TAIL2 + offsets, mask=key_mask[None, :], other=0.0
    ).to(tl.float32)
    for middle in range(1, BLOCKS):
        transport = ((outer * BLOCKS + middle) * K + keys) * 4
        f11 = tl.load(LOCAL_TRANSPORT + transport + 0, mask=key_mask, other=0.0)
        f12 = tl.load(LOCAL_TRANSPORT + transport + 1, mask=key_mask, other=0.0)
        f21 = tl.load(LOCAL_TRANSPORT + transport + 2, mask=key_mask, other=0.0)
        f22 = tl.load(LOCAL_TRANSPORT + transport + 3, mask=key_mask, other=0.0)
        next1 = value1 * f11[None, :] + value2 * f21[None, :]
        next2 = value1 * f12[None, :] + value2 * f22[None, :]
        apply = middle > source
        value1 = tl.where(apply, next1, value1)
        value2 = tl.where(apply, next2, value2)
    tl.store(FINAL_TAIL1 + offsets, value1, mask=key_mask[None, :])
    tl.store(FINAL_TAIL2 + offsets, value2, mask=key_mask[None, :])


@triton.jit
def _general_block_prefix_bwd_kernel(
    A1,
    A2,
    Q1,
    Q2,
    LOCAL_TRANSPORT,
    DGLOBAL_A1,
    DGLOBAL_A2,
    DGLOBAL_Q1,
    DGLOBAL_Q2,
    DFINAL_TRANSPORT,
    DA1,
    DA2,
    DQ1,
    DQ2,
    DLOCAL_TRANSPORT,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    nk = tl.cdiv(K, BK)
    key_block = pid % nk
    outer = pid // nk
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    rows = tl.arange(0, 16)
    final = (outer * K + keys) * 4
    df11 = tl.load(DFINAL_TRANSPORT + final + 0, mask=key_mask, other=0.0)
    df12 = tl.load(DFINAL_TRANSPORT + final + 1, mask=key_mask, other=0.0)
    df21 = tl.load(DFINAL_TRANSPORT + final + 2, mask=key_mask, other=0.0)
    df22 = tl.load(DFINAL_TRANSPORT + final + 3, mask=key_mask, other=0.0)
    for reverse in range(BLOCKS):
        block = BLOCKS - 1 - reverse
        f11 = tl.full((BK,), 1.0, tl.float32)
        f12 = tl.zeros((BK,), dtype=tl.float32)
        f21 = tl.zeros((BK,), dtype=tl.float32)
        f22 = tl.full((BK,), 1.0, tl.float32)
        for middle in range(BLOCKS):
            transport_middle = ((outer * BLOCKS + middle) * K + keys) * 4
            m11 = tl.load(
                LOCAL_TRANSPORT + transport_middle + 0, mask=key_mask, other=0.0
            )
            m12 = tl.load(
                LOCAL_TRANSPORT + transport_middle + 1, mask=key_mask, other=0.0
            )
            m21 = tl.load(
                LOCAL_TRANSPORT + transport_middle + 2, mask=key_mask, other=0.0
            )
            m22 = tl.load(
                LOCAL_TRANSPORT + transport_middle + 3, mask=key_mask, other=0.0
            )
            next11 = f11 * m11 + f12 * m21
            next12 = f11 * m12 + f12 * m22
            next21 = f21 * m11 + f22 * m21
            next22 = f21 * m12 + f22 * m22
            apply = middle < block
            f11 = tl.where(apply, next11, f11)
            f12 = tl.where(apply, next12, f12)
            f21 = tl.where(apply, next21, f21)
            f22 = tl.where(apply, next22, f22)
        token = outer * C + block * 16 + rows
        offsets = token[:, None] * K + keys[None, :]
        a1 = tl.load(A1 + offsets, mask=key_mask[None, :], other=0.0).to(tl.float32)
        a2 = tl.load(A2 + offsets, mask=key_mask[None, :], other=0.0).to(tl.float32)
        q1 = tl.load(Q1 + offsets, mask=key_mask[None, :], other=0.0).to(tl.float32)
        q2 = tl.load(Q2 + offsets, mask=key_mask[None, :], other=0.0).to(tl.float32)
        da_out1 = tl.load(
            DGLOBAL_A1 + offsets, mask=key_mask[None, :], other=0.0
        ).to(tl.float32)
        da_out2 = tl.load(
            DGLOBAL_A2 + offsets, mask=key_mask[None, :], other=0.0
        ).to(tl.float32)
        dq_out1 = tl.load(
            DGLOBAL_Q1 + offsets, mask=key_mask[None, :], other=0.0
        ).to(tl.float32)
        dq_out2 = tl.load(
            DGLOBAL_Q2 + offsets, mask=key_mask[None, :], other=0.0
        ).to(tl.float32)
        tl.store(
            DA1 + offsets,
            f11[None, :] * da_out1 + f21[None, :] * da_out2,
            mask=key_mask[None, :],
        )
        tl.store(
            DA2 + offsets,
            f12[None, :] * da_out1 + f22[None, :] * da_out2,
            mask=key_mask[None, :],
        )
        tl.store(
            DQ1 + offsets,
            f11[None, :] * dq_out1 + f21[None, :] * dq_out2,
            mask=key_mask[None, :],
        )
        tl.store(
            DQ2 + offsets,
            f12[None, :] * dq_out1 + f22[None, :] * dq_out2,
            mask=key_mask[None, :],
        )
        direct11 = tl.sum(da_out1 * a1 + dq_out1 * q1, axis=0)
        direct12 = tl.sum(da_out1 * a2 + dq_out1 * q2, axis=0)
        direct21 = tl.sum(da_out2 * a1 + dq_out2 * q1, axis=0)
        direct22 = tl.sum(da_out2 * a2 + dq_out2 * q2, axis=0)
        transport = ((outer * BLOCKS + block) * K + keys) * 4
        b11 = tl.load(LOCAL_TRANSPORT + transport + 0, mask=key_mask, other=0.0)
        b12 = tl.load(LOCAL_TRANSPORT + transport + 1, mask=key_mask, other=0.0)
        b21 = tl.load(LOCAL_TRANSPORT + transport + 2, mask=key_mask, other=0.0)
        b22 = tl.load(LOCAL_TRANSPORT + transport + 3, mask=key_mask, other=0.0)
        db11 = f11 * df11 + f21 * df21
        db12 = f11 * df12 + f21 * df22
        db21 = f12 * df11 + f22 * df21
        db22 = f12 * df12 + f22 * df22
        tl.atomic_add(DLOCAL_TRANSPORT + transport + 0, db11, mask=key_mask)
        tl.atomic_add(DLOCAL_TRANSPORT + transport + 1, db12, mask=key_mask)
        tl.atomic_add(DLOCAL_TRANSPORT + transport + 2, db21, mask=key_mask)
        tl.atomic_add(DLOCAL_TRANSPORT + transport + 3, db22, mask=key_mask)
        next_df11 = df11 * b11 + df12 * b12 + direct11
        next_df12 = df11 * b21 + df12 * b22 + direct12
        next_df21 = df21 * b11 + df22 * b12 + direct21
        next_df22 = df21 * b21 + df22 * b22 + direct22
        df11, df12 = next_df11, next_df12
        df21, df22 = next_df21, next_df22


@triton.jit
def _general_block_relative_bwd_kernel(
    C1,
    C2,
    LOCAL_TAIL1,
    LOCAL_TAIL2,
    LOCAL_TRANSPORT,
    DPAIR_C1,
    DPAIR_C2,
    DC1,
    DC2,
    DLOCAL_TAIL1,
    DLOCAL_TAIL2,
    DLOCAL_TRANSPORT,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAIRS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    nk = tl.cdiv(K, BK)
    key_block = pid % nk
    pair_system = pid // nk
    pair = pair_system % PAIRS
    outer = pair_system // PAIRS
    if BLOCKS == 2:
        target = tl.where(pair == 0, 0, 1)
        source = tl.where(pair == 2, 1, 0)
    else:
        target = tl.where(
            pair < 1, 0, tl.where(pair < 3, 1, tl.where(pair < 6, 2, 3))
        )
        source = pair - target * (target + 1) // 2
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    rows = tl.arange(0, 16)
    local_token = outer * C + source * 16 + rows
    local_offsets = local_token[:, None] * K + keys[None, :]
    diagonal = target == source
    c1 = tl.load(C1 + local_offsets, mask=key_mask[None, :], other=0.0)
    c2 = tl.load(C2 + local_offsets, mask=key_mask[None, :], other=0.0)
    tail1 = tl.load(
        LOCAL_TAIL1 + local_offsets, mask=key_mask[None, :], other=0.0
    )
    tail2 = tl.load(
        LOCAL_TAIL2 + local_offsets, mask=key_mask[None, :], other=0.0
    )
    pair_token = (outer * PAIRS + pair) * 16 + rows
    pair_offsets = pair_token[:, None] * K + keys[None, :]
    grad1 = tl.load(
        DPAIR_C1 + pair_offsets, mask=key_mask[None, :], other=0.0
    ).to(tl.float32)
    grad2 = tl.load(
        DPAIR_C2 + pair_offsets, mask=key_mask[None, :], other=0.0
    ).to(tl.float32)
    for reverse in range(BLOCKS - 1):
        middle = BLOCKS - 1 - reverse
        value1 = tl.where(diagonal, c1, tail1).to(tl.float32)
        value2 = tl.where(diagonal, c2, tail2).to(tl.float32)
        for prefix_middle in range(1, BLOCKS):
            transport_prefix = (
                (outer * BLOCKS + prefix_middle) * K + keys
            ) * 4
            p11 = tl.load(
                LOCAL_TRANSPORT + transport_prefix + 0,
                mask=key_mask,
                other=0.0,
            )
            p12 = tl.load(
                LOCAL_TRANSPORT + transport_prefix + 1,
                mask=key_mask,
                other=0.0,
            )
            p21 = tl.load(
                LOCAL_TRANSPORT + transport_prefix + 2,
                mask=key_mask,
                other=0.0,
            )
            p22 = tl.load(
                LOCAL_TRANSPORT + transport_prefix + 3,
                mask=key_mask,
                other=0.0,
            )
            next1 = value1 * p11[None, :] + value2 * p21[None, :]
            next2 = value1 * p12[None, :] + value2 * p22[None, :]
            apply_prefix = (
                (~diagonal)
                & (prefix_middle > source)
                & (prefix_middle < target)
                & (prefix_middle < middle)
            )
            value1 = tl.where(apply_prefix, next1, value1)
            value2 = tl.where(apply_prefix, next2, value2)
        transport = ((outer * BLOCKS + middle) * K + keys) * 4
        f11 = tl.load(LOCAL_TRANSPORT + transport + 0, mask=key_mask, other=0.0)
        f12 = tl.load(LOCAL_TRANSPORT + transport + 1, mask=key_mask, other=0.0)
        f21 = tl.load(LOCAL_TRANSPORT + transport + 2, mask=key_mask, other=0.0)
        f22 = tl.load(LOCAL_TRANSPORT + transport + 3, mask=key_mask, other=0.0)
        apply = (~diagonal) & (middle > source) & (middle < target)
        df11 = tl.sum(value1 * grad1, axis=0)
        df12 = tl.sum(value1 * grad2, axis=0)
        df21 = tl.sum(value2 * grad1, axis=0)
        df22 = tl.sum(value2 * grad2, axis=0)
        tl.atomic_add(
            DLOCAL_TRANSPORT + transport + 0,
            tl.where(apply, df11, 0.0),
            mask=key_mask,
        )
        tl.atomic_add(
            DLOCAL_TRANSPORT + transport + 1,
            tl.where(apply, df12, 0.0),
            mask=key_mask,
        )
        tl.atomic_add(
            DLOCAL_TRANSPORT + transport + 2,
            tl.where(apply, df21, 0.0),
            mask=key_mask,
        )
        tl.atomic_add(
            DLOCAL_TRANSPORT + transport + 3,
            tl.where(apply, df22, 0.0),
            mask=key_mask,
        )
        previous1 = grad1 * f11[None, :] + grad2 * f12[None, :]
        previous2 = grad1 * f21[None, :] + grad2 * f22[None, :]
        grad1 = tl.where(apply, previous1, grad1)
        grad2 = tl.where(apply, previous2, grad2)
    tl.store(DC1 + local_offsets, grad1, mask=diagonal & key_mask[None, :])
    tl.store(DC2 + local_offsets, grad2, mask=diagonal & key_mask[None, :])
    tl.atomic_add(
        DLOCAL_TAIL1 + local_offsets,
        tl.where(diagonal, 0.0, grad1),
        mask=key_mask[None, :],
    )
    tl.atomic_add(
        DLOCAL_TAIL2 + local_offsets,
        tl.where(diagonal, 0.0, grad2),
        mask=key_mask[None, :],
    )


@triton.jit
def _general_block_tail_bwd_kernel(
    LOCAL_TAIL1,
    LOCAL_TAIL2,
    LOCAL_TRANSPORT,
    DFINAL_TAIL1,
    DFINAL_TAIL2,
    DLOCAL_TAIL1,
    DLOCAL_TAIL2,
    DLOCAL_TRANSPORT,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    nk = tl.cdiv(K, BK)
    key_block = pid % nk
    source_system = pid // nk
    source = source_system % BLOCKS
    outer = source_system // BLOCKS
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    rows = tl.arange(0, 16)
    token = outer * C + source * 16 + rows
    offsets = token[:, None] * K + keys[None, :]
    initial1 = tl.load(
        LOCAL_TAIL1 + offsets, mask=key_mask[None, :], other=0.0
    )
    initial2 = tl.load(
        LOCAL_TAIL2 + offsets, mask=key_mask[None, :], other=0.0
    )
    grad1 = tl.load(
        DFINAL_TAIL1 + offsets, mask=key_mask[None, :], other=0.0
    ).to(tl.float32)
    grad2 = tl.load(
        DFINAL_TAIL2 + offsets, mask=key_mask[None, :], other=0.0
    ).to(tl.float32)
    for reverse in range(BLOCKS - 1):
        middle = BLOCKS - 1 - reverse
        value1 = initial1.to(tl.float32)
        value2 = initial2.to(tl.float32)
        for prefix_middle in range(1, BLOCKS):
            transport_prefix = (
                (outer * BLOCKS + prefix_middle) * K + keys
            ) * 4
            p11 = tl.load(
                LOCAL_TRANSPORT + transport_prefix + 0,
                mask=key_mask,
                other=0.0,
            )
            p12 = tl.load(
                LOCAL_TRANSPORT + transport_prefix + 1,
                mask=key_mask,
                other=0.0,
            )
            p21 = tl.load(
                LOCAL_TRANSPORT + transport_prefix + 2,
                mask=key_mask,
                other=0.0,
            )
            p22 = tl.load(
                LOCAL_TRANSPORT + transport_prefix + 3,
                mask=key_mask,
                other=0.0,
            )
            next1 = value1 * p11[None, :] + value2 * p21[None, :]
            next2 = value1 * p12[None, :] + value2 * p22[None, :]
            apply_prefix = (prefix_middle > source) & (prefix_middle < middle)
            value1 = tl.where(apply_prefix, next1, value1)
            value2 = tl.where(apply_prefix, next2, value2)
        transport = ((outer * BLOCKS + middle) * K + keys) * 4
        f11 = tl.load(LOCAL_TRANSPORT + transport + 0, mask=key_mask, other=0.0)
        f12 = tl.load(LOCAL_TRANSPORT + transport + 1, mask=key_mask, other=0.0)
        f21 = tl.load(LOCAL_TRANSPORT + transport + 2, mask=key_mask, other=0.0)
        f22 = tl.load(LOCAL_TRANSPORT + transport + 3, mask=key_mask, other=0.0)
        apply = middle > source
        df11 = tl.sum(value1 * grad1, axis=0)
        df12 = tl.sum(value1 * grad2, axis=0)
        df21 = tl.sum(value2 * grad1, axis=0)
        df22 = tl.sum(value2 * grad2, axis=0)
        tl.atomic_add(
            DLOCAL_TRANSPORT + transport + 0,
            tl.where(apply, df11, 0.0),
            mask=key_mask,
        )
        tl.atomic_add(
            DLOCAL_TRANSPORT + transport + 1,
            tl.where(apply, df12, 0.0),
            mask=key_mask,
        )
        tl.atomic_add(
            DLOCAL_TRANSPORT + transport + 2,
            tl.where(apply, df21, 0.0),
            mask=key_mask,
        )
        tl.atomic_add(
            DLOCAL_TRANSPORT + transport + 3,
            tl.where(apply, df22, 0.0),
            mask=key_mask,
        )
        previous1 = grad1 * f11[None, :] + grad2 * f12[None, :]
        previous2 = grad1 * f21[None, :] + grad2 * f22[None, :]
        grad1 = tl.where(apply, previous1, grad1)
        grad2 = tl.where(apply, previous2, grad2)
    tl.atomic_add(DLOCAL_TAIL1 + offsets, grad1, mask=key_mask[None, :])
    tl.atomic_add(DLOCAL_TAIL2 + offsets, grad2, mask=key_mask[None, :])


class _GeneralBlockTransport(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        (
            *inputs,
            direct,
            readout_gate,
            chunk_size,
            block_key,
            block_warps,
            block_stages,
            backward_block_key,
            backward_block_warps,
            backward_block_stages,
        ) = args
        if readout_gate and not direct:
            raise ValueError("the readout-gate transport is the direct form")
        for label, key, warps, stages in (
            ("forward", block_key, block_warps, block_stages),
            (
                "backward",
                backward_block_key,
                backward_block_warps,
                backward_block_stages,
            ),
        ):
            if key not in (8, 16, 32, 64):
                raise ValueError(
                    f"transport {label} block_key must be 8, 16, 32, or 64"
                )
            if warps not in (1, 2, 4, 8):
                raise ValueError(
                    f"transport {label} block_warps must be 1, 2, 4, or 8"
                )
            if stages not in (1, 2, 3):
                raise ValueError(
                    f"transport {label} block_stages must be 1, 2, or 3"
                )
        inputs = tuple(tensor.contiguous() for tensor in inputs)
        (
            a1,
            a2,
            c1,
            c2,
            q1,
            q2,
            local_tail1,
            local_tail2,
            local_transport,
        ) = inputs
        batch, heads, local_chunks, _, key_dim = a1.shape
        blocks = chunk_size // 16
        chunks = local_chunks // blocks
        pairs = (
            blocks * (blocks - 1) // 2
            if direct
            else blocks * (blocks + 1) // 2
        )
        vector_shape = (batch, heads, chunks, chunk_size, key_dim)
        pair_shape = (batch, heads, chunks, pairs, 16, key_dim)
        # The centered path keeps local A/Q in FP16 for its extra mantissa.
        # Global Q is consumed only by BF16 TensorCore scan products, so store
        # it as BF16 here. This folds the conversion into the transport stores
        # instead of paying two sequence-wide casts or per-scan load casts.
        global_a = tuple(
            torch.empty(vector_shape, device=a1.device, dtype=a1.dtype)
            for _ in range(2)
        )
        global_q_dtype = (
            torch.bfloat16 if q1.dtype == torch.float16 else q1.dtype
        )
        global_q = tuple(
            torch.empty(vector_shape, device=a1.device, dtype=global_q_dtype)
            for _ in range(2)
        )
        global_vectors = (*global_a, *global_q)
        pair_vectors = tuple(
            torch.empty(
                pair_shape,
                device=a1.device,
                dtype=local_tail1.dtype if direct else c1.dtype,
            )
            for _ in range(2)
        )
        tails = tuple(
            torch.empty(vector_shape, device=a1.device, dtype=local_tail1.dtype)
            for _ in range(2)
        )
        final_transport = torch.empty(
            batch,
            heads,
            chunks,
            key_dim,
            4,
            device=a1.device,
            dtype=local_transport.dtype,
        )
        systems = batch * heads * chunks
        nk = triton.cdiv(key_dim, block_key)
        _general_block_prefix_fwd_kernel[(systems * nk,)](
            a1,
            a2,
            q1,
            q2,
            local_transport,
            *global_vectors,
            final_transport,
            K=key_dim,
            BLOCKS=blocks,
            C=chunk_size,
            BK=block_key,
            READOUT_GATE=readout_gate,
            num_warps=block_warps,
            num_stages=block_stages,
        )
        if direct:
            from .general_direct_transport_fwd_triton import (
                _general_direct_relative_tail_fwd_kernel,
            )

            _general_direct_relative_tail_fwd_kernel[(systems * blocks * nk,)](
                local_tail1,
                local_tail2,
                local_transport,
                *pair_vectors,
                *tails,
                K=key_dim,
                BLOCKS=blocks,
                PAIRS=pairs,
                C=chunk_size,
                BK=block_key,
                num_warps=block_warps,
                num_stages=block_stages,
            )
        else:
            _general_block_relative_fwd_kernel[(systems * pairs * nk,)](
                c1,
                c2,
                local_tail1,
                local_tail2,
                local_transport,
                *pair_vectors,
                K=key_dim,
                BLOCKS=blocks,
                PAIRS=pairs,
                C=chunk_size,
                DIRECT=False,
                BK=block_key,
                num_warps=block_warps,
                num_stages=block_stages,
            )
            _general_block_tail_fwd_kernel[(systems * blocks * nk,)](
                local_tail1,
                local_tail2,
                local_transport,
                *tails,
                K=key_dim,
                BLOCKS=blocks,
                C=chunk_size,
                BK=block_key,
                num_warps=block_warps,
                num_stages=block_stages,
            )
        ctx.save_for_backward(*inputs)
        ctx.dimensions = (
            systems,
            key_dim,
            blocks,
            pairs,
            chunk_size,
            triton.cdiv(key_dim, backward_block_key),
            backward_block_key,
            backward_block_warps,
            backward_block_stages,
        )
        ctx.output_shapes = (
            *(vector_shape for _ in range(4)),
            *(pair_shape for _ in range(2)),
            *(vector_shape for _ in range(2)),
            (batch, heads, chunks, key_dim, 4),
        )
        ctx.input_dtypes = tuple(tensor.dtype for tensor in inputs)
        ctx.direct = bool(direct)
        ctx.readout_gate = bool(readout_gate)
        # Under readout_gate the second global A vector is not written; the
        # readout path discards it.
        outputs = (*global_vectors, *pair_vectors, *tails, final_transport)
        ctx.output_dtypes = tuple(output.dtype for output in outputs)
        return outputs

    @staticmethod
    def backward(ctx, *output_grads):
        (
            a1,
            a2,
            c1,
            c2,
            q1,
            q2,
            local_tail1,
            local_tail2,
            local_transport,
        ) = ctx.saved_tensors
        (
            systems,
            key_dim,
            blocks,
            pairs,
            chunk_size,
            nk,
            block_key,
            block_warps,
            block_stages,
        ) = ctx.dimensions
        if ctx.readout_gate:
            # The kernel reads neither the discarded vector's adjoint nor A2,
            # and does not write DA2: pass placeholders, skip the zero fill.
            output_grads = list(output_grads)
            if output_grads[1] is None:
                output_grads[1] = output_grads[0]
        output_grads = tuple(
            grad.contiguous()
            if grad is not None
            else torch.zeros(shape, device=a1.device, dtype=dtype)
            for grad, shape, dtype in zip(
                output_grads, ctx.output_shapes, ctx.output_dtypes
            )
        )
        da1, dc1, dc2, dq1, dq2 = (
            torch.empty_like(tensor) for tensor in (a1, c1, c2, q1, q2)
        )
        da2 = da1 if ctx.readout_gate else torch.empty_like(a2)
        dtail1 = torch.empty_like(local_tail1)
        dtail2 = torch.empty_like(local_tail2)
        dtransport = torch.empty_like(local_transport)
        from .general_block_bwd_fused_triton import (
            _general_block_transport_bwd_fused_kernel,
        )

        # BLOCKS is at most four. One program can retain all transport
        # adjoints and give every output a unique writer, removing all three
        # atomic backward launches and their FP32 zero fills.
        _general_block_transport_bwd_fused_kernel[(systems, nk)](
            a1,
            a2,
            c1,
            c2,
            q1,
            q2,
            local_tail1,
            local_tail2,
            local_transport,
            *output_grads,
            da1,
            da2,
            dc1,
            dc2,
            dq1,
            dq2,
            dtail1,
            dtail2,
            dtransport,
            K=key_dim,
            BLOCKS=blocks,
            PAIRS=pairs,
            C=chunk_size,
            DIRECT=ctx.direct,
            BK=block_key,
            READOUT_GATE=ctx.readout_gate,
            num_warps=block_warps,
            num_stages=block_stages,
        )
        grads = (
            da1,
            da2,
            dc1,
            dc2,
            dq1,
            dq2,
            dtail1,
            dtail2,
            dtransport,
        )
        grads = [grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)]
        if ctx.readout_gate:
            grads[1] = None  # A2 is a placeholder for the constant zero
        return tuple(grads) + (None, None, None, None, None, None, None, None, None)


def _compose_general_blocks_triton(
    *args,
    chunk_size,
    direct=False,
    readout_gate=False,
    block_key=None,
    block_warps=None,
    block_stages=None,
    backward_block_key=None,
    backward_block_warps=None,
    backward_block_stages=None,
):
    supplied = (block_key, block_warps, block_stages)
    backward_supplied = (
        backward_block_key,
        backward_block_warps,
        backward_block_stages,
    )
    if all(value is None for value in supplied):
        forward_table, backward_table = _transport_configs(args[0])
        try:
            block_key, block_warps, block_stages = forward_table[chunk_size]
            (
                backward_block_key,
                backward_block_warps,
                backward_block_stages,
            ) = backward_table[chunk_size]
        except KeyError as error:
            raise ValueError("transport supports only C32 and C64") from error
        if any(value is not None for value in backward_supplied):
            raise ValueError("backward tuning also requires a forward config")
    elif any(value is None for value in supplied):
        raise ValueError("transport tuning requires key, warp, and stage together")
    elif all(value is None for value in backward_supplied):
        (
            backward_block_key,
            backward_block_warps,
            backward_block_stages,
        ) = supplied
    elif any(value is None for value in backward_supplied):
        raise ValueError(
            "backward transport tuning requires key, warp, and stage together"
        )
    return _GeneralBlockTransport.apply(
        *args,
        direct,
        readout_gate,
        chunk_size,
        block_key,
        block_warps,
        block_stages,
        backward_block_key,
        backward_block_warps,
        backward_block_stages,
    )


_INTRA_CONFIGS = [
    triton.Config({"BK": block}, num_warps=warps, num_stages=stages)
    for block, warps, stages in (
        (64, 1, 2),
        (32, 2, 2),
        (32, 2, 3),
        (64, 4, 2),
        (64, 4, 3),
        (128, 8, 2),
        (128, 8, 3),
    )
]

_INTRA_BACKWARD_AUTOTUNE_CONFIGS = [
    triton.Config({"BK": block}, num_warps=warps, num_stages=stages)
    for block, warps, stages in (
        (32, 4, 2),
        (32, 8, 3),
        (64, 8, 2),
        (128, 2, 3),
        (128, 4, 1),
        (128, 4, 2),
        (128, 4, 3),
        (128, 8, 1),
    )
]


@triton.autotune(configs=_INTRA_CONFIGS, key=["K", "BLOCKS"])
@triton.jit
def _general_block_intra_fwd_kernel(
    A1,
    A2,
    Q1,
    Q2,
    PAIR_C1,
    PAIR_C2,
    MATRIX,
    CAUSAL,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAIRS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = pid % PAIRS
    outer = pid // PAIRS
    if BLOCKS == 2:
        target = tl.where(pair == 0, 0, 1)
        source = tl.where(pair == 2, 1, 0)
    else:
        target = tl.where(
            pair < 1, 0, tl.where(pair < 3, 1, tl.where(pair < 6, 2, 3))
        )
        start = target * (target + 1) // 2
        source = pair - start
    rows = tl.arange(0, 16)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    local_target = outer * C + target * 16 + rows
    pair_source = outer * PAIRS * 16 + pair * 16 + rows
    matrix = tl.zeros((16, 16), dtype=tl.float32)
    causal = tl.zeros((16, 16), dtype=tl.float32)
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        target_offsets = local_target[:, None] * K + keys[None, :]
        source_offsets = pair_source[:, None] * K + keys[None, :]
        a1 = tl.load(A1 + target_offsets, mask=key_mask[None, :], other=0.0)
        a2 = tl.load(A2 + target_offsets, mask=key_mask[None, :], other=0.0)
        q1 = tl.load(Q1 + target_offsets, mask=key_mask[None, :], other=0.0)
        q2 = tl.load(Q2 + target_offsets, mask=key_mask[None, :], other=0.0)
        c1 = tl.load(PAIR_C1 + source_offsets, mask=key_mask[None, :], other=0.0)
        c2 = tl.load(PAIR_C2 + source_offsets, mask=key_mask[None, :], other=0.0)
        matrix += safe_dot(a1, tl.trans(c1), allow_tf32=False)
        matrix += safe_dot(a2, tl.trans(c2), allow_tf32=False)
        causal += safe_dot(q1, tl.trans(c1), allow_tf32=False)
        causal += safe_dot(q2, tl.trans(c2), allow_tf32=False)
    diagonal = target == source
    matrix = tl.where((~diagonal) | (row_ids > col_ids), matrix, 0.0)
    causal = tl.where((~diagonal) | (row_ids >= col_ids), causal, 0.0)
    offsets = (
        outer * C * C
        + (target * 16 + row_ids) * C
        + source * 16
        + col_ids
    )
    tl.store(MATRIX + offsets, matrix)
    tl.store(CAUSAL + offsets, causal)
    if pair == 0:
        # The pair grid owns the lower block triangle. Its first program also
        # clears the small structurally zero upper triangle, avoiding two
        # full CxC zero-fill launches before every forward.
        for upper_target in range(BLOCKS):
            for upper_source in range(upper_target + 1, BLOCKS):
                upper_offsets = (
                    outer * C * C
                    + (upper_target * 16 + row_ids) * C
                    + upper_source * 16
                    + col_ids
                )
                tl.store(MATRIX + upper_offsets, 0.0)
                tl.store(CAUSAL + upper_offsets, 0.0)


@triton.autotune(
    configs=_INTRA_BACKWARD_AUTOTUNE_CONFIGS,
    key=["K", "BLOCKS", "TARGET"],
)
@triton.jit
def _general_block_intra_bwd_target_kernel(
    A1,
    A2,
    Q1,
    Q2,
    PAIR_C1,
    PAIR_C2,
    DMATRIX,
    DCAUSAL,
    DA1,
    DA2,
    DQ1,
    DQ2,
    DPAIR_C1,
    DPAIR_C2,
    K: tl.constexpr,
    TARGET: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAIRS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
):
    """Accumulate all source-block contributions for one target block.

    The target is a compile-time constant. This both keeps local reductions in
    registers and avoids executing masked GEMMs for source blocks above the
    diagonal (one wasted pair at C32 and six at C64).
    """
    outer = tl.program_id(0)
    rows = tl.arange(0, 16)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    local_target = outer * C + TARGET * 16 + rows

    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        target_offsets = local_target[:, None] * K + keys[None, :]
        a1 = tl.load(A1 + target_offsets, mask=key_mask[None, :], other=0.0)
        a2 = tl.load(A2 + target_offsets, mask=key_mask[None, :], other=0.0)
        q1 = tl.load(Q1 + target_offsets, mask=key_mask[None, :], other=0.0)
        q2 = tl.load(Q2 + target_offsets, mask=key_mask[None, :], other=0.0)
        da1 = tl.zeros((16, BK), dtype=tl.float32)
        da2 = tl.zeros((16, BK), dtype=tl.float32)
        dq1 = tl.zeros((16, BK), dtype=tl.float32)
        dq2 = tl.zeros((16, BK), dtype=tl.float32)

        for source in range(TARGET + 1):
            pair = TARGET * (TARGET + 1) // 2 + source
            pair_source = outer * PAIRS * 16 + pair * 16 + rows
            source_offsets = pair_source[:, None] * K + keys[None, :]
            c1 = tl.load(
                PAIR_C1 + source_offsets,
                mask=key_mask[None, :],
                other=0.0,
            )
            c2 = tl.load(
                PAIR_C2 + source_offsets,
                mask=key_mask[None, :],
                other=0.0,
            )
            matrix_offsets = (
                outer * C * C
                + (TARGET * 16 + row_ids) * C
                + source * 16
                + col_ids
            )
            grad_matrix = tl.load(DMATRIX + matrix_offsets).to(tl.float32)
            grad_causal = tl.load(DCAUSAL + matrix_offsets).to(tl.float32)
            off_diagonal = source < TARGET
            grad_matrix = tl.where(
                off_diagonal | (row_ids > col_ids), grad_matrix, 0.0
            )
            grad_causal = tl.where(
                off_diagonal | (row_ids >= col_ids), grad_causal, 0.0
            )
            grad_matrix = grad_matrix.to(tl.bfloat16)
            grad_causal = grad_causal.to(tl.bfloat16)

            da1 += safe_dot(grad_matrix, c1, allow_tf32=False)
            da2 += safe_dot(grad_matrix, c2, allow_tf32=False)
            dq1 += safe_dot(grad_causal, c1, allow_tf32=False)
            dq2 += safe_dot(grad_causal, c2, allow_tf32=False)
            dc1 = safe_dot(tl.trans(grad_matrix), a1, allow_tf32=False)
            dc1 += safe_dot(tl.trans(grad_causal), q1, allow_tf32=False)
            dc2 = safe_dot(tl.trans(grad_matrix), a2, allow_tf32=False)
            dc2 += safe_dot(tl.trans(grad_causal), q2, allow_tf32=False)
            tl.store(
                DPAIR_C1 + source_offsets,
                dc1,
                mask=key_mask[None, :],
            )
            tl.store(
                DPAIR_C2 + source_offsets,
                dc2,
                mask=key_mask[None, :],
            )

        tl.store(DA1 + target_offsets, da1, mask=key_mask[None, :])
        tl.store(DA2 + target_offsets, da2, mask=key_mask[None, :])
        tl.store(DQ1 + target_offsets, dq1, mask=key_mask[None, :])
        tl.store(DQ2 + target_offsets, dq2, mask=key_mask[None, :])


class _GeneralBlockedIntra(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a1, a2, q1, q2, pair_c1, pair_c2, chunk_size):
        batch, heads, local_chunks, _, key_dim = a1.shape
        blocks = chunk_size // _LOCAL_BLOCK
        chunks = local_chunks // blocks
        pairs = blocks * (blocks + 1) // 2
        tensors = tuple(
            tensor.contiguous()
            for tensor in (a1, a2, q1, q2, pair_c1, pair_c2)
        )
        a1, a2, q1, q2, pair_c1, pair_c2 = tensors
        shape = (batch, heads, chunks, chunk_size, chunk_size)
        matrix = torch.empty(*shape, device=a1.device, dtype=torch.float32)
        causal = torch.empty(*shape, device=a1.device, dtype=a1.dtype)
        systems = batch * heads * chunks
        _general_block_intra_fwd_kernel[(systems * pairs,)](
            a1,
            a2,
            q1,
            q2,
            pair_c1,
            pair_c2,
            matrix,
            causal,
            K=key_dim,
            BLOCKS=blocks,
            PAIRS=pairs,
            C=chunk_size,
        )
        ctx.save_for_backward(*tensors)
        ctx.dimensions = (systems, key_dim, blocks, pairs, chunk_size)
        ctx.input_dtypes = tuple(tensor.dtype for tensor in tensors)
        return matrix, causal

    @staticmethod
    def backward(ctx, dmatrix, dcausal):
        a1, a2, q1, q2, pair_c1, pair_c2 = ctx.saved_tensors
        systems, key_dim, blocks, pairs, chunk_size = ctx.dimensions
        local_grads = tuple(
            torch.empty_like(tensor)
            for tensor in (a1, a2, q1, q2)
        )
        pair_grads = tuple(
            torch.empty_like(tensor) for tensor in (pair_c1, pair_c2)
        )
        dmatrix = dmatrix.contiguous()
        dcausal = dcausal.contiguous()
        for target in range(blocks):
            launch = dict(
                K=key_dim,
                TARGET=target,
                BLOCKS=blocks,
                PAIRS=pairs,
                C=chunk_size,
            )
            _general_block_intra_bwd_target_kernel[(systems,)](
                a1,
                a2,
                q1,
                q2,
                pair_c1,
                pair_c2,
                dmatrix,
                dcausal,
                *local_grads,
                *pair_grads,
                **launch,
            )
        grads = (*local_grads, *pair_grads)
        return tuple(
            grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)
        ) + (None,)


@torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
def _blocked_write(v, w, chunk_size):
    batch, length, heads, value_dim = v.shape
    chunks = length // chunk_size
    return (v * w).transpose(1, 2).reshape(
        batch, heads, chunks, chunk_size, value_dim
    ).contiguous()


@torch.compiler.disable
def build_general_blocked_operands_triton(
    q, k, v, g, b, w, gamma, lam, scale, chunk_size
):
    if chunk_size not in (32, 64) or q.shape[1] % chunk_size:
        raise ValueError("blocked general builder requires a C32/C64 sequence")
    local = _GeneralCoefficients.apply(
        q,
        k,
        g,
        b,
        gamma,
        lam,
        scale,
        _LOCAL_BLOCK,
        _BLOCK_KEY,
        _BLOCK_WARPS,
    )
    (
        local_a1,
        local_a2,
        local_c1,
        local_c2,
        local_q1,
        local_q2,
        local_tail1,
        local_tail2,
        local_transport,
    ) = local
    compose_args = (
        local_a1,
        local_a2,
        local_c1,
        local_c2,
        local_q1,
        local_q2,
        local_tail1,
        local_tail2,
        local_transport,
    )
    # Keep this path bounded: a torch.compile C64 composition can spend
    # minutes compiling its generated backward (and appear hung) on the
    # first training step. The custom autograd has three fixed forward and
    # three fixed backward launches for both C32 and C64.
    transported = _compose_general_blocks_triton(
        *compose_args, chunk_size=chunk_size
    )
    (
        global_a1,
        global_a2,
        global_q1,
        global_q2,
        pair_c1,
        pair_c2,
        tail1,
        tail2,
        transport,
    ) = transported
    matrix, causal = _GeneralBlockedIntra.apply(
        local_a1,
        local_a2,
        local_q1,
        local_q2,
        pair_c1,
        pair_c2,
        chunk_size,
    )
    write = _blocked_write(v, w, chunk_size)
    u0, state_weight1, state_weight2 = solve_general_operands(
        matrix, write, global_a1, global_a2
    )
    return (
        u0,
        state_weight1,
        state_weight2,
        global_q1,
        global_q2,
        tail1,
        tail2,
        causal,
        transport,
    )


__all__ = ["build_general_blocked_operands_triton"]

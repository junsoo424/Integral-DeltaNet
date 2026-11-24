"""Shared-source direct block-pair and final-tail forward transport."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _general_direct_relative_tail_fwd_kernel(
    LOCAL_TAIL1,
    LOCAL_TAIL2,
    LOCAL_TRANSPORT,
    PAIR_C1,
    PAIR_C2,
    FINAL_TAIL1,
    FINAL_TAIL2,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAIRS: tl.constexpr,
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

    # Pair (target, source) observes the value before target's transport;
    # applying that transport then advances the same registers to the next
    # target.  Thus every source chain is evaluated exactly once.
    for target in range(1, BLOCKS):
        active = source < target
        pair = target * (target - 1) // 2 + source
        pair_token = (outer * PAIRS + pair) * 16 + rows
        pair_offsets = pair_token[:, None] * K + keys[None, :]
        tl.store(
            PAIR_C1 + pair_offsets,
            value1,
            mask=active & key_mask[None, :],
        )
        tl.store(
            PAIR_C2 + pair_offsets,
            value2,
            mask=active & key_mask[None, :],
        )
        transport = ((outer * BLOCKS + target) * K + keys) * 4
        f11 = tl.load(
            LOCAL_TRANSPORT + transport + 0, mask=key_mask, other=0.0
        )
        f12 = tl.load(
            LOCAL_TRANSPORT + transport + 1, mask=key_mask, other=0.0
        )
        f21 = tl.load(
            LOCAL_TRANSPORT + transport + 2, mask=key_mask, other=0.0
        )
        f22 = tl.load(
            LOCAL_TRANSPORT + transport + 3, mask=key_mask, other=0.0
        )
        next1 = value1 * f11[None, :] + value2 * f21[None, :]
        next2 = value1 * f12[None, :] + value2 * f22[None, :]
        value1 = tl.where(active, next1, value1)
        value2 = tl.where(active, next2, value2)

    tl.store(FINAL_TAIL1 + offsets, value1, mask=key_mask[None, :])
    tl.store(FINAL_TAIL2 + offsets, value2, mask=key_mask[None, :])


__all__ = ["_general_direct_relative_tail_fwd_kernel"]

"""Deterministic fused backward for C32/C64 channel-wise block transport."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _general_block_transport_bwd_fused_kernel(
    A1,
    A2,
    C1,
    C2,
    Q1,
    Q2,
    LOCAL_TAIL1,
    LOCAL_TAIL2,
    LOCAL_TRANSPORT,
    DGLOBAL_A1,
    DGLOBAL_A2,
    DGLOBAL_Q1,
    DGLOBAL_Q2,
    DPAIR_C1,
    DPAIR_C2,
    DFINAL_TAIL1,
    DFINAL_TAIL2,
    DFINAL_TRANSPORT,
    DA1,
    DA2,
    DC1,
    DC2,
    DQ1,
    DQ2,
    DLOCAL_TAIL1,
    DLOCAL_TAIL2,
    DLOCAL_TRANSPORT,
    K: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAIRS: tl.constexpr,
    C: tl.constexpr,
    DIRECT: tl.constexpr,
    BK: tl.constexpr,
    READOUT_GATE: tl.constexpr = False,
):
    """Differentiate every transport consumer with one unique writer.

    C32/C64 contain only two/four C16 blocks. Keeping their transport
    adjoints in registers avoids three launches, zero-filled FP32 buffers,
    and all cross-kernel atomic additions.

    READOUT_GATE: A2 is the readout path's constant zero and GLOBAL_A2 is
    discarded, so its adjoint is zero too.  Every term with A2 or
    DGLOBAL_A2 adds an exact zero; they are dropped, A2 / DGLOBAL_A2 are not
    read and DA2 (not needed) is not written.
    """
    outer = tl.program_id(0)
    key_block = tl.program_id(1)
    rows = tl.arange(0, 16)
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K
    vector_mask = key_mask[None, :]
    block_axis = tl.arange(0, BLOCKS)[:, None]

    dtransport11 = tl.zeros((BLOCKS, BK), dtype=tl.float32)
    dtransport12 = tl.zeros((BLOCKS, BK), dtype=tl.float32)
    dtransport21 = tl.zeros((BLOCKS, BK), dtype=tl.float32)
    dtransport22 = tl.zeros((BLOCKS, BK), dtype=tl.float32)

    # Materialize the tiny absolute-prefix history in registers once rather
    # than rebuilding it for every block.
    history11 = tl.zeros((BLOCKS, BK), dtype=tl.float32)
    history12 = tl.zeros((BLOCKS, BK), dtype=tl.float32)
    history21 = tl.zeros((BLOCKS, BK), dtype=tl.float32)
    history22 = tl.zeros((BLOCKS, BK), dtype=tl.float32)
    f11 = tl.full((BK,), 1.0, tl.float32)
    f12 = tl.zeros((BK,), dtype=tl.float32)
    f21 = tl.zeros((BK,), dtype=tl.float32)
    f22 = tl.full((BK,), 1.0, tl.float32)
    for block in range(BLOCKS):
        block_mask = block_axis == block
        history11 = tl.where(block_mask, f11[None, :], history11)
        history12 = tl.where(block_mask, f12[None, :], history12)
        history21 = tl.where(block_mask, f21[None, :], history21)
        history22 = tl.where(block_mask, f22[None, :], history22)
        transport = ((outer * BLOCKS + block) * K + keys) * 4
        b11 = tl.load(
            LOCAL_TRANSPORT + transport + 0, mask=key_mask, other=0.0
        )
        b12 = tl.load(
            LOCAL_TRANSPORT + transport + 1, mask=key_mask, other=0.0
        )
        b21 = tl.load(
            LOCAL_TRANSPORT + transport + 2, mask=key_mask, other=0.0
        )
        b22 = tl.load(
            LOCAL_TRANSPORT + transport + 3, mask=key_mask, other=0.0
        )
        next11 = f11 * b11 + f12 * b21
        next12 = f11 * b12 + f12 * b22
        next21 = f21 * b11 + f22 * b21
        next22 = f21 * b12 + f22 * b22
        f11, f12, f21, f22 = next11, next12, next21, next22

    final = (outer * K + keys) * 4
    df11 = tl.load(
        DFINAL_TRANSPORT + final + 0, mask=key_mask, other=0.0
    ).to(tl.float32)
    df12 = tl.load(
        DFINAL_TRANSPORT + final + 1, mask=key_mask, other=0.0
    ).to(tl.float32)
    df21 = tl.load(
        DFINAL_TRANSPORT + final + 2, mask=key_mask, other=0.0
    ).to(tl.float32)
    df22 = tl.load(
        DFINAL_TRANSPORT + final + 3, mask=key_mask, other=0.0
    ).to(tl.float32)

    # Absolute prefixes: global A/Q vectors and final transport.
    for reverse in range(BLOCKS):
        block = BLOCKS - 1 - reverse
        block_mask = block_axis == block
        f11 = tl.sum(tl.where(block_mask, history11, 0.0), axis=0)
        f12 = tl.sum(tl.where(block_mask, history12, 0.0), axis=0)
        f21 = tl.sum(tl.where(block_mask, history21, 0.0), axis=0)
        f22 = tl.sum(tl.where(block_mask, history22, 0.0), axis=0)
        token = outer * C + block * 16 + rows
        offsets = token[:, None] * K + keys[None, :]
        a1 = tl.load(A1 + offsets, mask=vector_mask, other=0.0).to(tl.float32)
        q1 = tl.load(Q1 + offsets, mask=vector_mask, other=0.0).to(tl.float32)
        q2 = tl.load(Q2 + offsets, mask=vector_mask, other=0.0).to(tl.float32)
        da_out1 = tl.load(
            DGLOBAL_A1 + offsets, mask=vector_mask, other=0.0
        ).to(tl.float32)
        dq_out1 = tl.load(
            DGLOBAL_Q1 + offsets, mask=vector_mask, other=0.0
        ).to(tl.float32)
        dq_out2 = tl.load(
            DGLOBAL_Q2 + offsets, mask=vector_mask, other=0.0
        ).to(tl.float32)
        if READOUT_GATE:
            tl.store(DA1 + offsets, f11[None, :] * da_out1, mask=vector_mask)
        else:
            a2 = tl.load(A2 + offsets, mask=vector_mask, other=0.0).to(tl.float32)
            da_out2 = tl.load(
                DGLOBAL_A2 + offsets, mask=vector_mask, other=0.0
            ).to(tl.float32)
            tl.store(
                DA1 + offsets,
                f11[None, :] * da_out1 + f21[None, :] * da_out2,
                mask=vector_mask,
            )
            tl.store(
                DA2 + offsets,
                f12[None, :] * da_out1 + f22[None, :] * da_out2,
                mask=vector_mask,
            )
        tl.store(
            DQ1 + offsets,
            f11[None, :] * dq_out1 + f21[None, :] * dq_out2,
            mask=vector_mask,
        )
        tl.store(
            DQ2 + offsets,
            f12[None, :] * dq_out1 + f22[None, :] * dq_out2,
            mask=vector_mask,
        )
        if READOUT_GATE:
            direct11 = tl.sum(da_out1 * a1 + dq_out1 * q1, axis=0)
            direct12 = tl.sum(dq_out1 * q2, axis=0)
            direct21 = tl.sum(dq_out2 * q1, axis=0)
            direct22 = tl.sum(dq_out2 * q2, axis=0)
        else:
            direct11 = tl.sum(da_out1 * a1 + dq_out1 * q1, axis=0)
            direct12 = tl.sum(da_out1 * a2 + dq_out1 * q2, axis=0)
            direct21 = tl.sum(da_out2 * a1 + dq_out2 * q1, axis=0)
            direct22 = tl.sum(da_out2 * a2 + dq_out2 * q2, axis=0)
        transport = ((outer * BLOCKS + block) * K + keys) * 4
        b11 = tl.load(
            LOCAL_TRANSPORT + transport + 0, mask=key_mask, other=0.0
        )
        b12 = tl.load(
            LOCAL_TRANSPORT + transport + 1, mask=key_mask, other=0.0
        )
        b21 = tl.load(
            LOCAL_TRANSPORT + transport + 2, mask=key_mask, other=0.0
        )
        b22 = tl.load(
            LOCAL_TRANSPORT + transport + 3, mask=key_mask, other=0.0
        )
        dtransport11 += tl.where(
            block_mask, (f11 * df11 + f21 * df21)[None, :], 0.0
        )
        dtransport12 += tl.where(
            block_mask, (f11 * df12 + f21 * df22)[None, :], 0.0
        )
        dtransport21 += tl.where(
            block_mask, (f12 * df11 + f22 * df21)[None, :], 0.0
        )
        dtransport22 += tl.where(
            block_mask, (f12 * df12 + f22 * df22)[None, :], 0.0
        )
        next_df11 = df11 * b11 + df12 * b12 + direct11
        next_df12 = df11 * b21 + df12 * b22 + direct12
        next_df21 = df21 * b11 + df22 * b12 + direct21
        next_df22 = df21 * b21 + df22 * b22 + direct22
        df11, df12, df21, df22 = (
            next_df11,
            next_df12,
            next_df21,
            next_df22,
        )

    # Relative pair chains and final tails, grouped by source block so each
    # local C16 tail has one register accumulator and one final store.
    for source in range(BLOCKS):
        token = outer * C + source * 16 + rows
        offsets = token[:, None] * K + keys[None, :]
        initial1 = tl.load(
            LOCAL_TAIL1 + offsets, mask=vector_mask, other=0.0
        ).to(tl.float32)
        initial2 = tl.load(
            LOCAL_TAIL2 + offsets, mask=vector_mask, other=0.0
        ).to(tl.float32)
        dtail1 = tl.zeros((16, BK), dtype=tl.float32)
        dtail2 = tl.zeros((16, BK), dtype=tl.float32)

        if not DIRECT:
            diagonal_pair = source * (source + 1) // 2 + source
            pair_token = (outer * PAIRS + diagonal_pair) * 16 + rows
            pair_offsets = pair_token[:, None] * K + keys[None, :]
            dc1 = tl.load(
                DPAIR_C1 + pair_offsets, mask=vector_mask, other=0.0
            )
            dc2 = tl.load(
                DPAIR_C2 + pair_offsets, mask=vector_mask, other=0.0
            )
            tl.store(DC1 + offsets, dc1, mask=vector_mask)
            tl.store(DC2 + offsets, dc2, mask=vector_mask)

        for target in range(source + 1, BLOCKS):
            if DIRECT:
                pair = target * (target - 1) // 2 + source
            else:
                pair = target * (target + 1) // 2 + source
            pair_token = (outer * PAIRS + pair) * 16 + rows
            pair_offsets = pair_token[:, None] * K + keys[None, :]
            grad1 = tl.load(
                DPAIR_C1 + pair_offsets, mask=vector_mask, other=0.0
            ).to(tl.float32)
            grad2 = tl.load(
                DPAIR_C2 + pair_offsets, mask=vector_mask, other=0.0
            ).to(tl.float32)
            for reverse_middle in range(target - source - 1):
                middle = target - 1 - reverse_middle
                value1 = initial1
                value2 = initial2
                for prefix in range(source + 1, middle):
                    prefix_offsets = (
                        (outer * BLOCKS + prefix) * K + keys
                    ) * 4
                    p11 = tl.load(
                        LOCAL_TRANSPORT + prefix_offsets + 0,
                        mask=key_mask,
                        other=0.0,
                    )
                    p12 = tl.load(
                        LOCAL_TRANSPORT + prefix_offsets + 1,
                        mask=key_mask,
                        other=0.0,
                    )
                    p21 = tl.load(
                        LOCAL_TRANSPORT + prefix_offsets + 2,
                        mask=key_mask,
                        other=0.0,
                    )
                    p22 = tl.load(
                        LOCAL_TRANSPORT + prefix_offsets + 3,
                        mask=key_mask,
                        other=0.0,
                    )
                    next1 = value1 * p11[None, :] + value2 * p21[None, :]
                    next2 = value1 * p12[None, :] + value2 * p22[None, :]
                    value1, value2 = next1, next2
                middle_offsets = (
                    (outer * BLOCKS + middle) * K + keys
                ) * 4
                m11 = tl.load(
                    LOCAL_TRANSPORT + middle_offsets + 0,
                    mask=key_mask,
                    other=0.0,
                )
                m12 = tl.load(
                    LOCAL_TRANSPORT + middle_offsets + 1,
                    mask=key_mask,
                    other=0.0,
                )
                m21 = tl.load(
                    LOCAL_TRANSPORT + middle_offsets + 2,
                    mask=key_mask,
                    other=0.0,
                )
                m22 = tl.load(
                    LOCAL_TRANSPORT + middle_offsets + 3,
                    mask=key_mask,
                    other=0.0,
                )
                middle_mask = block_axis == middle
                dtransport11 += tl.where(
                    middle_mask,
                    tl.sum(value1 * grad1, axis=0)[None, :],
                    0.0,
                )
                dtransport12 += tl.where(
                    middle_mask,
                    tl.sum(value1 * grad2, axis=0)[None, :],
                    0.0,
                )
                dtransport21 += tl.where(
                    middle_mask,
                    tl.sum(value2 * grad1, axis=0)[None, :],
                    0.0,
                )
                dtransport22 += tl.where(
                    middle_mask,
                    tl.sum(value2 * grad2, axis=0)[None, :],
                    0.0,
                )
                previous1 = grad1 * m11[None, :] + grad2 * m12[None, :]
                previous2 = grad1 * m21[None, :] + grad2 * m22[None, :]
                grad1, grad2 = previous1, previous2
            dtail1 += grad1
            dtail2 += grad2

        grad1 = tl.load(
            DFINAL_TAIL1 + offsets, mask=vector_mask, other=0.0
        ).to(tl.float32)
        grad2 = tl.load(
            DFINAL_TAIL2 + offsets, mask=vector_mask, other=0.0
        ).to(tl.float32)
        for reverse_middle in range(BLOCKS - source - 1):
            middle = BLOCKS - 1 - reverse_middle
            value1 = initial1
            value2 = initial2
            for prefix in range(source + 1, middle):
                prefix_offsets = (
                    (outer * BLOCKS + prefix) * K + keys
                ) * 4
                p11 = tl.load(
                    LOCAL_TRANSPORT + prefix_offsets + 0,
                    mask=key_mask,
                    other=0.0,
                )
                p12 = tl.load(
                    LOCAL_TRANSPORT + prefix_offsets + 1,
                    mask=key_mask,
                    other=0.0,
                )
                p21 = tl.load(
                    LOCAL_TRANSPORT + prefix_offsets + 2,
                    mask=key_mask,
                    other=0.0,
                )
                p22 = tl.load(
                    LOCAL_TRANSPORT + prefix_offsets + 3,
                    mask=key_mask,
                    other=0.0,
                )
                next1 = value1 * p11[None, :] + value2 * p21[None, :]
                next2 = value1 * p12[None, :] + value2 * p22[None, :]
                value1, value2 = next1, next2
            middle_offsets = (
                (outer * BLOCKS + middle) * K + keys
            ) * 4
            m11 = tl.load(
                LOCAL_TRANSPORT + middle_offsets + 0,
                mask=key_mask,
                other=0.0,
            )
            m12 = tl.load(
                LOCAL_TRANSPORT + middle_offsets + 1,
                mask=key_mask,
                other=0.0,
            )
            m21 = tl.load(
                LOCAL_TRANSPORT + middle_offsets + 2,
                mask=key_mask,
                other=0.0,
            )
            m22 = tl.load(
                LOCAL_TRANSPORT + middle_offsets + 3,
                mask=key_mask,
                other=0.0,
            )
            middle_mask = block_axis == middle
            dtransport11 += tl.where(
                middle_mask,
                tl.sum(value1 * grad1, axis=0)[None, :],
                0.0,
            )
            dtransport12 += tl.where(
                middle_mask,
                tl.sum(value1 * grad2, axis=0)[None, :],
                0.0,
            )
            dtransport21 += tl.where(
                middle_mask,
                tl.sum(value2 * grad1, axis=0)[None, :],
                0.0,
            )
            dtransport22 += tl.where(
                middle_mask,
                tl.sum(value2 * grad2, axis=0)[None, :],
                0.0,
            )
            previous1 = grad1 * m11[None, :] + grad2 * m12[None, :]
            previous2 = grad1 * m21[None, :] + grad2 * m22[None, :]
            grad1, grad2 = previous1, previous2
        dtail1 += grad1
        dtail2 += grad2
        tl.store(DLOCAL_TAIL1 + offsets, dtail1, mask=vector_mask)
        tl.store(DLOCAL_TAIL2 + offsets, dtail2, mask=vector_mask)

    for block in range(BLOCKS):
        block_mask = block_axis == block
        transport = ((outer * BLOCKS + block) * K + keys) * 4
        tl.store(
            DLOCAL_TRANSPORT + transport + 0,
            tl.sum(tl.where(block_mask, dtransport11, 0.0), axis=0),
            mask=key_mask,
        )
        tl.store(
            DLOCAL_TRANSPORT + transport + 1,
            tl.sum(tl.where(block_mask, dtransport12, 0.0), axis=0),
            mask=key_mask,
        )
        tl.store(
            DLOCAL_TRANSPORT + transport + 2,
            tl.sum(tl.where(block_mask, dtransport21, 0.0), axis=0),
            mask=key_mask,
        )
        tl.store(
            DLOCAL_TRANSPORT + transport + 3,
            tl.sum(tl.where(block_mask, dtransport22, 0.0), axis=0),
            mask=key_mask,
        )


__all__ = ["_general_block_transport_bwd_fused_kernel"]

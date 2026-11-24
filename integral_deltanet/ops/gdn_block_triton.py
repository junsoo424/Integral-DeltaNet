"""Stable BC32 operand construction for an outer C64 Integral GDN chunk.

The local coefficient recurrence is delegated to the numerically stable C32
kernel.  Two local transports are then joined at block granularity, while the
three lower-triangular QK/KK tiles are formed directly from sequence-layout BF16
inputs.  This mirrors FLA's large-chunk organization without changing the
outer C64 recurrence seen by the boundary scan.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import safe_dot


_BLOCK = 32
_CHUNK = 64
_BLOCKS = 2
_PAIRS = 3


def _matmul_2x2(left, right):
    a11, a12, a21, a22 = left
    b11, b12, b21, b22 = right
    return (
        a11 * b11 + a12 * b21,
        a11 * b12 + a12 * b22,
        a21 * b11 + a22 * b21,
        a21 * b12 + a22 * b22,
    )


def _row_transport(c1, c2, transport):
    t11, t12, t21, t22 = transport
    return (
        c1 * t11.unsqueeze(-1) + c2 * t21.unsqueeze(-1),
        c1 * t12.unsqueeze(-1) + c2 * t22.unsqueeze(-1),
    )


def _gdn_block_transport(
    pi1,
    pi2,
    sigma1,
    sigma2,
    eta1,
    eta2,
    tau1,
    tau2,
    transport,
):
    """Join two stable C32 coordinate systems into one C64 operand set."""
    batch, heads, subchunks, block = pi1.shape
    if block != _BLOCK or subchunks % _BLOCKS:
        raise ValueError("blocked GDN transport requires two C32 blocks")
    chunks = subchunks // _BLOCKS

    def local(tensor):
        return tensor.reshape(batch, heads, chunks, _BLOCKS, _BLOCK)

    pi1, pi2, sigma1, sigma2, eta1, eta2, tau1, tau2 = (
        local(tensor)
        for tensor in (pi1, pi2, sigma1, sigma2, eta1, eta2, tau1, tau2)
    )
    block_transports = transport.reshape(
        batch, heads, chunks, _BLOCKS, 4
    ).unbind(dim=-2)
    block_transports = tuple(tuple(t.unbind(dim=-1)) for t in block_transports)
    identity = (
        torch.ones_like(block_transports[0][0]),
        torch.zeros_like(block_transports[0][0]),
        torch.zeros_like(block_transports[0][0]),
        torch.ones_like(block_transports[0][0]),
    )
    before = []
    running = identity
    for block_transport in block_transports:
        before.append(running)
        running = _matmul_2x2(running, block_transport)
    final_transport = running

    global_pi1 = []
    global_pi2 = []
    global_eta1 = []
    global_eta2 = []
    for block_index, prefix in enumerate(before):
        p11, p12, p21, p22 = prefix
        global_pi1.append(
            p11.unsqueeze(-1) * pi1[..., block_index, :]
            + p12.unsqueeze(-1) * pi2[..., block_index, :]
        )
        global_pi2.append(
            p21.unsqueeze(-1) * pi1[..., block_index, :]
            + p22.unsqueeze(-1) * pi2[..., block_index, :]
        )
        global_eta1.append(
            p11.unsqueeze(-1) * eta1[..., block_index, :]
            + p12.unsqueeze(-1) * eta2[..., block_index, :]
        )
        global_eta2.append(
            p21.unsqueeze(-1) * eta1[..., block_index, :]
            + p22.unsqueeze(-1) * eta2[..., block_index, :]
        )

    # Triangular pair order: (0,0), (1,0), (1,1).
    pair_c1 = []
    pair_c2 = []
    for target in range(_BLOCKS):
        for source in range(target + 1):
            if source == target:
                c1 = sigma1[..., source, :]
                c2 = sigma2[..., source, :]
            else:
                c1 = tau1[..., source, :]
                c2 = tau2[..., source, :]
                for middle in range(source + 1, target):
                    c1, c2 = _row_transport(
                        c1, c2, block_transports[middle]
                    )
            pair_c1.append(c1)
            pair_c2.append(c2)

    tail1 = []
    tail2 = []
    for source in range(_BLOCKS):
        c1 = tau1[..., source, :]
        c2 = tau2[..., source, :]
        for middle in range(source + 1, _BLOCKS):
            c1, c2 = _row_transport(c1, c2, block_transports[middle])
        tail1.append(c1)
        tail2.append(c2)

    def flatten_blocks(tensors):
        return torch.stack(tensors, dim=-2).reshape(
            batch, heads, chunks, _CHUNK
        )

    return (
        flatten_blocks(global_pi1),
        flatten_blocks(global_pi2),
        flatten_blocks(global_eta1),
        flatten_blocks(global_eta2),
        torch.stack(pair_c1, dim=-2),
        torch.stack(pair_c2, dim=-2),
        flatten_blocks(tail1),
        flatten_blocks(tail2),
        torch.stack(final_transport, dim=-1),
    )


@triton.jit
def _gdn_block_transport_fwd_kernel(
    PI1,
    PI2,
    SIGMA1,
    SIGMA2,
    ETA1,
    ETA2,
    TAU1,
    TAU2,
    LOCAL_TRANSPORT,
    GLOBAL_PI1,
    GLOBAL_PI2,
    GLOBAL_ETA1,
    GLOBAL_ETA2,
    PAIR_C1,
    PAIR_C2,
    TAIL1,
    TAIL2,
    FINAL_TRANSPORT,
):
    """Fuse the complete two-C32 coordinate join into one launch."""
    outer = tl.program_id(0)
    rows = tl.arange(0, 32)
    block0 = outer * 64 + rows
    block1 = block0 + 32
    transport0 = outer * 8
    transport1 = transport0 + 4
    a11 = tl.load(LOCAL_TRANSPORT + transport0 + 0).to(tl.float32)
    a12 = tl.load(LOCAL_TRANSPORT + transport0 + 1).to(tl.float32)
    a21 = tl.load(LOCAL_TRANSPORT + transport0 + 2).to(tl.float32)
    a22 = tl.load(LOCAL_TRANSPORT + transport0 + 3).to(tl.float32)
    b11 = tl.load(LOCAL_TRANSPORT + transport1 + 0).to(tl.float32)
    b12 = tl.load(LOCAL_TRANSPORT + transport1 + 1).to(tl.float32)
    b21 = tl.load(LOCAL_TRANSPORT + transport1 + 2).to(tl.float32)
    b22 = tl.load(LOCAL_TRANSPORT + transport1 + 3).to(tl.float32)

    pi10 = tl.load(PI1 + block0).to(tl.float32)
    pi20 = tl.load(PI2 + block0).to(tl.float32)
    pi11 = tl.load(PI1 + block1).to(tl.float32)
    pi21 = tl.load(PI2 + block1).to(tl.float32)
    eta10 = tl.load(ETA1 + block0).to(tl.float32)
    eta20 = tl.load(ETA2 + block0).to(tl.float32)
    eta11 = tl.load(ETA1 + block1).to(tl.float32)
    eta21 = tl.load(ETA2 + block1).to(tl.float32)
    tau10 = tl.load(TAU1 + block0).to(tl.float32)
    tau20 = tl.load(TAU2 + block0).to(tl.float32)
    tau11 = tl.load(TAU1 + block1).to(tl.float32)
    tau21 = tl.load(TAU2 + block1).to(tl.float32)

    tl.store(GLOBAL_PI1 + block0, pi10)
    tl.store(GLOBAL_PI2 + block0, pi20)
    tl.store(GLOBAL_ETA1 + block0, eta10)
    tl.store(GLOBAL_ETA2 + block0, eta20)
    tl.store(GLOBAL_PI1 + block1, a11 * pi11 + a12 * pi21)
    tl.store(GLOBAL_PI2 + block1, a21 * pi11 + a22 * pi21)
    tl.store(GLOBAL_ETA1 + block1, a11 * eta11 + a12 * eta21)
    tl.store(GLOBAL_ETA2 + block1, a21 * eta11 + a22 * eta21)

    pair_base = outer * 3 * 32
    tl.store(PAIR_C1 + pair_base + rows, tl.load(SIGMA1 + block0))
    tl.store(PAIR_C2 + pair_base + rows, tl.load(SIGMA2 + block0))
    tl.store(PAIR_C1 + pair_base + 32 + rows, tau10)
    tl.store(PAIR_C2 + pair_base + 32 + rows, tau20)
    tl.store(PAIR_C1 + pair_base + 64 + rows, tl.load(SIGMA1 + block1))
    tl.store(PAIR_C2 + pair_base + 64 + rows, tl.load(SIGMA2 + block1))

    tl.store(TAIL1 + block0, tau10 * b11 + tau20 * b21)
    tl.store(TAIL2 + block0, tau10 * b12 + tau20 * b22)
    tl.store(TAIL1 + block1, tau11)
    tl.store(TAIL2 + block1, tau21)
    final = outer * 4
    tl.store(FINAL_TRANSPORT + final + 0, a11 * b11 + a12 * b21)
    tl.store(FINAL_TRANSPORT + final + 1, a11 * b12 + a12 * b22)
    tl.store(FINAL_TRANSPORT + final + 2, a21 * b11 + a22 * b21)
    tl.store(FINAL_TRANSPORT + final + 3, a21 * b12 + a22 * b22)


@triton.jit
def _gdn_block_transport_bwd_kernel(
    PI1,
    PI2,
    ETA1,
    ETA2,
    TAU1,
    TAU2,
    LOCAL_TRANSPORT,
    DGLOBAL_PI1,
    DGLOBAL_PI2,
    DGLOBAL_ETA1,
    DGLOBAL_ETA2,
    DPAIR_C1,
    DPAIR_C2,
    DTAIL1,
    DTAIL2,
    DFINAL_TRANSPORT,
    DPI1,
    DPI2,
    DSIGMA1,
    DSIGMA2,
    DETA1,
    DETA2,
    DTAU1,
    DTAU2,
    DLOCAL_TRANSPORT,
):
    outer = tl.program_id(0)
    rows = tl.arange(0, 32)
    block0 = outer * 64 + rows
    block1 = block0 + 32
    transport0 = outer * 8
    transport1 = transport0 + 4
    a11 = tl.load(LOCAL_TRANSPORT + transport0 + 0).to(tl.float32)
    a12 = tl.load(LOCAL_TRANSPORT + transport0 + 1).to(tl.float32)
    a21 = tl.load(LOCAL_TRANSPORT + transport0 + 2).to(tl.float32)
    a22 = tl.load(LOCAL_TRANSPORT + transport0 + 3).to(tl.float32)
    b11 = tl.load(LOCAL_TRANSPORT + transport1 + 0).to(tl.float32)
    b12 = tl.load(LOCAL_TRANSPORT + transport1 + 1).to(tl.float32)
    b21 = tl.load(LOCAL_TRANSPORT + transport1 + 2).to(tl.float32)
    b22 = tl.load(LOCAL_TRANSPORT + transport1 + 3).to(tl.float32)

    dgpi10 = tl.load(DGLOBAL_PI1 + block0).to(tl.float32)
    dgpi20 = tl.load(DGLOBAL_PI2 + block0).to(tl.float32)
    dgpi11 = tl.load(DGLOBAL_PI1 + block1).to(tl.float32)
    dgpi21 = tl.load(DGLOBAL_PI2 + block1).to(tl.float32)
    dgeta10 = tl.load(DGLOBAL_ETA1 + block0).to(tl.float32)
    dgeta20 = tl.load(DGLOBAL_ETA2 + block0).to(tl.float32)
    dgeta11 = tl.load(DGLOBAL_ETA1 + block1).to(tl.float32)
    dgeta21 = tl.load(DGLOBAL_ETA2 + block1).to(tl.float32)
    pi11 = tl.load(PI1 + block1).to(tl.float32)
    pi21 = tl.load(PI2 + block1).to(tl.float32)
    eta11 = tl.load(ETA1 + block1).to(tl.float32)
    eta21 = tl.load(ETA2 + block1).to(tl.float32)

    tl.store(DPI1 + block0, dgpi10)
    tl.store(DPI2 + block0, dgpi20)
    tl.store(DETA1 + block0, dgeta10)
    tl.store(DETA2 + block0, dgeta20)
    tl.store(DPI1 + block1, dgpi11 * a11 + dgpi21 * a21)
    tl.store(DPI2 + block1, dgpi11 * a12 + dgpi21 * a22)
    tl.store(DETA1 + block1, dgeta11 * a11 + dgeta21 * a21)
    tl.store(DETA2 + block1, dgeta11 * a12 + dgeta21 * a22)

    pair_base = outer * 3 * 32
    tl.store(DSIGMA1 + block0, tl.load(DPAIR_C1 + pair_base + rows))
    tl.store(DSIGMA2 + block0, tl.load(DPAIR_C2 + pair_base + rows))
    tl.store(DSIGMA1 + block1, tl.load(DPAIR_C1 + pair_base + 64 + rows))
    tl.store(DSIGMA2 + block1, tl.load(DPAIR_C2 + pair_base + 64 + rows))

    dtail10 = tl.load(DTAIL1 + block0).to(tl.float32)
    dtail20 = tl.load(DTAIL2 + block0).to(tl.float32)
    tau10 = tl.load(TAU1 + block0).to(tl.float32)
    tau20 = tl.load(TAU2 + block0).to(tl.float32)
    dtau10 = tl.load(DPAIR_C1 + pair_base + 32 + rows).to(tl.float32)
    dtau20 = tl.load(DPAIR_C2 + pair_base + 32 + rows).to(tl.float32)
    dtau10 += dtail10 * b11 + dtail20 * b12
    dtau20 += dtail10 * b21 + dtail20 * b22
    tl.store(DTAU1 + block0, dtau10)
    tl.store(DTAU2 + block0, dtau20)
    tl.store(DTAU1 + block1, tl.load(DTAIL1 + block1))
    tl.store(DTAU2 + block1, tl.load(DTAIL2 + block1))

    final = outer * 4
    df11 = tl.load(DFINAL_TRANSPORT + final + 0).to(tl.float32)
    df12 = tl.load(DFINAL_TRANSPORT + final + 1).to(tl.float32)
    df21 = tl.load(DFINAL_TRANSPORT + final + 2).to(tl.float32)
    df22 = tl.load(DFINAL_TRANSPORT + final + 3).to(tl.float32)
    da11 = tl.sum(dgpi11 * pi11 + dgeta11 * eta11, axis=0)
    da12 = tl.sum(dgpi11 * pi21 + dgeta11 * eta21, axis=0)
    da21 = tl.sum(dgpi21 * pi11 + dgeta21 * eta11, axis=0)
    da22 = tl.sum(dgpi21 * pi21 + dgeta21 * eta21, axis=0)
    da11 += df11 * b11 + df12 * b12
    da12 += df11 * b21 + df12 * b22
    da21 += df21 * b11 + df22 * b12
    da22 += df21 * b21 + df22 * b22
    db11 = a11 * df11 + a21 * df21 + tl.sum(dtail10 * tau10, axis=0)
    db12 = a11 * df12 + a21 * df22 + tl.sum(dtail20 * tau10, axis=0)
    db21 = a12 * df11 + a22 * df21 + tl.sum(dtail10 * tau20, axis=0)
    db22 = a12 * df12 + a22 * df22 + tl.sum(dtail20 * tau20, axis=0)
    tl.store(DLOCAL_TRANSPORT + transport0 + 0, da11)
    tl.store(DLOCAL_TRANSPORT + transport0 + 1, da12)
    tl.store(DLOCAL_TRANSPORT + transport0 + 2, da21)
    tl.store(DLOCAL_TRANSPORT + transport0 + 3, da22)
    tl.store(DLOCAL_TRANSPORT + transport1 + 0, db11)
    tl.store(DLOCAL_TRANSPORT + transport1 + 1, db12)
    tl.store(DLOCAL_TRANSPORT + transport1 + 2, db21)
    tl.store(DLOCAL_TRANSPORT + transport1 + 3, db22)


class _GDNBlockTransport(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        pi1, pi2, sigma1, sigma2, eta1, eta2, tau1, tau2, transport = (
            tensor.contiguous() for tensor in inputs
        )
        batch, heads, subchunks, block = pi1.shape
        if block != 32 or subchunks % 2:
            raise ValueError("blocked GDN transport requires two C32 blocks")
        chunks = subchunks // 2
        vector_shape = (batch, heads, chunks, 64)
        global_vectors = tuple(
            torch.empty(vector_shape, device=pi1.device, dtype=pi1.dtype)
            for _ in range(4)
        )
        pairs = tuple(
            torch.empty(
                batch, heads, chunks, 3, 32,
                device=pi1.device, dtype=pi1.dtype,
            )
            for _ in range(2)
        )
        tails = tuple(
            torch.empty(vector_shape, device=pi1.device, dtype=pi1.dtype)
            for _ in range(2)
        )
        final_transport = torch.empty(
            batch, heads, chunks, 4,
            device=pi1.device, dtype=pi1.dtype,
        )
        _gdn_block_transport_fwd_kernel[(batch * heads * chunks,)](
            pi1, pi2, sigma1, sigma2, eta1, eta2, tau1, tau2, transport,
            *global_vectors, *pairs, *tails, final_transport,
            num_warps=1,
        )
        ctx.save_for_backward(
            pi1, pi2, sigma1, sigma2, eta1, eta2, tau1, tau2, transport
        )
        ctx.output_shapes = (
            *(vector_shape for _ in range(4)),
            *((batch, heads, chunks, 3, 32) for _ in range(2)),
            *(vector_shape for _ in range(2)),
            (batch, heads, chunks, 4),
        )
        return (*global_vectors, *pairs, *tails, final_transport)

    @staticmethod
    def backward(ctx, *output_grads):
        pi1, pi2, sigma1, sigma2, eta1, eta2, tau1, tau2, transport = (
            ctx.saved_tensors
        )
        output_grads = tuple(
            grad.contiguous()
            if grad is not None
            else torch.zeros(shape, device=pi1.device, dtype=pi1.dtype)
            for grad, shape in zip(output_grads, ctx.output_shapes)
        )
        input_grads = tuple(
            torch.empty_like(tensor)
            for tensor in (pi1, pi2, sigma1, sigma2, eta1, eta2, tau1, tau2, transport)
        )
        outer_chunks = output_grads[-1].numel() // 4
        _gdn_block_transport_bwd_kernel[(outer_chunks,)](
            pi1, pi2, eta1, eta2, tau1, tau2, transport,
            *output_grads,
            *input_grads,
            num_warps=1,
        )
        return input_grads


def _gdn_block_transport_fast(*args):
    if not args[0].is_cuda:
        return _gdn_block_transport(*args)
    return _GDNBlockTransport.apply(*args)


@triton.autotune(
    configs=[
        triton.Config({"BK": bk}, num_warps=warps, num_stages=stages)
        for bk, warps in ((32, 2), (64, 4), (128, 8))
        for stages in (2, 3)
    ],
    key=["K"],
)
@triton.jit
def _gdn_block_intra_fwd_kernel(
    Q,
    K_IN,
    LOCAL_PI1,
    LOCAL_PI2,
    LOCAL_ETA1,
    LOCAL_ETA2,
    PAIR_C1,
    PAIR_C2,
    TAIL1_COEFF,
    TAIL2_COEFF,
    MATRIX,
    CAUSAL,
    Q_OUT,
    TAIL1_OUT,
    TAIL2_OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = pid % 3
    outer = pid // 3
    target = tl.where(pair < 1, 0, 1)
    start = tl.where(target == 0, 0, 1)
    source = pair - start
    chunk = outer % N
    head = (outer // N) % H
    batch = outer // (N * H)
    rows = tl.arange(0, 32)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    local_target = outer * 64 + target * 32 + rows
    pair_source = outer * 3 * 32 + pair * 32 + rows
    pi1 = tl.load(LOCAL_PI1 + local_target).to(tl.float32)
    pi2 = tl.load(LOCAL_PI2 + local_target).to(tl.float32)
    eta1 = tl.load(LOCAL_ETA1 + local_target).to(tl.float32)
    eta2 = tl.load(LOCAL_ETA2 + local_target).to(tl.float32)
    c1 = tl.load(PAIR_C1 + pair_source).to(tl.float32)
    c2 = tl.load(PAIR_C2 + pair_source).to(tl.float32)
    gram_kk = tl.zeros((32, 32), dtype=tl.float32)
    gram_qk = tl.zeros((32, 32), dtype=tl.float32)
    target_token = chunk * 64 + target * 32
    source_token = chunk * 64 + source * 32
    target_base = ((batch * T + target_token) * H + head) * K
    source_base = ((batch * T + source_token) * H + head) * K
    chunk_base = outer * 64 * K + target * 32 * K
    tail1 = tl.load(TAIL1_COEFF + local_target).to(tl.float32)
    tail2 = tl.load(TAIL2_COEFF + local_target).to(tl.float32)
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        target_offsets = target_base + rows[:, None] * H * K + keys[None, :]
        source_offsets = source_base + rows[:, None] * H * K + keys[None, :]
        key_target = tl.load(K_IN + target_offsets, mask=key_mask[None, :], other=0.0)
        key_source = tl.load(K_IN + source_offsets, mask=key_mask[None, :], other=0.0)
        query = tl.load(Q + target_offsets, mask=key_mask[None, :], other=0.0)
        query = (query * SCALE).to(tl.bfloat16)
        gram_kk += safe_dot(key_target, tl.trans(key_source), allow_tf32=False)
        gram_qk += safe_dot(query, tl.trans(key_source), allow_tf32=False)
        diagonal = target == source
        output_offsets = chunk_base + rows[:, None] * K + keys[None, :]
        tl.store(Q_OUT + output_offsets, query, mask=diagonal & key_mask[None, :])
        tl.store(
            TAIL1_OUT + output_offsets,
            tail1[:, None] * key_target,
            mask=diagonal & key_mask[None, :],
        )
        tl.store(
            TAIL2_OUT + output_offsets,
            tail2[:, None] * key_target,
            mask=diagonal & key_mask[None, :],
        )
    interaction_coeff = pi1[:, None] * c1[None, :] + pi2[:, None] * c2[None, :]
    output_coeff = eta1[:, None] * c1[None, :] + eta2[:, None] * c2[None, :]
    diagonal = target == source
    matrix = interaction_coeff * gram_kk
    causal = output_coeff * gram_qk
    matrix = tl.where((~diagonal) | (row_ids > col_ids), matrix, 0.0)
    causal = tl.where((~diagonal) | (row_ids >= col_ids), causal, 0.0)
    matrix_offsets = (
        outer * 64 * 64
        + (target * 32 + row_ids) * 64
        + source * 32
        + col_ids
    )
    tl.store(MATRIX + matrix_offsets, matrix)
    tl.store(CAUSAL + matrix_offsets, causal)


@triton.jit
def _gdn_block_intra_bwd_kernel(
    Q,
    K_IN,
    LOCAL_PI1,
    LOCAL_PI2,
    LOCAL_ETA1,
    LOCAL_ETA2,
    PAIR_C1,
    PAIR_C2,
    TAIL1_COEFF,
    TAIL2_COEFF,
    DMATRIX,
    DCAUSAL,
    DQ_DIRECT,
    DTAIL1,
    DTAIL2,
    DQ,
    DK,
    DLOCAL_PI1,
    DLOCAL_PI2,
    DLOCAL_ETA1,
    DLOCAL_ETA2,
    DPAIR_C1,
    DPAIR_C2,
    DTAIL1_COEFF,
    DTAIL2_COEFF,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = pid % 3
    outer = pid // 3
    target = tl.where(pair < 1, 0, 1)
    start = tl.where(target == 0, 0, 1)
    source = pair - start
    diagonal = target == source
    chunk = outer % N
    head = (outer // N) % H
    batch = outer // (N * H)
    rows = tl.arange(0, 32)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    local_target = outer * 64 + target * 32 + rows
    pair_source = outer * 3 * 32 + pair * 32 + rows
    pi1 = tl.load(LOCAL_PI1 + local_target).to(tl.float32)
    pi2 = tl.load(LOCAL_PI2 + local_target).to(tl.float32)
    eta1 = tl.load(LOCAL_ETA1 + local_target).to(tl.float32)
    eta2 = tl.load(LOCAL_ETA2 + local_target).to(tl.float32)
    c1 = tl.load(PAIR_C1 + pair_source).to(tl.float32)
    c2 = tl.load(PAIR_C2 + pair_source).to(tl.float32)
    matrix_offsets = (
        outer * 64 * 64
        + (target * 32 + row_ids) * 64
        + source * 32
        + col_ids
    )
    grad_matrix = tl.load(DMATRIX + matrix_offsets).to(tl.float32)
    grad_causal = tl.load(DCAUSAL + matrix_offsets).to(tl.float32)
    grad_matrix = tl.where((~diagonal) | (row_ids > col_ids), grad_matrix, 0.0)
    grad_causal = tl.where((~diagonal) | (row_ids >= col_ids), grad_causal, 0.0)
    interaction_coeff = pi1[:, None] * c1[None, :] + pi2[:, None] * c2[None, :]
    output_coeff = eta1[:, None] * c1[None, :] + eta2[:, None] * c2[None, :]
    grad_gram_kk = (grad_matrix * interaction_coeff).to(tl.bfloat16)
    grad_gram_qk = (grad_causal * output_coeff).to(tl.bfloat16)
    gram_kk = tl.zeros((32, 32), dtype=tl.float32)
    gram_qk = tl.zeros((32, 32), dtype=tl.float32)
    grad_tail1_coeff = tl.zeros((32,), dtype=tl.float32)
    grad_tail2_coeff = tl.zeros((32,), dtype=tl.float32)
    target_token = chunk * 64 + target * 32
    source_token = chunk * 64 + source * 32
    target_base = ((batch * T + target_token) * H + head) * K
    source_base = ((batch * T + source_token) * H + head) * K
    chunk_base = outer * 64 * K + target * 32 * K
    tail1 = tl.load(TAIL1_COEFF + local_target).to(tl.float32)
    tail2 = tl.load(TAIL2_COEFF + local_target).to(tl.float32)
    for key_block in range(tl.cdiv(K, BK)):
        keys = key_block * BK + tl.arange(0, BK)
        key_mask = keys < K
        target_offsets = target_base + rows[:, None] * H * K + keys[None, :]
        source_offsets = source_base + rows[:, None] * H * K + keys[None, :]
        key_target = tl.load(K_IN + target_offsets, mask=key_mask[None, :], other=0.0)
        key_source = tl.load(K_IN + source_offsets, mask=key_mask[None, :], other=0.0)
        query = tl.load(Q + target_offsets, mask=key_mask[None, :], other=0.0)
        query_scaled = (query * SCALE).to(tl.bfloat16)
        gram_kk += safe_dot(key_target, tl.trans(key_source), allow_tf32=False)
        gram_qk += safe_dot(query_scaled, tl.trans(key_source), allow_tf32=False)
        grad_query = safe_dot(grad_gram_qk, key_source, allow_tf32=False) * SCALE
        grad_key_target = safe_dot(grad_gram_kk, key_source, allow_tf32=False)
        grad_key_source = safe_dot(tl.trans(grad_gram_kk), key_target, allow_tf32=False)
        grad_key_source += safe_dot(
            tl.trans(grad_gram_qk), query_scaled, allow_tf32=False
        )
        output_offsets = chunk_base + rows[:, None] * K + keys[None, :]
        grad_query_direct = tl.load(
            DQ_DIRECT + output_offsets,
            mask=diagonal & key_mask[None, :],
            other=0.0,
        )
        grad_tail1 = tl.load(
            DTAIL1 + output_offsets,
            mask=diagonal & key_mask[None, :],
            other=0.0,
        )
        grad_tail2 = tl.load(
            DTAIL2 + output_offsets,
            mask=diagonal & key_mask[None, :],
            other=0.0,
        )
        grad_query += grad_query_direct * SCALE
        grad_key_target += (
            tail1[:, None] * grad_tail1 + tail2[:, None] * grad_tail2
        )
        grad_tail1_coeff += tl.sum(
            grad_tail1.to(tl.float32) * key_target.to(tl.float32), axis=1
        )
        grad_tail2_coeff += tl.sum(
            grad_tail2.to(tl.float32) * key_target.to(tl.float32), axis=1
        )
        tl.atomic_add(
            DQ + target_offsets,
            grad_query,
            mask=key_mask[None, :],
        )
        tl.atomic_add(
            DK + target_offsets,
            grad_key_target,
            mask=key_mask[None, :],
        )
        tl.atomic_add(
            DK + source_offsets,
            grad_key_source,
            mask=key_mask[None, :],
        )

    grad_interaction = grad_matrix * gram_kk
    grad_output = grad_causal * gram_qk
    grad_pi1 = tl.sum(grad_interaction * c1[None, :], axis=1)
    grad_pi2 = tl.sum(grad_interaction * c2[None, :], axis=1)
    grad_eta1 = tl.sum(grad_output * c1[None, :], axis=1)
    grad_eta2 = tl.sum(grad_output * c2[None, :], axis=1)
    grad_c1 = tl.sum(
        grad_interaction * pi1[:, None] + grad_output * eta1[:, None],
        axis=0,
    )
    grad_c2 = tl.sum(
        grad_interaction * pi2[:, None] + grad_output * eta2[:, None],
        axis=0,
    )
    tl.atomic_add(DLOCAL_PI1 + local_target, grad_pi1)
    tl.atomic_add(DLOCAL_PI2 + local_target, grad_pi2)
    tl.atomic_add(DLOCAL_ETA1 + local_target, grad_eta1)
    tl.atomic_add(DLOCAL_ETA2 + local_target, grad_eta2)
    tl.store(DPAIR_C1 + pair_source, grad_c1)
    tl.store(DPAIR_C2 + pair_source, grad_c2)
    tl.store(DTAIL1_COEFF + local_target, grad_tail1_coeff, mask=diagonal)
    tl.store(DTAIL2_COEFF + local_target, grad_tail2_coeff, mask=diagonal)


@triton.autotune(
    configs=[
        triton.Config({"BK": 32}, num_warps=8, num_stages=2),
        triton.Config({"BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BK": 64}, num_warps=8, num_stages=2),
    ],
    key=["K"],
    # Every key tile contributes to these smaller vectors with atomics. Triton
    # runs each candidate repeatedly while autotuning, so the destinations
    # must be cleared before every trial; otherwise the first cold-cache
    # backward returns benchmark-count-multiplied gradients.
    reset_to_zero=[
        "DLOCAL_PI1",
        "DLOCAL_PI2",
        "DLOCAL_ETA1",
        "DLOCAL_ETA2",
        "DPAIR_C1",
        "DPAIR_C2",
        "DTAIL1_COEFF",
        "DTAIL2_COEFF",
    ],
)
@triton.jit
def _gdn_block_intra_bwd_chunk_kernel(
    Q,
    K_IN,
    LOCAL_PI1,
    LOCAL_PI2,
    LOCAL_ETA1,
    LOCAL_ETA2,
    PAIR_C1,
    PAIR_C2,
    TAIL1_COEFF,
    TAIL2_COEFF,
    DMATRIX,
    DCAUSAL,
    DQ_DIRECT,
    DTAIL1,
    DTAIL2,
    DQ,
    DK,
    DLOCAL_PI1,
    DLOCAL_PI2,
    DLOCAL_ETA1,
    DLOCAL_ETA2,
    DPAIR_C1,
    DPAIR_C2,
    DTAIL1_COEFF,
    DTAIL2_COEFF,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    BK: tl.constexpr,
):
    """Own one complete C64/key tile and eliminate Q/K atomics.

    A pair-parallel backward would issue nine large atomic reductions per
    outer chunk. C64 contains only three lower block pairs, so processing all
    three in one program lets Q and K gradients stay in registers and be
    written once. Only the much smaller coefficient vectors are reduced
    across key tiles.
    """
    pid = tl.program_id(0)
    nk = tl.cdiv(K, BK)
    key_block = pid % nk
    outer = pid // nk
    chunk = outer % N
    head = (outer // N) % H
    batch = outer // (N * H)
    rows = tl.arange(0, 32)
    row_ids = rows[:, None]
    col_ids = rows[None, :]
    keys = key_block * BK + tl.arange(0, BK)
    key_mask = keys < K

    token0 = chunk * 64
    token1 = token0 + 32
    base0 = ((batch * T + token0) * H + head) * K
    base1 = ((batch * T + token1) * H + head) * K
    offsets0 = base0 + rows[:, None] * H * K + keys[None, :]
    offsets1 = base1 + rows[:, None] * H * K + keys[None, :]
    k0 = tl.load(K_IN + offsets0, mask=key_mask[None, :], other=0.0)
    k1 = tl.load(K_IN + offsets1, mask=key_mask[None, :], other=0.0)
    q0 = tl.load(Q + offsets0, mask=key_mask[None, :], other=0.0)
    q1 = tl.load(Q + offsets1, mask=key_mask[None, :], other=0.0)
    qs0 = (q0 * SCALE).to(tl.bfloat16)
    qs1 = (q1 * SCALE).to(tl.bfloat16)
    dq0 = tl.zeros((32, BK), dtype=tl.float32)
    dq1 = tl.zeros((32, BK), dtype=tl.float32)
    dk0 = tl.zeros((32, BK), dtype=tl.float32)
    dk1 = tl.zeros((32, BK), dtype=tl.float32)

    # Direct query/tail outputs are target-owned and therefore race-free.
    output0 = outer * 64 * K + rows[:, None] * K + keys[None, :]
    output1 = output0 + 32 * K
    direct_q0 = tl.load(
        DQ_DIRECT + output0, mask=key_mask[None, :], other=0.0
    )
    direct_q1 = tl.load(
        DQ_DIRECT + output1, mask=key_mask[None, :], other=0.0
    )
    grad_tail10 = tl.load(
        DTAIL1 + output0, mask=key_mask[None, :], other=0.0
    )
    grad_tail20 = tl.load(
        DTAIL2 + output0, mask=key_mask[None, :], other=0.0
    )
    grad_tail11 = tl.load(
        DTAIL1 + output1, mask=key_mask[None, :], other=0.0
    )
    grad_tail21 = tl.load(
        DTAIL2 + output1, mask=key_mask[None, :], other=0.0
    )
    local0 = outer * 64 + rows
    local1 = local0 + 32
    tail10 = tl.load(TAIL1_COEFF + local0).to(tl.float32)
    tail20 = tl.load(TAIL2_COEFF + local0).to(tl.float32)
    tail11 = tl.load(TAIL1_COEFF + local1).to(tl.float32)
    tail21 = tl.load(TAIL2_COEFF + local1).to(tl.float32)
    dq0 += direct_q0 * SCALE
    dq1 += direct_q1 * SCALE
    dk0 += tail10[:, None] * grad_tail10 + tail20[:, None] * grad_tail20
    dk1 += tail11[:, None] * grad_tail11 + tail21[:, None] * grad_tail21
    tl.atomic_add(
        DTAIL1_COEFF + local0,
        tl.sum(grad_tail10.to(tl.float32) * k0.to(tl.float32), axis=1),
    )
    tl.atomic_add(
        DTAIL2_COEFF + local0,
        tl.sum(grad_tail20.to(tl.float32) * k0.to(tl.float32), axis=1),
    )
    tl.atomic_add(
        DTAIL1_COEFF + local1,
        tl.sum(grad_tail11.to(tl.float32) * k1.to(tl.float32), axis=1),
    )
    tl.atomic_add(
        DTAIL2_COEFF + local1,
        tl.sum(grad_tail21.to(tl.float32) * k1.to(tl.float32), axis=1),
    )

    for pair in range(3):
        if pair == 0:
            target = 0
            source = 0
            key_target = k0
            key_source = k0
            query_scaled = qs0
        elif pair == 1:
            target = 1
            source = 0
            key_target = k1
            key_source = k0
            query_scaled = qs1
        else:
            target = 1
            source = 1
            key_target = k1
            key_source = k1
            query_scaled = qs1

        local_target = outer * 64 + target * 32 + rows
        pair_source = outer * 3 * 32 + pair * 32 + rows
        pi1 = tl.load(LOCAL_PI1 + local_target).to(tl.float32)
        pi2 = tl.load(LOCAL_PI2 + local_target).to(tl.float32)
        eta1 = tl.load(LOCAL_ETA1 + local_target).to(tl.float32)
        eta2 = tl.load(LOCAL_ETA2 + local_target).to(tl.float32)
        c1 = tl.load(PAIR_C1 + pair_source).to(tl.float32)
        c2 = tl.load(PAIR_C2 + pair_source).to(tl.float32)
        matrix_offsets = (
            outer * 64 * 64
            + (target * 32 + row_ids) * 64
            + source * 32
            + col_ids
        )
        grad_matrix = tl.load(DMATRIX + matrix_offsets).to(tl.float32)
        grad_causal = tl.load(DCAUSAL + matrix_offsets).to(tl.float32)
        if target == source:
            grad_matrix = tl.where(
                row_ids > col_ids, grad_matrix, 0.0
            )
            grad_causal = tl.where(
                row_ids >= col_ids, grad_causal, 0.0
            )
        interaction_coeff = pi1[:, None] * c1[None, :]
        interaction_coeff += pi2[:, None] * c2[None, :]
        output_coeff = eta1[:, None] * c1[None, :]
        output_coeff += eta2[:, None] * c2[None, :]
        grad_gram_kk = (grad_matrix * interaction_coeff).to(tl.bfloat16)
        grad_gram_qk = (grad_causal * output_coeff).to(tl.bfloat16)

        grad_query = safe_dot(
            grad_gram_qk, key_source, allow_tf32=False
        ) * SCALE
        grad_key_target = safe_dot(
            grad_gram_kk, key_source, allow_tf32=False
        )
        grad_key_source = safe_dot(
            tl.trans(grad_gram_kk), key_target, allow_tf32=False
        )
        grad_key_source += safe_dot(
            tl.trans(grad_gram_qk), query_scaled, allow_tf32=False
        )
        if pair == 0:
            dq0 += grad_query
            dk0 += grad_key_target + grad_key_source
        elif pair == 1:
            dq1 += grad_query
            dk1 += grad_key_target
            dk0 += grad_key_source
        else:
            dq1 += grad_query
            dk1 += grad_key_target + grad_key_source

        gram_kk = safe_dot(key_target, tl.trans(key_source), allow_tf32=False)
        gram_qk = safe_dot(query_scaled, tl.trans(key_source), allow_tf32=False)
        grad_interaction = grad_matrix * gram_kk
        grad_output = grad_causal * gram_qk
        tl.atomic_add(
            DLOCAL_PI1 + local_target,
            tl.sum(grad_interaction * c1[None, :], axis=1),
        )
        tl.atomic_add(
            DLOCAL_PI2 + local_target,
            tl.sum(grad_interaction * c2[None, :], axis=1),
        )
        tl.atomic_add(
            DLOCAL_ETA1 + local_target,
            tl.sum(grad_output * c1[None, :], axis=1),
        )
        tl.atomic_add(
            DLOCAL_ETA2 + local_target,
            tl.sum(grad_output * c2[None, :], axis=1),
        )
        tl.atomic_add(
            DPAIR_C1 + pair_source,
            tl.sum(
                grad_interaction * pi1[:, None]
                + grad_output * eta1[:, None],
                axis=0,
            ),
        )
        tl.atomic_add(
            DPAIR_C2 + pair_source,
            tl.sum(
                grad_interaction * pi2[:, None]
                + grad_output * eta2[:, None],
                axis=0,
            ),
        )

    tl.store(DQ + offsets0, dq0, mask=key_mask[None, :])
    tl.store(DQ + offsets1, dq1, mask=key_mask[None, :])
    tl.store(DK + offsets0, dk0, mask=key_mask[None, :])
    tl.store(DK + offsets1, dk1, mask=key_mask[None, :])


class _GDNBlockedIntra(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        local_pi1,
        local_pi2,
        local_eta1,
        local_eta2,
        pair_c1,
        pair_c2,
        tail1_coeff,
        tail2_coeff,
        scale,
    ):
        batch, length, heads, key_dim = q.shape
        chunks = length // _CHUNK
        tensors = tuple(
            tensor.contiguous()
            for tensor in (
                q,
                k,
                local_pi1,
                local_pi2,
                local_eta1,
                local_eta2,
                pair_c1,
                pair_c2,
                tail1_coeff,
                tail2_coeff,
            )
        )
        (
            q,
            k,
            local_pi1,
            local_pi2,
            local_eta1,
            local_eta2,
            pair_c1,
            pair_c2,
            tail1_coeff,
            tail2_coeff,
        ) = tensors
        shape = (batch, heads, chunks, _CHUNK)
        matrix = torch.empty(*shape, _CHUNK, device=q.device, dtype=torch.float32)
        causal = torch.zeros(*shape, _CHUNK, device=q.device, dtype=q.dtype)
        q_out = torch.empty(*shape, key_dim, device=q.device, dtype=q.dtype)
        tail1_out = torch.empty_like(q_out)
        tail2_out = torch.empty_like(q_out)
        _gdn_block_intra_fwd_kernel[(batch * heads * chunks * _PAIRS,)](
            q,
            k,
            local_pi1,
            local_pi2,
            local_eta1,
            local_eta2,
            pair_c1,
            pair_c2,
            tail1_coeff,
            tail2_coeff,
            matrix,
            causal,
            q_out,
            tail1_out,
            tail2_out,
            T=length,
            H=heads,
            N=chunks,
            K=key_dim,
            SCALE=float(scale),
        )
        ctx.save_for_backward(*tensors)
        ctx.dimensions = (batch, length, heads, key_dim, chunks)
        ctx.scale = float(scale)
        ctx.input_dtypes = tuple(tensor.dtype for tensor in tensors)
        return matrix, causal, q_out, tail1_out, tail2_out

    @staticmethod
    def backward(ctx, dmatrix, dcausal, dq_direct, dtail1, dtail2):
        (
            q,
            k,
            local_pi1,
            local_pi2,
            local_eta1,
            local_eta2,
            pair_c1,
            pair_c2,
            tail1_coeff,
            tail2_coeff,
        ) = ctx.saved_tensors
        batch, length, heads, key_dim, chunks = ctx.dimensions
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.empty_like(k, dtype=torch.float32)
        coefficient_grads = tuple(
            torch.zeros_like(tensor)
            for tensor in (local_pi1, local_pi2, local_eta1, local_eta2)
        )
        dpair_c1 = torch.zeros_like(pair_c1)
        dpair_c2 = torch.zeros_like(pair_c2)
        dtail1_coeff = torch.zeros_like(tail1_coeff)
        dtail2_coeff = torch.zeros_like(tail2_coeff)
        grid = lambda meta: (
            batch * heads * chunks * triton.cdiv(key_dim, meta["BK"]),
        )
        _gdn_block_intra_bwd_chunk_kernel[grid](
            q,
            k,
            local_pi1,
            local_pi2,
            local_eta1,
            local_eta2,
            pair_c1,
            pair_c2,
            tail1_coeff,
            tail2_coeff,
            dmatrix.contiguous(),
            dcausal.contiguous(),
            dq_direct.contiguous(),
            dtail1.contiguous(),
            dtail2.contiguous(),
            dq,
            dk,
            *coefficient_grads,
            dpair_c1,
            dpair_c2,
            dtail1_coeff,
            dtail2_coeff,
            T=length,
            H=heads,
            N=chunks,
            K=key_dim,
            SCALE=ctx.scale,
        )
        grads = (
            dq,
            dk,
            *coefficient_grads,
            dpair_c1,
            dpair_c2,
            dtail1_coeff,
            dtail2_coeff,
        )
        return tuple(
            grad.to(dtype) for grad, dtype in zip(grads, ctx.input_dtypes)
        ) + (None,)


def build_gdn_blocked_operands(q, k, v, g, beta, gamma, lam, scale):
    """Build stable C64 operands with C32 local coordinates."""
    if q.shape[1] % _CHUNK:
        raise ValueError("sequence length must be divisible by 64")
    from .delta_triton import _DeltaSolve
    from .gdn_coeff_triton import gdn_coefficients_triton

    local = gdn_coefficients_triton(
        g, beta, gamma, lam, chunk_size=_BLOCK
    )
    (
        local_pi1,
        local_pi2,
        local_sigma1,
        local_sigma2,
        local_eta1,
        local_eta2,
        local_tau1,
        local_tau2,
        _local_query,
        local_transport,
    ) = local
    (
        global_pi1,
        global_pi2,
        global_eta1,
        global_eta2,
        pair_c1,
        pair_c2,
        tail1_coeff,
        tail2_coeff,
        transport,
    ) = _gdn_block_transport_fast(
        local_pi1,
        local_pi2,
        local_sigma1,
        local_sigma2,
        local_eta1,
        local_eta2,
        local_tau1,
        local_tau2,
        local_transport,
    )
    matrix, causal, query, tail1, tail2 = _GDNBlockedIntra.apply(
        q,
        k,
        local_pi1,
        local_pi2,
        local_eta1,
        local_eta2,
        pair_c1,
        pair_c2,
        tail1_coeff,
        tail2_coeff,
        scale,
    )
    u0, state_weight1, state_weight2 = _DeltaSolve.apply(
        matrix, k, v, beta, global_pi1, global_pi2
    )
    query_coefficients = torch.stack(
        (global_eta1, global_eta2), dim=-1
    ).unsqueeze(-2)
    return (
        u0,
        state_weight1,
        state_weight2,
        query,
        query_coefficients,
        tail1,
        tail2,
        causal,
        transport,
    )


def build_deltanet_blocked_operands(q, k, v, beta, gamma, lam, scale):
    """Build stable C64 DeltaNet operands from two C32 local systems."""
    if q.shape[1] % _CHUNK:
        raise ValueError("sequence length must be divisible by 64")
    from .delta_triton import _DeltaCoefficients, _DeltaSolve

    local = _DeltaCoefficients.apply(beta, gamma, lam, _BLOCK)
    (
        local_pi1,
        local_pi2,
        local_sigma1,
        local_sigma2,
        local_eta1,
        local_eta2,
        local_tau1,
        local_tau2,
        _local_query,
        local_transport,
    ) = local
    (
        global_pi1,
        global_pi2,
        global_eta1,
        global_eta2,
        pair_c1,
        pair_c2,
        tail1_coeff,
        tail2_coeff,
        transport,
    ) = _gdn_block_transport_fast(
        local_pi1,
        local_pi2,
        local_sigma1,
        local_sigma2,
        local_eta1,
        local_eta2,
        local_tau1,
        local_tau2,
        local_transport,
    )
    matrix, causal, query, tail1, tail2 = _GDNBlockedIntra.apply(
        q,
        k,
        local_pi1,
        local_pi2,
        local_eta1,
        local_eta2,
        pair_c1,
        pair_c2,
        tail1_coeff,
        tail2_coeff,
        scale,
    )
    u0, state_weight1, state_weight2 = _DeltaSolve.apply(
        matrix, k, v, beta, global_pi1, global_pi2
    )
    query_coefficients = torch.stack(
        (global_eta1, global_eta2), dim=-1
    ).unsqueeze(-2)
    return (
        u0,
        state_weight1,
        state_weight2,
        query,
        query_coefficients,
        tail1,
        tail2,
        causal,
        transport,
    )


__all__ = ["build_deltanet_blocked_operands", "build_gdn_blocked_operands"]

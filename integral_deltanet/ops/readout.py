"""Exact chunkwise *readout-side* integration.

The write-side recurrence in :mod:`chunk` mixes the slow state into the
passive carrier (``M_{t-1} = gamma S_{t-1} + (1-gamma) X_{t-1}``), so the
active state forgets with timescale ``tau_gamma`` even when the base model
has no decay.  Readout-side integration keeps the active recurrence identical
to the base model and mixes at the readout instead::

    S_t = D_t S_{t-1} + k_t r_t^T,   r_t = z_t - (D_t S_{t-1})^T e_t
    X_t = (1 - lam_t) X_{t-1} + lam_t S_t
    o_t = gamma_t S_t^T q_t + (1 - gamma_t) X_t^T q_t

With ``gamma_t = 1`` the output is the base model's output exactly; ``X``
only ever adds information.  Because ``S`` no longer depends on ``X`` the
chunk transport is lower triangular, and the whole recurrence still fits the
operand contract of the fused two-state boundary scan in
:mod:`scan_triton`.  Only the operand builder below is new; the scan and its
backward are reused unchanged.

Per chunk with boundary states ``S, X`` (``[K, V]``), ``A_i = prod_{s<=i}
alpha_s`` (cumulative decay), ``l_i = prod_{s<=i} (1 - lam_s)`` and
``F_i = sum_{tau<=i} lam_tau A_tau / l_tau``::

    residual = u0 - w1 @ S                       (unit-lower solve, w2 = 0)
    q1_i  = gamma_i (q_i * A_i) + (1-gamma_i) l_i (q_i * F_i)        reads S
    q2_i  = (1-gamma_i) l_i q_i                                       reads X
    causal[i, j] = gamma_i q_i . (k_j * E_ij) + (1-gamma_i) l_i q_i . (k_j * P_ij)
    S' = A_end * S + k1^T residual,              k1_j = k_j * E_{end,j}
    X' = l_end F_end * S + l_end X + k2^T residual,
                                                 k2_j = l_end k_j * P_{end,j}

with ``E_ij = exp(G_i - G_j)`` (``i >= j``, channel-wise) and
``P_ij = sum_{tau=j}^{i} lam_tau / l_tau * E_{tau j}``.  Both are bounded, so
the builder has no exponent-range limit and its gradients are stable in bf16.
"""
from __future__ import annotations

import contextlib
import os

import torch

from .chunk import _chunk, _execute_operands, _validate_runtime, _validate_scalar_inputs
from .reference import _validate_inputs
from .solve import solve_unit_lower


def _validate_readout_inputs(q, k, v, g, b, w, gamma, lam):
    """Like :func:`reference._validate_inputs`, but ``gamma`` and ``lam`` may be per channel.

    The readout mix enters only through ``gamma * q`` and ``(1 - gamma) * q``,
    so a ``[B, T, H, K]`` gate (one mix per key channel) is admissible
    everywhere a ``[B, T, H]`` gate is. ``lambda`` supports both shapes too;
    each key channel may carry an independent slow-state update rate.
    """
    if lam.ndim == 4 and lam.shape != q.shape:
        raise ValueError("a per-channel lam must have the [B, T, H, K] shape of q")
    lam_head = lam[..., 0] if lam.ndim == 4 else lam
    if gamma.ndim == 4:
        if gamma.shape != q.shape:
            raise ValueError("a per-channel gamma must have the [B, T, H, K] shape of q")
        _validate_inputs(q, k, v, g, b, w, gamma[..., 0], lam_head)
    else:
        _validate_inputs(q, k, v, g, b, w, gamma, lam_head)


def _validate_readout_scalars(q, k, v, gamma, scalars):
    scalars = tuple(x[..., 0] if x.ndim == 4 and x.shape == q.shape else x for x in scalars)
    if gamma.ndim == 4:
        if gamma.shape != q.shape:
            raise ValueError("a per-channel gamma must have the [B, T, H, K] shape of q")
        _validate_scalar_inputs(q, k, v, scalars)
    else:
        _validate_scalar_inputs(q, k, v, (*scalars, gamma))


def readout_delta_reference(
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
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Token-loop oracle for the readout-side recurrence (state layout ``[B,H,K,V]``).

    ``gamma`` is ``[B, T, H]`` (one readout mix per head) or ``[B, T, H, K]``
    (one per key channel, i.e. per row of the ``[K, V]`` state).  ``lam`` takes
    the same two shapes: a per-channel ``lam`` gives every row of the state its
    own archive rate, so each key subspace carries an independent pair of
    timescales -- fast ``exp(g_t[k])`` and slow ``1 - lam_t[k]``.
    """
    _validate_readout_inputs(q, k, v, g, b, w, gamma, lam)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, g, b, w, gamma, lam = (
        x.to(acc_dtype) for x in (q, k, v, g, b, w, gamma, lam)
    )
    q = q * scale
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]

    if initial_state is None:
        active = torch.zeros(
            batch, heads, key_dim, value_dim, dtype=acc_dtype, device=q.device
        )
        slow = torch.zeros_like(active)
    else:
        active, slow = (state.to(acc_dtype) for state in initial_state)
        expected = (batch, heads, key_dim, value_dim)
        if active.shape != expected or slow.shape != expected:
            raise ValueError(f"initial states must both have shape {expected}")

    outputs = []
    for token in range(length):
        decay = g[:, token].exp().unsqueeze(-1)
        mix = gamma[:, token].unsqueeze(-1)                     # [B, H, K, 1] or [B, H, 1]
        if mix.ndim == 3:
            mix = mix.unsqueeze(-1)
        integrate = lam[:, token].unsqueeze(-1)          # [B, H, K, 1] or [B, H, 1]
        if integrate.ndim == 3:
            integrate = integrate.unsqueeze(-1)
        key = k[:, token]
        erase = b[:, token] * key
        write = w[:, token] * v[:, token]

        decayed_active = decay * active
        residual = write - (erase.unsqueeze(-1) * decayed_active).sum(-2)
        next_active = decayed_active + key.unsqueeze(-1) * residual.unsqueeze(-2)
        next_slow = (1.0 - integrate) * slow + integrate * next_active
        read = mix * next_active + (1.0 - mix) * next_slow
        outputs.append((q[:, token].unsqueeze(-1) * read).sum(-2))
        active, slow = next_active, next_slow

    output = torch.stack(outputs, dim=1).to(v.dtype)
    final_state = (active, slow) if output_final_state else None
    return output, final_state


def _local_readout_block(qc, kc, vc, gc, bc, wc, gamma, lam, scale):
    """Direct readout-side operands of one block relative to its own start.

    Every exponential is *relative* and at most 1.  The in-block decay only
    appears as ``E_ij = exp(G_i - G_j)`` with ``i >= j``, and the slow-state
    retention only as ``exp(L_i - L_tau)`` with ``i >= tau``.  The absolute
    forms are unstable: ``exp(-G_j)`` overflows for trained KDA decays
    (``G_16 ~ -800``), and ``1 / l_tau`` overflows once a learned ``lambda``
    approaches 1.

    The slow admission is the convex recurrence ``P[i] = (1 - lambda_i)
    P[i-1] + lambda_i E[i]`` (and ``B[i]`` likewise over ``A_i`` for the
    boundary state), i.e. ``sum_tau lambda_tau prod_{s>tau} (1 - lambda_s)
    E[tau, j]`` with every factor at most 1.  ``C x C x K`` tensors per block,
    reduced against ``q_i k_j``.
    """
    acc_dtype = torch.float64 if qc.dtype == torch.float64 else torch.float32
    gamma, lam = (tensor.to(acc_dtype) for tensor in (gamma, lam))
    gc = gc.to(acc_dtype)
    qc = qc.to(acc_dtype) * scale
    kc = kc.to(acc_dtype)
    bc = bc.to(acc_dtype)
    write = wc.to(acc_dtype) * vc.to(acc_dtype)
    chunk, key_dim = qc.shape[-2], qc.shape[-1]
    lower = torch.ones(chunk, chunk, dtype=torch.bool, device=qc.device).tril()

    log_decay = gc.cumsum(dim=-2)                                   # G_i   [.., C, K]
    decay = torch.exp(log_decay)                                    # A_i <= 1
    # gamma and lam are [.., C] (per head) or [.., C, K] (per key channel).
    # Broadcasting both to [.., C, K] here makes the two gate shapes share every
    # line below; a per-head lam simply repeats along the key axis.
    gamma_key = gamma.unsqueeze(-1) if gamma.ndim == qc.ndim - 1 else gamma
    lam_key = lam.unsqueeze(-1).expand_as(qc) if lam.ndim == qc.ndim - 1 else lam
    retain = 1.0 - lam_key                                          # 1 - lambda_i  [.., C, K]
    keep = retain.cumprod(dim=-2)                                   # l_i <= 1  [.., C, K]
    q_active = gamma_key * qc                                       # gamma_i q_i
    q_slow = (1.0 - gamma_key) * qc                                 # (1 - gamma_i) q_i

    # Pairwise relative decay E[i, j] = exp(G_i - G_j), i >= j.
    pair_log = log_decay.unsqueeze(-2) - log_decay.unsqueeze(-3)   # [.., i, j, K]
    relative = torch.exp(pair_log.clamp_max(0.0)) * lower.unsqueeze(-1)
    # Slow-state admission as a convex recurrence over the row axis:
    #   P[i] = (1 - lambda_i) P[i-1] + lambda_i E[i]      (in-block writes)
    #   B[i] = (1 - lambda_i) B[i-1] + lambda_i A_i       (boundary state)
    # so P[i, j] = sum_{tau<=i} lambda_tau prod_{s=tau+1..i} (1 - lambda_s) E[tau, j]
    # without ever forming 1 / l_tau, which overflows when lambda -> 1.  All
    # steps are pointwise on the same shape and fuse under torch.compile.
    admitted_rows, boundary_rows = [], []
    admitted = torch.zeros_like(relative[..., 0, :, :])
    boundary = torch.zeros_like(decay[..., 0, :])
    for i in range(chunk):
        lam_i = lam_key[..., i, :]                                  # [.., K]
        admitted = (1.0 - lam_i).unsqueeze(-2) * admitted + lam_i.unsqueeze(-2) * relative[..., i, :, :]
        boundary = (1.0 - lam_i) * boundary + lam_i * decay[..., i, :]
        admitted_rows.append(admitted)
        boundary_rows.append(boundary)
    admitted_pair = torch.stack(admitted_rows, dim=-3)              # [.., i, j, K]
    admit_boundary = torch.stack(boundary_rows, dim=-2)             # [.., C, K]

    interaction = ((bc * kc).unsqueeze(-2) * kc.unsqueeze(-3) * relative).sum(-1)
    causal_active = (q_active.unsqueeze(-2) * kc.unsqueeze(-3) * relative).sum(-1)
    causal_slow = (q_slow.unsqueeze(-2) * kc.unsqueeze(-3) * admitted_pair).sum(-1)
    causal = causal_active + causal_slow                            # [.., C, C] lower

    abar = decay * (bc * kc)
    qbar1 = q_active * decay + q_slow * admit_boundary
    qbar2 = q_slow * keep
    tail1 = kc * relative[..., -1, :, :]                            # k_j E_{end, j}
    tail2 = kc * admitted_pair[..., -1, :, :]                       # k_j P_{end, j}
    f11 = decay[..., -1, :]
    f12 = admit_boundary[..., -1, :]                                # sum_tau W[end, tau] A_tau
    f22 = keep[..., -1, :].expand_as(f11)   # already [.., K] for either lam shape
    return write, abar, qbar1, qbar2, tail1, tail2, f11, f12, f22, interaction, causal


def _local_readout_block_subblocks(qc, kc, vc, gc, bc, wc, gamma, lam, scale, sub_block=4):
    """Direct readout-side operands of one block relative to its own start.

    Every exponential is *relative* and at most 1: the in-block decay only
    appears as ``exp(G_i - G_j)`` with ``i >= j``, never as ``exp(-G_j)``.  A
    factorized form ``(q A_i) . (k / A_j)`` or the prefix difference
    ``(F_i - F_{j-1}) / A_j`` is exact in fp64 but its halves grow like
    ``exp(|G_j - G_tau|)``; trained KDA reaches ``G_16 ~ -800``, and with the
    scan's bf16 upstream gradients the cancellation fails.

    To keep the pairwise tensors small the block is split into sub-blocks of
    ``sub_block`` tokens (FLA's intra/inter split): pairs inside a sub-block
    use an explicit ``S x S x K`` tensor, pairs across sub-blocks pivot at the
    later sub-block's start, ``exp(G_i - G_j) = exp(G_i - Gp_b) exp(Gp_b - G_j)``,
    so they are plain products of two bounded ``[*, K]`` operands.

    Returns fp32/fp64 tensors; the caller decides the storage dtype.
    ``interaction`` and ``causal`` for the block itself are returned ready-made.
    """
    acc_dtype = torch.float64 if qc.dtype == torch.float64 else torch.float32
    gamma, lam = (tensor.to(acc_dtype) for tensor in (gamma, lam))
    gc = gc.to(acc_dtype)
    qc = qc.to(acc_dtype) * scale
    kc = kc.to(acc_dtype)
    bc = bc.to(acc_dtype)
    write = wc.to(acc_dtype) * vc.to(acc_dtype)
    chunk, key_dim = qc.shape[-2], qc.shape[-1]
    if chunk % sub_block:
        raise ValueError("sub_block must divide the block size")
    blocks = chunk // sub_block
    outer = qc.shape[:-2]
    device = qc.device
    zero = qc.new_zeros(())

    log_decay = gc.cumsum(dim=-2)                                   # G_i   [.., C, K]
    decay = torch.exp(log_decay)                                    # A_i <= 1
    if lam.ndim == qc.ndim:
        raise NotImplementedError("the sub-block builder has no per-channel lam form")
    log_keep = torch.log1p(-lam).cumsum(dim=-1)                     # L_i   [.., C]
    keep = torch.exp(log_keep).unsqueeze(-1)                        # l_i   [.., C, 1]
    lam_key = lam.unsqueeze(-1)
    gamma_key = gamma.unsqueeze(-1)
    weight = lam_key * torch.exp(-log_keep).unsqueeze(-1)           # w_i = lam_i / l_i
    admit_cum = (weight * decay).cumsum(dim=-2)                     # F_i = sum lam A / l

    # --- sub-block geometry -------------------------------------------------
    def blockify(tensor):                                           # [.., C, *] -> [.., n, S, *]
        return tensor.reshape(*outer, blocks, sub_block, tensor.shape[-1])

    token = torch.arange(chunk, device=device)
    start = torch.arange(blocks, device=device) * sub_block
    before_pivot = (token.unsqueeze(0) < start.unsqueeze(1))       # [n, C]  j <  s_b
    from_pivot = ~before_pivot                                      # [n, C]  i >= s_b
    lower_sub = torch.ones(sub_block, sub_block, dtype=torch.bool, device=device).tril()
    block_eye = torch.eye(blocks, dtype=acc_dtype, device=device)

    g_blk = blockify(log_decay)                                     # [.., n, S, K]
    pivot = torch.cat((g_blk.new_zeros(*outer, 1, key_dim), g_blk[..., :-1, -1, :]), dim=-2)  # Gp_b [.., n, K]

    # Bounded relative factors.
    # rel_from[b, i] = exp(G_i - Gp_b) for i >= s_b   (rows of later tokens)
    # rel_to[b, j]   = exp(Gp_b - G_j) for j <  s_b   (columns of earlier tokens)
    diff_from = log_decay.unsqueeze(-3) - pivot.unsqueeze(-2)       # [.., n, C, K]
    rel_from = torch.exp(torch.where(from_pivot.unsqueeze(-1), diff_from, zero)) * from_pivot.unsqueeze(-1)
    rel_to = torch.exp(torch.where(before_pivot.unsqueeze(-1), -diff_from, zero)) * before_pivot.unsqueeze(-1)
    k_piv = kc.unsqueeze(-3) * rel_to                               # [.., n, C, K]
    # Own-sub-block rows of rel_from: exp(G_i - Gp_{b(i)})  -> [.., C, K]
    rel_own = torch.diagonal(
        rel_from.reshape(*outer, blocks, blocks, sub_block, key_dim), dim1=-4, dim2=-3
    ).movedim(-1, -3).reshape(*outer, chunk, key_dim)

    # Pairwise inside a sub-block: E[b, i, j] = exp(G_i - G_j), i >= j.
    pair_log = g_blk.unsqueeze(-2) - g_blk.unsqueeze(-3)            # [.., n, i, j, K]
    intra = torch.exp(torch.where(lower_sub.unsqueeze(-1), pair_log, zero)) * lower_sub.unsqueeze(-1)
    w_blk = blockify(weight)                                        # [.., n, S, 1]
    intra_admit = (w_blk.unsqueeze(-2) * intra).cumsum(dim=-3)      # P[b, i, j] within the sub-block
    suffix = intra_admit[..., -1, :, :].reshape(*outer, chunk, key_dim)   # R_j: admission up to sub-block end

    # U[b, i] = sum_{tau = s_b}^{i} w_tau exp(G_tau - Gp_b), i >= s_b.
    admit_from = (weight.unsqueeze(-3) * rel_from).cumsum(dim=-2)   # [.., n, C, K]
    admit_from_own = torch.diagonal(
        admit_from.reshape(*outer, blocks, blocks, sub_block, key_dim), dim1=-4, dim2=-3
    ).movedim(-1, -3).reshape(*outer, chunk, key_dim)               # U_{b(i)}[i]
    # next_piv[j] = exp(Gp_{b(j)+1} - G_j): the factor that carries j's write to
    # the start of the following sub-block (zero for the last sub-block).
    rel_to_v = rel_to.reshape(*outer, blocks, blocks, sub_block, key_dim)  # [.., b, a, S, K]
    if blocks > 1:
        next_piv = torch.diagonal(rel_to_v[..., 1:, :-1, :, :], dim1=-4, dim2=-3).movedim(-1, -3)
        next_piv = torch.cat((next_piv, next_piv.new_zeros(*outer, 1, sub_block, key_dim)), dim=-3)
    else:
        next_piv = rel_to_v.new_zeros(*outer, 1, sub_block, key_dim)
    next_piv = next_piv.reshape(*outer, chunk, key_dim)
    # admission accumulated from sub-block a+1's start up to the end of the block
    # and up to each later sub-block start s_b - 1.
    admit_end = admit_from[..., -1, :]                              # U_m[end]      [.., n, K]
    admit_end_next = torch.cat((admit_end[..., 1:, :], admit_end.new_zeros(*outer, 1, key_dim)), dim=-2)
    admit_end_next = admit_end_next.unsqueeze(-2).expand(*outer, blocks, sub_block, key_dim)
    admit_total = suffix + next_piv * admit_end_next.reshape(*outer, chunk, key_dim)   # P_{end, j}
    if blocks > 1:
        # bound[m-1, b-1] = U_m[s_b - 1] for m, b >= 1        [.., n-1, n-1, K]
        bound = admit_from[..., 1:, start[1:] - 1, :]
        # P_{s_b - 1, j} for j in sub-block a < b: R_j + next_piv_j * U_{a+1}[s_b - 1]
        carried = bound.transpose(-3, -2).unsqueeze(-2)             # [.., b-1, a, 1, K]
        carried = (blockify(suffix)[..., :-1, :, :].unsqueeze(-4) + blockify(next_piv)[..., :-1, :, :].unsqueeze(-4) * carried)
        carried = carried * torch.tril(torch.ones(blocks - 1, blocks - 1, dtype=acc_dtype, device=device))[:, :, None, None]
        carried = torch.cat((carried.new_zeros(*outer, 1, blocks - 1, sub_block, key_dim), carried), dim=-4)   # b = 0 row
        carried = torch.cat((carried, carried.new_zeros(*outer, blocks, 1, sub_block, key_dim)), dim=-3)       # a = n-1 col
        admit_start = carried.reshape(*outer, blocks, chunk, key_dim)        # P_{s_b - 1, j}, j < s_b
    else:
        admit_start = rel_to.new_zeros(*outer, 1, chunk, key_dim)

    # --- C x C products ----------------------------------------------------
    def block_diagonal(tensor):                                     # [.., n, S, S] -> [.., C, C]
        full = tensor.unsqueeze(-2) * block_eye[:, None, :, None]
        return full.reshape(*outer, chunk, chunk)

    def cross(rows, cols):                                          # [.., C, K] x [.., n, C, K] -> [.., C, C]
        return torch.matmul(blockify(rows), cols.transpose(-1, -2)).reshape(*outer, chunk, chunk)

    q_blk, k_blk = blockify(qc), blockify(kc)
    erase = bc * kc
    qk_intra = q_blk.unsqueeze(-2) * k_blk.unsqueeze(-3)            # [.., n, i, j, K]
    ek_intra = blockify(erase).unsqueeze(-2) * k_blk.unsqueeze(-3)
    interaction = block_diagonal((ek_intra * intra).sum(-1)) + cross(erase * rel_own, k_piv)
    causal_active = block_diagonal((qk_intra * intra).sum(-1)) + cross(qc * rel_own, k_piv)
    causal_slow = (
        block_diagonal((qk_intra * intra_admit).sum(-1))
        + cross(qc, kc.unsqueeze(-3) * admit_start)
        + cross(qc * admit_from_own, k_piv)
    )
    read_weight = (1.0 - gamma_key) * keep                          # [.., C, 1]
    causal = gamma_key * causal_active + read_weight * causal_slow

    abar = decay * erase                                            # reads S in the residual
    qbar1 = gamma_key * (qc * decay) + read_weight * (qc * admit_cum)
    qbar2 = read_weight * qc
    decay_end = decay[..., -1, :]                                   # A_end   [.., K]
    keep_end = keep[..., -1, :]                                     # l_end   [.., 1]
    tail1 = kc * torch.exp(log_decay[..., -1:, :] - log_decay)     # k_j exp(G_end - G_j)
    tail2 = keep_end.unsqueeze(-2) * kc * admit_total
    f11 = decay_end
    f12 = keep_end * admit_cum[..., -1, :]
    f22 = keep_end.expand_as(f11)
    return write, abar, qbar1, qbar2, tail1, tail2, f11, f12, f22, interaction, causal


# Which in-block arithmetic the builders use.  "pairwise" is the C x C x K
# form; "subblock" is the sub-block pivot form of the same math.
LOCAL_BLOCK_IMPLEMENTATION = os.environ.get("INTEGRAL_READOUT_LOCAL", "pairwise")
if LOCAL_BLOCK_IMPLEMENTATION not in {"pairwise", "subblock"}:
    raise ValueError("INTEGRAL_READOUT_LOCAL must be pairwise or subblock")


def _local_block(qc, kc, vc, gc, bc, wc, gamma, lam, scale):
    if LOCAL_BLOCK_IMPLEMENTATION == "subblock":
        # Written with the separable lambda_tau / l_tau weights, which
        # overflow when lambda -> 1.
        raise NotImplementedError(
            "the sub-block builder predates the relative-lambda fix; use pairwise"
        )
    return _local_readout_block(qc, kc, vc, gc, bc, wc, gamma, lam, scale)


def _finish_readout_operands(write, abar, qbar1, qbar2, tail1, tail2,
                              f11, f12, f22, interaction, causal_read, operand_dtype):
    """Solve the residual system and pack operands in the scan's contract."""
    acc_dtype = write.dtype
    key_dim = abar.shape[-1]
    strict_lower = torch.tril(interaction, diagonal=-1)
    rhs = torch.cat((write, abar), dim=-1)
    solution = solve_unit_lower(strict_lower, rhs, matrix_is_strict_lower=True)
    u0, state_weight1 = solution.split((write.shape[-1], key_dim), dim=-1)
    state_weight2 = torch.zeros_like(state_weight1)
    causal_read = torch.tril(causal_read)
    f21 = torch.zeros_like(f11)
    final_transport = torch.stack((f11, f12, f21, f22), dim=-1)
    if operand_dtype is not None and operand_dtype != acc_dtype:
        # The scan converts every GEMM operand to bf16 on entry; emitting them
        # in that dtype here lets the compiled builder fuse the cast instead of
        # paying a separate full-tensor pass per operand.
        u0, state_weight1 = u0.to(operand_dtype), state_weight1.to(operand_dtype)
        state_weight2 = state_weight2.to(operand_dtype)
        qbar1, qbar2 = qbar1.to(operand_dtype), qbar2.to(operand_dtype)
        tail1, tail2 = tail1.to(operand_dtype), tail2.to(operand_dtype)
        causal_read = causal_read.to(operand_dtype)
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


def _gemm(a, b, data_dtype, acc_dtype):
    """Chunk-local product.  Inputs stay in the accumulation dtype.

    The readout-side operands mix scales (``q1`` carries the O(1) active read and
    the O(lambda) slow read in one tensor), and rounding the GEMM *inputs* to
    bf16 destroys the small part before it is multiplied by ``k_j / A_j``, and
    with it the gate gradients.  The scan still receives bf16
    operands, so only these small ``C x K x C`` products pay for fp32.
    """
    del data_dtype
    return (a.to(acc_dtype) @ b.to(acc_dtype).transpose(-1, -2)).to(acc_dtype)


def _storage_dtype(qc, acc_dtype, operand_dtype):
    if operand_dtype is not None:
        return operand_dtype
    return torch.bfloat16 if qc.is_cuda and acc_dtype == torch.float32 else acc_dtype


def _chunk_all(q, k, v, g, b, w, gamma, lam, chunk_size):
    """Token-major ``[B, T, H, *]`` inputs -> chunked ``[B, H, N, C, *]`` views.

    Called inside the compiled builders so the transposes fuse into their
    consumers instead of materializing eight copies of the layer's tensors.
    """
    batch, length, heads = q.shape[:3]
    chunks = length // chunk_size
    qc, kc, vc, gc, bc, wc = (_chunk(t, chunks, chunk_size) for t in (q, k, v, g, b, w))
    if gamma.ndim == 4:
        gamma_chunks = _chunk(gamma, chunks, chunk_size)            # [B, H, N, C, K]
    else:
        gamma_chunks = gamma.transpose(1, 2).reshape(batch, heads, chunks, chunk_size)
    if lam.ndim == 4:
        lambda_chunks = _chunk(lam, chunks, chunk_size)             # [B, H, N, C, K]
        return qc, kc, vc, gc, bc, wc, gamma_chunks, lambda_chunks
    lambda_chunks = lam.transpose(1, 2).reshape(batch, heads, chunks, chunk_size)
    return qc, kc, vc, gc, bc, wc, gamma_chunks, lambda_chunks


def _build_readout_operands(q, k, v, g, b, w, gamma, lam, scale, chunk_size, operand_dtype=None):
    """Direct readout-side operands for the fused two-state scan (one block per chunk).

    Inputs are token-major: ``q, k, g, b`` are ``[B, T, H, K]``, ``v, w`` are
    ``[B, T, H, V]``, ``gamma, lam`` are ``[B, T, H]``.  ``operand_dtype``
    overrides the storage dtype of the operands handed to the scan (default:
    bfloat16 on CUDA, the accumulation dtype elsewhere).  Exact for any
    ``C``; numerically intended for ``C = 16`` because the in-chunk decay is
    factorized as ``A_i / A_j``.  Larger chunks should use the blocked builder.
    """
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    data_dtype = _storage_dtype(q, acc_dtype, operand_dtype)
    qc, kc, vc, gc, bc, wc, gamma, lam = _chunk_all(q, k, v, g, b, w, gamma, lam, chunk_size)
    (write, abar, qbar1, qbar2, tail1, tail2, f11, f12, f22, interaction, causal) = _local_block(
        qc, kc, vc, gc, bc, wc, gamma, lam, scale
    )
    return _finish_readout_operands(
        write, abar, qbar1, qbar2, tail1, tail2, f11, f12, f22,
        interaction, causal, data_dtype if data_dtype != acc_dtype else None,
    )


def _build_readout_operands_blocked(q, k, v, g, b, w, gamma, lam, scale, chunk_size, operand_dtype=None):
    """Stable C32/C64 readout-side operands composed from exact C16 blocks.

    Every block computes its own direct operands relative to its start, so no
    exponent spans more than 16 tokens.  Block boundaries are crossed only
    with the block transports ``T_b = [[f11, 0], [f12, f22]]`` (S -> S,
    S -> X, X -> X), which are products of bounded per-token factors.  For
    ``S`` the chunk transport is a plain product; for ``X`` the recurrence
    ``p12' = f12 p11 + f22 p12`` accumulates what each block admitted.
    """
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    data_dtype = _storage_dtype(q, acc_dtype, operand_dtype)
    qc, kc, vc, gc, bc, wc, gamma, lam = _chunk_all(q, k, v, g, b, w, gamma, lam, chunk_size)
    block_size = 16
    if chunk_size % block_size:
        raise ValueError("blocked readout-side chunks must be a multiple of 16")
    blocks = chunk_size // block_size
    outer = qc.shape[:-2]

    def split(tensor):
        return tensor.reshape(*outer, blocks, block_size, tensor.shape[-1])

    def split_scalar(tensor):
        return tensor.reshape(*outer, blocks, block_size)

    (write, abar, qbar1, qbar2, tail1, tail2, f11, f12, f22, local_inter, local_causal) = _local_block(
        split(qc), split(kc), split(vc), split(gc), split(bc), split(wc),
        split(gamma) if gamma.ndim == qc.ndim else split_scalar(gamma),
        split(lam) if lam.ndim == qc.ndim else split_scalar(lam), scale,
    )
    # Per-block quantities now carry a leading block axis at dim -3 (vectors)
    # or -2 (transports: [.., blocks, K]).
    key_dim = kc.shape[-1]
    ones = torch.ones_like(f11[..., 0, :])
    zeros = torch.zeros_like(ones)

    # Transport from the chunk start to each block start: S_b = p11 S,
    # X_b = p12 S + p22 X.
    p11, p12, p22 = [ones], [zeros], [ones]
    for b in range(1, blocks):
        p11.append(f11[..., b - 1, :] * p11[-1])
        p12.append(f12[..., b - 1, :] * p11[-2] + f22[..., b - 1, :] * p12[-1])
        p22.append(f22[..., b - 1, :] * p22[-1])
    # Transport from each block end to the chunk end.
    r11, r12, r22 = [None] * blocks, [None] * blocks, [None] * blocks
    r11[-1], r12[-1], r22[-1] = ones, zeros, ones
    for b in range(blocks - 2, -1, -1):
        r11[b] = r11[b + 1] * f11[..., b + 1, :]
        r12[b] = r12[b + 1] * f11[..., b + 1, :] + r22[b + 1] * f12[..., b + 1, :]
        r22[b] = r22[b + 1] * f22[..., b + 1, :]

    chunk_abar, chunk_q1, chunk_q2, chunk_k1, chunk_k2 = [], [], [], [], []
    # (S, X) components at the current block start contributed by every
    # earlier block's residuals, kept per source block: lists of [.., 16, K].
    carried_s, carried_x = [], []
    interaction_rows, causal_rows = [], []
    for b in range(blocks):
        a_b = abar[..., b, :, :]
        q1_b, q2_b = qbar1[..., b, :, :], qbar2[..., b, :, :]
        chunk_abar.append(a_b * p11[b].unsqueeze(-2))
        chunk_q1.append(q1_b * p11[b].unsqueeze(-2) + q2_b * p12[b].unsqueeze(-2))
        chunk_q2.append(q2_b * p22[b].unsqueeze(-2))
        chunk_k1.append(tail1[..., b, :, :] * r11[b].unsqueeze(-2))
        chunk_k2.append(tail1[..., b, :, :] * r12[b].unsqueeze(-2) + tail2[..., b, :, :] * r22[b].unsqueeze(-2))
        # Row blocks of the C x C matrices: earlier blocks then the diagonal.
        inter_cols = [_gemm(a_b, s_comp, data_dtype, acc_dtype) for s_comp in carried_s]
        causal_cols = [
            _gemm(q1_b, s_comp, data_dtype, acc_dtype) + _gemm(q2_b, x_comp, data_dtype, acc_dtype)
            for s_comp, x_comp in zip(carried_s, carried_x)
        ]
        inter_cols.append(local_inter[..., b, :, :])
        causal_cols.append(local_causal[..., b, :, :])
        pad = blocks - b - 1
        if pad:
            zero_cols = torch.zeros_like(inter_cols[0]).repeat_interleave(pad, dim=-1)
            inter_cols.append(zero_cols)
            causal_cols.append(zero_cols)
        interaction_rows.append(torch.cat(inter_cols, dim=-1))
        causal_rows.append(torch.cat(causal_cols, dim=-1))
        # Carry this block's writes (and everything carried so far) through T_b.
        t11, t12, t22 = (f[..., b, :].unsqueeze(-2) for f in (f11, f12, f22))
        carried_x = [t12 * s_comp + t22 * x_comp for s_comp, x_comp in zip(carried_s, carried_x)]
        carried_s = [t11 * s_comp for s_comp in carried_s]
        carried_s.append(tail1[..., b, :, :])
        carried_x.append(tail2[..., b, :, :])

    merged_f11 = r11[0] * f11[..., 0, :]
    merged_f12 = r12[0] * f11[..., 0, :] + r22[0] * f12[..., 0, :]
    merged_f22 = r22[0] * f22[..., 0, :]
    cat_tokens = lambda parts: torch.cat(parts, dim=-2)
    del key_dim
    return _finish_readout_operands(
        write.reshape(*outer, chunk_size, write.shape[-1]),
        cat_tokens(chunk_abar),
        cat_tokens(chunk_q1),
        cat_tokens(chunk_q2),
        cat_tokens(chunk_k1),
        cat_tokens(chunk_k2),
        merged_f11,
        merged_f12,
        merged_f22,
        torch.cat(interaction_rows, dim=-2),
        torch.cat(causal_rows, dim=-2),
        data_dtype if data_dtype != acc_dtype else None,
    )


_COMPILED_INTEGRAL_BUILD = None
_COMPILED_INTEGRAL_BLOCKED_BUILD = None


def _triton_builder_available(q, chunk_size, lam=None):
    # The generated row kernels read ``lam`` through strides, so a per-head lam
    # (stride 0 on the key axis) and a per-channel lam share them.  The hand
    # written tile kernels build a scalar ``R[i, tau]`` table and cannot.
    import os
    if lam is not None and lam.ndim == 4 and os.environ.get("INTEGRAL_READOUT_KERNEL", "rows") != "rows":
        return False
    try:
        from .readout_triton import readout_builder_available
    except ImportError:  # pragma: no cover - triton missing
        return False
    return readout_builder_available(q, chunk_size)
_COMPILE_OPTIONS = {"dynamic": False, "mode": "max-autotune-no-cudagraphs"}


def _build_readout_operands_fast(*args):
    global _COMPILED_INTEGRAL_BUILD
    if _COMPILED_INTEGRAL_BUILD is None:
        try:
            _COMPILED_INTEGRAL_BUILD = torch.compile(
                _build_readout_operands, **_COMPILE_OPTIONS
            )
        except Exception:  # pragma: no cover - environment-specific fallback
            _COMPILED_INTEGRAL_BUILD = _build_readout_operands
    try:
        return _COMPILED_INTEGRAL_BUILD(*args)
    except Exception:  # pragma: no cover - safe eager fallback
        _COMPILED_INTEGRAL_BUILD = _build_readout_operands
        return _build_readout_operands(*args)


def _build_readout_operands_blocked_fast(*args):
    global _COMPILED_INTEGRAL_BLOCKED_BUILD
    if _COMPILED_INTEGRAL_BLOCKED_BUILD is None:
        try:
            _COMPILED_INTEGRAL_BLOCKED_BUILD = torch.compile(
                _build_readout_operands_blocked, **_COMPILE_OPTIONS
            )
        except Exception:  # pragma: no cover - environment-specific fallback
            _COMPILED_INTEGRAL_BLOCKED_BUILD = _build_readout_operands_blocked
    try:
        return _COMPILED_INTEGRAL_BLOCKED_BUILD(*args)
    except Exception:  # pragma: no cover - safe eager fallback
        _COMPILED_INTEGRAL_BLOCKED_BUILD = _build_readout_operands_blocked
        return _build_readout_operands_blocked(*args)


def chunk_readout_delta(
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
    operand_dtype: torch.dtype | None = None,
    blocked: bool | None = None,
    builder: str = "auto",
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Exact readout-side integration in the general (GDN-2) gate form.

    Args follow :func:`readout_delta_reference`.  ``implementation='auto'``
    selects the fused Triton boundary scan on CUDA and the eager scan on CPU.
    ``blocked`` chooses the C16-block composition (default for C32/C64) or
    the direct single-block builder (default for C16).  ``builder`` picks the
    operand builder: ``'triton'`` uses stable C16 kernels and native C32/C64 composition in
    :mod:`readout_triton`, ``'torch'`` the compiled pairwise builder, and
    ``'auto'`` takes the Triton builder wherever it applies (CUDA, C16/C32/C64, not
    fp64; ``INTEGRAL_READOUT_BUILDER=torch`` forces the torch builder).
    """
    _validate_readout_inputs(q, k, v, g, b, w, gamma, lam)
    _validate_runtime(q, chunk_size, implementation, allow_key_padding=True)
    if builder not in ("auto", "torch", "triton"):
        raise ValueError("builder must be auto, torch, or triton")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    autocast_context = (
        torch.amp.autocast(device_type="cuda", enabled=False)
        if q.is_cuda
        else contextlib.nullcontext()
    )
    with autocast_context:
        use_blocked = chunk_size > 16 if blocked is None else blocked
        if builder == "triton" or (
            builder == "auto" and (chunk_size == 16 or use_blocked) and _triton_builder_available(q, chunk_size, lam)
        ):
            from .readout_triton import build_readout_operands_triton

            operands = build_readout_operands_triton(
                q, k, v, g, b, w, gamma, lam, scale, chunk_size, operand_dtype
            )
            return _execute_operands(
                operands, q, v, initial_state, output_final_state, implementation,
                readout_gate=os.environ.get("INTEGRAL_READOUT_SCAN", "triangular") != "general",
            )
        if use_blocked:
            build = (
                _build_readout_operands_blocked_fast
                if compile_precompute and q.is_cuda
                else _build_readout_operands_blocked
            )
        else:
            build = (
                _build_readout_operands_fast
                if compile_precompute and q.is_cuda
                else _build_readout_operands
            )
        operands = build(q, k, v, g, b, w, gamma, lam, scale, chunk_size, operand_dtype)
        return _execute_operands(
            operands, q, v, initial_state, output_final_state, implementation,
                readout_gate=os.environ.get("INTEGRAL_READOUT_SCAN", "triangular") != "general",
        )


def chunk_readout_kda(q, k, v, g, beta, gamma, lam, **kwargs):
    """Readout-side integral KDA: channel-wise decay, scalar erase/write."""
    _validate_readout_scalars(q, k, v, gamma, (beta, lam))
    return chunk_readout_delta(
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


def chunk_readout_gdn2(q, k, v, g, b, w, gamma, lam, **kwargs):
    """Named entry point for the general readout-side GDN-2 form."""
    return chunk_readout_delta(q, k, v, g, b, w, gamma, lam, **kwargs)


__all__ = [
    "readout_delta_reference",
    "chunk_readout_delta",
    "chunk_readout_kda",
    "chunk_readout_gdn2",
]

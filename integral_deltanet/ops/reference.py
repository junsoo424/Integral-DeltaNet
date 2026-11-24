"""Readable recurrent oracle for the Integral DeltaNet recurrence.

The state layout is ``[B, H, K, V]`` (the transpose of the value-by-key
notation in integral_deltanet.tex).  This implementation intentionally
uses a token loop; it is the correctness oracle for the chunkwise kernel, not
the training path.
"""
from __future__ import annotations

import math

import torch


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
) -> None:
    if q.ndim != 4:
        raise ValueError("q must have shape [B, T, H, K]")
    if q.shape != k.shape or q.shape != g.shape or q.shape != b.shape:
        raise ValueError("q, k, g, and b must have identical [B, T, H, K] shapes")
    if v.ndim != 4 or w.shape != v.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("v and w must have identical [B, T, H, V] shapes")
    if gamma.shape != q.shape[:3] or lam.shape != q.shape[:3]:
        raise ValueError("gamma and lam must have shape [B, T, H]")
    tensors = (q, k, v, g, b, w, gamma, lam)
    if any(t.device != q.device for t in tensors):
        raise ValueError("all inputs must be on the same device")


def integral_delta_reference(
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
    r"""Evaluate the exact two-state recurrence one token at a time.

    ``g`` is log-decay, ``b`` is the key-channel erase gate, and ``w`` is the
    value-channel write gate, matching the FLA GDN-2 low-level API.  The paper
    variables are ``D=diag(exp(g))``, ``e=b*k`` and ``z=w*v``.  ``gamma``
    mixes active and slow memory before the edit; ``lam`` integrates the
    post-update active state.

    The four paper variants are obtained without changing this function:

    * DeltaNet: ``g=0`` and scalar ``b=w=beta``.
    * GDN: head-scalar ``g=log(alpha)`` and scalar ``b=w=beta``.
    * KDA: channel-wise ``g`` and scalar ``b=w=beta``.
    * GDN-2: channel-wise ``g``, ``b``, and ``w``.
    """
    _validate_inputs(q, k, v, g, b, w, gamma, lam)
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
        mix = gamma[:, token].unsqueeze(-1).unsqueeze(-1)
        integrate = lam[:, token].unsqueeze(-1).unsqueeze(-1)
        key = k[:, token]
        erase = b[:, token] * key
        write = w[:, token] * v[:, token]

        decayed_active = decay * active
        decayed_slow = decay * slow
        residual = write - (erase.unsqueeze(-1) * decayed_active).sum(-2)
        next_active = (
            mix * decayed_active
            + (1.0 - mix) * decayed_slow
            + key.unsqueeze(-1) * residual.unsqueeze(-2)
        )
        next_slow = (1.0 - integrate) * slow + integrate * next_active
        outputs.append((q[:, token].unsqueeze(-1) * next_active).sum(-2))
        active, slow = next_active, next_slow

    output = torch.stack(outputs, dim=1).to(v.dtype)
    final_state = (active, slow) if output_final_state else None
    return output, final_state


__all__ = ["integral_delta_reference"]

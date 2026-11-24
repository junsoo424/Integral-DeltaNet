"""Ordered-rate integration gates from ``integral_deltanet.tex``."""
from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .ops.arch import FUSED_GATE_ARCHS, on_arch


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("inverse softplus is defined here only for positive values")
    return math.log(math.expm1(value))


class OrderedIntegrationGates(nn.Module):
    r"""Emit paper-constrained scalar ``gamma`` and ``lambda`` per state head.

    One fused linear contains two independent row groups: the positive
    slow-to-fast rate and the explicit timescale ratio. ``lambda`` is derived,
    so there is no third learned projection. Rate arithmetic stays in FP32.

    ``coupling="independent"`` drops the ordering constraint: the second row
    group then parameterizes the slow-write rate ``rho_lambda`` directly and
    ``chi_min`` is unused.  Readout-side integration uses this because its
    ``gamma`` is a readout mix, not a transport rate, so there is no
    stability reason to order the two timescales.

    ``gamma_channels=K`` emits one ``gamma`` per key channel instead of one
    per head (``[B, T, H, K]``): the readout then mixes the active and the
    slow state per row of the ``[K, V]`` state, i.e. per key subspace.  The
    gamma row group grows to ``H * K`` logits, initialized to the same
    per-head timescale on every channel, so a channel-gated model starts
    exactly where the scalar one does.  ``lambda`` stays per head.
    """

    def __init__(
        self,
        hidden_size: int,
        num_state_heads: int,
        *,
        chi_min: float | Sequence[float] = 4.0,
        tau_gamma: float | Sequence[float] = 64.0,
        tau_lambda: float | Sequence[float] = 1024.0,
        bias: bool = True,
        mode: str = "learned",
        bias_init: str = "timescale",
        coupling: str = "ordered",
        gamma_channels: int | None = None,
        gamma_rank: int | None = None,
        lambda_channels: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_state_heads <= 0:
            raise ValueError("hidden_size and num_state_heads must be positive")
        if mode not in {"learned", "fixed"}:
            raise ValueError("mode must be learned or fixed")
        if bias_init not in {"timescale", "zero"}:
            raise ValueError("bias_init must be timescale or zero")
        if coupling not in {"ordered", "independent"}:
            raise ValueError("coupling must be ordered or independent")
        def per_head(value, name):
            values = [float(value)] * num_state_heads if isinstance(value, (int, float)) else [float(v) for v in value]
            if len(values) != num_state_heads or any(v <= 0 for v in values):
                raise ValueError(f"{name} must be positive and have one value per state head")
            return torch.tensor(values, dtype=torch.float32)
        chi_min = per_head(chi_min, "chi_min")
        tau_gamma = per_head(tau_gamma, "tau_gamma")
        tau_lambda = per_head(tau_lambda, "tau_lambda")
        initial_chi = tau_lambda / tau_gamma
        if coupling == "ordered" and torch.any(initial_chi <= chi_min):
            raise ValueError("tau_lambda / tau_gamma must be greater than chi_min per head")
        if not bias:
            raise ValueError("the paper initialization requires projection biases")
        if gamma_channels is not None and int(gamma_channels) <= 0:
            raise ValueError("gamma_channels must be a positive channel count or None")
        if gamma_rank is not None:
            if gamma_channels is None:
                raise ValueError("gamma_rank needs a per-channel gamma (gamma_channels)")
            if int(gamma_rank) <= 0:
                raise ValueError("gamma_rank must be a positive integer or None")
        if lambda_channels is not None:
            if int(lambda_channels) <= 0:
                raise ValueError("lambda_channels must be a positive channel count or None")
            if gamma_rank is None:
                raise ValueError("a per-channel lambda is only wired for the low-rank gate")

        self.hidden_size = hidden_size
        self.num_state_heads = num_state_heads
        self.gamma_channels = None if gamma_channels is None else int(gamma_channels)
        self.gamma_width = num_state_heads * (self.gamma_channels or 1)
        self.lambda_channels = None if lambda_channels is None else int(lambda_channels)
        self.lambda_width = num_state_heads * (self.lambda_channels or 1)
        self.register_buffer("chi_min", chi_min)
        self.register_buffer("tau_gamma", tau_gamma)
        self.register_buffer("tau_lambda", tau_lambda)
        self.mode = mode
        self.bias_init = bias_init
        self.coupling = coupling
        self.gamma_rank = None if gamma_rank is None else int(gamma_rank)
        if self.gamma_rank is None:
            self.projection = nn.Linear(hidden_size, self.gamma_width + num_state_heads, bias=True)
            init_modules = (self.projection,)
        else:
            # ``projection`` then carries only the lambda logits; gamma goes
            # through the bottleneck.  ``project`` re-assembles the fused layout.
            self.gamma_down = nn.Linear(hidden_size, self.gamma_rank, bias=False)
            self.gamma_up = nn.Linear(self.gamma_rank, self.gamma_width, bias=True)
            if self.lambda_channels is None:
                self.projection = nn.Linear(hidden_size, num_state_heads, bias=True)
                init_modules = (self.projection, self.gamma_down, self.gamma_up)
            else:
                # lambda gets its own bottleneck of the same rank, so both
                # channel gates of the layer have the shape of KDA's f_proj.
                self.projection = nn.Identity()
                self.lambda_down = nn.Linear(hidden_size, self.gamma_rank, bias=False)
                self.lambda_up = nn.Linear(self.gamma_rank, self.lambda_width, bias=True)
                init_modules = (self.gamma_down, self.gamma_up, self.lambda_down, self.lambda_up)

        # CausalLM applies one common initializer to ordinary Linear modules.
        # Keep the paper's timescale initialization intact for these projections.
        for module in init_modules:
            module._preserve_integration_init = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore the ordered paper initialization."""

        initial_chi = self.tau_lambda / self.tau_gamma
        gamma_bias = torch.tensor([_inverse_softplus(float(v)) for v in (1.0 / self.tau_gamma)])
        if self.coupling == "independent":
            ratio_bias = torch.tensor([_inverse_softplus(float(v)) for v in (1.0 / self.tau_lambda)])
        else:
            ratio_bias = torch.tensor([_inverse_softplus(float(v)) for v in torch.log(initial_chi / self.chi_min)])
        if self.gamma_channels is not None:
            gamma_bias = gamma_bias.repeat_interleave(self.gamma_channels)   # head-major [H * K]
        with torch.no_grad():
            if not isinstance(self.projection, nn.Identity):
                self.projection.weight.zero_()
            if self.gamma_rank is None:
                if self.bias_init == "timescale":
                    self.projection.bias[:self.gamma_width].copy_(gamma_bias)
                    self.projection.bias[self.gamma_width:].copy_(ratio_bias)
                else:
                    self.projection.bias.zero_()
            else:
                # LoRA-style: random down, zero up, so the output is the bias
                # at initialization and both factors receive gradient.
                nn.init.normal_(self.gamma_down.weight, mean=0.0, std=0.02)
                self.gamma_up.weight.zero_()
                if self.lambda_channels is not None:
                    nn.init.normal_(self.lambda_down.weight, mean=0.0, std=0.02)
                    self.lambda_up.weight.zero_()
                    ratio_bias = ratio_bias.repeat_interleave(self.lambda_channels)
                if self.bias_init == "timescale":
                    self.gamma_up.bias.copy_(gamma_bias)
                    (self.lambda_up if self.lambda_channels is not None else self.projection).bias.copy_(ratio_bias)
                else:
                    self.gamma_up.bias.zero_()
                    (self.lambda_up if self.lambda_channels is not None else self.projection).bias.zero_()
        if self.mode == "fixed":
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def project(self, normalized_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states without materializing activation intermediates."""

        if normalized_hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden width {self.hidden_size}, got "
                f"{normalized_hidden_states.shape[-1]}"
            )
        if self.gamma_rank is None:
            if self.mode == "fixed":
                return self.projection.bias.expand(*normalized_hidden_states.shape[:-1], -1)
            return self.projection(normalized_hidden_states)
        lam_bias = self.lambda_up.bias if self.lambda_channels is not None else self.projection.bias
        if self.mode == "fixed":
            bias = torch.cat((self.gamma_up.bias, lam_bias))
            return bias.expand(*normalized_hidden_states.shape[:-1], -1)
        gamma_logits = self.gamma_up(self.gamma_down(normalized_hidden_states))
        if self.lambda_channels is not None:
            ratio_logits = self.lambda_up(self.lambda_down(normalized_hidden_states))
        else:
            ratio_logits = self.projection(normalized_hidden_states)
        return torch.cat((gamma_logits, ratio_logits), dim=-1)

    def activate(self, logits: torch.Tensor):
        """Reference activation used by CPU and unsupported GPU shapes."""

        gamma_logits, ratio_logits = logits.float().split(
            (self.gamma_width, self.lambda_width), dim=-1
        )
        if self.gamma_channels is not None:
            gamma_logits = gamma_logits.reshape(
                *gamma_logits.shape[:-1], self.num_state_heads, self.gamma_channels
            )
        if self.lambda_channels is not None:
            ratio_logits = ratio_logits.reshape(
                *ratio_logits.shape[:-1], self.num_state_heads, self.lambda_channels
            )
        rho_gamma = F.softplus(gamma_logits)
        if self.coupling == "independent":
            rho_lambda = F.softplus(ratio_logits)
        else:
            if self.gamma_channels is not None:
                raise ValueError("ordered gate coupling needs a per-head gamma")
            log_chi = self.chi_min.log() + F.softplus(ratio_logits)
            rho_lambda = rho_gamma * torch.exp(-log_chi)
        gamma = torch.exp(-rho_gamma)
        lam = -torch.expm1(-rho_lambda)
        return gamma, lam

    def can_fuse_channel_pair(self, hidden_states):
        return (os.environ.get("INTEGRAL_FUSE_GATE_PAIR", "0") == "1"
                and hidden_states.is_cuda and hidden_states.dtype != torch.float64
                and on_arch(hidden_states, FUSED_GATE_ARCHS)
                and self.mode == "learned" and self.coupling == "independent"
                and self.gamma_rank == 128 and self.gamma_channels == 128
                and self.lambda_channels == 128)

    def activate_down(self, gamma_down: torch.Tensor, lambda_down: torch.Tensor):
        """``forward`` for a caller that already ran gamma_down/lambda_down."""

        gx = self.gamma_up(gamma_down)
        lx = self.lambda_up(lambda_down)
        shape = (*gx.shape[:-1], self.num_state_heads, self.gamma_channels)
        if self.can_fuse_channel_pair(gx):
            from .ops.channel_gate_pair import independent_gate_pair
            gamma, lam = independent_gate_pair(gx, lx)
            return gamma.reshape(shape), lam.reshape(shape)
        return self.activate(torch.cat((gx, lx), dim=-1))

    def forward(self, normalized_hidden_states: torch.Tensor):
        if self.can_fuse_channel_pair(normalized_hidden_states):
            from .ops.channel_gate_pair import independent_gate_pair
            gx = self.gamma_up(self.gamma_down(normalized_hidden_states))
            lx = self.lambda_up(self.lambda_down(normalized_hidden_states))
            gamma, lam = independent_gate_pair(gx, lx)
            shape = (*gx.shape[:-1], self.num_state_heads, self.gamma_channels)
            return gamma.reshape(shape), lam.reshape(shape)
        return self.activate(self.project(normalized_hidden_states))

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, mode={self.mode}, bias_init={self.bias_init}, "
            f"coupling={self.coupling}, gamma_channels={self.gamma_channels}, gamma_rank={self.gamma_rank}, "
            f"lambda_channels={self.lambda_channels}, "
            f"num_state_heads={self.num_state_heads}, chi_min={self.chi_min}, "
            f"tau_gamma={self.tau_gamma}, tau_lambda={self.tau_lambda}"
        )


__all__ = ["OrderedIntegrationGates"]

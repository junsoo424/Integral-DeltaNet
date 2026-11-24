from __future__ import annotations

import math
from typing import Any


import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from fla.layers import GatedDeltaNet2, KimiDeltaAttention
from fla.modules import FusedLinearCrossEntropyLoss, GatedMLP

from fla.modules import FusedRMSNormSwishGate

from .config import validate_model_config
from .norm import RMSNorm
from .integral_mixer import IntegralMixer


class _ConfiguredKimiDeltaAttention(KimiDeltaAttention):
    """Honor an explicitly configured chunk kernel during inference.

    FLA 0.5.2 otherwise replaces ``mode="chunk"`` with its fused recurrent
    kernel for evaluation sequences of at most 64 tokens. Training already
    uses the configured chunk path, so keep inference on that same operator.
    Only this module's mode-selection flag is changed for the duration of the
    call; its child modules remain in evaluation mode.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        force_configured_chunk = self.mode == "chunk" and not self.training
        if not force_configured_chunk:
            return super().forward(*args, **kwargs)

        self.training = True
        try:
            return super().forward(*args, **kwargs)
        finally:
            self.training = False


class _PaperGatedDeltaNet2(GatedDeltaNet2):
    """Restore the two architecture contracts NVIDIA's released layer carries.

    FLA 0.5.2's generic GDN-2 wires the output gate through
    ``FusedRMSNormGated(activation="sigmoid")`` and leaves every internal
    ``nn.Linear`` on the package default initializer.  ``lit_gpt/gdn2.py`` in
    NVlabs/GatedDeltaNet-2 uses ``FusedRMSNormSwishGate`` -- ``RMSNorm(o) *
    silu(g)`` -- and Xavier uniform at gain ``2 ** -2.5`` with zeroed biases.
    Neither changes the parameter count, so the config's matched
    ``intermediate_size`` still holds; both change what the layer computes.
    The recurrence kernels stay FLA's.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.o_norm = FusedRMSNormSwishGate(
            self.head_v_dim, eps=kwargs.get("norm_eps", 1e-5)
        )
        self.apply(self._init_paper_weights)

    @staticmethod
    def _init_paper_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Matches the guard on NVIDIA's layer: the outer initializer must not
        # revisit weights this module already owns.
        module._is_hf_initialized = True


def _build_mixer(config: dict[str, Any], layer_idx: int) -> nn.Module:
    model_type = config["model_type"]
    common = {
        "hidden_size": config["hidden_size"],
        "expand_v": config["expand_v"],
        "num_heads": config["num_heads"],
        "mode": config.get("attn_mode", "chunk"),
        "use_short_conv": config.get("use_short_conv", True),
        "allow_neg_eigval": config.get("allow_neg_eigval", False),
        "conv_size": config.get("conv_size", 4),
        "conv_bias": False,
        "layer_idx": layer_idx,
        "norm_eps": config.get("norm_eps", 1e-6),
    }
    common["head_dim"] = config["head_dim"]
    common["num_v_heads"] = config.get("num_v_heads")
    if model_type == "kda":
        mixer = _ConfiguredKimiDeltaAttention(
            **common,
            safe_gate=config.get("safe_gate", False),
            lower_bound=config.get("lower_bound"),
        )
    elif model_type == "gdn2":
        mixer = _PaperGatedDeltaNet2(**common)
    else:
        raise ValueError(f"unsupported model_type: {model_type}")
    if config.get("mixer_implementation", "fla") == "integral":
        return IntegralMixer(mixer, model_type, config)
    return mixer


class Block(nn.Module):
    def __init__(self, config: dict[str, Any], layer_idx: int) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        eps = config.get("norm_eps", 1e-6)
        self.attn_norm = RMSNorm(hidden_size, eps=eps)
        self.mixer = _build_mixer(config, layer_idx)
        self.use_mlp = bool(config.get("use_mlp", True))
        if self.use_mlp:
            self.mlp_norm = RMSNorm(hidden_size, eps=eps)
            self.mlp = GatedMLP(
                hidden_size=hidden_size,
                intermediate_size=config["intermediate_size"],
                hidden_act="swish",
                fuse_swiglu=True,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.mixer(hidden_states=hidden_states, use_cache=False)[0]
        if not self.use_mlp:
            return residual + hidden_states
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class CausalLM(nn.Module):
    """Common LM skeleton; only the official FLA token mixer changes."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        validate_model_config(config)
        self.config = dict(config)
        hidden_size = config["hidden_size"]
        self.embeddings = nn.Embedding(config["vocab_size"], hidden_size)
        self.layers = nn.ModuleList(
            Block(config, layer_idx) for layer_idx in range(config["num_hidden_layers"])
        )
        self.norm = RMSNorm(hidden_size, eps=config.get("norm_eps", 1e-6))
        self.lm_head = nn.Linear(hidden_size, config["vocab_size"], bias=False)
        self.criterion = FusedLinearCrossEntropyLoss(use_l2warp=False)
        self.gradient_checkpointing = config.get("gradient_checkpointing", True)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if self.config["model_type"] == "gdn2":
            self._init_gdn2_weights(module)
            return
        if isinstance(module, nn.Linear):
            if getattr(module, "_preserve_integration_init", False):
                return
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and not getattr(module.bias, "_is_hf_initialized", False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_gdn2_weights(self, module: nn.Module) -> None:
        """Match NVIDIA's GPT initializer with ``mamba_init=True``.

        The mixer owns its Xavier initialization. Other Linear weights keep
        their constructor initialization, except the MLP and attention output
        projections, which use the reference residual scaling. In our module
        names those are ``down_proj`` (SwiGLU.w3) and ``o_proj`` (attention.proj).
        The GDN-2 mixer's own o_proj keeps its Xavier initialization.
        """
        if getattr(module, "_preserve_integration_init", False):
            return
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, GatedMLP):
            projection = module.down_proj if isinstance(module, GatedMLP) else module.o_proj
            nn.init.kaiming_uniform_(projection.weight, a=math.sqrt(5))
            residuals = 2 if self.config.get("use_mlp", True) else 1
            with torch.no_grad():
                projection.weight.div_(math.sqrt(residuals * self.config["num_hidden_layers"]))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        num_logits_to_keep: int | None = None,
        logit_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                hidden_states = checkpoint(layer, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states)
        if labels is None:
            if logit_positions is not None:
                if num_logits_to_keep is not None:
                    raise ValueError(
                        "num_logits_to_keep and logit_positions are mutually exclusive"
                    )
                if (
                    logit_positions.ndim != 1
                    or logit_positions.shape[0] != input_ids.shape[0]
                ):
                    raise ValueError("logit_positions must have shape [batch_size]")
                batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
                hidden_states = hidden_states[
                    batch_indices,
                    logit_positions.to(device=input_ids.device, dtype=torch.long),
                ].unsqueeze(1)
            elif num_logits_to_keep is not None:
                if num_logits_to_keep <= 0:
                    raise ValueError("num_logits_to_keep must be positive")
                hidden_states = hidden_states[:, -num_logits_to_keep:]

        hidden_states = self.norm(hidden_states)
        if labels is None:
            return self.lm_head(hidden_states)
        shifted = torch.cat(
            (labels[..., 1:], torch.full_like(labels[..., :1], self.criterion.ignore_index)),
            dim=1,
        )
        return self.criterion(hidden_states, shifted, self.lm_head.weight, self.lm_head.bias)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def learning_rate(step: int, max_steps: int, warmup_steps: int, peak: float, minimum: float) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps - 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * min(progress, 1.0)))

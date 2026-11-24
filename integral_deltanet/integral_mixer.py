"""Training-ready FLA layer wrappers for the integral recurrences."""
from __future__ import annotations

from typing import Any
import functools
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from fla.modules.convolution import causal_conv1d
from fla.modules.l2norm import l2norm

from .gates import OrderedIntegrationGates
from .kernel_mode import MEMORY_FIRST, kernel_mode as resolve_kernel_mode
from .ops.arch import (
    FUSED_GATE_ARCHS,
    READOUT_RECOMPUTE_ARCHS,
    SEPARATE_QKV_CONV_ARCHS,
    on_arch,
    tuned_arch,
)
from .ops.erase_write_sigmoid import erase_write_gates
from .ops import (
    chunk_integral_gdn2,
    chunk_integral_kda,
    chunk_readout_gdn2,
    chunk_readout_kda,
    fused_channel_preprocess,
    fused_scalar_preprocess,
)

try:
    from fla.ops.gated_delta_rule.gate import fused_gdn_gate as _fused_gdn_gate
except (ImportError, ModuleNotFoundError):  # FLA < 0.5.2 compatibility
    _fused_gdn_gate = None

try:
    from fla.ops.kda.gate import fused_kda_gate as _fused_kda_gate
except (ImportError, ModuleNotFoundError):  # defensive CPU/minimal installs
    _fused_kda_gate = None


def _checkpoint_keywords(fn, **kwargs):
    """``fn(**kwargs)`` under a non-reentrant checkpoint, its tensors passed positionally.

    The checkpoint saves positional inputs through the saved-tensor hooks, so
    an enclosing block checkpoint drops them with the rest of the block; it
    holds keyword arguments in its recompute closure until backward, which
    under block checkpointing keeps every layer's readout inputs alive.
    """
    names = [name for name, value in kwargs.items() if isinstance(value, torch.Tensor)]
    rest = {name: value for name, value in kwargs.items() if name not in names}

    def call(*tensors):
        return fn(**dict(zip(names, tensors)), **rest)

    return torch.utils.checkpoint.checkpoint(
        call, *(kwargs[name] for name in names), use_reentrant=False
    )


def _l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    # Reuse FLA's fused forward/backward normalization on the training path.
    if tensor.is_cuda:
        return l2norm(tensor, eps=1e-6)
    rstd = torch.rsqrt(tensor.float().square().sum(dim=-1, keepdim=True) + 1e-6)
    return (tensor.float() * rstd).to(tensor.dtype)


class IntegralMixer(nn.Module):
    """Keep an official FLA layer's projections while replacing its recurrence.

    ``integration_readout_gate: true`` selects the readout-side recurrence
    (:mod:`integral_deltanet.ops.readout`): the active state follows the
    base model exactly and the slow state is mixed in at the readout.  Both
    forms share the gate projection and the fused boundary scan.

    Integral training currently targets fixed-length dense batches. Inputs
    are causally right-padded to the selected C16/C32/C64 specialization and
    cropped after the output projection, so arbitrary evaluation lengths are
    still safe. Cache and packed-variable-length execution are deliberately
    rejected because the two-state cache ABI is not part of FLA v0.5.2.
    """

    def __init__(self, base: nn.Module, model_type: str, config: dict[str, Any]) -> None:
        super().__init__()
        if model_type not in {"kda", "gdn2"}:
            raise ValueError(f"unsupported integral model_type: {model_type}")
        chunk_size = int(config["chunk_size"])
        if chunk_size not in (16, 32, 64):
            raise ValueError("integral chunk_size must be 16, 32, or 64")
        if getattr(base, "num_v_heads", base.num_heads) != base.num_heads:
            raise ValueError("integral training currently requires num_v_heads == num_heads")

        self.base = base
        self.model_type = model_type
        self.chunk_size = chunk_size
        self.readout_gate = bool(config.get("integration_readout_gate", False))
        gate_coupling = config.get(
            "integration_gate_coupling", "independent" if self.readout_gate else "ordered"
        )
        gamma_gate = config.get("integration_gamma_gate", "scalar")
        if gamma_gate not in {"scalar", "channel"}:
            raise ValueError("integration_gamma_gate must be scalar or channel")
        if gamma_gate == "channel" and not (self.readout_gate and gate_coupling == "independent"):
            raise ValueError("a per-channel gamma needs readout-side integration with independent gates")
        self.gamma_gate = gamma_gate
        self.kernel_implementation = config.get("kernel_implementation", "auto")
        self.compile_precompute = bool(config.get("compile_precompute", True))
        self.fuse_qkv_conv = bool(config.get("fuse_qkv_conv", True))
        # QKV fusion is algebraically exact. Fused nonlinear activation is an
        # approximate BF16 optimization and is enabled only by recipes that set
        # allow_approximate_fusions.
        allow_approximate = bool(
            config.get("allow_approximate_fusions", False)
        )
        # The fused preprocess kernels bake the ordered per-head gate layout.
        self.fuse_preprocess = (
            bool(config.get("fuse_preprocess", True)) and allow_approximate
            and gate_coupling == "ordered" and gamma_gate == "scalar"
            and not any(isinstance(config.get(key, default), list) for key, default in (("integration_chi_min", 4.0), ("integration_tau_gamma", 64.0), ("integration_tau_lambda", 1024.0)))
        )
        self.fuse_decay_gate = (
            bool(config.get("fuse_decay_gate", True))
            and allow_approximate
        )
        self.allow_approximate_fusions = allow_approximate
        # Which kernel the readout path runs.  Stored, not resolved, so
        # INTEGRAL_KERNEL_MODE can still switch an already-built model.
        self._kernel_mode_setting = config.get("kernel_mode")
        self.integration_gates = OrderedIntegrationGates(
            hidden_size=base.hidden_size,
            num_state_heads=base.num_heads,
            chi_min=config.get("integration_chi_min", 4.0),
            tau_gamma=config.get("integration_tau_gamma", 64.0),
            tau_lambda=config.get("integration_tau_lambda", 1024.0),
            mode=config.get("integration_gate_mode", "learned"),
            bias_init=config.get("integration_gate_bias_init", "timescale"),
            coupling=gate_coupling,
            gamma_channels=base.head_k_dim if gamma_gate == "channel" else None,
            gamma_rank=config.get("integration_gamma_rank"),
            lambda_channels=base.head_k_dim if config.get("integration_lambda_gate") == "channel" else None,
        )

    def _input_projections(self, hidden_states: torch.Tensor):
        """Every projection that reads ``hidden_states`` as one GEMM, or None.

        q/k/v, the decay and output-gate bottlenecks, the erase gate(s) and
        the two integration-gate bottlenecks all read the same input.  One
        GEMM over their stacked weights gives dx in one backward GEMM instead
        of eight or nine GEMMs and the bf16 adds that sum their dx.  The
        outputs are column views, which
        FLA's Triton convolution reads by stride; the narrow bottleneck views
        are made contiguous, since a strided input drops the up-projections'
        bias out of the GEMM epilogue into separate add/sum kernels.
        Opt-in with INTEGRAL_FUSE_INPUT_PROJ=1.
        """
        if os.environ.get("INTEGRAL_FUSE_INPUT_PROJ", "0") != "1" or not tuned_arch(hidden_states):
            return None
        base, gates = self.base, self.integration_gates
        convs = [getattr(base, f"{name}_conv1d", None) for name in "qkv"]
        bottlenecks = [getattr(base, name, None) for name in ("f_proj", "g_proj")]
        if not (
            base.use_short_conv
            and all(conv is not None and conv.backend == "triton" for conv in convs)
            and all(isinstance(seq, nn.Sequential) and len(seq) == 2 for seq in bottlenecks)
            and gates.mode == "learned" and gates.gamma_rank is not None
            and gates.lambda_channels is not None and not self.fuse_preprocess
        ):
            return None
        layers = {"q": base.q_proj, "k": base.k_proj, "v": base.v_proj,
                  "f": base.f_proj[0], "g": base.g_proj[0], "b": base.b_proj}
        if self.model_type == "gdn2":
            layers["w"] = base.w_proj
        layers.update(gamma=gates.gamma_down, lam=gates.lambda_down)
        if any(not isinstance(layer, nn.Linear) or layer.bias is not None for layer in layers.values()):
            return None
        weight = torch.cat([layer.weight for layer in layers.values()])
        out = F.linear(hidden_states, weight)
        widths = {name: layer.out_features for name, layer in layers.items()}
        if self.model_type == "gdn2":
            # b and w sit side by side and leave the split as one piece, read
            # by one gate kernel (a second view of them would cost the
            # backward a zero-filled full-width gradient).
            widths["b"] += widths.pop("w")
        projected = dict(zip(widths, out.split(list(widths.values()), dim=-1)))
        if self.model_type == "gdn2":
            projected["bw"] = projected.pop("b")
            projected["b"], projected["w"] = projected["bw"].split(
                (base.b_proj.out_features, base.w_proj.out_features), dim=-1
            )
        for name in ("f", "g", "gamma", "lam"):
            projected[name] = projected[name].contiguous()
        return projected

    def _project_qkv(self, hidden_states: torch.Tensor):
        base = self.base
        convs = (
            getattr(base, "q_conv1d", None),
            getattr(base, "k_conv1d", None),
            getattr(base, "v_conv1d", None),
        )
        same_dense_conv = (
            base.use_short_conv
            and all(convolution is not None for convolution in convs)
            and convs[0].activation == convs[1].activation == convs[2].activation
            and convs[0].backend == convs[1].backend == convs[2].backend
            and convs[0].kernel_size == convs[1].kernel_size == convs[2].kernel_size
        )
        # FLA v0.5.2 introduced this dense fused QKV convolution for GDN.
        # DeltaNet/KDA/GDN-2 use the same ShortConvolution ABI and, for the
        # shipped recipes, the same SiLU activation on all three branches.
        # On SEPARATE_QKV_CONV_ARCHS the dense convolution costs as much as the
        # three separate ones, so the concatenations around it are overhead.
        if (self.fuse_qkv_conv and same_dense_conv
                and not on_arch(hidden_states, SEPARATE_QKV_CONV_ARCHS)):
            qkv = torch.cat(
                (
                    base.q_proj(hidden_states),
                    base.k_proj(hidden_states),
                    base.v_proj(hidden_states),
                ),
                dim=-1,
            )
            weight = torch.cat(
                (
                    base.q_conv1d.weight.squeeze(1),
                    base.k_conv1d.weight.squeeze(1),
                    base.v_conv1d.weight.squeeze(1),
                ),
                dim=0,
            )
            bias = (
                torch.cat((base.q_conv1d.bias, base.k_conv1d.bias, base.v_conv1d.bias), dim=0)
                if base.conv_bias
                else None
            )
            qkv, _ = causal_conv1d(
                x=qkv,
                weight=weight,
                bias=bias,
                activation=base.q_conv1d.activation,
                backend=base.q_conv1d.backend,
            )
            return torch.split(qkv, (base.key_dim, base.key_dim, base.value_dim), dim=-1)

        if base.use_short_conv:
            projected = []
            for projection, convolution in (
                (base.q_proj, base.q_conv1d),
                (base.k_proj, base.k_conv1d),
                (base.v_proj, base.v_conv1d),
            ):
                value, _ = convolution(
                    x=projection(hidden_states),
                    cache=None,
                    output_final_state=False,
                    cu_seqlens=None,
                )
                projected.append(value)
            return tuple(projected)

        q = base.q_proj(hidden_states)
        k = base.k_proj(hidden_states)
        v = base.v_proj(hidden_states)
        q, k = F.silu(q), F.silu(k)
        return q, k, F.silu(v)

    def _reshape_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        base = self.base
        q = q.reshape(*q.shape[:-1], base.num_heads, base.head_k_dim)
        k = k.reshape(*k.shape[:-1], base.num_heads, base.head_k_dim)
        value_heads = getattr(base, "num_v_heads", base.num_heads)
        v = v.reshape(*v.shape[:-1], value_heads, base.head_v_dim)
        return q, k, v

    def _run_recurrence(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gamma: torch.Tensor,
        lam: torch.Tensor,
        *,
        beta: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        w: torch.Tensor | None = None,
        f_down: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.base
        decay = base.f_proj(hidden_states) if f_down is None else base.f_proj[1](f_down)
        common = {
            "q": q,
            "k": k,
            "v": v,
            "gamma": gamma,
            "lam": lam,
            "scale": base.head_k_dim ** -0.5,
            "chunk_size": self._op_chunk(q),
            "implementation": self.kernel_implementation,
            "compile_precompute": self.compile_precompute,
        }
        if self.readout_gate:
            kernels = {"kda": chunk_readout_kda, "gdn2": chunk_readout_gdn2}
        else:
            kernels = {"kda": chunk_integral_kda, "gdn2": chunk_integral_gdn2}
        recurrence = kernels[self.model_type]
        if self._recomputes_readout_only(q) and torch.is_grad_enabled():
            # memory-first on READOUT_RECOMPUTE_ARCHS: the backward pass
            # recomputes the readout operator instead of keeping its
            # intermediates; the rest of the mixer stays stored.
            recurrence = functools.partial(_checkpoint_keywords, recurrence)
        if self.model_type == "kda":
            raw_g = decay.reshape(
                *hidden_states.shape[:-1], base.num_v_heads, base.head_k_dim
            )
            lower_bound = base.lower_bound if base.safe_gate else None
            if base.safe_gate and lower_bound is None:
                raise ValueError("safe_gate=True requires lower_bound")
            if (os.environ.get("INTEGRAL_FUSE_DECAY_GATE", "0") == "1"
                    and raw_g.is_cuda and raw_g.dtype != torch.float64
                    and on_arch(raw_g, FUSED_GATE_ARCHS)
                    and not getattr(base, "safe_gate", False)
                    and base.A_log.dtype != torch.float64 and base.dt_bias.dtype != torch.float64):
                from .ops.channel_decay_gate import channel_decay_gate
                g = channel_decay_gate(raw_g, base.A_log, base.dt_bias)
            elif (
                self.fuse_decay_gate
                and raw_g.is_cuda
                and _fused_kda_gate is not None
            ):
                g = _fused_kda_gate(
                    raw_g,
                    base.A_log,
                    base.dt_bias,
                    lower_bound=lower_bound,
                    output_dtype=torch.float32,
                )
            else:
                raw_g = raw_g.float() + base.dt_bias.float().reshape(
                    base.num_v_heads, base.head_k_dim
                )
                if base.safe_gate:
                    g = lower_bound * torch.sigmoid(
                        base.A_log.float().exp()[None, None, :, None] * raw_g
                    )
                else:
                    g = -base.A_log.float().exp()[None, None, :, None] * F.softplus(raw_g)
            if beta is None:
                raise RuntimeError("KDA erase gate was not preprocessed")
            return recurrence(g=g, beta=beta, **common)[0]

        raw_g = decay.reshape(
            *hidden_states.shape[:-1], base.num_heads, base.head_k_dim
        )
        if (os.environ.get("INTEGRAL_FUSE_DECAY_GATE", "0") == "1"
                and raw_g.is_cuda and raw_g.dtype != torch.float64
                and on_arch(raw_g, FUSED_GATE_ARCHS)
                and not getattr(base, "safe_gate", False)
                and base.A_log.dtype != torch.float64 and base.dt_bias.dtype != torch.float64):
            from .ops.channel_decay_gate import channel_decay_gate
            g = channel_decay_gate(raw_g, base.A_log, base.dt_bias)
        elif (
            self.fuse_decay_gate
            and raw_g.is_cuda
            and _fused_kda_gate is not None
        ):
            g = _fused_kda_gate(
                raw_g,
                base.A_log,
                base.dt_bias,
                output_dtype=torch.float32,
            )
        else:
            raw_g = raw_g.float() + base.dt_bias.float().reshape(
                base.num_heads, base.head_k_dim
            )
            g = -base.A_log.float().exp()[None, None, :, None] * F.softplus(raw_g)
        if b is None or w is None:
            raise RuntimeError("GDN-2 channel gates were not preprocessed")
        return recurrence(g=g, b=b, w=w, **common)[0]

    def _op_chunk(self, q: torch.Tensor) -> int:
        """Chunk size of the recurrence kernels; the same function at any size
        dividing the padded length, only the rounding differs.

        INTEGRAL_OP_CHUNK pins it.  INTEGRAL_AUTO_OP_CHUNK=1 takes C16 for a C32
        mixer on GH200 unless the grid is small and the sequence long: the
        direct C16 readout is faster than the C32 composition there.
        """
        pinned = os.environ.get("INTEGRAL_OP_CHUNK")
        if pinned:
            return int(pinned)
        if (self.chunk_size == 32 and os.environ.get("INTEGRAL_AUTO_OP_CHUNK", "0") == "1"
                and q.is_cuda and torch.cuda.get_device_capability(q.device) == (9, 0)):
            batch, length, heads = q.shape[:3]
            if batch * heads >= 32 or length <= 4096:
                return 16
        return self.chunk_size

    def _output(
        self, hidden_states: torch.Tensor, output: torch.Tensor, g_down: torch.Tensor | None = None
    ) -> torch.Tensor:
        base = self.base
        gate = base.g_proj(hidden_states) if g_down is None else base.g_proj[1](g_down)
        gate = gate.reshape(
            *hidden_states.shape[:-1], base.num_v_heads, base.head_v_dim
        )
        output = base.o_norm(output, gate)
        return base.o_proj(output.reshape(*output.shape[:-2], base.value_dim))

    @property
    def kernel_mode(self) -> str:
        """``throughput-first`` or ``memory-first``; see ``kernel_mode.py``."""
        return resolve_kernel_mode({"kernel_mode": self._kernel_mode_setting})

    def _recomputes_readout_only(self, tensor: torch.Tensor) -> bool:
        """Whether memory-first recomputes only the readout operator here."""
        return (self.kernel_mode == MEMORY_FIRST and self.readout_gate
                and on_arch(tensor, READOUT_RECOMPUTE_ARCHS))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ):
        del output_attentions
        if attention_mask is not None or kwargs.get("cu_seqlens") is not None:
            raise NotImplementedError(
                "integral mixer supports dense fixed-length batches only"
            )
        if past_key_values is not None or use_cache:
            raise NotImplementedError("two-state generation cache is not implemented")

        original_length = hidden_states.shape[1]
        padding = (-original_length) % self.chunk_size
        if padding:
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding))

        # memory-first recomputes the body below in the backward pass instead
        # of storing what it produces (on READOUT_RECOMPUTE_ARCHS only the
        # readout operator, in _run_recurrence); throughput-first stores it.
        if (self.kernel_mode == MEMORY_FIRST and torch.is_grad_enabled()
                and not self._recomputes_readout_only(hidden_states)):
            if (os.environ.get("INTEGRAL_MF_NOGRAD_FWD", "0") == "1"
                    and hidden_states.requires_grad
                    and hidden_states.shape[:-1].numel() <= 16384):
                # The reentrant form runs the first pass without grad: no
                # autograd graph to record and no per-chunk states for the
                # backward (which recomputes them).  That pays where the host
                # bounds the step; on larger inputs its slower backward gives
                # back the forward saving.
                output = torch.utils.checkpoint.checkpoint(
                    self._mixer_body, hidden_states, use_reentrant=True
                )
            else:
                output = torch.utils.checkpoint.checkpoint(
                    self._mixer_body, hidden_states, use_reentrant=False
                )
        else:
            output = self._mixer_body(hidden_states)
        return output[:, :original_length], None, None

    def _mixer_body(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._output(hidden_states, *self._mixer_core(hidden_states))

    def _mixer_core(self, hidden_states: torch.Tensor):
        """The recurrence output and, when merged, the output-gate bottleneck."""
        projected = self._input_projections(hidden_states)
        if projected is None:
            q, k, v = self._reshape_qkv(*self._project_qkv(hidden_states))
            down = {}
        else:
            q, k, v = self._reshape_qkv(*(
                getattr(self.base, f"{name}_conv1d")(
                    x=projected[name], cache=None, output_final_state=False, cu_seqlens=None
                )[0]
                for name in "qkv"
            ))
            down = {"f_down": projected["f"]}
        gate_values = None
        if projected is not None:
            gate_logits = None
            gate_values = self.integration_gates.activate_down(projected["gamma"], projected["lam"])
        elif not self.fuse_preprocess and self.integration_gates.can_fuse_channel_pair(hidden_states):
            gate_logits = None
            gate_values = self.integration_gates(hidden_states)
        else:
            gate_logits = self.integration_gates.project(hidden_states)
        erase_scale = 2.0 if self.base.allow_neg_eigval else 1.0
        if self.model_type == "gdn2":
            b_logits = (self.base.b_proj(hidden_states) if projected is None else projected["b"]).reshape_as(k)
            w_logits = (self.base.w_proj(hidden_states) if projected is None else projected["w"]).reshape_as(v)
            if self.fuse_preprocess:
                q, k, b, w, gamma, lam = fused_channel_preprocess(
                    q,
                    k,
                    b_logits,
                    w_logits,
                    gate_logits,
                    chi_min=float(self.integration_gates.chi_min[0]),
                    b_scale=erase_scale,
                )
            else:
                q, k = _l2_normalize(q), _l2_normalize(k)
                if projected is not None:
                    b, w = erase_write_gates(projected["bw"], self.base.b_proj.out_features, erase_scale)
                    b, w = b.reshape_as(k), w.reshape_as(v)
                else:
                    b = b_logits.sigmoid()
                    if erase_scale != 1.0:  # a multiply by 1.0 is exact, but is one more pass
                        b = b * erase_scale
                    w = w_logits.sigmoid()
                gamma, lam = gate_values if gate_values is not None else self.integration_gates.activate(gate_logits)
            output = self._run_recurrence(
                hidden_states, q, k, v, gamma, lam, b=b, w=w, **down
            )
        else:
            beta_logits = (
                self.base.b_proj(hidden_states) if projected is None else projected["b"]
            ).reshape(q.shape[:-1])
            if self.fuse_preprocess:
                q, k, beta, gamma, lam = fused_scalar_preprocess(
                    q,
                    k,
                    beta_logits,
                    gate_logits,
                    chi_min=float(self.integration_gates.chi_min[0]),
                    beta_scale=erase_scale,
                )
            else:
                q, k = _l2_normalize(q), _l2_normalize(k)
                beta = beta_logits.sigmoid()
                if erase_scale != 1.0:
                    beta = beta * erase_scale
                gamma, lam = gate_values if gate_values is not None else self.integration_gates.activate(gate_logits)
            output = self._run_recurrence(
                hidden_states, q, k, v, gamma, lam, beta=beta, **down
            )
        return output, None if projected is None else projected["g"]


__all__ = ["IntegralMixer"]

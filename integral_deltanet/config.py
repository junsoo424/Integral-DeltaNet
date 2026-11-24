from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# The two backbones this repository publishes, each buildable on its own or
# with the integral readout (`mixer_implementation: integral`), which is
# what makes Integral KDA and Integral GDN-2.
MODEL_TYPES = {"kda", "gdn2"}
LLAMA2_TOKENIZER_ID = "meta-llama/Llama-2-7b-hf"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def validate_model_config(config: dict[str, Any]) -> None:
    required = {
        "model_type",
        "scale",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_heads",
        "head_dim",
        "expand_v",
        "intermediate_size",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"model config is missing: {', '.join(missing)}")
    if config["model_type"] not in MODEL_TYPES:
        raise ValueError(f"unknown model_type: {config['model_type']}")
    # The parameter-matched comparison pins every model to expand_v=1 and to a
    # head layout that tiles hidden_size exactly.  A config may opt out with
    # reference_shape=true when it deliberately reproduces an upstream
    # reference architecture instead of joining that comparison.
    reference_shape = bool(config.get("reference_shape", False))
    if not reference_shape:
        if config["hidden_size"] != config["num_heads"] * config["head_dim"]:
            raise ValueError("hidden_size must equal num_heads * head_dim")
        if config["expand_v"] != 1:
            raise ValueError("this comparison fixes expand_v=1 for all models")
    if config.get("use_mlp", True) is False and config["intermediate_size"] != 0:
        raise ValueError("a mixer-only block (use_mlp=false) must set intermediate_size: 0")
    # The FFN width is the parameter-matching knob.  A multiple of 8 keeps every
    # width tensor-core aligned while still letting a match land inside 0.05 % --
    # the granularity the GDN-2 reference configs themselves use (e.g. 6264).
    if config["intermediate_size"] % 8:
        raise ValueError("intermediate_size must be a multiple of 8")
    # YAML parses `1e-05` as a string (a float needs the dot, e.g. `1.0e-05`),
    # and a string eps reaches the Triton norm kernels as an unspecializable
    # argument -- a failure that surfaces only at the first forward pass.
    for name in ("norm_eps", "rope_theta"):
        value = config.get(name)
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}; "
                             f"write it as 1.0e-05, not 1e-05")
    if config["vocab_size"] != 32000:
        raise ValueError("all comparison configs must use the 32k Llama 2 vocabulary")
    implementation = config.get("mixer_implementation", "fla")
    if implementation not in {"fla", "integral"}:
        raise ValueError("mixer_implementation must be 'fla' or 'integral'")
    if implementation == "integral":
        if config.get("attn_mode", "chunk") != "chunk":
            raise ValueError("integral training requires attn_mode=chunk")
        if config.get("chunk_size") not in {16, 32, 64}:
            raise ValueError("integral chunk_size must be 16, 32, or 64")
        if config.get("num_v_heads") not in {None, config["num_heads"]}:
            raise ValueError("integral training requires num_v_heads == num_heads")
        if config["model_type"] == "deltanet" and config.get("qk_norm", "l2") != "l2":
            raise ValueError("integral DeltaNet requires qk_norm=l2")
        gate_mode = config.get("integration_gate_mode", "learned")
        if gate_mode not in {"learned", "fixed"}:
            raise ValueError("integration_gate_mode must be learned or fixed")
        bias_init = config.get("integration_gate_bias_init", "timescale")
        if bias_init not in {"timescale", "zero"}:
            raise ValueError("integration_gate_bias_init must be timescale or zero")
        readout_gate = config.get("integration_readout_gate", False)
        if not isinstance(readout_gate, bool):
            raise ValueError("integration_readout_gate must be a boolean")
        coupling = config.get(
            "integration_gate_coupling", "independent" if readout_gate else "ordered"
        )
        if coupling not in {"ordered", "independent"}:
            raise ValueError("integration_gate_coupling must be ordered or independent")
        gamma_gate = config.get("integration_gamma_gate", "scalar")
        if gamma_gate not in {"scalar", "channel"}:
            raise ValueError("integration_gamma_gate must be scalar or channel")
        if gamma_gate == "channel" and not readout_gate:
            raise ValueError("a per-channel gamma is only defined for readout-side integration")
        if gamma_gate == "channel" and coupling != "independent":
            raise ValueError("a per-channel gamma requires independent gate coupling")
        lambda_gate = config.get("integration_lambda_gate", "scalar")
        if lambda_gate not in {"scalar", "channel"}:
            raise ValueError("integration_lambda_gate must be scalar or channel")
        if lambda_gate == "channel" and not (readout_gate and gamma_gate == "channel"):
            raise ValueError("a per-channel lambda needs readout-side integration with a channel gamma")
        gamma_rank = config.get("integration_gamma_rank")
        if gamma_rank is not None:
            if isinstance(gamma_rank, bool) or not isinstance(gamma_rank, int) or gamma_rank <= 0:
                raise ValueError("integration_gamma_rank must be a positive integer")
            if gamma_gate != "channel":
                raise ValueError("integration_gamma_rank needs integration_gamma_gate: channel")
        if coupling == "ordered":
            tau_gamma = config.get("integration_tau_gamma", 64.0)
            tau_lambda = config.get("integration_tau_lambda", 1024.0)
            chi_min = config.get("integration_chi_min", 4.0)
            if not any(isinstance(v, list) for v in (tau_gamma, tau_lambda, chi_min)):
                if tau_lambda / tau_gamma <= chi_min:
                    raise ValueError("ordered gates need tau_lambda / tau_gamma > chi_min")
        for name in ("integration_chi_min", "integration_tau_gamma", "integration_tau_lambda"):
            value = config.get(name)
            if isinstance(value, list) and len(value) != config["num_heads"]:
                raise ValueError(f"{name} list must have num_heads entries")


def validate_training_config(config: dict[str, Any]) -> None:
    for section in ("data", "optimization", "runtime", "checkpoint"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"training config needs a '{section}' mapping")
    data = config["data"]
    opt = config["optimization"]
    runtime = config["runtime"]
    for section, keys in (
        ("data", ("sequence_length", "shuffle_buffer")),
        ("optimization", ("max_steps", "target_tokens")),
        ("runtime", ("expected_world_size", "micro_batch_size", "gradient_accumulation_steps",
                     "global_batch_tokens", "log_every")),
        ("checkpoint", ("save_every", "keep_last")),
    ):
        for key in keys:
            value = config[section][key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer")
    if type(opt["warmup_steps"]) is not int or opt["warmup_steps"] < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if runtime["precision"] != "bf16":
        raise ValueError("this training loop requires runtime.precision=bf16")
    if data["tokenizer"] != LLAMA2_TOKENIZER_ID:
        raise ValueError(
            f"the comparison is fixed to the Llama 2 tokenizer ({LLAMA2_TOKENIZER_ID})"
        )
    if data.get("document_separator", "bos") not in {"bos", "eos"}:
        raise ValueError("data.document_separator must be 'bos' or 'eos'")
    if data["sequence_length"] % 256:
        raise ValueError("sequence_length must be a multiple of 256")
    global_tokens = (
        runtime["expected_world_size"]
        * runtime["micro_batch_size"]
        * runtime["gradient_accumulation_steps"]
        * data["sequence_length"]
    )
    if global_tokens != runtime["global_batch_tokens"]:
        raise ValueError(f"global batch mismatch: derived {global_tokens:,}")
    if opt["warmup_steps"] >= opt["max_steps"]:
        raise ValueError("warmup_steps must be smaller than max_steps")

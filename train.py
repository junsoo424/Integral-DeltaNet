from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchdata.stateful_dataloader import StatefulDataLoader

from integral_deltanet.config import (
    load_yaml,
    validate_model_config,
    validate_training_config,
)
from integral_deltanet.data import FineWebEduTokens
from integral_deltanet.model import CausalLM, learning_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Integral DeltaNet model or its backbone")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint path or 'auto'")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def resolve_num_workers(runtime_config: dict[str, Any], override: int | None) -> int:
    """Validate the loader policy before allocating GPUs and record its effective value."""
    workers = runtime_config["num_workers"] if override is None else override
    if type(workers) is not int or workers != 0:
        raise ValueError(
            "FineWebEduTokens requires num_workers=0 for checkpoint-exact resume; "
            "set runtime.num_workers: 0 or pass --num-workers 0"
        )
    runtime_config["num_workers"] = workers
    return workers


def resolve_runtime_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Record CLI overrides before validation, checkpointing, and logging."""
    for argument, section, key in (
        ("max_steps", "optimization", "max_steps"),
        ("micro_batch_size", "runtime", "micro_batch_size"),
        ("gradient_accumulation_steps", "runtime", "gradient_accumulation_steps"),
    ):
        value = getattr(args, argument)
        if value is not None:
            config[section][key] = value
    resolve_num_workers(config["runtime"], args.num_workers)
    validate_training_config(config)


def validate_resume_config(state: dict[str, Any], model_config: dict[str, Any],
                           training_config: dict[str, Any]) -> None:
    # Scale labels and expected counts are metadata; they do not change a model.
    descriptive = {"scale", "expected_parameters", "expected_body_parameters"}
    saved_model = {k: v for k, v in state["model_config"].items() if k not in descriptive}
    current_model = {k: v for k, v in model_config.items() if k not in descriptive}
    mismatches = []
    if saved_model != current_model:
        mismatches.append("model_config")
    previous = state["training_config"]
    for section in ("data", "optimization"):
        if previous[section] != training_config[section]:
            mismatches.append(section)
    # Older checkpoints recorded YAML workers=4 despite the CLI using 0.
    # Loader format is checked separately. Logging/save frequency may change.
    for key in ("precision", "expected_world_size", "micro_batch_size",
                "gradient_accumulation_steps", "global_batch_tokens", "seed"):
        if previous["runtime"].get(key) != training_config["runtime"].get(key):
            mismatches.append(f"runtime.{key}")
    if mismatches:
        raise ValueError("resume configuration changed: " + ", ".join(mismatches))


def create_data_iterator(loader: StatefulDataLoader, *, resuming: bool):
    # StatefulDataLoader samples a fresh CPU base seed even when restoring a
    # single-process iterator. That construction must not advance training RNG.
    context = torch.random.fork_rng(devices=[]) if resuming else nullcontext()
    with context:
        return iter(loader)


def trim_metrics_for_resume(output_dir: Path, step: int) -> Path | None:
    """Archive rolled-back metrics, then atomically keep only checkpointed steps."""
    path = output_dir / "metrics.jsonl"
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = {}
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
            row_step = int(row["step"])
        except (ValueError, KeyError, TypeError):
            if index != len(lines) - 1:
                raise ValueError(f"invalid metrics record at {path}:{index + 1}")
            continue  # A terminated write can leave one incomplete final line.
        if row_step <= step:
            kept[row_step] = line.rstrip("\n") + "\n"
    retained = "".join(kept[key] for key in sorted(kept))
    if retained == "".join(lines):
        return None
    archive = output_dir / f"metrics.before-resume-{step}-{time.time_ns()}.jsonl"
    shutil.copy2(path, archive)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(retained, encoding="utf-8")
    os.replace(temporary, path)
    return archive


def distributed_setup() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required by the FLA Triton training kernels")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or getattr(parameter, "_no_weight_decay", False):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(output_dir: Path) -> Path | None:
    paths = list(output_dir.glob("checkpoint-*.pt"))
    return max(paths, key=checkpoint_step) if paths else None


def iter_named_tensors(value: Any, prefix: str = ""):
    """Yield tensors from nested checkpoint state with stable diagnostic names."""

    if isinstance(value, torch.Tensor):
        yield prefix or "<root>", value
    elif isinstance(value, dict):
        for key, nested in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_named_tensors(nested, name)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            name = f"{prefix}.{index}" if prefix else str(index)
            yield from iter_named_tensors(nested, name)


def assert_finite_named_tensors(named_tensors, *, stage: str) -> None:
    """Reject poisoned model/checkpoint tensors before they can propagate."""

    for name, value in named_tensors:
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite tensor at {stage}: {name}")


def assert_finite_model_distributed(
    model: torch.nn.Module,
    *,
    stage: str,
    world_size: int,
) -> None:
    """Make every DDP rank reject a poisoned model before checkpoint I/O."""

    named_tensors = tuple(model.state_dict().items())
    device = next(model.parameters()).device
    finite = torch.ones((), dtype=torch.int32, device=device)
    for _, value in named_tensors:
        if value.is_floating_point():
            finite *= torch.isfinite(value).all().to(device=device, dtype=torch.int32)
    if world_size > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if bool(finite.item()):
        return

    local_bad = next(
        (
            name
            for name, value in named_tensors
            if value.is_floating_point() and not bool(torch.isfinite(value).all())
        ),
        None,
    )
    location = local_bad if local_bad is not None else "another DDP rank"
    raise FloatingPointError(f"non-finite tensor at {stage}: {location}")


def assert_finite_optimizer_distributed(
    optimizer: torch.optim.Optimizer,
    *,
    stage: str,
    world_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Reject non-finite optimizer moments collectively before checkpoint I/O."""

    state = optimizer.state_dict()
    named_tensors = tuple(
        iter_named_tensors(state, prefix="optimizer")
    )
    finite = torch.ones((), dtype=torch.int32, device=device)
    for _, value in named_tensors:
        if value.is_floating_point():
            finite *= torch.isfinite(value).all().to(
                device=device, dtype=torch.int32
            )
    if world_size > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if bool(finite.item()):
        return state

    local_bad = next(
        (
            name
            for name, value in named_tensors
            if value.is_floating_point()
            and not bool(torch.isfinite(value).all())
        ),
        None,
    )
    location = local_bad if local_bad is not None else "another DDP rank"
    raise FloatingPointError(f"non-finite tensor at {stage}: {location}")


def assert_finite_loss(
    loss: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    step: int,
    micro_step: int,
) -> None:
    """Fail every DDP rank together if any rank observes a non-finite loss."""

    finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
    if world_size > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        local_loss = float(loss.detach())
        raise FloatingPointError(
            f"non-finite loss at step={step + 1}, micro_step={micro_step + 1}, "
            f"rank={rank}: {local_loss}"
        )


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: StatefulDataLoader,
    rank: int,
    world_size: int,
    keep_last: int,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> None:
    raw_model = (
        model.module if isinstance(model, DistributedDataParallel) else model
    )
    assert_finite_model_distributed(
        raw_model, stage=f"checkpoint-{step}", world_size=world_size
    )
    optimizer_state = assert_finite_optimizer_distributed(
        optimizer,
        stage=f"checkpoint-{step}:optimizer",
        world_size=world_size,
        device=next(raw_model.parameters()).device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_path = output_dir / f"dataloader-{step}-rank{rank}.pt"
    rank_tmp = rank_path.with_suffix(".tmp")
    torch.save(
        {
            "loader": loader.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "world_size": world_size,
        },
        rank_tmp,
    )
    os.replace(rank_tmp, rank_path)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        path = output_dir / f"checkpoint-{step}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "step": step,
                "model": raw_model.state_dict(),
                "optimizer": optimizer_state,
                "world_size": world_size,
                "model_config": model_config,
                "training_config": training_config,
            },
            temporary,
        )
        os.replace(temporary, path)
        old = sorted(output_dir.glob("checkpoint-*.pt"), key=checkpoint_step)[:-keep_last]
        for old_path in old:
            old_step = checkpoint_step(old_path)
            old_path.unlink(missing_ok=True)
            for data_path in output_dir.glob(f"dataloader-{old_step}-rank*.pt"):
                data_path.unlink(missing_ok=True)
    if world_size > 1:
        dist.barrier()


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: StatefulDataLoader,
    rank: int,
    world_size: int,
    device: torch.device,
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
) -> int:
    state = torch.load(path, map_location=device, weights_only=False)
    if state["world_size"] != world_size:
        raise ValueError("resume requires the same world size as the saved run")
    if model_config is not None and training_config is not None:
        validate_resume_config(state, model_config, training_config)
    assert_finite_named_tensors(
        state["model"].items(), stage=f"resume:{path}:model"
    )
    assert_finite_named_tensors(
        iter_named_tensors(state["optimizer"], prefix="optimizer"),
        stage=f"resume:{path}:optimizer",
    )
    rank_path = path.parent / f"dataloader-{state['step']}-rank{rank}.pt"
    if not rank_path.exists():
        raise FileNotFoundError(f"missing rank-local dataloader state: {rank_path}")
    rank_state = torch.load(rank_path, map_location="cpu", weights_only=False)
    if rank_state.get("world_size") != world_size:
        raise ValueError(f"rank-local world size differs from checkpoint: {rank_path}")
    if "_snapshot" in rank_state["loader"]:
        raise ValueError("resume expects a num_workers=0 dataloader checkpoint")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    loader.load_state_dict(rank_state["loader"])
    random.setstate(rank_state["python_rng"])
    torch.set_rng_state(rank_state["torch_rng"])
    torch.cuda.set_rng_state(rank_state["cuda_rng"], device)
    return int(state["step"])


def main() -> None:
    args = parse_args()
    model_config = load_yaml(args.model_config)
    training_config = load_yaml(args.training_config)
    validate_model_config(model_config)
    resolve_runtime_config(training_config, args)
    num_workers = training_config["runtime"]["num_workers"]
    expected_world_size = training_config["runtime"]["expected_world_size"]
    if int(os.environ.get("WORLD_SIZE", "1")) != expected_world_size:
        raise ValueError(f"WORLD_SIZE must match runtime.expected_world_size={expected_world_size}")
    rank, local_rank, world_size, device = distributed_setup()

    data_config = training_config["data"]
    opt_config = training_config["optimization"]
    runtime_config = training_config["runtime"]
    checkpoint_config = training_config["checkpoint"]
    max_steps = opt_config["max_steps"]
    micro_batch_size = runtime_config["micro_batch_size"]
    accumulation_steps = runtime_config["gradient_accumulation_steps"]
    seed_everything(runtime_config["seed"], rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dataset = FineWebEduTokens(
        dataset_name=data_config["dataset"],
        dataset_config=data_config["dataset_config"],
        tokenizer_name=data_config["tokenizer"],
        sequence_length=data_config["sequence_length"],
        shuffle_buffer=data_config["shuffle_buffer"],
        seed=runtime_config["seed"],
        rank=rank,
        world_size=world_size,
        document_separator=data_config.get("document_separator", "bos"),
    )
    loader = StatefulDataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        snapshot_every_n_steps=100,
    )

    with torch.device(device):
        raw_model = CausalLM(model_config)
    optimizer = torch.optim.AdamW(
        parameter_groups(raw_model, opt_config["weight_decay"]),
        lr=opt_config["peak_lr"],
        betas=tuple(opt_config["betas"]),
        eps=opt_config["eps"],
        fused=True,
    )
    model: torch.nn.Module = raw_model
    if world_size > 1:
        model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = latest_checkpoint(output_dir) if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    if args.resume is None and latest_checkpoint(output_dir) is not None:
        raise ValueError("output directory already has checkpoints; pass --resume auto")
    start_step = 0
    if resume_path is not None:
        start_step = load_checkpoint(
            resume_path, raw_model, optimizer, loader, rank, world_size, device,
            model_config, training_config,
        )
    data_iterator = create_data_iterator(loader, resuming=resume_path is not None)
    if resume_path is not None and rank == 0:
        print(f"[resume] loaded {resume_path} at step={start_step}; "
              f"num_workers={num_workers}, world_size={world_size}", flush=True)

    actual_global_batch = world_size * micro_batch_size * accumulation_steps * data_config["sequence_length"]
    if rank == 0:
        if resume_path is not None:
            archive = trim_metrics_for_resume(output_dir, start_step)
            if archive is not None:
                print(f"[resume] archived rolled-back metrics: {archive}", flush=True)
        run_metadata = {
            "model_config": model_config,
            "training_config": training_config,
            "world_size": world_size,
            "actual_global_batch_tokens": actual_global_batch,
            "parameters": raw_model.num_parameters(),
        }
        with (output_dir / "run.json").open("w", encoding="utf-8") as handle:
            json.dump(run_metadata, handle, indent=2)
        print(json.dumps(run_metadata), flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_time = time.perf_counter()
    last_log_step = start_step
    for step in range(start_step, max_steps):
        total_loss = torch.zeros((), device=device)
        for micro_step in range(accumulation_steps):
            input_ids = next(data_iterator).to(device, non_blocking=True)
            synchronize = micro_step == accumulation_steps - 1
            sync_context = nullcontext() if synchronize or world_size == 1 else model.no_sync()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=input_ids, labels=input_ids)
                    scaled_loss = loss / accumulation_steps
                assert_finite_loss(
                    loss,
                    rank=rank,
                    world_size=world_size,
                    step=step,
                    micro_step=micro_step,
                )
                scaled_loss.backward()
            total_loss += loss.detach()

        lr = learning_rate(
            step,
            max_steps,
            opt_config["warmup_steps"],
            opt_config["peak_lr"],
            opt_config["min_lr"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            opt_config["max_grad_norm"],
            error_if_nonfinite=True,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if world_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        mean_loss = total_loss.item() / (accumulation_steps * world_size)
        completed_step = step + 1
        if rank == 0 and completed_step % runtime_config["log_every"] == 0:
            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            metrics = {
                "step": completed_step,
                "tokens": completed_step * actual_global_batch,
                "loss": mean_loss,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "tokens_per_second": actual_global_batch * (completed_step - last_log_step) / elapsed,
            }
            last_log_step = completed_step
            print(json.dumps(metrics), flush=True)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")

        if completed_step % checkpoint_config["save_every"] == 0 or completed_step == max_steps:
            save_checkpoint(
                output_dir,
                completed_step,
                model,
                optimizer,
                loader,
                rank,
                world_size,
                checkpoint_config["keep_last"],
                model_config,
                training_config,
            )

    # A finite training run leaves a streaming parquet reader unfinished.
    # Close it while Python and Arrow are alive instead of at interpreter exit.
    dataset.close()
    del data_iterator, loader, dataset
    gc.collect()
    torch.cuda.synchronize(device)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

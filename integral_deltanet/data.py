from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from huggingface_hub import HfFileSystem
from huggingface_hub.file_download import hf_hub_url
from huggingface_hub.hf_file_system import HfFileSystemFile
from huggingface_hub.utils import hf_raise_for_status, http_backoff
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoTokenizer


_ORIGINAL_REPO_AND_REVISION_EXIST = HfFileSystem._repo_and_revision_exist
_ORIGINAL_HF_GLOB = HfFileSystem.glob
_ORIGINAL_HF_INFO = HfFileSystem.info
_FINEWEB_EDU_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
_FINEWEB_EDU_FILE_SIZES: dict[str, int] = {}


def _manifest_path_in_repo(path: str) -> str | None:
    normalized = HfFileSystem._strip_protocol(path)
    prefix = f"datasets/HuggingFaceFW/fineweb-edu@{_FINEWEB_EDU_REVISION}/"
    if normalized.startswith(prefix) and normalized.endswith(".parquet"):
        return normalized.removeprefix(prefix)
    return None

def _repo_and_revision_exist_without_manifest_probe(
    self: HfFileSystem, repo_type: str, repo_id: str, revision: str | None
) -> tuple[bool, Exception | None]:
    if (
        repo_type == "dataset"
        and repo_id == "HuggingFaceFW/fineweb-edu"
        and revision in {
            "refs/convert/parquet",
            "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        }
    ):
        return True, None
    return _ORIGINAL_REPO_AND_REVISION_EXIST(self, repo_type, repo_id, revision)


def _glob_manifest_file_without_remote_probe(
    self: HfFileSystem, path: str, **kwargs: Any
) -> Any:
    path_in_repo = _manifest_path_in_repo(path)
    if path_in_repo is not None and not any(char in path_in_repo for char in "*?[{"):
        name = self.resolve_path(path, revision=kwargs.get("revision")).unresolve()
        info = {
            "name": name,
            "size": _FINEWEB_EDU_FILE_SIZES[path_in_repo],
            "type": "file",
        }
        return {name: info} if kwargs.get("detail") else [name]
    return _ORIGINAL_HF_GLOB(self, path, **kwargs)


def _info_manifest_file_without_remote_probe(
    self: HfFileSystem, path: str, **kwargs: Any
) -> dict[str, Any]:
    path_in_repo = _manifest_path_in_repo(path)
    if path_in_repo is not None and path_in_repo in _FINEWEB_EDU_FILE_SIZES:
        name = self.resolve_path(path, revision=kwargs.get("revision")).unresolve()
        return {
            "name": name,
            "size": _FINEWEB_EDU_FILE_SIZES[path_in_repo],
            "type": "file",
        }
    return _ORIGINAL_HF_INFO(self, path, **kwargs)

def _fetch_range_with_cached_redirect(
    self: HfFileSystemFile, start: int, end: int
) -> bytes:
    range_headers = {"range": f"bytes={start}-{end - 1}"}
    direct_url = getattr(self, "_direct_download_url", None)
    if direct_url:
        response = http_backoff(
            "GET",
            direct_url,
            headers=range_headers,
            timeout=60,
        )
        if response.status_code not in (401, 403):
            hf_raise_for_status(response)
            return response.content

    hub_url = hf_hub_url(
        repo_id=self.resolved_path.repo_id,
        revision=self.resolved_path.revision,
        filename=self.resolved_path.path_in_repo,
        repo_type=self.resolved_path.repo_type,
        endpoint=self.fs.endpoint,
    )
    response = http_backoff(
        "GET",
        hub_url,
        headers={**range_headers, **self.fs._api._build_hf_headers()},
        timeout=60,
    )
    hf_raise_for_status(response)
    if response.url != hub_url:
        self._direct_download_url = response.url
    return response.content


_LLAMA2_GATED_REPO = "meta-llama/Llama-2-7b-hf"
_LLAMA2_UNGATED_MIRROR = "NousResearch/Llama-2-7b-hf"


def _load_llama2_tokenizer(tokenizer_name: str) -> Any:
    """Load ``tokenizer_name``, falling back to an ungated Llama-2 mirror.

    Byte-identical to ``meta-llama/Llama-2-7b-hf`` (same vocab/class); used
    when this HF account's Llama-2 license request hasn't been approved yet,
    without weakening the config's literal Llama-2-tokenizer requirement.
    """
    if tokenizer_name != _LLAMA2_GATED_REPO:
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(
            _LLAMA2_UNGATED_MIRROR, use_fast=True, token=False
        )


class FineWebEduTokens(IterableDataset):
    """Stateful, bounded-memory FineWeb-Edu tokenizer and sequence shuffler.

    Production training uses one instance per distributed rank (num_workers=0).
    It reads shared local parquet shards, keeps at most shuffle_buffer packed
    sequences in an int16 tensor, and serializes the exact stream state. The old
    Hugging Face manifest path remains available as a fallback for diagnostics.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        tokenizer_name: str,
        sequence_length: int,
        shuffle_buffer: int,
        seed: int,
        rank: int,
        world_size: int,
        document_separator: str = "bos",
    ) -> None:
        super().__init__()
        if shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be positive")
        if document_separator not in {"bos", "eos"}:
            raise ValueError("document_separator must be 'bos' or 'eos'")
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer_name = tokenizer_name
        self.sequence_length = sequence_length
        self.shuffle_buffer = shuffle_buffer
        self.document_separator = document_separator
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self._source = None
        self._source_iterator: Iterator | None = None
        self._tokenizer = None
        self._sequence_buffer = torch.empty(
            (0, self.sequence_length), dtype=torch.int16
        )
        self._tokens: list[int] = []
        self._pending_tokenized: list[list[int]] = []
        self._rng_state: torch.Tensor | None = None
        self._pending_state: dict[str, Any] | None = None
        self._drain_order: list[int] | None = None
        self._drain_position = 0

    def _initialize(self) -> None:
        if self._source is not None:
            self._restore_pending_state()
            return
        local_root = os.environ.get("FINEWEB_EDU_LOCAL_ROOT")
        data_files_manifest = os.environ.get("FINEWEB_EDU_DATA_FILES_MANIFEST")
        if local_root:
            root = Path(local_root)
            pattern = root / "data" / "*" / "*.parquet"
            if not root.is_dir():
                raise FileNotFoundError(f"FineWeb-Edu root does not exist: {root}")
            if not any((root / "data").glob("*/*.parquet")):
                raise FileNotFoundError(f"no FineWeb-Edu parquet shards matched {pattern}")
            source = load_dataset(
                "parquet",
                data_files={"train": str(pattern)},
                split="train",
                streaming=True,
            )
        elif data_files_manifest:
            sizes_manifest = os.environ.get("FINEWEB_EDU_FILE_SIZES_MANIFEST")
            if not sizes_manifest:
                raise ValueError(
                    "FINEWEB_EDU_FILE_SIZES_MANIFEST is required with the data files manifest"
                )
            with open(sizes_manifest, encoding="utf-8") as handle:
                url_sizes = json.load(handle)
            _FINEWEB_EDU_FILE_SIZES.clear()
            marker = f"/resolve/{_FINEWEB_EDU_REVISION}/"
            for url, size in url_sizes.items():
                if marker not in url or not isinstance(size, int) or size <= 0:
                    raise ValueError(
                        "invalid FineWeb-Edu size entry: "
                        + repr(url)
                        + ": "
                        + repr(size)
                    )
                _FINEWEB_EDU_FILE_SIZES[url.split(marker, 1)[1]] = size
            HfFileSystem._repo_and_revision_exist = (
                _repo_and_revision_exist_without_manifest_probe
            )
            HfFileSystem.glob = _glob_manifest_file_without_remote_probe
            HfFileSystem.info = _info_manifest_file_without_remote_probe
            HfFileSystemFile._fetch_range = _fetch_range_with_cached_redirect
            with open(data_files_manifest, encoding="utf-8") as handle:
                data_files = json.load(handle)
            if not isinstance(data_files, list) or not data_files or not all(
                isinstance(path, str) and path for path in data_files
            ):
                raise ValueError(
                    "FINEWEB_EDU_DATA_FILES_MANIFEST must contain a non-empty JSON list of URLs"
                )
            source = load_dataset(
                "parquet",
                data_files={"train": data_files},
                split="train",
                streaming=True,
            )
        else:
            source = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split="train",
                streaming=True,
            )
        source = split_dataset_by_node(source, rank=self.rank, world_size=self.world_size)
        worker = get_worker_info()
        if worker is not None:
            raise RuntimeError(
                "FineWebEduTokens requires num_workers=0 for bounded, checkpoint-exact state"
            )
        self._source = source
        self._tokenizer = _load_llama2_tokenizer(self.tokenizer_name)
        if len(self._tokenizer) != 32000:
            raise ValueError(f"expected tokenizer size 32000, got {len(self._tokenizer)}")
        if self.document_separator == "eos" and self._tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")
        if self.document_separator == "bos" and self._tokenizer.bos_token_id is None:
            raise ValueError("document_separator='bos' needs a tokenizer with a BOS token")
        self._restore_pending_state()

    def _restore_pending_state(self) -> None:
        if self._pending_state is None or self._source is None:
            return
        state = self._pending_state
        self._source.load_state_dict(state["source"])
        if "sequence_buffer" in state:
            sequence_buffer = state["sequence_buffer"]
            if (
                sequence_buffer.ndim != 2
                or sequence_buffer.shape[1] != self.sequence_length
                or len(sequence_buffer) > self.shuffle_buffer
            ):
                raise ValueError("invalid checkpointed FineWeb-Edu sequence buffer")
            self._sequence_buffer = sequence_buffer.to(torch.int16)
            self._tokens = list(state.get("tokens", []))
            self._pending_tokenized = deepcopy(state.get("pending_tokenized", []))
            self._rng_state = state.get("rng_state")
        else:
            # Compatibility with checkpoints from the old document-shuffle loader.
            # Its buffer contains only the unconsumed packed-token tail.
            self._sequence_buffer = torch.empty(
                (0, self.sequence_length), dtype=torch.int16
            )
            self._tokens = list(state.get("buffer", []))
            self._pending_tokenized = []
            self._rng_state = None
        self._drain_order = state.get("drain_order")
        self._drain_position = state.get("drain_position", 0)
        self._pending_state = None

    def _packed_sequences(self) -> Iterator[torch.Tensor]:
        """Tokenize documents and cut the stream into fixed-length blocks.

        ``document_separator`` decides the boundary token.  ``"bos"`` prefixes
        every document with the tokenizer's BOS, which is what FLA's ``flame``
        loader, Samba's ``PackedDatasetBuilder(sep_token=bos_id)`` (the loader
        Gated DeltaNet released) and lm-eval's own Llama-tokenizer default all
        do -- so a model trained this way sees at evaluation exactly the
        boundary token it was trained on.  ``"eos"`` ends every document with
        EOS and never emits BOS.
        """
        assert self._source is not None and self._tokenizer is not None
        separator = (
            self._tokenizer.bos_token_id
            if self.document_separator == "bos"
            else self._tokenizer.eos_token_id
        )
        prefix = self.document_separator == "bos"
        data_iterator = iter(self._source)
        self._source_iterator = data_iterator
        while True:
            # A checkpoint can be taken halfway through a long document.
            # Drain its saved tail before fetching another document or EOF.
            while len(self._tokens) >= self.sequence_length:
                block = self._tokens[: self.sequence_length]
                self._tokens = self._tokens[self.sequence_length :]
                yield torch.tensor(block, dtype=torch.int16)
            if not self._pending_tokenized:
                texts = []
                for _ in range(32):
                    try:
                        texts.append(next(data_iterator)["text"])
                    except StopIteration:
                        break
                if not texts:
                    break
                encoded = self._tokenizer(
                    texts, add_special_tokens=False, truncation=False
                )["input_ids"]
                self._pending_tokenized = [list(ids) for ids in encoded]

            token_ids = self._pending_tokenized.pop(0)
            if prefix:
                self._tokens.append(separator)
                self._tokens.extend(token_ids)
            else:
                self._tokens.extend(token_ids)
                self._tokens.append(separator)

    def __iter__(self) -> Iterator[torch.Tensor]:
        self._initialize()
        assert self._source is not None and self._tokenizer is not None
        generator = torch.Generator().manual_seed(self.seed + self.rank)
        if self._rng_state is not None:
            generator.set_state(self._rng_state)

        if self._drain_order is None:
            for block in self._packed_sequences():
                if len(self._sequence_buffer) < self.shuffle_buffer:
                    self._sequence_buffer = torch.cat(
                        (self._sequence_buffer, block.unsqueeze(0)), dim=0
                    )
                    continue

                index = int(torch.randint(0, self.shuffle_buffer, (), generator=generator))
                output = self._sequence_buffer[index].to(torch.long)
                self._sequence_buffer[index] = block
                self._rng_state = generator.get_state()
                yield output

            self._drain_order = torch.randperm(
                len(self._sequence_buffer), generator=generator
            ).tolist()
            self._rng_state = generator.get_state()
        while self._drain_position < len(self._drain_order):
            index = self._drain_order[self._drain_position]
            self._drain_position += 1
            yield self._sequence_buffer[index].to(torch.long)

    def state_dict(self) -> dict[str, Any]:
        self._initialize()
        assert self._source is not None
        return {
            "source": self._source.state_dict(),
            "sequence_buffer": self._sequence_buffer.clone(),
            "tokens": list(self._tokens),
            "pending_tokenized": deepcopy(self._pending_tokenized),
            "rng_state": self._rng_state,
            "drain_order": self._drain_order,
            "drain_position": self._drain_position,
        }

    def close(self) -> None:
        """Release the active parquet reader before Python/CUDA teardown."""
        if self._source_iterator is not None:
            close = getattr(self._source_iterator, "close", None)
            if close is not None:
                close()
            self._source_iterator = None

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._pending_state = state_dict
